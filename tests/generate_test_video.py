"""用 FFmpeg 合成「已知每一格內容」的測試影片，驗證 AT-008（單格精確）與 AT-009（影音同步）。

畫面設計
--------
- 最上方 24px 條碼列：16 個黑白方塊編碼 **原始 0-based 影格編號**（bit k 在第 k 塊），
  可在輸出檔逐格自動解碼，不必 OCR。
- 下方 drawtext 印出 ``F <frame_num>``，方便人眼檢查。
- 廣告區間 ``[start, end)``：整個畫面被紅色圖卡取代（含移動中的黃色方塊模擬局部動畫），
  條碼列仍保留，因此可以確認「廣告格 0 殘留」。

聲音設計
--------
- 正片：每個整數秒開頭響 0.1 秒 1kHz 嗶聲（嗶聲起點 = 第 k*fps 格的時間）。
- 廣告：連續 440Hz 音。

``--vfr-drop-mod M``：丟掉 ``n % M == M//2`` 的影格並保留原時間戳 -> 產生 VFR 素材（AT-012）。

用法::

    uv run python tests/generate_test_video.py --out tests/output/synthetic_ads.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

CODE_BITS = 16
CODE_STRIP_HEIGHT = 24

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


@dataclass
class SyntheticSpec:
    duration: int = 150  # 秒（需為整數，總影格數 = duration * fps）
    fps: int = 30
    width: int = 640
    height: int = 360
    ad_intervals: list[tuple[int, int]] = field(default_factory=lambda: [(1234, 1984), (3001, 3751)])
    sample_rate: int = 48000
    beep_hz: int = 1000
    beep_len: float = 0.1
    ad_tone_hz: int = 440
    vfr_drop_mod: int = 0  # > 0 時產生 VFR
    gop: int = 250  # 長 GOP，確保剪輯點不會剛好落在關鍵格

    @property
    def total_frames(self) -> int:
        return self.duration * self.fps

    def validate(self) -> None:
        if self.width % CODE_BITS:
            raise ValueError(f"寬度需為 {CODE_BITS} 的倍數（條碼方塊等寬）")
        if self.total_frames >= 2**CODE_BITS:
            raise ValueError(f"總影格數需 < {2**CODE_BITS}")
        for s, e in self.ad_intervals:
            if not 0 <= s < e <= self.total_frames:
                raise ValueError(f"廣告區間 [{s}, {e}) 超出範圍 [0, {self.total_frames}]")

    def is_ad_frame(self, original_index: int) -> bool:
        return any(s <= original_index < e for s, e in self.ad_intervals)


def _font_option() -> str:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            # 在單引號內，"\:" 會在選項解析層被還原為 ":"（Windows 磁碟代號）
            return f"fontfile='{candidate.replace(':', chr(92) + ':')}':"
    return ""  # 交給 fontconfig 預設字型


def build_filtergraph(spec: SyntheticSpec) -> str:
    d, r, w, h = spec.duration, spec.fps, spec.width, spec.height
    size = f"{w}x{h}"
    font = _font_option()
    enable = "+".join(f"between(n,{s},{e - 1})" for s, e in spec.ad_intervals) or "0"

    parts = [
        f"testsrc2=s={size}:r={r}:d={d},format=yuv420p[main]",
        (
            f"color=c=0xB03A2E:s={size}:r={r}:d={d},format=yuv420p,"
            f"drawtext={font}text='ADVERTISEMENT':fontcolor=white:fontsize={h // 7}"
            f":x=(w-tw)/2:y=(h-th)/2,"
            f"drawbox=x='mod(t*240,iw-80)':y=ih-90:w=80:h=40:color=yellow:t=fill[adcard]"
        ),
        f"[main][adcard]overlay=0:0:enable='{enable}'[base]",
        (
            f"nullsrc=s={CODE_BITS}x2:r={r}:d={d},format=gray,"
            f"geq=lum='if(mod(floor(N/pow(2,X)),2),255,0)',"
            f"scale={w}:{CODE_STRIP_HEIGHT}:flags=neighbor,format=yuv420p[code]"
        ),
    ]
    vfr = ""
    if spec.vfr_drop_mod > 0:
        m = spec.vfr_drop_mod
        vfr = f",select='not(eq(mod(n,{m}),{m // 2}))'"
    parts.append(
        f"[base][code]overlay=0:0,"
        f"drawtext={font}text='F %{{frame_num}}':fontcolor=white:fontsize={h // 12}"
        f":box=1:boxcolor=black@0.6:x=10:y={CODE_STRIP_HEIGHT + 8}{vfr}[vout]"
    )

    main_expr = f"0.5*sin(2*PI*{spec.beep_hz}*t)*lt(mod(t,1),{spec.beep_len})"
    if spec.ad_intervals:
        cond = "+".join(
            f"gte(t,{s / r:.6f})*lt(t,{e / r:.6f})" for s, e in spec.ad_intervals
        )
        expr = f"if({cond},0.3*sin(2*PI*{spec.ad_tone_hz}*t),{main_expr})"
    else:
        expr = main_expr
    parts.append(f"aevalsrc=exprs='{expr}|{expr}':s={spec.sample_rate}:d={d}[aout]")
    return ";".join(parts)


def build_command(spec: SyntheticSpec, output: Path, ffmpeg: str = "ffmpeg") -> list[str]:
    return [
        ffmpeg, "-hide_banner", "-y", "-v", "warning",
        "-filter_complex", build_filtergraph(spec),
        "-map", "[vout]", "-map", "[aout]",
        "-t", str(spec.duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
        "-g", str(spec.gop), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-fps_mode", "passthrough",
        "-movflags", "+faststart",
        str(output),
    ]


def generate(spec: SyntheticSpec, output: str | Path, ffmpeg: str = "ffmpeg") -> Path:
    """產生測試影片，並寫出同名 ``.json`` 描述檔（含廣告區間真值）。"""
    spec.validate()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(build_command(spec, output, ffmpeg), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"合成測試影片失敗：\n{proc.stderr[-2000:]}")
    sidecar = output.with_suffix(".json")
    sidecar.write_text(
        json.dumps({**asdict(spec), "total_frames": spec.total_frames}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output


def parse_ads(text: str) -> list[tuple[int, int]]:
    if not text.strip():
        return []
    result = []
    for item in text.split(","):
        start, end = item.split(":")
        result.append((int(start), int(end)))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="合成帶影格編號條碼與嗶聲的廣告測試影片")
    parser.add_argument("--out", default="tests/output/synthetic_ads.mp4")
    parser.add_argument("--duration", type=int, default=150)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--size", default="640x360")
    parser.add_argument("--ads", default="1234:1984,3001:3751", help="廣告區間 start:end,...（半開）")
    parser.add_argument("--vfr-drop-mod", type=int, default=0)
    args = parser.parse_args(argv)

    width, height = (int(v) for v in args.size.lower().split("x"))
    spec = SyntheticSpec(
        duration=args.duration,
        fps=args.fps,
        width=width,
        height=height,
        ad_intervals=parse_ads(args.ads),
        vfr_drop_mod=args.vfr_drop_mod,
    )
    out = generate(spec, args.out)
    print(f"已產生 {out}（{spec.total_frames} 格，廣告 {spec.ad_intervals}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
