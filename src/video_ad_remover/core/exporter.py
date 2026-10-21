"""[模組4] FFmpeg 精確剪輯引擎：trim/atrim + setpts/asetpts + concat 濾鏡管線與重編碼。

精確度設計
----------
- **CFR 視訊**：``trim=start_frame=S:end_frame=E``，以實際解碼影格「計數」切割，不受浮點誤差影響。
- **VFR 視訊**：``trim=start=..:end=..``，邊界取「相鄰兩格 PTS 的中點」，
  避免 6 位小數四捨五入把邊界格切錯（trim 的規則：pts >= start 保留、pts >= end 丟棄）。
- **音訊**：``atrim`` 使用邊界影格的精確 PTS（樣本級精度），最後一段以最後一格的結束時間收尾，
  每段長度 = 對應視訊段長度，``concat`` 合併後不會累積影音偏移（FR-010 / AT-009）。
- ``aresample=async=1:first_pts=0`` 先把音訊對齊到時間軸 0（處理音訊起點晚於視訊或中途缺口）。
- 輸出使用 ``-fps_mode passthrough``，禁止 FFmpeg 為了湊 CFR 而補格／丟格；VFR 來源保持 VFR。
- 先寫入同目錄暫存檔 ``.{name}.partial.mp4``，成功後才 ``os.replace`` 成正式檔名；取消或失敗會刪除暫存檔。
"""

from __future__ import annotations

import os
import platform
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
import logging
from pathlib import Path
from typing import IO, Literal

from video_ad_remover.core.env_checker import hidden_window_kwargs, resolve_tool
from video_ad_remover.core.interval_calc import compute_keep_segments
from video_ad_remover.models.interval import FrameInterval, KeepSegment
from video_ad_remover.models.video_meta import VideoMetadata

TrimMode = Literal["frame", "time"]

logger = logging.getLogger(__name__)

_DETECTED_ENCODER: str | None = None


def detect_optimal_video_codec(ffmpeg: str | None = None, *, reset_cache: bool = False) -> str:
    """依系統環境與硬體自動偵測最佳的 H.264 編碼器：

    1. macOS (Apple Silicon / Intel Mac) -> h264_videotoolbox
    2. Windows / Linux (Intel Lunar Lake / QSV) -> h264_qsv
    3. 其他 / 安全降級 -> libx264 (CPU)
    """
    global _DETECTED_ENCODER
    if _DETECTED_ENCODER is not None and not reset_cache:
        return _DETECTED_ENCODER

    ffmpeg_bin = ffmpeg or resolve_tool("ffmpeg")
    encoders_out = ""
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            **hidden_window_kwargs(),
        )
        encoders_out = proc.stdout
    except Exception:
        pass

    sys_name = platform.system()
    if sys_name == "Darwin" and "h264_videotoolbox" in encoders_out:
        _DETECTED_ENCODER = "h264_videotoolbox"
    elif sys_name in ["Windows", "Linux"] and "h264_qsv" in encoders_out:
        try:
            test_proc = subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-v",
                    "quiet",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=64x64:d=0.04",
                    "-c:v",
                    "h264_qsv",
                    "-global_quality",
                    "24",
                    "-f",
                    "null",
                    "-",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **hidden_window_kwargs(),
            )
            if test_proc.returncode == 0:
                _DETECTED_ENCODER = "h264_qsv"
            else:
                _DETECTED_ENCODER = "libx264"
        except Exception:
            _DETECTED_ENCODER = "libx264"
    else:
        _DETECTED_ENCODER = "libx264"

    return _DETECTED_ENCODER


class ExportError(RuntimeError):
    """FFmpeg 輸出失敗；``log_tail`` 帶有最後幾行 FFmpeg 診斷訊息（FR-012）。"""

    def __init__(self, message: str, log_tail: str = "") -> None:
        super().__init__(message)
        self.log_tail = log_tail

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base}\n--- FFmpeg log ---\n{self.log_tail}" if self.log_tail else base


class ExportCancelled(ExportError):
    """使用者取消輸出。"""


@dataclass(frozen=True)
class ExportSettings:
    video_codec: str = "auto"
    crf: int = 24
    video_bitrate: str | None = None  # 設定後改用位元率模式（如 "8M"），忽略 crf
    preset: str | None = None
    pix_fmt: str | None = "yuv420p"  # None = 保留濾鏡輸出格式
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"
    faststart: bool = True
    overwrite: bool = False
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class FilterGraph:
    text: str
    video_label: str
    audio_label: str | None
    mode: TrimMode


@dataclass(frozen=True)
class ExportPlan:
    source: Path
    output: Path
    partial_path: Path
    segments: tuple[KeepSegment, ...]
    graph: FilterGraph
    command: tuple[str, ...]
    expected_duration: float
    expected_frames: int
    settings: ExportSettings
    meta: VideoMetadata | None = None


@dataclass(frozen=True)
class ExportProgress:
    fraction: float  # 0.0 ~ 1.0
    out_time: float  # 已輸出秒數
    total_time: float  # 預估輸出總秒數
    frame: int
    speed: float | None  # 例如 3.2 代表 3.2x
    eta_seconds: float | None


@dataclass
class ExportResult:
    output: Path
    plan: ExportPlan
    elapsed: float
    log_tail: str = ""
    verification: object | None = field(default=None)  # verifier.VerificationReport


ProgressCallback = Callable[[ExportProgress], None]


# ---------------------------------------------------------------------------
# 規劃：濾鏡圖與命令列
# ---------------------------------------------------------------------------
def default_output_path(source: str | Path) -> Path:
    """預設輸出檔名 ``{原始檔名}_ad_removed.mp4``（FR-010）。"""
    source = Path(source)
    return source.with_name(f"{source.stem}_ad_removed.mp4")


def _fmt(seconds: float) -> str:
    return f"{seconds:.6f}"


def _midpoint(meta: VideoMetadata, frame: int) -> float:
    """第 ``frame-1`` 與第 ``frame`` 格 PTS 的中點，作為 VFR trim 的安全切點。"""
    return (meta.time_of(frame - 1) + meta.time_of(frame)) / 2


def _video_trim_args(meta: VideoMetadata, seg: KeepSegment, mode: TrimMode) -> list[str]:
    n = meta.frame_count
    args: list[str] = []
    if mode == "frame":
        if seg.start_frame > 0:
            args.append(f"start_frame={seg.start_frame}")
        if seg.end_frame < n:
            args.append(f"end_frame={seg.end_frame}")
    else:
        if seg.start_frame > 0:
            args.append(f"start={_fmt(_midpoint(meta, seg.start_frame))}")
        if seg.end_frame < n:
            args.append(f"end={_fmt(_midpoint(meta, seg.end_frame))}")
    return args


def build_filtergraph(
    meta: VideoMetadata,
    segments: Iterable[KeepSegment],
    *,
    mode: TrimMode | None = None,
) -> FilterGraph:
    segs = list(segments)
    if not segs:
        raise ValueError("沒有任何保留區間")
    for seg in segs:
        if seg.end_frame > meta.frame_count:
            raise ValueError(f"保留區間 {seg} 超出影片總影格數 {meta.frame_count}")
    mode = mode or ("time" if meta.is_vfr else "frame")
    k = len(segs)
    parts: list[str] = []

    # ---- 視訊 -----------------------------------------------------------
    vin = f"[0:{meta.video_index}]"
    if k == 1:
        v_src = [vin]
    else:
        v_src = [f"[v{i}]" for i in range(k)]
        parts.append(f"{vin}split={k}{''.join(v_src)}")
    for i, seg in enumerate(segs):
        args = _video_trim_args(meta, seg, mode)
        trim = f"trim={':'.join(args)}" if args else "trim"
        parts.append(f"{v_src[i]}{trim},setpts=PTS-STARTPTS[vs{i}]")

    # ---- 音訊 -----------------------------------------------------------
    audio_label: str | None = None
    if meta.audio is not None:
        ain = f"[0:{meta.audio.index}]"
        pre = "aresample=async=1:first_pts=0"
        if k == 1:
            a_src = ["[a0]"]
            parts.append(f"{ain}{pre}[a0]")
        else:
            a_src = [f"[a{i}]" for i in range(k)]
            parts.append(f"{ain}{pre},asplit={k}{''.join(a_src)}")
        for i, seg in enumerate(segs):
            start = meta.time_of(seg.start_frame)
            end = meta.time_of(seg.end_frame)  # end_frame == frame_count 時為最後一格結束時間
            parts.append(
                f"{a_src[i]}atrim=start={_fmt(start)}:end={_fmt(end)},asetpts=PTS-STARTPTS[as{i}]"
            )
        audio_label = "[aout]"

    # ---- 合併 -----------------------------------------------------------
    if audio_label:
        inputs = "".join(f"[vs{i}][as{i}]" for i in range(k))
        parts.append(f"{inputs}concat=n={k}:v=1:a=1[vout][aout]")
    else:
        inputs = "".join(f"[vs{i}]" for i in range(k))
        parts.append(f"{inputs}concat=n={k}:v=1:a=0[vout]")

    return FilterGraph(text=";".join(parts), video_label="[vout]", audio_label=audio_label, mode=mode)


def resolve_video_codec(settings: ExportSettings, *, ffmpeg: str) -> str:
    """決定具體使用的視訊編碼器（若為 auto 則自動偵測最佳硬體加速）。"""
    if settings.video_codec and settings.video_codec != "auto":
        return settings.video_codec
    return detect_optimal_video_codec(ffmpeg)


def build_command(
    meta: VideoMetadata,
    graph: FilterGraph,
    output: Path,
    settings: ExportSettings,
    *,
    ffmpeg: str,
) -> list[str]:
    codec = resolve_video_codec(settings, ffmpeg=ffmpeg)

    cmd = [
        ffmpeg, "-hide_banner", "-nostdin", "-y", "-v", "warning",
        "-i", str(meta.path),
        "-filter_complex", graph.text,
        "-map", graph.video_label,
    ]
    if graph.audio_label:
        cmd += ["-map", graph.audio_label]

    cmd += ["-c:v", codec]
    if settings.video_bitrate:
        cmd += ["-b:v", settings.video_bitrate]
    else:
        if codec == "h264_qsv":
            cmd += ["-global_quality", str(settings.crf)]
        elif codec == "h264_videotoolbox":
            # CRF 24 對應品質 60；CRF 越小（品質要求越高）對應 q:v 越高
            q_v = max(30, min(85, int(60 - (settings.crf - 24) * 2.5)))
            cmd += ["-q:v", str(q_v)]
        else:
            cmd += ["-crf", str(settings.crf)]

    # Preset 設定適配
    if codec == "h264_qsv":
        qsv_presets = {"veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}
        preset = settings.preset
        if preset == "ultrafast":
            preset = "veryfast"
        elif preset not in qsv_presets:
            preset = "fast"
        cmd += ["-preset", preset]
    elif codec == "h264_videotoolbox":
        # VideoToolbox 不支援標準 x264 preset，不傳入 -preset
        pass
    else:
        cmd += ["-preset", settings.preset or "veryfast"]

    if settings.pix_fmt:
        pix_fmt = "nv12" if codec == "h264_qsv" and settings.pix_fmt == "yuv420p" else settings.pix_fmt
        cmd += ["-pix_fmt", pix_fmt]

    if graph.audio_label:
        cmd += ["-c:a", settings.audio_codec, "-b:a", settings.audio_bitrate]
        if meta.audio and meta.audio.sample_rate:
            cmd += ["-ar", str(meta.audio.sample_rate)]

    fps_mode = "vfr" if meta.is_vfr else "auto"
    cmd += [
        "-fps_mode", fps_mode,
        "-avoid_negative_ts", "make_zero",
        "-max_muxing_queue_size", "2048",
        "-map_chapters", "-1",
    ]
    if settings.faststart:
        cmd += ["-movflags", "+faststart"]
    cmd += list(settings.extra_args)
    cmd += ["-progress", "pipe:1", "-nostats", "-f", "mp4", str(output)]
    return cmd


def _is_same_file(a: Path, b: Path) -> bool:
    try:
        if a.exists() and b.exists():
            return os.path.samefile(a, b)
    except OSError:
        pass
    return a.resolve() == b.resolve()


def plan_export(
    meta: VideoMetadata,
    ad_intervals: Iterable[FrameInterval],
    output: str | Path | None = None,
    settings: ExportSettings | None = None,
    *,
    ffmpeg: str | None = None,
    mode: TrimMode | None = None,
) -> ExportPlan:
    """把「已確認廣告區間」轉成可執行的輸出計畫（含防呆：重疊、越界、覆寫原檔）。"""
    settings = settings or ExportSettings()
    output = Path(output) if output else default_output_path(meta.path)
    if _is_same_file(output, meta.path):
        raise ValueError("輸出路徑不可與原始影片相同（原始檔全程唯讀）")

    segments = compute_keep_segments(ad_intervals, meta.frame_count, strict=True)
    graph = build_filtergraph(meta, segments, mode=mode)
    partial = output.with_name(f".{output.stem}.partial.mp4")
    command = build_command(meta, graph, partial, settings, ffmpeg=ffmpeg or resolve_tool("ffmpeg"))
    return ExportPlan(
        source=meta.path,
        output=output,
        partial_path=partial,
        segments=tuple(segments),
        graph=graph,
        command=tuple(command),
        expected_duration=sum(meta.segment_duration(s.start_frame, s.end_frame) for s in segments),
        expected_frames=sum(s.length for s in segments),
        settings=settings,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# 執行：進度回報與安全取消
# ---------------------------------------------------------------------------
def _drain(stream: IO[str], sink: deque[str]) -> None:
    for line in stream:
        line = line.rstrip()
        if line:
            sink.append(line)


def _terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _watch_cancel(proc: subprocess.Popen[str], cancel_event: threading.Event) -> None:
    while proc.poll() is None:
        if cancel_event.wait(0.2):
            _terminate(proc)
            return


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _to_int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "", "N/A") else None
    except ValueError:
        return None


def _consume_progress(
    stream: IO[str],
    total: float,
    callback: ProgressCallback | None,
    started: float,
) -> None:
    state: dict[str, str] = {}
    for raw in stream:
        line = raw.strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        state[key] = value
        if key != "progress" or callback is None:
            continue
        out_us = _to_int(state.get("out_time_us"))
        out_time = max(out_us / 1e6, 0.0) if out_us is not None else 0.0
        fraction = 1.0 if value == "end" else (min(out_time / total, 0.999) if total > 0 else 0.0)
        speed_text = state.get("speed", "").rstrip("x").strip()
        try:
            speed = float(speed_text) if speed_text and speed_text != "N/A" else None
        except ValueError:
            speed = None
        elapsed = time.monotonic() - started
        eta = elapsed * (1 - fraction) / fraction if 0 < fraction < 1 else (0.0 if fraction >= 1 else None)
        callback(
            ExportProgress(
                fraction=fraction,
                out_time=out_time,
                total_time=total,
                frame=_to_int(state.get("frame")) or 0,
                speed=speed,
                eta_seconds=eta,
            )
        )


def run_export(
    plan: ExportPlan,
    *,
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    log_lines: int = 200,
    allow_fallback: bool = True,
) -> ExportResult:
    """執行 FFmpeg 輸出。可由另一執行緒設定 ``cancel_event`` 安全取消（FR-012）。"""
    if plan.output.exists() and not plan.settings.overwrite:
        raise FileExistsError(f"輸出檔已存在：{plan.output}（請確認是否覆寫）")
    plan.partial_path.parent.mkdir(parents=True, exist_ok=True)
    cancel_event = cancel_event or threading.Event()
    log_tail: deque[str] = deque(maxlen=log_lines)
    started = time.monotonic()

    proc = subprocess.Popen(
        list(plan.command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **hidden_window_kwargs(),
    )
    assert proc.stdout is not None and proc.stderr is not None
    stderr_thread = threading.Thread(target=_drain, args=(proc.stderr, log_tail), daemon=True)
    watcher = threading.Thread(target=_watch_cancel, args=(proc, cancel_event), daemon=True)
    stderr_thread.start()
    watcher.start()

    try:
        _consume_progress(proc.stdout, plan.expected_duration, progress_cb, started)
        returncode = proc.wait()
    except BaseException:
        _terminate(proc)
        _safe_unlink(plan.partial_path)
        raise
    finally:
        stderr_thread.join(timeout=5)

    tail = "\n".join(log_tail)
    if cancel_event.is_set():
        _safe_unlink(plan.partial_path)
        raise ExportCancelled("使用者已取消輸出，暫存檔已清除", tail)
    if returncode != 0:
        _safe_unlink(plan.partial_path)
        is_hw = (
            "-c:v" in plan.command
            and plan.command[plan.command.index("-c:v") + 1]
            in ("h264_qsv", "h264_nvenc", "h264_amf", "h264_videotoolbox")
        )
        if allow_fallback and is_hw and plan.meta is not None:
            logger.warning(
                "硬體加速編碼失敗（exit code %s），自動降級切換至 libx264 (CPU) 重新嘗試輸出...",
                returncode,
            )
            fallback_settings = replace(plan.settings, video_codec="libx264")
            fallback_cmd = build_command(
                plan.meta,
                plan.graph,
                plan.partial_path,
                fallback_settings,
                ffmpeg=plan.command[0],
            )
            fallback_plan = replace(plan, command=tuple(fallback_cmd), settings=fallback_settings)
            return run_export(
                fallback_plan,
                progress_cb=progress_cb,
                cancel_event=cancel_event,
                log_lines=log_lines,
                allow_fallback=False,
            )
        raise ExportError(f"FFmpeg 執行失敗（exit code {returncode}）", tail)
    if not plan.partial_path.is_file() or plan.partial_path.stat().st_size == 0:
        raise ExportError("FFmpeg 結束但沒有產生輸出檔", tail)

    os.replace(plan.partial_path, plan.output)
    return ExportResult(
        output=plan.output, plan=plan, elapsed=time.monotonic() - started, log_tail=tail
    )


def export_video(
    meta: VideoMetadata,
    ad_intervals: Iterable[FrameInterval],
    output: str | Path | None = None,
    settings: ExportSettings | None = None,
    *,
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    verify: bool = True,
) -> ExportResult:
    """便利函式：規劃 -> 執行 -> （選用）FFprobe 自動驗證（FR-013）。"""
    plan = plan_export(meta, ad_intervals, output, settings)
    result = run_export(plan, progress_cb=progress_cb, cancel_event=cancel_event)
    if verify:
        from video_ad_remover.core.verifier import verify_output

        result.verification = verify_output(
            result.output,
            expected_duration=plan.expected_duration,
            expected_frames=plan.expected_frames,
            expect_audio=meta.has_audio,
        )
    return result
