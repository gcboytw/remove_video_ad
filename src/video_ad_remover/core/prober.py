"""[模組1] FFprobe 封裝：讀取影片資訊並建立影格 PTS 時間戳記索引。

PTS 索引的建立方式
------------------
以 ``ffprobe -show_entries packet=pts_time,duration_time,flags`` 讀取視訊封包
（只解封裝、不解碼，長片也只需數秒），濾掉 MP4 edit list 標記丟棄（``D``）的封包，
再依 PTS 排序成「呈現順序」。每個封包對應一格，因此 ``frame_times[i]`` 就是
0-based 第 i 格的實際時間戳記 —— VFR 影片也不會用 ``i / fps`` 亂推算（FR-002）。
"""

from __future__ import annotations

import json
import statistics
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from video_ad_remover.core.env_checker import hidden_window_kwargs, resolve_tool
from video_ad_remover.models.video_meta import AudioStreamInfo, VideoMetadata
from video_ad_remover.utils.time_conv import parse_float, parse_fraction

# 相鄰影格間隔偏離中位數超過此比例即視為 VFR
VFR_TOLERANCE_RATIO = 0.25
# 允許少量異常間隔（例如片頭片尾單一格），超過此比例才判定 VFR
VFR_OUTLIER_RATIO = 0.001


class ProbeError(RuntimeError):
    """FFprobe 執行失敗或影片無法解析。"""


def _run(cmd: list[str], timeout: float | None) -> str:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            **hidden_window_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"FFprobe 逾時（{timeout}s）") from exc
    except OSError as exc:
        raise ProbeError(f"無法執行 FFprobe：{exc}") from exc
    if proc.returncode != 0:
        raise ProbeError(f"FFprobe 失敗（exit {proc.returncode}）：{proc.stderr.strip()[-500:]}")
    return proc.stdout


def probe_streams(path: Path, *, ffprobe: str | None = None, timeout: float = 60) -> dict[str, Any]:
    """回傳 ``ffprobe -show_format -show_streams`` 的 JSON 結果。"""
    ffprobe = ffprobe or resolve_tool("ffprobe")
    out = _run(
        [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        timeout,
    )
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"FFprobe 輸出不是合法 JSON：{exc}") from exc


def read_frame_index(
    path: Path,
    stream_index: int,
    *,
    ffprobe: str | None = None,
    timeout: float | None = 600,
) -> tuple[list[float], list[float]]:
    """讀取視訊封包 PTS，回傳依呈現順序排序的 ``(pts_list, duration_list)``（絕對秒數）。"""
    ffprobe = ffprobe or resolve_tool("ffprobe")
    out = _run(
        [
            ffprobe, "-v", "error",
            "-select_streams", str(stream_index),
            "-show_entries", "packet=pts_time,duration_time,flags",
            "-of", "compact=p=0",
            str(path),
        ],
        timeout,
    )
    entries: list[tuple[float, float | None]] = []
    for line in out.splitlines():
        fields = dict(part.split("=", 1) for part in line.strip().split("|") if "=" in part)
        if "D" in fields.get("flags", ""):
            continue  # edit list 丟棄的封包，不會被呈現
        pts = parse_float(fields.get("pts_time"))
        if pts is None:
            continue
        entries.append((pts, parse_float(fields.get("duration_time"))))
    if not entries:
        raise ProbeError("無法從視訊串流讀取任何影格時間戳記")

    entries.sort(key=lambda e: e[0])
    pts_list = [e[0] for e in entries]
    deltas = [b - a for a, b in zip(pts_list, pts_list[1:]) if b > a]
    fallback = statistics.median(deltas) if deltas else 0.0

    durations: list[float] = []
    for i, (pts, dur) in enumerate(entries):
        if i + 1 < len(entries):
            # 以下一格 PTS 決定本格顯示時長（最可靠，也符合 VFR 定義）
            gap = pts_list[i + 1] - pts
            durations.append(gap if gap > 0 else (dur or fallback))
        else:
            durations.append(dur if dur and dur > 0 else fallback)
    return pts_list, durations


def detect_vfr(frame_times: list[float]) -> bool:
    """依實際影格間隔判斷是否為可變影格率。"""
    deltas = [b - a for a, b in zip(frame_times, frame_times[1:])]
    if len(deltas) < 2:
        return False
    median = statistics.median(deltas)
    if median <= 0:
        return True
    tolerance = median * VFR_TOLERANCE_RATIO
    outliers = sum(1 for d in deltas if abs(d - median) > tolerance)
    return outliers > max(2, int(len(deltas) * VFR_OUTLIER_RATIO))


def _pick_video_stream(streams: list[dict[str, Any]]) -> dict[str, Any]:
    for s in streams:
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic"):
            return s
    raise ProbeError("找不到視訊串流（封面圖片不算）")


def _pick_audio_stream(streams: list[dict[str, Any]]) -> dict[str, Any] | None:
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        return None
    default = [s for s in audio if s.get("disposition", {}).get("default")]
    return (default or audio)[0]


def probe_video(
    path: str | Path,
    *,
    ffprobe: str | None = None,
    timeout: float | None = 600,
) -> VideoMetadata:
    """讀取影片資訊並建立 PTS 索引（原始檔全程唯讀）。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到影片檔案：{path}")

    info = probe_streams(path, ffprobe=ffprobe)
    streams = info.get("streams", [])
    fmt = info.get("format", {})
    vstream = _pick_video_stream(streams)
    astream = _pick_audio_stream(streams)

    origin = parse_float(fmt.get("start_time"), 0.0) or 0.0
    pts_list, durations = read_frame_index(
        path, int(vstream["index"]), ffprobe=ffprobe, timeout=timeout
    )
    frame_times = [t - origin for t in pts_list]
    end_time = frame_times[-1] + durations[-1]

    audio = None
    if astream is not None:
        audio = AudioStreamInfo(
            index=int(astream["index"]),
            codec=astream.get("codec_name", "unknown"),
            sample_rate=int(astream.get("sample_rate") or 0),
            channels=int(astream.get("channels") or 0),
            channel_layout=astream.get("channel_layout", ""),
            start_time=parse_float(astream.get("start_time"), origin) or origin,
            duration=parse_float(astream.get("duration")),
        )

    nb_frames = vstream.get("nb_frames")
    return VideoMetadata(
        path=path,
        file_size=path.stat().st_size,
        format_name=fmt.get("format_name", ""),
        duration=parse_float(fmt.get("duration"), end_time) or end_time,
        video_index=int(vstream["index"]),
        width=int(vstream.get("width") or 0),
        height=int(vstream.get("height") or 0),
        video_codec=vstream.get("codec_name", "unknown"),
        pix_fmt=vstream.get("pix_fmt", ""),
        time_base=Fraction(vstream.get("time_base", "1/1000")),
        avg_frame_rate=parse_fraction(vstream.get("avg_frame_rate")),
        r_frame_rate=parse_fraction(vstream.get("r_frame_rate")),
        is_vfr=detect_vfr(frame_times),
        timeline_origin=origin,
        video_start_time=parse_float(vstream.get("start_time"), origin) or origin,
        frame_times=frame_times,
        end_time=end_time,
        audio=audio,
        reported_frame_count=int(nb_frames) if str(nb_frames).isdigit() else None,
    )
