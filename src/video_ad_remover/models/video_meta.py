"""影片資訊模型：FFprobe 解析結果 + 影格 PTS 時間戳記索引。

時間軸約定（重要）
------------------
``frame_times[i]`` 是第 i 格（0-based、依呈現順序）的 PTS 秒數，
並已減去 ``timeline_origin``（容器 format start_time）。
這正是 FFmpeg CLI 在預設（未加 ``-copyts``）時濾鏡所看到的時間軸，
因此可直接用於 ``trim`` / ``atrim`` 的 start / end 參數。

``time_of(frame_count)`` 回傳最後一格的「結束時間」(``end_time``)，
讓半開區間 ``[s, e)`` 的時長永遠可用 ``time_of(e) - time_of(s)`` 計算。
"""

from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from video_ad_remover.utils.time_conv import format_timestamp


@dataclass(frozen=True, slots=True)
class AudioStreamInfo:
    index: int
    codec: str
    sample_rate: int
    channels: int
    channel_layout: str = ""
    start_time: float = 0.0  # 絕對 PTS 秒數（未減 timeline_origin）
    duration: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioStreamInfo:
        return cls(**data)


@dataclass(slots=True)
class VideoMetadata:
    path: Path
    file_size: int
    format_name: str
    duration: float  # 容器時長（秒）
    video_index: int  # 視訊串流的絕對 stream index
    width: int
    height: int
    video_codec: str
    pix_fmt: str
    time_base: Fraction
    avg_frame_rate: Fraction | None
    r_frame_rate: Fraction | None
    is_vfr: bool
    timeline_origin: float  # 容器 start_time（絕對秒數）
    video_start_time: float  # 視訊串流 start_time（絕對秒數）
    frame_times: list[float] = field(default_factory=list, repr=False)
    end_time: float = 0.0  # 最後一格結束時間（同 frame_times 時間軸）
    audio: AudioStreamInfo | None = None
    reported_frame_count: int | None = None  # 容器宣告的 nb_frames（僅供參考）

    # ---- 基本屬性 -------------------------------------------------------
    @property
    def frame_count(self) -> int:
        return len(self.frame_times)

    @property
    def has_audio(self) -> bool:
        return self.audio is not None

    @property
    def nominal_fps(self) -> float:
        rate = self.avg_frame_rate or self.r_frame_rate
        return float(rate) if rate else 0.0

    @property
    def frame_mode(self) -> str:
        return "VFR" if self.is_vfr else "CFR"

    @property
    def video_duration(self) -> float:
        """視訊實際涵蓋時長（第一格起點到最後一格結束）。"""
        if not self.frame_times:
            return 0.0
        return self.end_time - self.frame_times[0]

    # ---- 影格 <-> 時間 ---------------------------------------------------
    def _check_boundary(self, frame: int) -> None:
        if not 0 <= frame <= self.frame_count:
            raise IndexError(f"影格邊界 {frame} 超出範圍 [0, {self.frame_count}]")

    def time_of(self, frame: int) -> float:
        """影格邊界時間；``frame == frame_count`` 時回傳最後一格結束時間。"""
        self._check_boundary(frame)
        if frame == self.frame_count:
            return self.end_time
        return self.frame_times[frame]

    def frame_duration(self, frame: int) -> float:
        if not 0 <= frame < self.frame_count:
            raise IndexError(f"影格 {frame} 超出範圍 [0, {self.frame_count})")
        return self.time_of(frame + 1) - self.time_of(frame)

    def segment_duration(self, start_frame: int, end_frame: int) -> float:
        """半開區間 ``[start, end)`` 的實際時長（依 PTS，而非 frame/fps 推算）。"""
        return self.time_of(end_frame) - self.time_of(start_frame)

    def frame_at_time(self, seconds: float) -> int:
        """回傳在 ``seconds`` 時刻正在顯示的影格索引（以實際 PTS 查表，VFR 安全）。"""
        if not self.frame_times:
            raise IndexError("影片沒有任何影格")
        idx = bisect.bisect_right(self.frame_times, seconds + 1e-9) - 1
        return min(max(idx, 0), self.frame_count - 1)

    def timestamp_of(self, frame: int) -> str:
        return format_timestamp(self.time_of(frame))

    # ---- 顯示 / 序列化 ---------------------------------------------------
    def summary(self) -> dict[str, str]:
        """GUI 頂部資訊列用的可讀摘要（FR-001）。"""
        audio = (
            f"{self.audio.codec} {self.audio.sample_rate} Hz {self.audio.channels}ch"
            if self.audio
            else "無音訊"
        )
        return {
            "檔案名稱": self.path.name,
            "檔案大小": f"{self.file_size / 1024 / 1024:.1f} MB",
            "總時長": format_timestamp(self.video_duration),
            "解析度": f"{self.width}x{self.height}",
            "影格率": f"{self.nominal_fps:.3f} fps ({self.frame_mode})",
            "總影格數": str(self.frame_count),
            "視訊編碼": f"{self.video_codec} / {self.pix_fmt}",
            "音訊編碼": audio,
        }

    def to_dict(self, *, include_frame_times: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "path": str(self.path),
            "file_size": self.file_size,
            "format_name": self.format_name,
            "duration": self.duration,
            "video_index": self.video_index,
            "width": self.width,
            "height": self.height,
            "video_codec": self.video_codec,
            "pix_fmt": self.pix_fmt,
            "time_base": str(self.time_base),
            "avg_frame_rate": str(self.avg_frame_rate) if self.avg_frame_rate else None,
            "r_frame_rate": str(self.r_frame_rate) if self.r_frame_rate else None,
            "is_vfr": self.is_vfr,
            "timeline_origin": self.timeline_origin,
            "video_start_time": self.video_start_time,
            "frame_count": self.frame_count,
            "end_time": self.end_time,
            "audio": self.audio.to_dict() if self.audio else None,
            "reported_frame_count": self.reported_frame_count,
        }
        if include_frame_times:
            data["frame_times"] = list(self.frame_times)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoMetadata:
        def frac(value: str | None) -> Fraction | None:
            return Fraction(value) if value else None

        return cls(
            path=Path(data["path"]),
            file_size=int(data["file_size"]),
            format_name=data["format_name"],
            duration=float(data["duration"]),
            video_index=int(data["video_index"]),
            width=int(data["width"]),
            height=int(data["height"]),
            video_codec=data["video_codec"],
            pix_fmt=data["pix_fmt"],
            time_base=Fraction(data["time_base"]),
            avg_frame_rate=frac(data.get("avg_frame_rate")),
            r_frame_rate=frac(data.get("r_frame_rate")),
            is_vfr=bool(data["is_vfr"]),
            timeline_origin=float(data["timeline_origin"]),
            video_start_time=float(data["video_start_time"]),
            frame_times=[float(t) for t in data.get("frame_times", [])],
            end_time=float(data["end_time"]),
            audio=AudioStreamInfo.from_dict(data["audio"]) if data.get("audio") else None,
            reported_frame_count=data.get("reported_frame_count"),
        )
