"""資料模型層：全系統共用的資料結構（0-based、半開區間）。"""

from video_ad_remover.models.interval import (
    AdCandidate,
    ConfirmedInterval,
    DetectionSource,
    FrameInterval,
    IntervalStatus,
    InvalidIntervalError,
    KeepSegment,
)
from video_ad_remover.models.video_meta import AudioStreamInfo, VideoMetadata

__all__ = [
    "AdCandidate",
    "AudioStreamInfo",
    "ConfirmedInterval",
    "DetectionSource",
    "FrameInterval",
    "IntervalStatus",
    "InvalidIntervalError",
    "KeepSegment",
    "VideoMetadata",
]
