"""區間資料模型：全系統統一使用 0-based 影格索引與半開區間 ``[start_frame, end_frame)``。

- ``FrameInterval``    ：基礎半開區間（不可變、建立時即驗證）。
- ``ConfirmedInterval``：已確認刪除的廣告區間。
- ``KeepSegment``      ：正片保留區間（由 interval_calc 換算產生）。
- ``AdCandidate``      ：候選廣告（可變），帶狀態、來源與評分，供 GUI 與專案檔使用。
"""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self


class InvalidIntervalError(ValueError):
    """無效區間（如 start >= end、負數索引、超出影片範圍、已確認區間互相重疊）。"""


class IntervalStatus(StrEnum):
    """候選區間的三種人工確認狀態（FR-006）。"""

    UNCONFIRMED = "unconfirmed"  # 未確認
    CONFIRMED_AD = "confirmed_ad"  # 已確認為廣告
    EXCLUDED = "excluded"  # 已排除

    @property
    def label(self) -> str:
        return {
            IntervalStatus.UNCONFIRMED: "未確認",
            IntervalStatus.CONFIRMED_AD: "已確認為廣告",
            IntervalStatus.EXCLUDED: "已排除",
        }[self]


class DetectionSource(StrEnum):
    """候選區間的偵測來源（NFR 可追溯性）。"""

    AUTO_SCAN = "auto_scan"  # 三階段自動掃描
    TEMPLATE_MATCH = "template_match"  # 同片樣本比對
    MANUAL = "manual"  # 手動新增


@dataclass(frozen=True)
class FrameInterval:
    """0-based 半開影格區間 ``[start_frame, end_frame)``，包含 start、不包含 end。"""

    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        for name in ("start_frame", "end_frame"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise InvalidIntervalError(f"{name} 必須是整數，收到 {value!r}")
        if self.start_frame < 0:
            raise InvalidIntervalError(f"start_frame 不可為負數：{self}")
        if self.start_frame >= self.end_frame:
            raise InvalidIntervalError(
                f"無效區間 {self}：必須滿足 start_frame < end_frame（不允許空區間）"
            )

    # ---- 基本屬性 -------------------------------------------------------
    @property
    def length(self) -> int:
        """區間內影格數（= end - start）。"""
        return self.end_frame - self.start_frame

    @property
    def last_frame(self) -> int:
        """區間內最後一格的索引（= end - 1）。"""
        return self.end_frame - 1

    def contains_frame(self, frame: int) -> bool:
        return self.start_frame <= frame < self.end_frame

    def overlaps(self, other: FrameInterval) -> bool:
        """是否有共同影格（相鄰區間 ``[a,b)`` 與 ``[b,c)`` 不算重疊）。"""
        return self.start_frame < other.end_frame and other.start_frame < self.end_frame

    def is_adjacent(self, other: FrameInterval) -> bool:
        return self.end_frame == other.start_frame or other.end_frame == self.start_frame

    def intersection_length(self, other: FrameInterval) -> int:
        return max(0, min(self.end_frame, other.end_frame) - max(self.start_frame, other.start_frame))

    def iou(self, other: FrameInterval) -> float:
        """交集 / 聯集（Intersection over Union），供候選去重使用。"""
        inter = self.intersection_length(other)
        union = self.length + other.length - inter
        return inter / union if union else 0.0

    # ---- 產生新區間（不可變） --------------------------------------------
    def with_bounds(self, start_frame: int, end_frame: int) -> Self:
        return dataclasses.replace(self, start_frame=start_frame, end_frame=end_frame)

    def shifted(self, start_delta: int = 0, end_delta: int = 0) -> Self:
        """回傳起點／終點各自位移後的新區間（±1 格微調用）。"""
        return self.with_bounds(self.start_frame + start_delta, self.end_frame + end_delta)

    # ---- 序列化 ---------------------------------------------------------
    def to_list(self) -> list[int]:
        return [self.start_frame, self.end_frame]

    @classmethod
    def from_list(cls, values: list[int] | tuple[int, int]) -> Self:
        start, end = values
        return cls(int(start), int(end))

    def __str__(self) -> str:
        return f"[{self.start_frame}, {self.end_frame})"


@dataclass(frozen=True)
class ConfirmedInterval(FrameInterval):
    """已確認要刪除的廣告區間。"""


@dataclass(frozen=True)
class KeepSegment(FrameInterval):
    """正片保留區間。"""


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class AdCandidate:
    """候選廣告區間（GUI 候選清單的一列）。"""

    interval: FrameInterval
    status: IntervalStatus = IntervalStatus.UNCONFIRMED
    source: DetectionSource = DetectionSource.AUTO_SCAN
    score: float = 0.0
    score_detail: dict[str, float] = field(default_factory=dict)
    note: str = ""
    id: str = field(default_factory=_new_id)

    # ---- 便利屬性 -------------------------------------------------------
    @property
    def start_frame(self) -> int:
        return self.interval.start_frame

    @property
    def end_frame(self) -> int:
        return self.interval.end_frame

    @property
    def length(self) -> int:
        return self.interval.length

    @property
    def is_confirmed(self) -> bool:
        return self.status == IntervalStatus.CONFIRMED_AD

    # ---- 編輯（驗證交由 FrameInterval 處理，失敗時原值不變） --------------
    def set_bounds(self, start_frame: int, end_frame: int) -> None:
        self.interval = FrameInterval(start_frame, end_frame)

    def adjust_start(self, delta: int) -> None:
        self.interval = self.interval.shifted(start_delta=delta)

    def adjust_end(self, delta: int) -> None:
        self.interval = self.interval.shifted(end_delta=delta)

    def to_confirmed(self) -> ConfirmedInterval:
        if not self.is_confirmed:
            raise InvalidIntervalError(f"候選 {self.id} 尚未確認為廣告（狀態：{self.status.label}）")
        return ConfirmedInterval(self.start_frame, self.end_frame)

    # ---- 序列化 ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "status": self.status.value,
            "source": self.source.value,
            "score": self.score,
            "score_detail": dict(self.score_detail),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdCandidate:
        return cls(
            interval=FrameInterval(int(data["start_frame"]), int(data["end_frame"])),
            status=IntervalStatus(data.get("status", IntervalStatus.UNCONFIRMED)),
            source=DetectionSource(data.get("source", DetectionSource.MANUAL)),
            score=float(data.get("score", 0.0)),
            score_detail={k: float(v) for k, v in data.get("score_detail", {}).items()},
            note=str(data.get("note", "")),
            id=str(data.get("id") or _new_id()),
        )
