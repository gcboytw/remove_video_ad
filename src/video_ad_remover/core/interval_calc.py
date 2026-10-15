"""半開區間換算器：廣告刪除區間 ``[start, end)`` -> 正片保留區間，含防呆、合併與去重。

範例（SRS §6.1）::

    >>> compute_keep_segments([FrameInterval(10000, 10750), FrameInterval(30000, 30750)], 50000)
    [KeepSegment(0, 10000), KeepSegment(10750, 30000), KeepSegment(30750, 50000)]
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from video_ad_remover.models.interval import (
    AdCandidate,
    ConfirmedInterval,
    FrameInterval,
    InvalidIntervalError,
    IntervalStatus,
    KeepSegment,
)


def sort_intervals[T: FrameInterval](intervals: Iterable[T]) -> list[T]:
    return sorted(intervals, key=lambda iv: (iv.start_frame, iv.end_frame))


def find_overlaps[T: FrameInterval](intervals: Iterable[T]) -> list[tuple[T, T]]:
    """找出所有互相重疊的區間對（相鄰不算重疊）。"""
    ordered = sort_intervals(intervals)
    pairs: list[tuple[T, T]] = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if b.start_frame >= a.end_frame:
                break
            pairs.append((a, b))
    return pairs


def validate_ad_intervals(intervals: Iterable[FrameInterval], total_frames: int) -> None:
    """FR-008 防呆：區間必須落在 ``[0, total_frames]`` 內，且彼此不可重疊。"""
    if total_frames <= 0:
        raise InvalidIntervalError(f"影片總影格數必須 > 0，收到 {total_frames}")
    items = list(intervals)
    for iv in items:
        if iv.end_frame > total_frames:
            raise InvalidIntervalError(
                f"區間 {iv} 超出影片範圍（總影格數 {total_frames}，最後一格索引 {total_frames - 1}）"
            )
    overlaps = find_overlaps(items)
    if overlaps:
        desc = "、".join(f"{a} 與 {b}" for a, b in overlaps[:5])
        raise InvalidIntervalError(f"已確認的廣告區間互相重疊：{desc}")


def merge_intervals(
    intervals: Iterable[FrameInterval],
    *,
    merge_adjacent: bool = True,
    max_gap: int = 0,
) -> list[FrameInterval]:
    """合併重疊（及相鄰／間距 <= max_gap）的區間，回傳排序後的新清單。"""
    merged: list[FrameInterval] = []
    for iv in sort_intervals(intervals):
        if merged:
            last = merged[-1]
            gap = iv.start_frame - last.end_frame  # < 0 表示重疊，0 表示相鄰
            if gap < 0 or (merge_adjacent and gap <= max_gap):
                merged[-1] = FrameInterval(last.start_frame, max(last.end_frame, iv.end_frame))
                continue
        merged.append(FrameInterval(iv.start_frame, iv.end_frame))
    return merged


def collect_confirmed(candidates: Iterable[AdCandidate]) -> list[ConfirmedInterval]:
    """只收集「已確認為廣告」的候選（未確認、已排除一律不刪，FR-008）。"""
    return sort_intervals(c.to_confirmed() for c in candidates if c.status == IntervalStatus.CONFIRMED_AD)


def compute_keep_segments(
    ad_intervals: Iterable[FrameInterval],
    total_frames: int,
    *,
    strict: bool = True,
) -> list[KeepSegment]:
    """將廣告刪除區間反轉為正片保留區間。

    Args:
        ad_intervals: 要刪除的廣告半開區間。
        total_frames: 影片總影格數（合法索引為 ``0 .. total_frames-1``）。
        strict: True 時重疊或越界直接拋錯（輸出前的最終檢查）；
                False 時自動合併重疊並裁掉越界部分（預覽用）。
    """
    if total_frames <= 0:
        raise InvalidIntervalError(f"影片總影格數必須 > 0，收到 {total_frames}")

    ads = list(ad_intervals)
    if strict:
        validate_ad_intervals(ads, total_frames)
        ordered: Sequence[FrameInterval] = sort_intervals(ads)
    else:
        clipped = [
            FrameInterval(iv.start_frame, min(iv.end_frame, total_frames))
            for iv in ads
            if iv.start_frame < total_frames
        ]
        ordered = merge_intervals(clipped, merge_adjacent=True)

    keeps: list[KeepSegment] = []
    cursor = 0
    for ad in ordered:
        if ad.start_frame > cursor:
            keeps.append(KeepSegment(cursor, ad.start_frame))
        cursor = max(cursor, ad.end_frame)
    if cursor < total_frames:
        keeps.append(KeepSegment(cursor, total_frames))

    if not keeps:
        raise InvalidIntervalError("所有影格都被標記為廣告，沒有任何正片可輸出")
    return keeps


def count_frames(segments: Iterable[FrameInterval]) -> int:
    return sum(s.length for s in segments)


def split_interval[T: FrameInterval](interval: T, at_frame: int) -> tuple[T, T]:
    """在 ``at_frame`` 拆成 ``[start, at)`` 與 ``[at, end)``（兩段都不可為空）。"""
    if not interval.start_frame < at_frame < interval.end_frame:
        raise InvalidIntervalError(
            f"拆分點 {at_frame} 必須位於 {interval} 內部（{interval.start_frame} < at < {interval.end_frame}）"
        )
    return (
        interval.with_bounds(interval.start_frame, at_frame),
        interval.with_bounds(at_frame, interval.end_frame),
    )


def union_interval[T: FrameInterval](a: T, b: FrameInterval) -> T:
    """合併兩個區間為涵蓋兩者的最小區間（FR-008「合併相鄰區間」）。"""
    return a.with_bounds(min(a.start_frame, b.start_frame), max(a.end_frame, b.end_frame))


def dedupe_candidates(
    candidates: Iterable[AdCandidate],
    *,
    iou_threshold: float = 0.5,
) -> list[AdCandidate]:
    """依分數由高到低保留候選，丟棄與已保留者 IoU >= 門檻的重複候選（FR-005）。"""
    kept: list[AdCandidate] = []
    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        if all(cand.interval.iou(k.interval) < iou_threshold for k in kept):
            kept.append(cand)
    return sorted(kept, key=lambda c: (c.start_frame, c.end_frame))
