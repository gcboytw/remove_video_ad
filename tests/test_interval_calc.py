"""半開區間換算、防呆、合併與邊界條件測試（AT-007 / AT-008 純數學部分）。"""

from __future__ import annotations

import pytest

from video_ad_remover.core.interval_calc import (
    collect_confirmed,
    compute_keep_segments,
    count_frames,
    dedupe_candidates,
    find_overlaps,
    merge_intervals,
    split_interval,
    union_interval,
    validate_ad_intervals,
)
from video_ad_remover.models.interval import (
    AdCandidate,
    ConfirmedInterval,
    DetectionSource,
    FrameInterval,
    IntervalStatus,
    InvalidIntervalError,
    KeepSegment,
)

FI = FrameInterval


# ---- FrameInterval 本身 ----------------------------------------------------
@pytest.mark.parametrize("start,end", [(5, 5), (6, 5), (-1, 3)])
def test_invalid_interval_rejected(start, end):
    with pytest.raises(InvalidIntervalError):
        FI(start, end)


def test_non_int_rejected():
    with pytest.raises(InvalidIntervalError):
        FI(1.0, 3)  # type: ignore[arg-type]
    with pytest.raises(InvalidIntervalError):
        FI(True, 3)  # type: ignore[arg-type]


def test_interval_properties():
    iv = FI(10, 20)
    assert iv.length == 10
    assert iv.last_frame == 19
    assert iv.contains_frame(10) and iv.contains_frame(19)
    assert not iv.contains_frame(20)
    assert not iv.overlaps(FI(20, 30))  # 相鄰不重疊
    assert iv.is_adjacent(FI(20, 30))
    assert iv.overlaps(FI(19, 30))
    assert iv.iou(FI(15, 25)) == pytest.approx(5 / 15)
    assert str(iv) == "[10, 20)"


def test_shift_keeps_subclass_and_validates():
    c = ConfirmedInterval(10, 20)
    shifted = c.shifted(start_delta=-1, end_delta=+1)
    assert isinstance(shifted, ConfirmedInterval)
    assert shifted.to_list() == [9, 21]
    with pytest.raises(InvalidIntervalError):
        FI(10, 11).shifted(start_delta=1)


# ---- 換算：SRS §6.1 範例 ----------------------------------------------------
def test_srs_example():
    keeps = compute_keep_segments([FI(10000, 10750), FI(30000, 30750)], 50000)
    assert keeps == [KeepSegment(0, 10000), KeepSegment(10750, 30000), KeepSegment(30750, 50000)]
    assert count_frames(keeps) == 50000 - 1500


def test_unsorted_input():
    keeps = compute_keep_segments([FI(30000, 30750), FI(10000, 10750)], 50000)
    assert keeps[0] == KeepSegment(0, 10000)


def test_ad_at_start_and_end():
    keeps = compute_keep_segments([FI(0, 100), FI(900, 1000)], 1000)
    assert keeps == [KeepSegment(100, 900)]


def test_adjacent_ads_ok():
    keeps = compute_keep_segments([FI(100, 200), FI(200, 300)], 1000)
    assert keeps == [KeepSegment(0, 100), KeepSegment(300, 1000)]


def test_no_ads_keeps_everything():
    assert compute_keep_segments([], 500) == [KeepSegment(0, 500)]


def test_single_frame_ad():
    keeps = compute_keep_segments([FI(42, 43)], 100)
    assert keeps == [KeepSegment(0, 42), KeepSegment(43, 100)]
    assert count_frames(keeps) == 99


# ---- 防呆 --------------------------------------------------------------------
def test_overlap_rejected_in_strict():
    with pytest.raises(InvalidIntervalError, match="重疊"):
        compute_keep_segments([FI(100, 200), FI(150, 300)], 1000)


def test_out_of_range_rejected():
    with pytest.raises(InvalidIntervalError, match="超出"):
        validate_ad_intervals([FI(900, 1001)], 1000)


def test_all_removed_rejected():
    with pytest.raises(InvalidIntervalError, match="沒有任何正片"):
        compute_keep_segments([FI(0, 1000)], 1000)


def test_non_strict_merges_and_clips():
    keeps = compute_keep_segments([FI(100, 200), FI(150, 300), FI(950, 2000)], 1000, strict=False)
    assert keeps == [KeepSegment(0, 100), KeepSegment(300, 950)]


def test_find_overlaps():
    pairs = find_overlaps([FI(0, 10), FI(5, 15), FI(15, 20), FI(14, 16)])
    assert (FI(0, 10), FI(5, 15)) in pairs
    assert (FI(5, 15), FI(14, 16)) in pairs
    assert len(pairs) == 3  # 還有 [14,16) 與 [15,20)


# ---- 合併 / 拆分 / 去重 -----------------------------------------------------
def test_merge_intervals():
    merged = merge_intervals([FI(0, 10), FI(10, 20), FI(25, 30), FI(28, 40)])
    assert merged == [FI(0, 20), FI(25, 40)]
    assert merge_intervals([FI(0, 10), FI(10, 20)], merge_adjacent=False) == [FI(0, 10), FI(10, 20)]
    assert merge_intervals([FI(0, 10), FI(15, 20)], max_gap=5) == [FI(0, 20)]


def test_split_and_union():
    a, b = split_interval(ConfirmedInterval(100, 200), 150)
    assert (a, b) == (ConfirmedInterval(100, 150), ConfirmedInterval(150, 200))
    with pytest.raises(InvalidIntervalError):
        split_interval(FI(100, 200), 100)
    assert union_interval(FI(0, 10), FI(20, 30)) == FI(0, 30)


def test_dedupe_keeps_highest_score():
    cands = [
        AdCandidate(FI(100, 850), score=0.6),
        AdCandidate(FI(105, 850), score=0.9),
        AdCandidate(FI(3000, 3750), score=0.5),
    ]
    kept = dedupe_candidates(cands)
    assert [c.score for c in kept] == [0.9, 0.5]


# ---- 候選與確認狀態 ----------------------------------------------------------
def test_only_confirmed_are_collected():
    cands = [
        AdCandidate(FI(300, 400), status=IntervalStatus.CONFIRMED_AD),
        AdCandidate(FI(100, 200), status=IntervalStatus.CONFIRMED_AD),
        AdCandidate(FI(500, 600), status=IntervalStatus.UNCONFIRMED),
        AdCandidate(FI(700, 800), status=IntervalStatus.EXCLUDED),
    ]
    assert collect_confirmed(cands) == [ConfirmedInterval(100, 200), ConfirmedInterval(300, 400)]


def test_candidate_adjust_and_roundtrip():
    cand = AdCandidate(FI(100, 850), source=DetectionSource.MANUAL, score=0.7)
    cand.adjust_start(-1)
    cand.adjust_end(+1)
    assert (cand.start_frame, cand.end_frame) == (99, 851)
    with pytest.raises(InvalidIntervalError):
        cand.set_bounds(500, 400)
    assert (cand.start_frame, cand.end_frame) == (99, 851)  # 失敗時原值不變

    restored = AdCandidate.from_dict(cand.to_dict())
    assert restored == cand
    with pytest.raises(InvalidIntervalError):
        cand.to_confirmed()
