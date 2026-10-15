"""[模組2] Stage 3：局部邊界逐格解碼與單格精確化（連續解碼優化版）。

設計規範（SRS §5）：
- 針對 Stage 2 候選區間之起點與終點前後 +-1 秒局部範圍。
- 改採「一次 Seek + 連續解碼」，徹底消除長影片長 GOP 重複 Seek 解碼卡頓。
- 優先對 Top-N 高分候選進行精確化，支援進度回報與取消。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from video_ad_remover.core.detector.stage1_scan import (
    compare_frames,
    compute_frame_features,
)
from video_ad_remover.core.frame_extractor import FrameExtractor
from video_ad_remover.models.interval import AdCandidate, FrameInterval
from video_ad_remover.models.video_meta import VideoMetadata


def _find_exact_cut_sequential(
    extractor: FrameExtractor,
    center_frame: int,
    window_frames: int,
    total_frames: int,
) -> int:
    """在 [center - window, center + window] 內以連續讀取尋找相鄰差異最大之切點。"""
    start_search = max(0, center_frame - window_frames)
    end_search = min(total_frames - 1, center_frame + window_frames)

    if start_search >= end_search:
        return center_frame

    # 連續解碼，尺寸縮小至 320x180 即可達到非常高的特徵對比度且極速
    frames_data = extractor.get_frame_range(start_search, end_search, target_size=(320, 180))
    if len(frames_data) < 2:
        return center_frame

    features = [(f, compute_frame_features(img)) for f, img in frames_data]

    max_diff = -1.0
    best_cut_frame = center_frame

    for i in range(len(features) - 1):
        f_curr, feat_curr = features[i]
        f_next, feat_next = features[i + 1]
        diff = compare_frames(feat_curr, feat_next)
        if diff > max_diff:
            max_diff = diff
            best_cut_frame = f_next

    return best_cut_frame


def refine_candidate_boundaries(
    candidates: list[AdCandidate],
    path: str | Path,
    meta: VideoMetadata,
    *,
    search_window_sec: float = 1.0,
    max_refine_candidates: int = 15,
    progress_cb: Callable[[float], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[AdCandidate]:
    """針對候選區間進行 Stage 3 邊界逐格精確化。"""
    if not candidates:
        return []

    cancel_event = cancel_event or threading.Event()
    fps = meta.nominal_fps if meta.nominal_fps > 0 else 30.0
    window_frames = max(3, round(fps * search_window_sec))

    # 依分數排序，優先精確化 Top 候選
    sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
    to_refine = sorted_cands[:max_refine_candidates]
    untouched = sorted_cands[max_refine_candidates:]

    refined: list[AdCandidate] = []
    extractor = FrameExtractor(path, meta)

    total_tasks = len(to_refine)

    with extractor:
        for idx, cand in enumerate(to_refine):
            if cancel_event.is_set():
                break

            exact_start = _find_exact_cut_sequential(
                extractor, cand.start_frame, window_frames, meta.frame_count
            )
            exact_end = _find_exact_cut_sequential(
                extractor, cand.end_frame, window_frames, meta.frame_count
            )

            # 防呆：精確化後若 start >= end，則退回原粗略切點
            if exact_start >= exact_end:
                exact_start = cand.start_frame
                exact_end = cand.end_frame

            refined_cand = AdCandidate(
                interval=FrameInterval(exact_start, exact_end),
                status=cand.status,
                source=cand.source,
                score=cand.score,
                score_detail=cand.score_detail,
                note=cand.note,
                id=cand.id,
            )
            refined.append(refined_cand)

            if progress_cb and total_tasks > 0:
                progress_cb((idx + 1) / total_tasks)

    # 合併未精確化的候選
    final_list = refined + untouched
    return sorted(final_list, key=lambda c: c.score, reverse=True)
