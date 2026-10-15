"""[模組2] 三階段廣告偵測總控管線（Pipeline）。

依序執行：
1. Stage 1：全片低解析度快速掃描與粗略切點集合 T_cuts 產出。
2. Stage 2：時間窗配對、圖卡特徵穩定度分析、評分排序與 IoU 去重。
3. Stage 3：局部邊界 +-1s 原始解析度逐格解碼精確化。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from video_ad_remover.core.detector.stage1_scan import scan_cut_points
from video_ad_remover.core.detector.stage2_pair import PairConfig, pair_and_score
from video_ad_remover.core.detector.stage3_refiner import refine_candidate_boundaries
from video_ad_remover.models.interval import AdCandidate
from video_ad_remover.models.video_meta import VideoMetadata


@dataclass(frozen=True)
class PipelineConfig:
    sample_fps: float = 3.0  # Stage 1 取樣頻率
    diff_threshold: float = 0.42  # 提高切點門檻，過濾正片正常鏡頭轉場
    pair_config: PairConfig = PairConfig()  # Stage 2 參數（時長窗、內部穩定度等）
    refine_window_sec: float = 1.0  # Stage 3 搜尋窗秒數


def detect_ads(
    path: str | Path,
    meta: VideoMetadata,
    *,
    config: PipelineConfig | None = None,
    stage_cb: Callable[[str, float], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[AdCandidate]:
    """執行完整三階段廣告偵測管線。

    Args:
        path: 影片路徑。
        meta: 影片詮釋資料。
        config: 管線設定參數。
        stage_cb: 階段與進度回報回呼 (stage_name, progress_0_to_1)。
        cancel_event: 取消信號。
    """
    cfg = config or PipelineConfig()
    cancel_event = cancel_event or threading.Event()

    def report(stage: str, frac: float):
        if stage_cb is not None:
            stage_cb(stage, frac)

    # ---- Stage 1：快速切點掃描 ----
    report("Stage 1: 全片快速掃描", 0.0)
    cuts = scan_cut_points(
        path,
        meta,
        sample_fps=cfg.sample_fps,
        diff_threshold=cfg.diff_threshold,
        progress_cb=lambda p: report("Stage 1: 全片快速掃描", p),
        cancel_event=cancel_event,
    )
    if cancel_event.is_set():
        return []

    # ---- Stage 2：時間窗配對與評分 ----
    report("Stage 2: 廣告區間配對與評分", 0.3)
    raw_candidates = pair_and_score(cuts, path, meta, config=cfg.pair_config)
    report("Stage 2: 廣告區間配對與評分", 1.0)
    if cancel_event.is_set() or not raw_candidates:
        return raw_candidates

    # ---- Stage 3：局部邊界逐格精確化 ----
    report("Stage 3: 邊界逐格精確化", 0.0)
    refined_candidates = refine_candidate_boundaries(
        raw_candidates,
        path,
        meta,
        search_window_sec=cfg.refine_window_sec,
        progress_cb=lambda p: report("Stage 3: 邊界逐格精確化", p),
        cancel_event=cancel_event,
    )
    report("Stage 3: 邊界逐格精確化", 1.0)

    # 依分數由高到低排序，優先凸顯最可能的候選（FR-005）
    return sorted(refined_candidates, key=lambda c: c.score, reverse=True)
