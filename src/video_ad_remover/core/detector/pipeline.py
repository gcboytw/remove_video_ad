"""[模組2] 三階段廣告偵測總控管線（Pipeline）。

依序執行：
1. Stage 1：全片低解析度快速掃描與粗略切點集合 T_cuts 產出。
2. Stage 2：時間窗配對、圖卡特徵穩定度分析、評分排序與 IoU 去重。
3. Stage 3：局部邊界 +-1s 原始解析度逐格解碼精確化。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from video_ad_remover.core.detector.stage1_scan import scan_cut_points
from video_ad_remover.core.detector.stage2_pair import PairConfig, pair_and_score
from video_ad_remover.core.detector.stage3_refiner import refine_candidate_boundaries
from video_ad_remover.core.detector.stage_audio_visual import (
    AudioVisualAdConfig,
    AudioVisualDetector,
)
from video_ad_remover.models.interval import AdCandidate
from video_ad_remover.models.video_meta import VideoMetadata


@dataclass(frozen=True)
class PipelineConfig:
    sample_fps: float = 3.0  # Stage 1 取樣頻率
    diff_threshold: float = 0.35  # 平衡切點門檻，確保特殊壓製與低碼率影片切點不被誤殺
    pair_config: PairConfig = PairConfig()  # Stage 2 參數（時長窗、內部穩定度等）
    refine_window_sec: float = 1.0  # Stage 3 搜尋窗秒數
    enable_static_card: bool = True  # 是否啟用圖卡型廣告偵測（Stage 1~3）
    enable_animation_ad: bool = True  # 是否啟用動畫型商業廣告偵測（音視融合）
    audio_visual_config: AudioVisualAdConfig = field(default_factory=AudioVisualAdConfig)


def _merge_candidates(
    av_cands: list[AdCandidate], static_cands: list[AdCandidate]
) -> list[AdCandidate]:
    """將音視動畫廣告候選與視覺圖卡候選進行無衝突聯集與去重。"""
    merged: list[AdCandidate] = list(av_cands)
    for s_cand in static_cands:
        is_overlap = False
        for m_cand in merged:
            if s_cand.interval.iou(m_cand.interval) > 0.25 or s_cand.interval.overlaps(m_cand.interval):
                is_overlap = True
                # 若重疊，保留分數較高者或標記註解
                if s_cand.score > m_cand.score:
                    m_cand.score = s_cand.score
                break
        if not is_overlap:
            merged.append(s_cand)
    return sorted(merged, key=lambda c: c.score, reverse=True)


def detect_ads(
    path: str | Path,
    meta: VideoMetadata,
    *,
    config: PipelineConfig | None = None,
    stage_cb: Callable[[str, float], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[AdCandidate]:
    """執行廣告偵測管線（支援動畫廣告跨模態快篩與三階段圖卡偵測）。

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

    av_candidates: list[AdCandidate] = []
    static_candidates: list[AdCandidate] = []

    # ---- 1. 動畫型廣告跨模態快篩 ----
    if cfg.enable_animation_ad:
        report("動畫廣告: 全片音視響度快篩", 0.0)
        av_detector = AudioVisualDetector(config=cfg.audio_visual_config)
        av_segments = av_detector.detect(
            path,
            meta,
            progress_callback=lambda p, msg: report(f"動畫廣告: {msg}", p),
        )
        if cancel_event.is_set():
            return []
        av_candidates = av_detector.to_ad_candidates(av_segments)
        report("動畫廣告: 偵測完成", 1.0)

    # ---- 2. 圖卡型廣告三階段掃描 ----
    if cfg.enable_static_card:
        # Stage 1：快速切點掃描
        report("圖卡 Stage 1: 全片快速掃描", 0.0)
        cuts = scan_cut_points(
            path,
            meta,
            sample_fps=cfg.sample_fps,
            diff_threshold=cfg.diff_threshold,
            progress_cb=lambda p: report("圖卡 Stage 1: 全片快速掃描", p),
            cancel_event=cancel_event,
        )
        if cancel_event.is_set():
            return av_candidates

        # Stage 2：時間窗配對與評分
        report("圖卡 Stage 2: 廣告區間配對與評分", 0.0)
        raw_candidates = pair_and_score(
            cuts,
            path,
            meta,
            config=cfg.pair_config,
            progress_cb=lambda frac, msg: report(f"圖卡 Stage 2: {msg}", frac),
            cancel_event=cancel_event,
        )
        report("圖卡 Stage 2: 配對完成", 1.0)
        if cancel_event.is_set():
            return av_candidates

        # Stage 3：局部邊界逐格精確化
        if raw_candidates:
            report("圖卡 Stage 3: 邊界逐格精確化", 0.0)
            static_candidates = refine_candidate_boundaries(
                raw_candidates,
                path,
                meta,
                search_window_sec=cfg.refine_window_sec,
                progress_cb=lambda p: report("圖卡 Stage 3: 邊界逐格精確化", p),
                cancel_event=cancel_event,
            )
            report("圖卡 Stage 3: 精確化完成", 1.0)

    # ---- 3. 結果合併與排序 ----
    final_candidates = _merge_candidates(av_candidates, static_candidates)
    return final_candidates
