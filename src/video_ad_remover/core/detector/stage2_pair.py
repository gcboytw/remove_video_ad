"""[模組2] Stage 2：持續性分析、時間窗配對、特徵評分與去重（精準度強化版）。

設計規範（SRS §5 / FR-004 / FR-005）：
- 配對滿足 min_duration <= (t_out - t_in) <= max_duration 之切點對（預設 18～32 秒，以 25 秒為峰值）。
- 廣告內部特徵檢驗：在區間內均勻取樣 5~7 格，評估內部版面底色穩定度（過濾正片運鏡與鏡頭頻繁切換）。
- 前後正片呼應度：比對 t_in 前與 t_out 後之畫面相似度（廣告結束接回同一場景則大幅加分）。
- 綜合計算優先級分數並依 IoU 門檻去重。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from video_ad_remover.core.detector.stage1_scan import (
    CutPoint,
    compare_frames,
    compute_frame_features,
)
from video_ad_remover.core.frame_extractor import FrameExtractor
from video_ad_remover.core.interval_calc import dedupe_candidates
from video_ad_remover.models.interval import (
    AdCandidate,
    DetectionSource,
    FrameInterval,
    IntervalStatus,
)
from video_ad_remover.models.video_meta import VideoMetadata


@dataclass(frozen=True)
class PairConfig:
    target_duration: float = 25.0  # 目標廣告秒數
    min_duration: float = 18.0  # 縮小搜尋範圍下限，過濾過短正片切換
    max_duration: float = 32.0  # 縮小搜尋範圍上限
    duration_sigma: float = 3.5  # 緊密聚焦在 25 秒附近
    iou_threshold: float = 0.40  # 降低重複候選門檻
    max_internal_variance: float = 0.45  # 內部取樣平均差異上限，超過視為正片運鏡淘汰


def _duration_score(duration: float, target: float, sigma: float) -> float:
    """以高斯分佈評估時長與 target 的接近程度 (0.0 ~ 1.0)。"""
    diff = duration - target
    return math.exp(-0.5 * (diff / sigma) ** 2)


def pair_and_score(
    cut_points: list[CutPoint],
    path: str | Path,
    meta: VideoMetadata,
    config: PairConfig | None = None,
) -> list[AdCandidate]:
    """將切點集合配對成候選廣告區間，並評分與去重。"""
    cfg = config or PairConfig()
    if len(cut_points) < 2:
        return []

    candidates: list[AdCandidate] = []
    extractor = FrameExtractor(path, meta)
    fps = meta.nominal_fps if meta.nominal_fps > 0 else 30.0

    with extractor:
        n = len(cut_points)
        for i in range(n):
            cp_in = cut_points[i]
            for j in range(i + 1, n):
                cp_out = cut_points[j]
                dur = cp_out.timestamp - cp_in.timestamp

                if dur < cfg.min_duration:
                    continue
                if dur > cfg.max_duration:
                    break

                # 1. 時長分數（25 秒峰值）
                score_dur = _duration_score(dur, cfg.target_duration, cfg.duration_sigma)

                # 2. 切點強度分數
                score_cut = 0.5 * (cp_in.diff_score + cp_out.diff_score)

                # 3. 採樣正片前後（各 1 秒處）
                f_before = max(0, int(cp_in.frame_index - fps))
                f_after = min(meta.frame_count - 1, int(cp_out.frame_index + fps))

                img_before = extractor.get_frame(f_before, target_size=(160, 90))
                img_after = extractor.get_frame(f_after, target_size=(160, 90))

                score_context = 0.5
                feat_b = None
                feat_a = None
                if img_before is not None and img_after is not None:
                    feat_b = compute_frame_features(img_before)
                    feat_a = compute_frame_features(img_after)
                    # 前後正片若連續（同場景/色調相似），相似度越高加分越多
                    diff_context = compare_frames(feat_b, feat_a)
                    score_context = 1.0 - diff_context

                # 4. 廣告區間內部「版面色彩穩定度」取樣檢驗（取 5 個時間點）
                s_frame = cp_in.frame_index
                e_frame = cp_out.frame_index
                internal_samples = [
                    int(s_frame + (e_frame - s_frame) * r) for r in (0.2, 0.4, 0.6, 0.8)
                ]

                internal_feats = []
                for f_idx in internal_samples:
                    img = extractor.get_frame(f_idx, target_size=(160, 90))
                    if img is not None:
                        internal_feats.append(compute_frame_features(img))

                # 計算內部各取樣點之間的平均變化率
                internal_diffs = []
                for k in range(len(internal_feats) - 1):
                    internal_diffs.append(
                        compare_frames(internal_feats[k], internal_feats[k + 1])
                    )

                avg_internal_diff = (
                    sum(internal_diffs) / len(internal_diffs) if internal_diffs else 0.5
                )

                # 若內部畫面變化太劇烈（鏡頭一直換），直接判定為正片劇情淘汰！
                if avg_internal_diff > cfg.max_internal_variance:
                    continue

                # 內部版面穩定度分數：變化越少，分數越高
                score_stability = max(0.0, 1.0 - (avg_internal_diff / cfg.max_internal_variance))

                # 5. 內部與正片的反差度（廣告圖卡與前後正片完全不同風格）
                score_contrast = 0.5
                if internal_feats and feat_b and feat_a:
                    mid_feat = internal_feats[len(internal_feats) // 2]
                    contrast_b = compare_frames(mid_feat, feat_b)
                    contrast_a = compare_frames(mid_feat, feat_a)
                    score_contrast = 0.5 * (contrast_b + contrast_a)

                # 綜合加權：時長(30%) + 內部穩定度(25%) + 切點強度(20%) + 前後正片呼應(15%) + 反差(10%)
                total_score = (
                    0.30 * score_dur
                    + 0.25 * score_stability
                    + 0.20 * score_cut
                    + 0.15 * score_context
                    + 0.10 * score_contrast
                )

                score_detail = {
                    "duration": round(score_dur, 3),
                    "stability": round(score_stability, 3),
                    "cut_strength": round(score_cut, 3),
                    "context": round(score_context, 3),
                    "contrast": round(score_contrast, 3),
                }

                cand = AdCandidate(
                    interval=FrameInterval(s_frame, e_frame),
                    status=IntervalStatus.UNCONFIRMED,
                    source=DetectionSource.AUTO_SCAN,
                    score=round(total_score, 3),
                    score_detail=score_detail,
                )
                candidates.append(cand)

    # 去重
    return dedupe_candidates(candidates, iou_threshold=cfg.iou_threshold)
