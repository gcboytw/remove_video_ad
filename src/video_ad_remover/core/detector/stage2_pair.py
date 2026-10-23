"""[模組2] Stage 2：持續性分析、時間窗配對、特徵評分與去重（精準度強化版）。

設計規範（SRS §5 / FR-004 / FR-005）：
- 配對滿足 min_duration <= (t_out - t_in) <= max_duration 之切點對（預設 18～32 秒，以 25 秒為峰值）。
- 廣告內部特徵檢驗：在區間內均勻取樣 5~7 格，評估內部版面底色穩定度（過濾正片運鏡與鏡頭頻繁切換）。
- 前後正片呼應度：比對 t_in 前與 t_out 後之畫面相似度（廣告結束接回同一場景則大幅加分）。
- 綜合計算優先級分數並依 IoU 門檻去重。
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

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
from video_ad_remover.utils.time_conv import format_timestamp


@dataclass
class Stage2Stats:
    total_pairs_checked: int = 0
    duration_filtered: int = 0
    boundary_score_filtered: int = 0
    evaluated_pairs: int = 0
    early_stopped_pairs: int = 0
    insufficient_samples_pairs: int = 0
    get_frame_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    decode_time: float = 0.0
    feature_compute_time: float = 0.0
    wall_time: float = 0.0
    surviving_candidates: int = 0


@dataclass(frozen=True)
class PairConfig:
    target_duration: float = 25.0  # 目標廣告秒數
    min_duration: float = 18.0  # 縮小搜尋範圍下限，過濾過短正片切換
    max_duration: float = 32.0  # 縮小搜尋範圍上限
    duration_sigma: float = 3.5  # 緊密聚焦在 25 秒附近
    iou_threshold: float = 0.40  # 降低重複候選門檻
    max_internal_variance: float = 0.45  # 內部取樣平均差異上限，超過視為正片運鏡淘汰
    stability_sigma: float = 0.12  # 圖卡內部穩定度衰減常數（凸顯靜態圖卡優勢，壓制正片運鏡）
    cache_max_size: int = 1024  # 區域特徵快取上限容量 (LRU)
    min_valid_samples: int = 3  # 最少需要 3 個有效內部特徵樣本（至少 2 次相鄰比較）
    min_boundary_diff_score: float = 0.0  # 起迄點最小轉場強度先驗門檻
    enable_early_stopping: bool = True  # 啟用數學安全型不可逆超標早停


def compute_aspect_preserved_size(
    orig_w: int, orig_h: int, max_w: int = 160, max_h: int = 90
) -> tuple[int, int]:
    """計算維持長寬比的等比例縮圖尺寸（避免非 16:9 畫面拉伸失真）。"""
    if orig_w <= 0 or orig_h <= 0:
        return (max_w, max_h)
    scale = min(max_w / orig_w, max_h / orig_h)
    new_w = max(2, int(round(orig_w * scale)) // 2 * 2)
    new_h = max(2, int(round(orig_h * scale)) // 2 * 2)
    return (new_w, new_h)


def _duration_score(duration: float, target: float, sigma: float) -> float:
    """以高斯分佈評估時長與 target 的接近程度 (0.0 ~ 1.0)。"""
    diff = duration - target
    return math.exp(-0.5 * (diff / sigma) ** 2)


def pair_and_score(
    cut_points: list[CutPoint],
    path: str | Path,
    meta: VideoMetadata,
    config: PairConfig | None = None,
    stats: Stage2Stats | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[AdCandidate]:
    """將切點集合配對成候選廣告區間，並評分與去重。"""
    cfg = config or PairConfig()
    if len(cut_points) < 2:
        return []

    t_start = time.perf_counter()
    candidates: list[AdCandidate] = []
    extractor = FrameExtractor(path, meta)
    feat_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray] | None] = OrderedDict()
    target_size = compute_aspect_preserved_size(meta.width, meta.height, 160, 90)
    cancel_event = cancel_event or threading.Event()
    last_cb_time = 0.0

    def _get_frame(f_idx: int):
        if stats is not None:
            stats.get_frame_calls += 1
            t0 = time.perf_counter()
            f = extractor.get_frame(f_idx, target_size=target_size)
            stats.decode_time += time.perf_counter() - t0
        else:
            f = extractor.get_frame(f_idx, target_size=target_size)
        if f is None or f.size == 0:
            return None
        return f

    def _calc_feat(img):
        if stats is not None:
            t0 = time.perf_counter()
            res = compute_frame_features(img)
            stats.feature_compute_time += time.perf_counter() - t0
            return res
        return compute_frame_features(img)

    def _get_cached_feature(f_idx: int):
        if f_idx in feat_cache:
            if stats is not None:
                stats.cache_hits += 1
            feat_cache.move_to_end(f_idx)
            return feat_cache[f_idx]

        if stats is not None:
            stats.cache_misses += 1

        img = _get_frame(f_idx)
        feat = _calc_feat(img) if img is not None else None
        feat_cache[f_idx] = feat
        if len(feat_cache) > cfg.cache_max_size:
            feat_cache.popitem(last=False)
        return feat

    def _comp_frames(f1, f2) -> float:
        if stats is not None:
            t0 = time.perf_counter()
            res = compare_frames(f1, f2)
            stats.feature_compute_time += time.perf_counter() - t0
            return res
        return compare_frames(f1, f2)

    with extractor:
        n = len(cut_points)
        for i in range(n):
            if cancel_event.is_set():
                break

            cp_in = cut_points[i]

            # ETA 倒數與候選心跳狀態回報 (節流約每 0.25 秒或最後一格發送一次)
            now = time.perf_counter()
            if progress_cb is not None and (now - last_cb_time >= 0.25 or i == n - 1):
                last_cb_time = now
                frac = (i + 1) / n
                elapsed = now - t_start
                if frac > 0.02 and elapsed > 1.0:
                    eta_sec = max(0.0, (elapsed / frac) - elapsed)
                    m, s = divmod(int(eta_sec), 60)
                    eta_str = f"{m:02d}:{s:02d}"
                else:
                    eta_str = "--:--"
                t_str = format_timestamp(cp_in.timestamp)[:8]
                status_msg = f"切點 {i+1}/{n} ({t_str}) | 鎖定 {len(candidates)} 候選 | 預估剩餘 {eta_str}"
                progress_cb(frac, status_msg)

            for j in range(i + 1, n):
                if stats is not None:
                    stats.total_pairs_checked += 1
                cp_out = cut_points[j]
                dur = cp_out.timestamp - cp_in.timestamp

                if dur < cfg.min_duration:
                    if stats is not None:
                        stats.duration_filtered += 1
                    continue
                if dur > cfg.max_duration:
                    if stats is not None:
                        stats.duration_filtered += 1
                    break

                if stats is not None:
                    stats.evaluated_pairs += 1

                # 1. 時長分數（25 秒峰值）
                score_dur = _duration_score(dur, cfg.target_duration, cfg.duration_sigma)

                # 2. 切點強度分數
                score_cut = 0.5 * (cp_in.diff_score + cp_out.diff_score)
                if cfg.min_boundary_diff_score > 0.0 and score_cut < cfg.min_boundary_diff_score:
                    if stats is not None:
                        stats.boundary_score_filtered += 1
                    continue

                # 3. 採樣正片前後（各 1 秒處），改為時間軸標準映射（VFR/CFR 安全）
                t_in = cp_in.timestamp
                t_out = cp_out.timestamp
                f_before = meta.frame_at_time(max(0.0, t_in - 1.0))
                f_after = meta.frame_at_time(min(meta.end_time, t_out + 1.0))

                feat_b = _get_cached_feature(f_before)
                feat_a = _get_cached_feature(f_after)

                score_context = 0.5
                if feat_b is not None and feat_a is not None:
                    # 前後正片若連續（同場景/色調相似），相似度越高加分越多
                    diff_context = _comp_frames(feat_b, feat_a)
                    score_context = 1.0 - diff_context

                # 4. 廣告區間內部「版面色彩穩定度」取樣檢驗（依時間軸均勻採樣 4 個時間點）
                s_frame = cp_in.frame_index
                e_frame = cp_out.frame_index
                sample_times = [
                    t_in + (t_out - t_in) * r for r in (0.2, 0.4, 0.6, 0.8)
                ]
                internal_samples = [meta.frame_at_time(t) for t in sample_times]

                internal_feats = []
                internal_diffs = []
                early_stop = False
                expected_diffs = 3  # 4 samples -> 3 adjacent comparisons
                sum_limit = cfg.max_internal_variance * expected_diffs
                accumulated_diff = 0.0

                for f_idx in internal_samples:
                    feat = _get_cached_feature(f_idx)
                    if feat is not None:
                        internal_feats.append(feat)
                        if len(internal_feats) >= 2:
                            diff_val = _comp_frames(internal_feats[-2], internal_feats[-1])
                            internal_diffs.append(diff_val)
                            accumulated_diff += diff_val
                            # 數學安全早停：即使剩餘未測差異全部為 0，累積總和仍必定超標
                            if cfg.enable_early_stopping and accumulated_diff > sum_limit:
                                early_stop = True
                                if stats is not None:
                                    stats.early_stopped_pairs += 1
                                break

                if early_stop:
                    continue

                # 若內部有效樣本少於門檻（例如 < 3），代表過半影格解碼失敗，判定為不可信區間淘汰
                if len(internal_feats) < cfg.min_valid_samples:
                    if stats is not None:
                        stats.insufficient_samples_pairs += 1
                    continue

                avg_internal_diff = (
                    sum(internal_diffs) / len(internal_diffs) if internal_diffs else 0.5
                )

                # 若內部畫面變化太劇烈（鏡頭一直換），直接判定為正片劇情淘汰！
                if avg_internal_diff > cfg.max_internal_variance:
                    continue

                # 內部版面穩定度分數：以高斯衰減重賞高度靜止之圖卡特徵
                # 靜態圖卡 (avg_diff ~ 0.0) 拿滿分 1.0；正片微幅運鏡 (avg_diff >= 0.12) 分數迅速衰減
                score_stability = math.exp(-0.5 * (avg_internal_diff / cfg.stability_sigma) ** 2)

                # 5. 內部與正片的反差度（廣告圖卡與前後正片完全不同風格）
                score_contrast = 0.5
                if internal_feats and feat_b and feat_a:
                    mid_feat = internal_feats[len(internal_feats) // 2]
                    contrast_b = _comp_frames(mid_feat, feat_b)
                    contrast_a = _comp_frames(mid_feat, feat_a)
                    score_contrast = 0.5 * (contrast_b + contrast_a)

                # 綜合加權：內部穩定度(35%) + 時長(25%) + 切點強度(20%) + 前後正片呼應(10%) + 反差(10%)
                total_score = (
                    0.35 * score_stability
                    + 0.25 * score_dur
                    + 0.20 * score_cut
                    + 0.10 * score_context
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
    result = dedupe_candidates(candidates, iou_threshold=cfg.iou_threshold)
    if stats is not None:
        stats.wall_time = time.perf_counter() - t_start
        stats.surviving_candidates = len(result)
    return result
