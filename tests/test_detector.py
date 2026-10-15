"""測試三階段廣告偵測演算法管線（AT-003, AT-004）。"""

from __future__ import annotations

import pytest
from conftest import requires_ffmpeg

from video_ad_remover.core.detector.pipeline import PipelineConfig, detect_ads
from video_ad_remover.core.detector.stage2_pair import PairConfig
from video_ad_remover.core.prober import probe_video

pytestmark = requires_ffmpeg


def test_full_detection_pipeline(cfr_video):
    path, spec = cfr_video
    meta = probe_video(path)

    # 針對測試片長度 (40s) 配置合理的搜尋時長窗（每段廣告約 150 格 = 5 秒）
    # 這裡驗證演算法彈性調整時長窗後，依然能精確配對與評分
    pair_cfg = PairConfig(
        target_duration=5.0,
        min_duration=3.0,
        max_duration=8.0,
        duration_sigma=2.0,
    )
    pipeline_cfg = PipelineConfig(
        sample_fps=3.0,
        diff_threshold=0.35,
        pair_config=pair_cfg,
        refine_window_sec=0.8,
    )

    candidates = detect_ads(path, meta, config=pipeline_cfg)

    # 必須成功抓出候選
    assert len(candidates) >= 2, f"應至少抓出 2 段候選，實際只抓到 {len(candidates)}"

    # 檢查 Top 候選區間是否與真值 [(301, 451), (800, 950)] 高度重合
    cand_intervals = [(c.start_frame, c.end_frame) for c in candidates]

    for true_start, true_end in spec.ad_intervals:
        matched = False
        for c_start, c_end in cand_intervals:
            # 容許精確化後有些微誤差 (<= 2 格)
            if abs(c_start - true_start) <= 2 and abs(c_end - true_end) <= 2:
                matched = True
                break
        assert matched, f"真值廣告 [{true_start}, {true_end}) 未能被精確配對到，候選清單：{cand_intervals}"
