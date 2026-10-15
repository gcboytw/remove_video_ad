"""測試 Stage 1 快速掃描與 FrameExtractor（AT-003 全片掃描基礎）。"""

from __future__ import annotations

import pytest
from conftest import requires_ffmpeg

from video_ad_remover.core.detector.stage1_scan import scan_cut_points
from video_ad_remover.core.frame_extractor import FrameExtractor
from video_ad_remover.core.prober import probe_video

pytestmark = requires_ffmpeg


def test_frame_extractor(cfr_video):
    path, spec = cfr_video
    meta = probe_video(path)

    extractor = FrameExtractor(path, meta)
    with extractor:
        # 單格隨機讀取
        f0 = extractor.get_frame(0)
        assert f0 is not None
        assert f0.shape == (spec.height, spec.width, 3)

        # 縮圖讀取
        f_small = extractor.get_frame(10, target_size=(160, 90))
        assert f_small is not None
        assert f_small.shape == (90, 160, 3)

        # 超出範圍回傳 None
        assert extractor.get_frame(spec.total_frames + 100) is None


def test_stage1_detects_cuts(cfr_video):
    path, spec = cfr_video
    meta = probe_video(path)

    cuts = scan_cut_points(path, meta, sample_fps=3.0, diff_threshold=0.35)

    # cfr_video 內有兩段廣告: [(301, 451), (800, 950)]，每段有切入與切出，至少有 4 個切點
    assert len(cuts) >= 4

    # 驗證每個廣告的邊界附近都有被 Stage 1 的取樣切點捕捉到
    cut_frames = [c.frame_index for c in cuts]
    step = round(spec.fps / 3.0)  # 約 10 格容差

    for start_f, end_f in spec.ad_intervals:
        has_near_start = any(abs(cf - start_f) <= step * 1.5 for cf in cut_frames)
        has_near_end = any(abs(cf - end_f) <= step * 1.5 for cf in cut_frames)
        assert has_near_start, f"廣告起點 {start_f} 附近未找到切點"
        assert has_near_end, f"廣告終點 {end_f} 附近未找到切點"
