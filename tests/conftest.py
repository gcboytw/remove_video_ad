"""pytest 共用設定：FFmpeg 不存在時自動略過需要影片的測試；合成測試片 fixture。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_test_video import SyntheticSpec, generate  # noqa: E402

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="需要 FFmpeg / FFprobe")

# 測試用短片：40 秒 30fps = 1200 格；兩段非整秒對齊的廣告
CFR_SPEC = SyntheticSpec(duration=40, fps=30, ad_intervals=[(301, 451), (800, 950)])
VFR_SPEC = SyntheticSpec(duration=40, fps=30, ad_intervals=[(301, 451), (800, 950)], vfr_drop_mod=7)


@pytest.fixture(scope="session")
def cfr_video(tmp_path_factory) -> tuple[Path, SyntheticSpec]:
    if not HAS_FFMPEG:
        pytest.skip("需要 FFmpeg / FFprobe")
    path = tmp_path_factory.mktemp("media") / "synthetic_cfr.mp4"
    return generate(CFR_SPEC, path), CFR_SPEC


@pytest.fixture(scope="session")
def vfr_video(tmp_path_factory) -> tuple[Path, SyntheticSpec]:
    if not HAS_FFMPEG:
        pytest.skip("需要 FFmpeg / FFprobe")
    path = tmp_path_factory.mktemp("media") / "synthetic_vfr.mp4"
    return generate(VFR_SPEC, path), VFR_SPEC
