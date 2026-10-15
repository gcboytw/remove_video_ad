"""精確剪輯驗收測試：AT-001 / AT-008 / AT-009 / AT-010 / AT-012，以及 FR-012 取消與原檔保護。"""

from __future__ import annotations

import hashlib
import threading

import pytest
from conftest import requires_ffmpeg
from run_cut_check import cut_and_check

from video_ad_remover.core.exporter import (
    ExportCancelled,
    ExportSettings,
    plan_export,
    run_export,
)
from video_ad_remover.core.prober import probe_video
from video_ad_remover.models.interval import FrameInterval

pytestmark = requires_ffmpeg
FAST = ExportSettings(overwrite=True, preset="ultrafast", crf=20)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- AT-001 / AT-012：影片資訊與 PTS 索引 ----------------------------------
def test_probe_cfr(cfr_video):
    path, spec = cfr_video
    meta = probe_video(path)
    assert meta.frame_count == spec.total_frames
    assert not meta.is_vfr
    assert meta.nominal_fps == pytest.approx(spec.fps)
    assert (meta.width, meta.height) == (spec.width, spec.height)
    assert meta.has_audio
    assert meta.time_of(spec.fps) == pytest.approx(1.0, abs=1e-3)
    assert meta.frame_at_time(1.0) == spec.fps
    assert meta.segment_duration(0, meta.frame_count) == pytest.approx(spec.duration, abs=1e-3)


def test_probe_vfr(vfr_video):
    path, spec = vfr_video
    meta = probe_video(path)
    dropped = sum(1 for n in range(spec.total_frames) if n % 7 == 3)
    assert meta.is_vfr
    assert meta.frame_count == spec.total_frames - dropped


# ---- AT-008 / AT-009 / AT-010：CFR 精確剪輯 --------------------------------
@pytest.mark.parametrize(
    "cuts",
    [
        [(301, 451), (800, 950)],  # 兩段中插廣告
        [(0, 45), (301, 451), (1150, 1200)],  # 含片頭、片尾邊界
        [(100, 101), (599, 600), (1000, 1001)],  # 單格刪除
    ],
    ids=["two-ads", "head-tail", "single-frames"],
)
def test_cfr_frame_accurate_cut(cfr_video, tmp_path, cuts):
    path, spec = cfr_video
    before = _sha256(path)
    result = cut_and_check(
        path,
        [FrameInterval(s, e) for s, e in cuts],
        tmp_path / "out.mp4",
        spec=spec,
        settings=FAST,
        verbose=False,
    )
    assert result.verification.ok, result.verification.summary()
    assert result.frames is not None and result.frames.ok, result.frames.summary()
    assert result.sync is not None and len(result.sync.measured) >= 20
    assert result.sync_ok, result.sync.summary(spec.fps)
    assert _sha256(path) == before  # AT-011：原始檔不變


# ---- AT-012：VFR 精確剪輯 --------------------------------------------------
def test_vfr_frame_accurate_cut(vfr_video, tmp_path):
    path, spec = vfr_video
    meta = probe_video(path)
    cuts = [
        FrameInterval(meta.frame_at_time(s / spec.fps), meta.frame_at_time(e / spec.fps))
        for s, e in spec.ad_intervals
    ]
    result = cut_and_check(path, cuts, tmp_path / "out.mp4", spec=spec, settings=FAST, verbose=False)
    assert result.plan.graph.mode == "time"
    assert result.verification.ok, result.verification.summary()
    assert result.frames is not None and result.frames.ok, result.frames.summary()
    assert result.sync_ok, result.sync.summary(spec.fps)
    assert result.verification.metadata is not None and result.verification.metadata.is_vfr


# ---- FR-012 取消 / FR-001 原檔保護 -----------------------------------------
def test_cancel_removes_partial(cfr_video, tmp_path):
    path, _ = cfr_video
    meta = probe_video(path)
    plan = plan_export(meta, [FrameInterval(301, 451)], tmp_path / "cancel.mp4", FAST)
    cancel = threading.Event()
    with pytest.raises(ExportCancelled):
        run_export(plan, progress_cb=lambda p: cancel.set(), cancel_event=cancel)
    assert not plan.partial_path.exists()
    assert not plan.output.exists()


def test_refuse_overwrite_source(cfr_video):
    path, _ = cfr_video
    meta = probe_video(path)
    with pytest.raises(ValueError, match="原始"):
        plan_export(meta, [FrameInterval(301, 451)], path)


def test_refuse_existing_output_without_overwrite(cfr_video, tmp_path):
    path, _ = cfr_video
    meta = probe_video(path)
    out = tmp_path / "exists.mp4"
    out.write_bytes(b"x")
    plan = plan_export(meta, [FrameInterval(301, 451)], out, ExportSettings(overwrite=False))
    with pytest.raises(FileExistsError):
        run_export(plan)
