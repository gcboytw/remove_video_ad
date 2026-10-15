"""命令列端到端驗證：合成測試片 -> 剪掉廣告區間 -> FFprobe 驗證 + 逐格比對 + 影音同步量測。

用法::

    uv run python tests/run_cut_check.py                 # CFR，兩段 25 秒廣告
    uv run python tests/run_cut_check.py --vfr           # VFR 來源
    uv run python tests/run_cut_check.py --cut 0:45 --cut 1234:1984 --cut 4400:4500
    uv run python tests/run_cut_check.py --source 影片.mp4 --cut 10000:10750 --cut 30000:30750
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from avcheck import (  # noqa: E402
    FrameCheck,
    SyncCheck,
    check_frame_accuracy,
    decode_frame_codes,
    measure_av_sync,
)
from generate_test_video import SyntheticSpec, generate, parse_ads  # noqa: E402

from video_ad_remover.core.env_checker import check_environment  # noqa: E402
from video_ad_remover.core.exporter import (  # noqa: E402
    ExportPlan,
    ExportProgress,
    ExportSettings,
    plan_export,
    run_export,
)
from video_ad_remover.core.prober import probe_video  # noqa: E402
from video_ad_remover.core.verifier import VerificationReport, verify_output  # noqa: E402
from video_ad_remover.models.interval import FrameInterval  # noqa: E402
from video_ad_remover.models.video_meta import VideoMetadata  # noqa: E402
from video_ad_remover.utils.time_conv import format_timestamp  # noqa: E402


@dataclass
class CutCheckResult:
    source_meta: VideoMetadata
    plan: ExportPlan
    verification: VerificationReport
    frames: FrameCheck | None = None
    sync: SyncCheck | None = None
    fps: int = 0

    @property
    def sync_ok(self) -> bool:
        if self.sync is None:
            return True
        # 允許誤差：半格（遠小於人可察覺的 ~40ms）
        return bool(self.sync.measured) and self.sync.max_abs_offset() < 0.5 / self.fps

    @property
    def ok(self) -> bool:
        return self.verification.ok and (self.frames is None or self.frames.ok) and self.sync_ok


def _progress_printer(p: ExportProgress) -> None:
    eta = f"{p.eta_seconds:5.1f}s" if p.eta_seconds is not None else "  ?  "
    speed = f"{p.speed:.2f}x" if p.speed else "?"
    print(f"\r  輸出中 {p.fraction * 100:5.1f}%  frame={p.frame:6d}  speed={speed:>7s}  ETA={eta}", end="")
    if p.fraction >= 1:
        print()


def cut_and_check(
    source: Path,
    cuts: list[FrameInterval],
    output: Path,
    *,
    spec: SyntheticSpec | None = None,
    settings: ExportSettings | None = None,
    verbose: bool = True,
) -> CutCheckResult:
    log = print if verbose else (lambda *a, **k: None)
    meta = probe_video(source)
    log("\n[1] 來源影片資訊")
    for key, value in meta.summary().items():
        log(f"  {key}：{value}")

    settings = settings or ExportSettings(overwrite=True, preset="veryfast")
    plan = plan_export(meta, cuts, output, settings)
    log(f"\n[2] 剪輯計畫（trim 模式：{plan.graph.mode}）")
    for c in sorted(cuts, key=lambda c: c.start_frame):
        log(f"  刪除 {c}  {format_timestamp(meta.time_of(c.start_frame))} ~ "
            f"{format_timestamp(meta.time_of(c.end_frame))}  ({c.length} 格)")
    for s in plan.segments:
        log(f"  保留 {s}  ({s.length} 格, {meta.segment_duration(s.start_frame, s.end_frame):.3f}s)")
    log(f"  預估輸出：{plan.expected_frames} 格 / {plan.expected_duration:.3f}s")

    log("\n[3] FFmpeg 輸出")
    result = run_export(plan, progress_cb=_progress_printer if verbose else None)
    log(f"  完成：{result.output}（{result.elapsed:.1f}s）")

    report = verify_output(
        result.output,
        expected_duration=plan.expected_duration,
        expected_frames=plan.expected_frames,
        expect_audio=meta.has_audio,
    )
    log("\n[4] " + report.summary())
    check = CutCheckResult(source_meta=meta, plan=plan, verification=report)

    if spec is not None:
        check.fps = spec.fps
        src_codes = decode_frame_codes(source, width=spec.width)
        if len(src_codes) != meta.frame_count:
            raise RuntimeError(f"來源條碼解碼格數 {len(src_codes)} 與 PTS 索引 {meta.frame_count} 不符")
        out_codes = decode_frame_codes(result.output, width=spec.width)
        check.frames = check_frame_accuracy(out_codes, src_codes, plan.segments)
        log("\n[5] " + check.frames.summary())

        tail_start = sum(s.length for s in plan.segments[:-1])
        check.sync = measure_av_sync(
            result.output,
            out_codes=out_codes,
            fps=spec.fps,
            sample_rate=spec.sample_rate,
            is_ad=spec.is_ad_frame,
            tail_start_frame=tail_start,
        )
        log("[6] " + check.sync.summary(spec.fps))

    log(f"\n==> 總結：{'全部通過 ✅' if check.ok else '有項目失敗 ❌'}")
    return check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="精確剪輯端到端驗證")
    parser.add_argument("--workdir", default="tests/output")
    parser.add_argument("--source", help="改用既有影片（只做 FFprobe 驗證）")
    parser.add_argument("--cut", action="append", default=[], help="要刪除的區間 start:end（可重複）")
    parser.add_argument("--duration", type=int, default=150)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--ads", default="1234:1984,3001:3751", help="合成片中的廣告位置")
    parser.add_argument("--vfr", action="store_true", help="合成 VFR 來源（每 7 格丟 1 格）")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    args = parser.parse_args(argv)

    for info in check_environment().values():
        print(f"{info.name}: {info.version}")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cuts = [FrameInterval(s, e) for s, e in parse_ads(",".join(args.cut))]
    settings = ExportSettings(overwrite=True, crf=args.crf, preset=args.preset)

    if args.source:
        if not cuts:
            parser.error("使用 --source 時必須以 --cut 指定刪除區間")
        source = Path(args.source)
        output = workdir / f"{source.stem}_ad_removed.mp4"
        result = cut_and_check(source, cuts, output, settings=settings)
        return 0 if result.ok else 1

    spec = SyntheticSpec(
        duration=args.duration,
        fps=args.fps,
        ad_intervals=parse_ads(args.ads),
        vfr_drop_mod=7 if args.vfr else 0,
    )
    tag = "vfr" if args.vfr else "cfr"
    source = workdir / f"synthetic_{tag}.mp4"
    print(f"\n[0] 合成測試片 {source}（廣告真值 {spec.ad_intervals}）")
    generate(spec, source)

    if not cuts:
        if args.vfr:
            # VFR 的影格索引 != 原始編號；依時間換算出對應的實際影格索引
            meta = probe_video(source)
            cuts = [
                FrameInterval(meta.frame_at_time(s / spec.fps), meta.frame_at_time(e / spec.fps))
                for s, e in spec.ad_intervals
            ]
        else:
            cuts = [FrameInterval(s, e) for s, e in spec.ad_intervals]

    result = cut_and_check(source, cuts, workdir / f"synthetic_{tag}_ad_removed.mp4", spec=spec, settings=settings)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
