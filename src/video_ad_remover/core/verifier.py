"""[模組5] 輸出驗證器：以 FFprobe 檢查輸出檔串流、總時長、影格數與時間戳記（FR-013）。"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

from video_ad_remover.core.prober import ProbeError, probe_video
from video_ad_remover.models.video_meta import VideoMetadata

# AAC 一個封包 1024 samples（48kHz 約 21ms），音訊尾端允許多出一點
AUDIO_EXTRA_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class VerificationReport:
    output: Path
    checks: list[CheckResult] = field(default_factory=list)
    metadata: VideoMetadata | None = None

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(CheckResult(name, passed, detail))

    def summary(self) -> str:
        lines = [f"驗證結果：{'通過' if self.ok else '失敗'}  ({self.output.name})"]
        lines += [f"  [{'OK' if c.passed else 'NG'}] {c.name}：{c.detail}" for c in self.checks]
        return "\n".join(lines)


def verify_output(
    output: str | Path,
    *,
    expected_duration: float,
    expected_frames: int | None = None,
    expect_audio: bool = True,
    tolerance: float | None = None,
    ffprobe: str | None = None,
) -> VerificationReport:
    output = Path(output)
    report = VerificationReport(output=output)

    # 1. 檔案存在且可解析
    if not output.is_file():
        report.add("檔案存在", False, "輸出檔不存在")
        return report
    try:
        meta = probe_video(output, ffprobe=ffprobe)
    except (ProbeError, FileNotFoundError) as exc:
        report.add("FFprobe 解析", False, str(exc))
        return report
    report.metadata = meta
    report.add("FFprobe 解析", True, f"{meta.video_codec} {meta.width}x{meta.height}, {meta.frame_mode}")

    # 2. 串流完整
    report.add("視訊串流", meta.frame_count > 0, f"{meta.frame_count} 格")
    if expect_audio:
        report.add(
            "音訊串流",
            meta.audio is not None,
            f"{meta.audio.codec} {meta.audio.sample_rate}Hz" if meta.audio else "缺少音訊串流",
        )

    # 3. 時長與影格數
    deltas = [b - a for a, b in zip(meta.frame_times, meta.frame_times[1:])]
    median = statistics.median(deltas) if deltas else 0.0
    tol = tolerance if tolerance is not None else max(1.5 * median, 0.02)
    actual = meta.video_duration
    diff = actual - expected_duration
    report.add(
        "輸出總時長",
        abs(diff) <= tol,
        f"實際 {actual:.4f}s / 預期 {expected_duration:.4f}s（差 {diff * 1000:+.1f} ms，容許 ±{tol * 1000:.1f} ms）",
    )
    if expected_frames is not None:
        report.add(
            "輸出影格數",
            meta.frame_count == expected_frames,
            f"實際 {meta.frame_count} / 預期 {expected_frames}",
        )
    if expect_audio and meta.audio is not None and meta.audio.duration is not None:
        a_diff = meta.audio.duration - actual
        report.add(
            "音訊長度對齊",
            abs(a_diff) <= tol + AUDIO_EXTRA_TOLERANCE,
            f"音訊 {meta.audio.duration:.4f}s vs 視訊 {actual:.4f}s（差 {a_diff * 1000:+.1f} ms）",
        )

    # 4. 時間戳記單調遞增、無異常跳躍
    non_increasing = sum(1 for d in deltas if d <= 0)
    jump_threshold = max(10 * median, 0.5)
    jumps = sum(1 for d in deltas if d > jump_threshold)
    report.add(
        "時間戳記",
        non_increasing == 0 and jumps == 0,
        f"重複/倒退 {non_increasing} 處，跳躍(>{jump_threshold:.3f}s) {jumps} 處",
    )
    return report
