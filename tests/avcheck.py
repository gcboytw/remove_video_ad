"""輸出檔逐格／影音同步檢查工具（搭配 generate_test_video.py 的條碼與嗶聲）。"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from generate_test_video import CODE_BITS, CODE_STRIP_HEIGHT
from video_ad_remover.core.prober import probe_video
from video_ad_remover.models.interval import FrameInterval

BEEP_THRESHOLD = 0.25  # 嗶聲振幅 0.5，取 0.25 作為起音判斷門檻


def decode_frame_codes(path: str | Path, *, width: int, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """逐格解碼頂端條碼，回傳每一格的「原始影格編號」陣列（依輸出呈現順序）。"""
    cmd = [
        ffmpeg, "-v", "error", "-nostdin", "-i", str(path),
        "-map", "0:v:0",
        "-vf", f"crop={width}:{CODE_STRIP_HEIGHT}:0:0,format=gray",
        "-fps_mode", "passthrough",
        "-f", "rawvideo", "pipe:1",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, CODE_STRIP_HEIGHT, width)
    block = width // CODE_BITS
    cy = CODE_STRIP_HEIGHT // 2
    bits = np.stack(
        [
            frames[:, cy - 3 : cy + 3, k * block + block // 2 - 4 : k * block + block // 2 + 4].mean(
                axis=(1, 2)
            )
            > 128
            for k in range(CODE_BITS)
        ],
        axis=1,
    )
    weights = (1 << np.arange(CODE_BITS)).astype(np.int64)
    return (bits.astype(np.int64) * weights).sum(axis=1)


def decode_audio(path: str | Path, *, sample_rate: int, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """解碼第一聲道為 float32 PCM（sample 0 = 音訊串流 start_time）。"""
    cmd = [
        ffmpeg, "-v", "error", "-nostdin", "-i", str(path),
        "-map", "0:a:0", "-af", "pan=mono|c0=c0",
        "-ar", str(sample_rate), "-f", "f32le", "pipe:1",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


@dataclass
class FrameCheck:
    expected_count: int
    actual_count: int
    missing: list[int]  # 應保留卻遺失的原始影格
    residue: list[int]  # 應刪除卻殘留的原始影格
    exact_sequence: bool  # 輸出序列與預期完全一致（含順序、無重複）

    @property
    def ok(self) -> bool:
        return self.exact_sequence and not self.missing and not self.residue

    def summary(self) -> str:
        return (
            f"逐格檢查：{'通過' if self.ok else '失敗'}｜預期 {self.expected_count} 格、實際 {self.actual_count} 格｜"
            f"遺失 {len(self.missing)} 格 {self.missing[:10]}｜殘留 {len(self.residue)} 格 {self.residue[:10]}"
        )


def check_frame_accuracy(
    out_codes: np.ndarray,
    src_codes: np.ndarray,
    keep_segments: Sequence[FrameInterval],
) -> FrameCheck:
    expected = np.concatenate([src_codes[s.start_frame : s.end_frame] for s in keep_segments])
    expected_set = set(expected.tolist())
    removed = set(src_codes.tolist()) - expected_set
    out_set = set(out_codes.tolist())
    return FrameCheck(
        expected_count=len(expected),
        actual_count=len(out_codes),
        missing=sorted(expected_set - out_set),
        residue=sorted(out_set & removed),
        exact_sequence=bool(np.array_equal(out_codes, expected)),
    )


@dataclass
class SyncSample:
    out_frame: int
    code: int
    video_time: float
    audio_time: float | None

    @property
    def offset(self) -> float | None:
        return None if self.audio_time is None else self.audio_time - self.video_time


@dataclass
class SyncCheck:
    samples: list[SyncSample] = field(default_factory=list)
    tail_start_frame: int = 0  # 最後一段正片在輸出中的起始格

    @property
    def measured(self) -> list[SyncSample]:
        return [s for s in self.samples if s.offset is not None]

    @property
    def missing_beeps(self) -> int:
        return sum(1 for s in self.samples if s.offset is None)

    def max_abs_offset(self, tail_only: bool = False) -> float:
        items = [s for s in self.measured if not tail_only or s.out_frame >= self.tail_start_frame]
        return max((abs(s.offset) for s in items), default=float("nan"))  # type: ignore[arg-type]

    def summary(self, fps: float) -> str:
        tail = [s for s in self.measured if s.out_frame >= self.tail_start_frame]
        return (
            f"影音同步：量測 {len(self.measured)} 個嗶聲（片尾段 {len(tail)} 個），漏偵測 {self.missing_beeps}｜"
            f"最大偏移 {self.max_abs_offset() * 1000:.2f} ms，片尾段最大偏移 "
            f"{self.max_abs_offset(tail_only=True) * 1000:.2f} ms（1 格 = {1000 / fps:.1f} ms）"
        )


def measure_av_sync(
    path: str | Path,
    *,
    out_codes: np.ndarray,
    fps: int,
    sample_rate: int,
    is_ad: Callable[[int], bool],
    tail_start_frame: int = 0,
    window: float = 0.05,
    ffmpeg: str = "ffmpeg",
) -> SyncCheck:
    """對每個「整數秒起點且附近無剪接」的正片影格，量測嗶聲起音時間與該格視訊時間的差。"""
    meta = probe_video(path)
    assert meta.audio is not None, "輸出缺少音訊串流"
    audio = decode_audio(path, sample_rate=sample_rate, ffmpeg=ffmpeg)
    audio_t0 = meta.audio.start_time - meta.timeline_origin
    result = SyncCheck(tail_start_frame=tail_start_frame)

    codes = out_codes.tolist()
    for j, code in enumerate(codes):
        if code % fps or j < 3 or j + 4 > len(codes):
            continue
        neighborhood = codes[j - 3 : j + 4]
        contiguous = all(b > a for a, b in zip(neighborhood, neighborhood[1:])) and (
            neighborhood[-1] - neighborhood[0] <= 12
        )
        if not contiguous or any(is_ad(c) for c in neighborhood):
            continue  # 嗶聲前後 0.1s 內有剪接點或廣告，不適合量測

        video_time = meta.time_of(j)
        i0 = max(int((video_time - window - audio_t0) * sample_rate), 0)
        i1 = min(int((video_time + window - audio_t0) * sample_rate), len(audio))
        hits = np.nonzero(np.abs(audio[i0:i1]) > BEEP_THRESHOLD)[0]
        audio_time = audio_t0 + (i0 + int(hits[0])) / sample_rate if len(hits) else None
        result.samples.append(SyncSample(j, code, video_time, audio_time))
    return result
