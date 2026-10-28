"""[動畫型廣告跨模態偵測器] Audio-Visual Boundary Fusion Detector

【偵測邏輯簡述】
----------------------------------------------------------------------
1. 適用場景與定位：
   專為解決高動態、快節奏、多鏡頭切換且跨區間非同源的「商業動畫型廣告（時長約 15~20 秒）」設計。
   本模組完全獨立於既有的「靜態圖卡偵測（Stage 1~3）」，兩者特徵互補、互不干擾。

2. 階段一：FFmpeg 全片音訊瞬時響度快篩（Loudness Jump Detection）
   - 商業動畫短片為吸引注意力，音訊動態經過高度壓縮（Loudness War 響度戰爭），能量高度飽和。
   - 透過 FFmpeg `ebur128` 濾鏡以 100ms 為採樣解析全片音訊瞬時響度（Momentary Loudness, LUFS）。
   - 只解碼音訊串流，極度輕量高效（40 分鐘影片全片快篩僅需約 5~8 秒）。
   - 自動計算正片音量中位數作為基準線，鎖定平均音量突增（高於基準線 +8dB ~ +13dB）的高能區間。

3. 階段二：商業廣告時長常模約束過濾（Duration Constraints）
   - 電視與網路廣告時段受標準採購規格約束（通常為 15 秒、20 秒或 30 秒）。
   - 設定時長過濾窗口（預設 12.0s ~ 28.0s），精準排除短暫槍砲爆炸聲（<10s）與長篇配樂（>30s）。

4. 階段三：視訊轉場邊界精確吸附（Boundary Snapping）
   - 在粗篩候選起點周圍（±2.5 秒）執行 FFmpeg `blackdetect` 與 `scene_score` 偵測：
     優先匹配黑畫面結束點（Black End）或強分鏡硬切點（Scene Cut > 0.8），精準吸附廣告起始影格。
   - 在粗篩候選終點周圍（±2.5 秒）執行 `silencedetect` 與 `scene_score` 偵測：
     匹配靜音過渡帶（Silence）或強分鏡硬切點，精準吸附廣告離開影格。

5. 輸出標準：
   產出 0-based 半開影格區間 [start_frame, end_frame) 與秒數時間戳，
   可直接轉為 AdCandidate 或無縫輸入 exporter 執行無失真切除。
----------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from video_ad_remover.core.env_checker import hidden_window_kwargs, resolve_tool
from video_ad_remover.core.prober import probe_video
from video_ad_remover.models.interval import AdCandidate, DetectionSource, FrameInterval
from video_ad_remover.models.video_meta import VideoMetadata


@dataclass(slots=True)
class AudioVisualAdConfig:
    """動畫型廣告偵測設定參數。"""

    # 響度門檻設定 (LUFS)
    loudness_delta_db: float = 8.0  # 高出正片基準線多少 dB 視為廣告候選
    min_loudness_lufs: float = -23.0  # 廣告最低絕對響度門檻 (避免整體都極安靜時誤判)
    
    # 時長約束 (秒)
    min_duration_sec: float = 12.0  # 最短廣告時長 (過濾瞬時音效)
    max_duration_sec: float = 28.0  # 最長廣告時長 (過濾連續長配樂)
    
    # 平滑與容許中斷 (秒)
    loudness_sample_interval_sec: float = 0.5  # 響度時序採樣聚合窗口
    max_gap_tolerance_sec: float = 1.0  # 允許響度暫時掉落的最大間隔
    
    # 邊界吸附設定 (秒)
    boundary_search_window_sec: float = 2.5  # 邊界搜尋窗口半徑
    scene_change_threshold: float = 0.60  # 場景切換分數門檻 (0.0 ~ 1.0)
    black_pixel_threshold: float = 0.15  # 黑畫面像素亮度門檻
    black_min_duration: float = 0.03  # 黑畫面最短長度


@dataclass(slots=True)
class DetectedAdSegment:
    """偵測到的廣告時段詳細資訊。"""

    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    mean_lufs: float
    snapped_start: bool = False
    snapped_end: bool = False
    start_cut_type: str = "none"  # "black_frame", "scene_cut", "audio_step"
    end_cut_type: str = "none"  # "silence", "scene_cut", "audio_step"

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


class AudioVisualDetector:
    """動畫廣告跨模態音視融合偵測器。"""

    def __init__(
        self,
        config: AudioVisualAdConfig | None = None,
        ffmpeg_bin: str | None = None,
        ffprobe_bin: str | None = None,
    ) -> None:
        self.config = config or AudioVisualAdConfig()
        self.ffmpeg = ffmpeg_bin or resolve_tool("ffmpeg")
        self.ffprobe = ffprobe_bin or resolve_tool("ffprobe")

    def detect(
        self,
        video_path: Path | str,
        meta: VideoMetadata | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[DetectedAdSegment]:
        """對影片執行完整的音視融合廣告偵測。

        :param video_path: 影片檔案路徑
        :param meta: 預先探測的 VideoMetadata（若無則自動探測）
        :param progress_callback: 進度回報回呼 (0.0 ~ 1.0, 說明文字)
        :return: 偵測到的廣告時段列表
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"影片檔案不存在：{video_path}")

        # 1. 確保有基礎視訊規格
        if meta is None:
            if progress_callback:
                progress_callback(0.05, "讀取影片基礎中繼資料...")
            meta = probe_video(video_path, ffmpeg=self.ffmpeg, ffprobe=self.ffprobe)

        fps = float(meta.r_frame_rate or meta.avg_frame_rate or 24.0)

        # 2. 全片音訊瞬時響度快篩
        if progress_callback:
            progress_callback(0.15, "執行全片音訊響度階躍快篩...")
        loudness_series = self._extract_loudness_series(video_path)
        if not loudness_series:
            return []

        # 3. 響度基準線計算與候選粗篩
        if progress_callback:
            progress_callback(0.40, "分析音訊基線與篩選商業廣告長度區間...")
        raw_intervals = self._find_loudness_step_intervals(loudness_series)

        # 4. 針對候選區間執行邊界吸附
        results: list[DetectedAdSegment] = []
        total_cands = len(raw_intervals)

        for idx, (approx_s, approx_e, mean_lufs) in enumerate(raw_intervals):
            pct = 0.50 + (0.45 * (idx / max(1, total_cands)))
            if progress_callback:
                progress_callback(pct, f"精確微調廣告邊界 ({idx + 1}/{total_cands})...")

            snapped_s, s_type = self._snap_start_boundary(video_path, approx_s)
            snapped_e, e_type = self._snap_end_boundary(video_path, approx_e)

            # 時長再次防護
            duration = snapped_e - snapped_s
            if not (self.config.min_duration_sec <= duration <= self.config.max_duration_sec):
                continue

            # 轉換為 0-based 影格索引
            start_f = self._time_to_frame(snapped_s, meta, fps)
            end_f = self._time_to_frame(snapped_e, meta, fps)

            if end_f <= start_f:
                continue

            results.append(
                DetectedAdSegment(
                    start_time=snapped_s,
                    end_time=snapped_e,
                    start_frame=start_f,
                    end_frame=end_f,
                    mean_lufs=mean_lufs,
                    snapped_start=(s_type != "audio_step"),
                    snapped_end=(e_type != "audio_step"),
                    start_cut_type=s_type,
                    end_cut_type=e_type,
                )
            )

        if progress_callback:
            progress_callback(1.0, f"偵測完成，共發現 {len(results)} 個動畫廣告區間。")

        return results

    def to_ad_candidates(self, segments: list[DetectedAdSegment]) -> list[AdCandidate]:
        """將偵測結果轉換為系統標準的 AdCandidate 物件。"""
        candidates: list[AdCandidate] = []
        for seg in segments:
            score = 0.95 if (seg.snapped_start and seg.snapped_end) else 0.85
            interval = FrameInterval(start_frame=seg.start_frame, end_frame=seg.end_frame)
            cand = AdCandidate(
                interval=interval,
                score=score,
                score_detail={"mean_lufs": seg.mean_lufs},
                source=DetectionSource.AUTO_SCAN,
                note=f"動畫廣告 (起: {seg.start_cut_type}, 迄: {seg.end_cut_type})",
            )
            candidates.append(cand)
        return candidates

    # ------------------------------------------------------------------
    # 內部實作：音訊響度時序萃取與區間識別
    # ------------------------------------------------------------------

    def _extract_loudness_series(self, video_path: Path) -> list[tuple[float, float]]:
        """使用 FFmpeg ebur128 取得 (時間_秒, Momentary_LUFS) 序列。"""
        cmd = [
            self.ffmpeg,
            "-nostats",
            "-i",
            str(video_path),
            "-filter_complex",
            "ebur128=meter=18",
            "-f",
            "null",
            "-",
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                **hidden_window_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            return []

        series: list[tuple[float, float]] = []
        # 正則匹配：t: 0.100000 TARGET:-23 LUFS M: -30.4 ...
        pattern = re.compile(r"t:\s*([\d\.]+)\s+TARGET:.*?M:\s*([-\d\.]+)")

        for line in proc.stderr.splitlines():
            m = pattern.search(line)
            if m:
                t = float(m.group(1))
                m_lufs = float(m.group(2))
                series.append((t, m_lufs))

        return series

    def _find_loudness_step_intervals(
        self, series: list[tuple[float, float]]
    ) -> list[tuple[float, float, float]]:
        """分析響度時序，找出高於基線且符合廣告長度的區間。

        :return: list of (approx_start, approx_end, mean_lufs)
        """
        if not series:
            return []

        # 1. 計算正片基線（排除極安靜的靜音片段 <-60 LUFS）
        valid_lufs = [lufs for _, lufs in series if lufs > -60.0]
        if not valid_lufs:
            return []

        valid_lufs.sort()
        baseline_lufs = valid_lufs[len(valid_lufs) // 2]  # 中位數

        # 判定門檻：高出基線 delta，且至少達到最低絕對響度
        threshold = max(
            baseline_lufs + self.config.loudness_delta_db,
            self.config.min_loudness_lufs,
        )

        # 2. 標記高能時段並以滑動窗口聚合
        intervals: list[tuple[float, float, float]] = []
        in_high = False
        start_t = 0.0
        last_high_t = 0.0
        lufs_collector: list[float] = []

        for t, lufs in series:
            if lufs >= threshold:
                if not in_high:
                    in_high = True
                    start_t = t
                    lufs_collector = [lufs]
                else:
                    lufs_collector.append(lufs)
                last_high_t = t
            else:
                if in_high and (t - last_high_t > self.config.max_gap_tolerance_sec):
                    duration = last_high_t - start_t
                    if self.config.min_duration_sec <= duration <= self.config.max_duration_sec:
                        avg_l = sum(lufs_collector) / len(lufs_collector)
                        intervals.append((start_t, last_high_t, avg_l))
                    in_high = False
                    lufs_collector = []

        if in_high:
            duration = last_high_t - start_t
            if self.config.min_duration_sec <= duration <= self.config.max_duration_sec:
                avg_l = sum(lufs_collector) / len(lufs_collector)
                intervals.append((start_t, last_high_t, avg_l))

        return intervals

    # ------------------------------------------------------------------
    # 內部實作：轉場邊界吸附 (Boundary Snapping)
    # ------------------------------------------------------------------

    def _snap_start_boundary(
        self, video_path: Path, approx_time: float
    ) -> tuple[float, str]:
        """尋找廣告起點附近的黑畫面結束點或強場景切換。"""
        win = self.config.boundary_search_window_sec
        start_search = max(0.0, approx_time - win)
        end_search = approx_time + win

        # 1. 優先檢查黑畫面 (blackdetect)
        black_end = self._detect_black_frame_end(video_path, start_search, end_search)
        if black_end is not None and abs(black_end - approx_time) <= win:
            return black_end, "black_frame"

        # 2. 次之檢查場景切換 (scene cut)
        best_cut = self._detect_best_scene_cut(video_path, start_search, end_search)
        if best_cut is not None:
            return best_cut, "scene_cut"

        return approx_time, "audio_step"

    def _snap_end_boundary(
        self, video_path: Path, approx_time: float
    ) -> tuple[float, str]:
        """尋找廣告結束點附近的場景切換或靜音過渡帶。"""
        win = self.config.boundary_search_window_sec
        start_search = max(0.0, approx_time - win)
        end_search = approx_time + win

        # 1. 優先檢查場景切換 (scene cut)
        best_cut = self._detect_best_scene_cut(video_path, start_search, end_search)
        if best_cut is not None:
            return best_cut, "scene_cut"

        # 2. 次之檢查靜音起始/結束點 (silencedetect)
        silence_point = self._detect_silence_boundary(video_path, start_search, end_search)
        if silence_point is not None:
            return silence_point, "silence"

        return approx_time, "audio_step"

    def _detect_black_frame_end(
        self, video_path: Path, start_t: float, end_t: float
    ) -> float | None:
        """偵測搜尋窗口內黑畫面結束時間戳。"""
        duration = end_t - start_t
        cmd = [
            self.ffmpeg,
            "-nostats",
            "-ss",
            f"{start_t:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(video_path),
            "-vf",
            f"blackdetect=d={self.config.black_min_duration}:pix_th={self.config.black_pixel_threshold}",
            "-an",
            "-f",
            "null",
            "-",
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                **hidden_window_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            return None

        pattern = re.compile(r"black_end:([\d\.]+)")
        ends: list[float] = []
        for line in proc.stderr.splitlines():
            m = pattern.search(line)
            if m:
                ends.append(start_t + float(m.group(1)))

        return ends[-1] if ends else None

    def _detect_best_scene_cut(
        self, video_path: Path, start_t: float, end_t: float
    ) -> float | None:
        """偵測搜尋窗口內 scene_score 最高的場景切換點。"""
        duration = end_t - start_t
        cmd = [
            self.ffmpeg,
            "-nostats",
            "-ss",
            f"{start_t:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(video_path),
            "-vf",
            f"select='gt(scene,{self.config.scene_change_threshold})',metadata=print",
            "-an",
            "-f",
            "null",
            "-",
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                **hidden_window_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            return None

        pts_pattern = re.compile(r"pts_time:([\d\.]+)")
        score_pattern = re.compile(r"lavfi\.scene_score=([\d\.]+)")

        cuts: list[tuple[float, float]] = []  # (time, score)
        cur_t: float | None = None

        for line in proc.stderr.splitlines():
            m_t = pts_pattern.search(line)
            if m_t:
                cur_t = start_t + float(m_t.group(1))
            m_s = score_pattern.search(line)
            if m_s and cur_t is not None:
                cuts.append((cur_t, float(m_s.group(1))))
                cur_t = None

        if not cuts:
            return None

        # 回傳分數最高（切換最劇烈）的時間點
        cuts.sort(key=lambda x: x[1], reverse=True)
        return cuts[0][0]

    def _detect_silence_boundary(
        self, video_path: Path, start_t: float, end_t: float
    ) -> float | None:
        """偵測搜尋窗口內的靜音起訖點。"""
        duration = end_t - start_t
        cmd = [
            self.ffmpeg,
            "-nostats",
            "-ss",
            f"{start_t:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(video_path),
            "-af",
            "silencedetect=noise=-25dB:d=0.08",
            "-f",
            "null",
            "-",
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                **hidden_window_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            return None

        pattern = re.compile(r"silence_(start|end):\s*([\d\.]+)")
        points: list[float] = []
        for line in proc.stderr.splitlines():
            m = pattern.search(line)
            if m:
                points.append(start_t + float(m.group(2)))

        return points[-1] if points else None

    # ------------------------------------------------------------------
    # 輔助函式：時間轉影格索引
    # ------------------------------------------------------------------

    def _time_to_frame(self, t: float, meta: VideoMetadata, fps: float) -> int:
        if meta.frame_times:
            import bisect
            idx = bisect.bisect_right(meta.frame_times, t) - 1
            return max(0, min(idx, len(meta.frame_times) - 1))
        return max(0, round(t * fps))


# ----------------------------------------------------------------------
# CLI 獨立手動測試入口
# ----------------------------------------------------------------------

def _format_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:06.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="動畫型商業廣告跨模態偵測器（手動測試 CLI 工具）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="待測影片檔案路徑")
    parser.add_argument(
        "--loudness-delta",
        type=float,
        default=8.0,
        help="高於正片基線的判定門檻 (dB)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=12.0,
        help="廣告最短長度限制 (秒)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=28.0,
        help="廣告最長長度限制 (秒)",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=0.60,
        help="場景切換吸附門檻 (0.0~1.0)",
    )

    args = parser.parse_args()
    video_path = args.video

    if not video_path.exists():
        print(f"錯誤：找不到影片檔案 {video_path}", file=sys.stderr)
        return 1

    cfg = AudioVisualAdConfig(
        loudness_delta_db=args.loudness_delta,
        min_duration_sec=args.min_duration,
        max_duration_sec=args.max_duration,
        scene_change_threshold=args.scene_threshold,
    )

    print("=" * 70)
    print("【動畫型廣告跨模態偵測手動測試】")
    print(f"目標影片：{video_path.name}")
    print(f"門檻設定：Delta = +{cfg.loudness_delta_db} dB | 時長範圍 = {cfg.min_duration_sec}s ~ {cfg.max_duration_sec}s")
    print("=" * 70)

    detector = AudioVisualDetector(config=cfg)

    def on_progress(pct: float, msg: str) -> None:
        bar_len = 25
        filled = int(bar_len * pct)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r[{bar}] {pct*100:5.1f}% | {msg}", end="", flush=True)

    segments = detector.detect(video_path, progress_callback=on_progress)
    print("\n" + "=" * 70)

    if not segments:
        print("未偵測到任何符合條件的動畫廣告區間。")
        print("=" * 70)
        return 0

    print(f"偵測完成！共發現 {len(segments)} 個廣告區段：\n")
    print(f"{'序號':<4} | {'起始時間 (影格)':<24} | {'結束時間 (影格)':<24} | {'時長':<8} | {'平均響度':<9} | {'轉場吸附特徵'}")
    print("-" * 105)

    for i, seg in enumerate(segments, 1):
        s_str = f"{_format_time(seg.start_time)} ({seg.start_frame})"
        e_str = f"{_format_time(seg.end_time)} ({seg.end_frame})"
        dur_str = f"{seg.duration:.2f}s"
        lufs_str = f"{seg.mean_lufs:.1f} LUFS"
        feature_str = f"起: {seg.start_cut_type} | 迄: {seg.end_cut_type}"
        print(f"{i:<4} | {s_str:<24} | {e_str:<24} | {dur_str:<8} | {lufs_str:<9} | {feature_str}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
