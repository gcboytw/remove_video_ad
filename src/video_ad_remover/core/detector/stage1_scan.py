"""[模組2] Stage 1：全片快速掃描與畫面切換點（Cut Point）粗略偵測。

設計規範（SRS §5）：
- 全片時間軸從頭到尾完整掃描，不因找到第一處而中斷。
- 降採樣至長邊 320～480px，每秒取樣 2～5 格。
- 計算相鄰取樣格之 HSV 色彩直方圖差異與亮度/邊緣結構變化。
- 輸出全片所有劇烈切換之粗略切點集合 T_cuts。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from video_ad_remover.core.frame_extractor import FrameExtractor
from video_ad_remover.models.video_meta import VideoMetadata


@dataclass(frozen=True, slots=True)
class CutPoint:
    frame_index: int  # 0-based 取樣影格索引
    timestamp: float  # 秒數
    diff_score: float  # 切換差異度 (0.0 ~ 1.0)


def compute_frame_features(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """計算影格特徵：(HSV 直方圖, 邊緣強度直方圖)。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # H: 30 bins, S: 32 bins
    hist_hsv = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist_hsv, hist_hsv, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # 灰階邊緣
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.array([np.mean(edges) / 255.0], dtype=np.float32)

    return hist_hsv, edge_density


def compare_frames(
    feat1: tuple[np.ndarray, np.ndarray],
    feat2: tuple[np.ndarray, np.ndarray],
) -> float:
    """計算兩影格特徵的相異度 (0.0 ~ 1.0，越大代表差異越劇烈)。"""
    h1, e1 = feat1
    h2, e2 = feat2

    # 巴氏距離 (Bhattacharyya distance)，0 表示完全相同，1 表示完全相異
    bhat_dist = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

    # 邊緣結構差異
    edge_diff = float(abs(e1[0] - e2[0]))

    # 綜合評分：以色彩直方圖為主，邊緣變化為輔
    diff = 0.85 * bhat_dist + 0.15 * edge_diff
    return min(max(diff, 0.0), 1.0)


def scan_cut_points(
    path: str | Path,
    meta: VideoMetadata,
    *,
    sample_fps: float = 3.0,
    target_long_edge: int = 400,
    diff_threshold: float = 0.40,
    progress_cb: Callable[[float], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[CutPoint]:
    """執行 Stage 1 全片快速掃描。

    Args:
        path: 影片路徑。
        meta: 影片詮釋資料。
        sample_fps: 每秒取樣格數（預設 3.0 fps）。
        target_long_edge: 降採樣長邊像素。
        diff_threshold: 判定畫面劇烈切換之相異度門檻。
        progress_cb: 進度回報回呼 (0.0 ~ 1.0)。
        cancel_event: 取消信號。
    """
    total_frames = meta.frame_count
    nominal_fps = meta.nominal_fps if meta.nominal_fps > 0 else 30.0
    step = max(1, round(nominal_fps / sample_fps))

    cut_points: list[CutPoint] = []
    cancel_event = cancel_event or threading.Event()

    extractor = FrameExtractor(path, meta)
    with extractor:
        prev_feat: tuple[np.ndarray, np.ndarray] | None = None

        for frame_idx, frame in extractor.read_sampled_frames(
            step=step,
            target_long_edge=target_long_edge,
            sample_fps=sample_fps,
            cancel_event=cancel_event,
        ):
            if cancel_event.is_set():
                break

            curr_feat = compute_frame_features(frame)
            if prev_feat is not None:
                diff = compare_frames(prev_feat, curr_feat)
                if diff >= diff_threshold:
                    t = meta.time_of(frame_idx)
                    cut_points.append(CutPoint(frame_index=frame_idx, timestamp=t, diff_score=diff))

            prev_feat = curr_feat

            if progress_cb is not None and total_frames > 0:
                progress_cb(min(frame_idx / total_frames, 1.0))

        if progress_cb is not None:
            progress_cb(1.0)

    return cut_points
