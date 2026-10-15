"""[核心工具] 影格擷取器：封裝 OpenCV VideoCapture，支援低解析度取樣串流與指定單格精確解碼。

特性：
- 支援低解析度模式（縮減長邊至 target_long_edge，降低記憶體與計算負擔）。
- 支援指定 0-based Frame 精確讀取（搭配 VideoMetadata 的 PTS 或影格索引）。
- Context Manager 與自動資源釋放。
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import cv2
import numpy as np

from video_ad_remover.models.video_meta import VideoMetadata


class FrameExtractor:
    """影格擷取器，提供循序讀取與隨機存取。"""

    def __init__(self, path: str | Path, meta: VideoMetadata | None = None) -> None:
        self.path = Path(path)
        self.meta = meta
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def open(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise RuntimeError(f"無法透過 OpenCV 開啟影片：{self.path}")

    def close(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            self._cap = None

    def get_frame(self, frame_index: int, *, target_size: tuple[int, int] | None = None) -> np.ndarray | None:
        """讀取指定 0-based 影格。

        Args:
            frame_index: 0-based 影格編號。
            target_size: (寬, 高) 縮放大小，若為 None 則維持原始解析度。
        """
        self.open()
        assert self._cap is not None

        # 優先以 frame index 定位
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            # 若為 VFR 或 frame seek 失敗，且有 PTS 資訊，嘗試以時間 (毫秒) 定位
            if self.meta is not None and frame_index < self.meta.frame_count:
                pts_ms = self.meta.time_of(frame_index) * 1000.0
                self._cap.set(cv2.CAP_PROP_POS_MSEC, pts_ms)
                ret, frame = self._cap.read()

        if not ret or frame is None:
            return None

        if target_size is not None:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

        return frame

    def get_frame_range(
        self,
        start_frame: int,
        end_frame: int,
        *,
        target_size: tuple[int, int] | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """循序讀取連續區間 [start_frame, end_frame] 內的影格。

        只執行一次 Seek，後面全走硬體/軟體連續解碼，效能比連續隨機 Seek 快數十倍。
        """
        self.open()
        assert self._cap is not None

        start_frame = max(0, start_frame)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        results: list[tuple[int, np.ndarray]] = []
        curr = start_frame

        while curr <= end_frame:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                break

            if target_size is not None:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

            results.append((curr, frame))
            curr += 1

        return results

    def read_sampled_frames(
        self,
        step: int = 1,
        target_long_edge: int = 400,
    ):
        """產生器：以指定步長循序產出 (frame_index, frame_bgr)。

        Args:
            step: 取樣間隔影格數（例如 30fps 取樣 3fps 則 step=10）。
            target_long_edge: 降採樣長邊像素。
        """
        self.open()
        assert self._cap is not None

        orig_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_size: tuple[int, int] | None = None
        if orig_w > 0 and orig_h > 0:
            scale = target_long_edge / max(orig_w, orig_h)
            if scale < 1.0:
                target_size = (int(orig_w * scale), int(orig_h * scale))

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        curr_frame = 0

        while True:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                break

            if curr_frame % step == 0:
                if target_size is not None:
                    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                yield curr_frame, frame

            curr_frame += 1
