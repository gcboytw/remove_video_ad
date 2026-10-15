"""[模組3] 四格邊界檢查與微調視圖（4-Frame Boundary View）。

設計規範（FR-006）：
- 並排顯示四張縮圖：
  1. 廣告前一格 (start - 1)
  2. 廣告第一格 (start)
  3. 廣告最後一格 (end - 1)
  4. 正片恢復第一格 (end)
- 附帶起點與終點的 [-1 格]、[+1 格] 獨立微調按鈕，點擊時即時更新。
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from video_ad_remover.core.frame_extractor import FrameExtractor
from video_ad_remover.models.interval import AdCandidate
from video_ad_remover.models.video_meta import VideoMetadata
from video_ad_remover.utils.time_conv import format_timestamp


def _mat_to_pixmap(mat: np.ndarray, width: int = 180, height: int = 100) -> QPixmap:
    """將 OpenCV BGR 轉為 QPixmap 縮圖。"""
    h, w, ch = mat.shape
    bytes_per_line = ch * w
    # BGR -> RGB
    rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        width, height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )


class SingleFrameCard(QFrame):
    """單一影格縮圖卡片（包含標題、影格索引、時間戳記與縮圖）。"""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("background-color: #2b2b2b; border-radius: 4px; padding: 4px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("font-weight: bold; color: #e0e0e0; font-size: 11px;")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_info = QLabel("Frame: -")
        self.lbl_info.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_img = QLabel()
        self.lbl_img.setFixedSize(180, 100)
        self.lbl_img.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444;")
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_info)
        layout.addWidget(self.lbl_img)

    def set_content(self, frame_idx: int, timestamp: float, pixmap: QPixmap | None) -> None:
        self.lbl_info.setText(f"Frame {frame_idx} ({format_timestamp(timestamp)})")
        if pixmap is not None:
            self.lbl_img.setPixmap(pixmap)
        else:
            self.lbl_img.setText("無畫面")


class BoundaryView(QWidget):
    """四格邊界檢查與微調區。"""

    interval_changed = Signal(AdCandidate)  # 微調後發射新區間

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.meta: VideoMetadata | None = None
        self.extractor: FrameExtractor | None = None
        self.current_candidate: AdCandidate | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # 頂部標題
        lbl_header = QLabel("🔍 四格邊界檢查與微調 (4-Frame Boundary View)")
        lbl_header.setStyleSheet("font-weight: bold; font-size: 12px; color: #4fa3d1;")
        layout.addWidget(lbl_header)

        # 四張縮圖橫向排列
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(6)

        self.card_prev = SingleFrameCard("1. 廣告前一格 (正片最後)")
        self.card_start = SingleFrameCard("2. 廣告第一格 (切入)")
        self.card_last = SingleFrameCard("3. 廣告最後一格 (圖卡最後)")
        self.card_resume = SingleFrameCard("4. 正片第一格 (恢復)")

        cards_layout.addWidget(self.card_prev)
        cards_layout.addWidget(self.card_start)
        cards_layout.addWidget(self.card_last)
        cards_layout.addWidget(self.card_resume)
        layout.addLayout(cards_layout)

        # 微調控制按鈕列
        ctrl_layout = QGridLayout()

        # 起點微調
        btn_start_dec = QPushButton("◀ 起點 -1格")
        btn_start_inc = QPushButton("起點 +1格 ▶")
        btn_start_dec.clicked.connect(lambda: self._adjust_start(-1))
        btn_start_inc.clicked.connect(lambda: self._adjust_start(+1))

        # 終點微調
        btn_end_dec = QPushButton("◀ 終點 -1格")
        btn_end_inc = QPushButton("終點 +1格 ▶")
        btn_end_dec.clicked.connect(lambda: self._adjust_end(-1))
        btn_end_inc.clicked.connect(lambda: self._adjust_end(+1))

        ctrl_layout.addWidget(btn_start_dec, 0, 0)
        ctrl_layout.addWidget(btn_start_inc, 0, 1)
        ctrl_layout.addWidget(btn_end_dec, 0, 2)
        ctrl_layout.addWidget(btn_end_inc, 0, 3)

        layout.addLayout(ctrl_layout)

    def set_context(self, meta: VideoMetadata, extractor: FrameExtractor) -> None:
        self.meta = meta
        self.extractor = extractor

    def load_candidate(self, candidate: AdCandidate | None) -> None:
        self.current_candidate = candidate
        self.refresh()

    def refresh(self) -> None:
        if self.current_candidate is None or self.meta is None or self.extractor is None:
            return

        cand = self.current_candidate
        total = self.meta.frame_count

        # 1. 廣告前一格 (start - 1)
        f_prev = max(0, cand.start_frame - 1)
        # 2. 廣告第一格 (start)
        f_start = min(total - 1, cand.start_frame)
        # 3. 廣告最後一格 (end - 1)
        f_last = max(0, cand.end_frame - 1)
        # 4. 正片第一格 (end)
        f_resume = min(total - 1, cand.end_frame)

        for card, f_idx in [
            (self.card_prev, f_prev),
            (self.card_start, f_start),
            (self.card_last, f_last),
            (self.card_resume, f_resume),
        ]:
            img = self.extractor.get_frame(f_idx, target_size=(180, 100))
            pm = _mat_to_pixmap(img) if img is not None else None
            t = self.meta.time_of(f_idx) if f_idx < total else 0.0
            card.set_content(f_idx, t, pm)

    def _adjust_start(self, delta: int) -> None:
        if self.current_candidate is None:
            return
        new_start = self.current_candidate.start_frame + delta
        if 0 <= new_start < self.current_candidate.end_frame:
            self.current_candidate.adjust_start(delta)
            self.refresh()
            self.interval_changed.emit(self.current_candidate)

    def _adjust_end(self, delta: int) -> None:
        if self.current_candidate is None or self.meta is None:
            return
        new_end = self.current_candidate.end_frame + delta
        if self.current_candidate.start_frame < new_end <= self.meta.frame_count:
            self.current_candidate.adjust_end(delta)
            self.refresh()
            self.interval_changed.emit(self.current_candidate)
