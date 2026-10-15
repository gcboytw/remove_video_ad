"""[模組3] 主播放器與逐格控制面板。

設計規範（FR-002）：
- 顯示目前影格畫面、0-based 影格索引與 PTS 時間戳記。
- 逐格定位控制：[-30格] [-5格] [-1格] [+1格] [+5格] [+30格]。
- 指定 Frame 跳轉。
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from video_ad_remover.core.frame_extractor import FrameExtractor
from video_ad_remover.models.video_meta import VideoMetadata
from video_ad_remover.utils.time_conv import format_timestamp


class PlayerPanel(QWidget):
    """主預覽畫面與播放/逐格控制面板。"""

    frame_changed = Signal(int)  # 目前影格變更

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.meta: VideoMetadata | None = None
        self.extractor: FrameExtractor | None = None
        self.current_frame: int = 0

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # 畫面顯示區
        self.lbl_screen = QLabel()
        self.lbl_screen.setMinimumSize(480, 270)
        self.lbl_screen.setStyleSheet("background-color: #000; border: 1px solid #333;")
        self.lbl_screen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_screen)

        # 資訊顯示列（Frame Index 與時間戳記）
        info_layout = QHBoxLayout()
        self.lbl_frame_info = QLabel("Frame: 0 / 0")
        self.lbl_time_info = QLabel("時間: 00:00:00.000")
        self.lbl_frame_info.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.lbl_time_info.setStyleSheet("color: #4fa3d1; font-size: 12px;")
        info_layout.addWidget(self.lbl_frame_info)
        info_layout.addStretch()
        info_layout.addWidget(self.lbl_time_info)
        layout.addLayout(info_layout)

        # 控制按鈕列
        ctrl_layout = QHBoxLayout()

        btn_m30 = QPushButton("-30格")
        btn_m5 = QPushButton("-5格")
        btn_m1 = QPushButton("-1格")
        btn_p1 = QPushButton("+1格")
        btn_p5 = QPushButton("+5格")
        btn_p30 = QPushButton("+30格")

        btn_m30.clicked.connect(lambda: self.seek_relative(-30))
        btn_m5.clicked.connect(lambda: self.seek_relative(-5))
        btn_m1.clicked.connect(lambda: self.seek_relative(-1))
        btn_p1.clicked.connect(lambda: self.seek_relative(+1))
        btn_p5.clicked.connect(lambda: self.seek_relative(+5))
        btn_p30.clicked.connect(lambda: self.seek_relative(+30))

        ctrl_layout.addWidget(btn_m30)
        ctrl_layout.addWidget(btn_m5)
        ctrl_layout.addWidget(btn_m1)
        ctrl_layout.addWidget(btn_p1)
        ctrl_layout.addWidget(btn_p5)
        ctrl_layout.addWidget(btn_p30)

        # 跳至指定影格
        ctrl_layout.addSpacing(10)
        self.spin_goto = QSpinBox()
        self.spin_goto.setRange(0, 0)
        btn_goto = QPushButton("跳轉")
        btn_goto.clicked.connect(lambda: self.seek_frame(self.spin_goto.value()))
        ctrl_layout.addWidget(QLabel("跳至格:"))
        ctrl_layout.addWidget(self.spin_goto)
        ctrl_layout.addWidget(btn_goto)

        layout.addLayout(ctrl_layout)

    def set_context(self, meta: VideoMetadata, extractor: FrameExtractor) -> None:
        self.meta = meta
        self.extractor = extractor
        self.spin_goto.setRange(0, max(0, meta.frame_count - 1))
        self.seek_frame(0)

    def seek_relative(self, delta: int) -> None:
        self.seek_frame(self.current_frame + delta)

    def seek_frame(self, frame_idx: int) -> None:
        if not self.meta or not self.extractor:
            return

        target = max(0, min(frame_idx, self.meta.frame_count - 1))
        self.current_frame = target

        img = self.extractor.get_frame(target, target_size=(480, 270))
        if img is not None:
            h, w, ch = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.lbl_screen.setPixmap(QPixmap.fromImage(qimg))

        t = self.meta.time_of(target)
        self.lbl_frame_info.setText(f"Frame: {target} / {self.meta.frame_count - 1}")
        self.lbl_time_info.setText(f"時間: {format_timestamp(t)}")
        self.spin_goto.setValue(target)
        self.frame_changed.emit(target)
