"""[模組3] 主播放器與逐格控制面板。

設計規範（FR-002）：
- 顯示目前影格畫面、0-based 影格索引與 PTS 時間戳記。
- 逐格定位控制：[-30格] [-5格] [-1格] [+1格] [+5格] [+30格]。
- 指定 Frame 跳轉。
"""

from __future__ import annotations

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
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
        self._current_pixmap: QPixmap | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # 1. 畫面顯示區（高度加大，彈性伸展）
        self.lbl_screen = QLabel()
        self.lbl_screen.setMinimumSize(640, 360)
        self.lbl_screen.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_screen.setStyleSheet("background-color: #000; border: 1px solid #333;")
        self.lbl_screen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_screen, stretch=1)

        # 2. 緊湊單列控制列：資訊、微調按鈕、跳轉整合為一行
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(2, 0, 2, 0)
        bottom_bar.setSpacing(4)

        # 左側資訊顯示
        self.lbl_frame_info = QLabel("Frame: 0 / 0")
        self.lbl_time_info = QLabel("00:00:00.000")
        self.lbl_frame_info.setStyleSheet("font-weight: bold; font-size: 11px;")
        self.lbl_time_info.setStyleSheet("color: #4fa3d1; font-size: 11px;")
        bottom_bar.addWidget(self.lbl_frame_info)
        bottom_bar.addWidget(QLabel("|"))
        bottom_bar.addWidget(self.lbl_time_info)

        bottom_bar.addSpacing(8)

        # 中間相對跳轉按鈕（精簡高度）
        btn_m30 = QPushButton("-30格")
        btn_m5 = QPushButton("-5格")
        btn_m1 = QPushButton("-1格")
        btn_p1 = QPushButton("+1格")
        btn_p5 = QPushButton("+5格")
        btn_p30 = QPushButton("+30格")

        for btn in (btn_m30, btn_m5, btn_m1, btn_p1, btn_p5, btn_p30):
            btn.setFixedHeight(24)
            btn.setStyleSheet("padding: 2px 6px; font-size: 11px;")
            bottom_bar.addWidget(btn)

        btn_m30.clicked.connect(lambda: self.seek_relative(-30))
        btn_m5.clicked.connect(lambda: self.seek_relative(-5))
        btn_m1.clicked.connect(lambda: self.seek_relative(-1))
        btn_p1.clicked.connect(lambda: self.seek_relative(+1))
        btn_p5.clicked.connect(lambda: self.seek_relative(+5))
        btn_p30.clicked.connect(lambda: self.seek_relative(+30))

        bottom_bar.addSpacing(16)
        lbl_divider = QLabel("|")
        lbl_divider.setStyleSheet("color: #555;")
        bottom_bar.addWidget(lbl_divider)
        bottom_bar.addSpacing(16)

        # 右側跳至指定影格
        lbl_goto = QLabel("跳至格:")
        lbl_goto.setStyleSheet("font-size: 11px;")
        bottom_bar.addWidget(lbl_goto)

        self.spin_goto = QSpinBox()
        self.spin_goto.setRange(0, 0)
        self.spin_goto.setFixedHeight(24)
        self.spin_goto.setStyleSheet("font-size: 11px;")
        bottom_bar.addWidget(self.spin_goto)

        btn_goto = QPushButton("跳轉")
        btn_goto.setFixedHeight(24)
        btn_goto.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        btn_goto.clicked.connect(lambda: self.seek_frame(self.spin_goto.value()))
        bottom_bar.addWidget(btn_goto)

        bottom_bar.addStretch()
        layout.addLayout(bottom_bar)

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

        img = self.extractor.get_frame(target, target_size=(960, 540))
        if img is not None:
            h, w, ch = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self._current_pixmap = QPixmap.fromImage(qimg)
            self._update_screen_pixmap()

        t = self.meta.time_of(target)
        self.lbl_frame_info.setText(f"Frame: {target} / {self.meta.frame_count - 1}")
        self.lbl_time_info.setText(f"時間: {format_timestamp(t)}")
        self.spin_goto.setValue(target)
        self.frame_changed.emit(target)

    def _update_screen_pixmap(self) -> None:
        if self._current_pixmap and not self._current_pixmap.isNull():
            scaled_pm = self._current_pixmap.scaled(
                self.lbl_screen.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.lbl_screen.setPixmap(scaled_pm)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_screen_pixmap()
