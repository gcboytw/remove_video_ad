"""[模組3] 底部視覺化時間軸與輸出面板。

設計規範（FR-009 / FR-010 / FR-012）：
- 色塊時間軸：正片保留（綠）、已確認刪除廣告（紅）、未確認候選（黃）。
- 顯示保留片段清單、預估輸出時長。
- 輸出控制：輸出路徑選擇、CRF 品質設定、轉檔進度條、取消與輸出按鈕。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from video_ad_remover.core.exporter import ExportProgress, default_output_path
from video_ad_remover.core.interval_calc import collect_confirmed, compute_keep_segments
from video_ad_remover.models.interval import AdCandidate, IntervalStatus
from video_ad_remover.models.video_meta import VideoMetadata
from video_ad_remover.utils.time_conv import format_timestamp


class TimelineBar(QFrame):
    """色塊長條時間軸。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(24)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444; border-radius: 2px;")
        self.meta: VideoMetadata | None = None
        self.candidates: list[AdCandidate] = []

    def set_data(self, meta: VideoMetadata | None, candidates: list[AdCandidate]) -> None:
        self.meta = meta
        self.candidates = candidates
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if not self.meta or self.meta.frame_count <= 0:
            return

        painter = QPainter(self)
        w = self.width()
        h = self.height()
        total_frames = self.meta.frame_count

        # 1. 預設整條畫成綠色（正片保留）
        painter.fillRect(0, 0, w, h, QColor("#2e7d32"))

        # 2. 疊加候選色塊
        for cand in self.candidates:
            x_start = int((cand.start_frame / total_frames) * w)
            x_end = int((cand.end_frame / total_frames) * w)
            block_w = max(2, x_end - x_start)

            if cand.status == IntervalStatus.CONFIRMED_AD:
                color = QColor("#d32f2f")  # 紅色：確認刪除廣告
            elif cand.status == IntervalStatus.UNCONFIRMED:
                color = QColor("#fbc02d")  # 黃色：未確認
            else:
                continue  # 已排除的不畫或維持綠色

            painter.fillRect(x_start, 0, block_w, h, color)


class TimelineExportPanel(QWidget):
    """底部時間軸與輸出設定控制面板。"""

    export_requested = Signal(Path, int)  # output_path, crf
    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.meta: VideoMetadata | None = None
        self.candidates: list[AdCandidate] = []
        self.custom_output_path: Path | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # 時間軸
        self.timeline_bar = TimelineBar()
        layout.addWidget(self.timeline_bar)

        # 摘要資訊列
        self.lbl_summary = QLabel("保留片段: - | 預估時長: -")
        self.lbl_summary.setStyleSheet("color: #cccccc; font-size: 11px;")
        layout.addWidget(self.lbl_summary)

        # 轉檔控制列
        export_layout = QHBoxLayout()

        self.spin_crf = QSpinBox()
        self.spin_crf.setRange(10, 35)
        self.spin_crf.setValue(24)
        self.spin_crf.setPrefix("品質 (CRF): ")
        self.spin_crf.setToolTip("推薦值 23~25（兼顧畫質與容量）。數值越小畫質越高但檔案越大；18 會使檔案暴增。")

        self.btn_select_out = QPushButton("選擇輸出檔名...")
        self.btn_select_out.clicked.connect(self._select_output_file)

        self.btn_export = QPushButton("🚀 開始輸出 MP4")
        self.btn_export.setStyleSheet("background-color: #388e3c; color: white; font-weight: bold;")
        self.btn_export.clicked.connect(self._on_export_clicked)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(lambda: self.cancel_requested.emit())

        export_layout.addWidget(self.spin_crf)
        export_layout.addWidget(self.btn_select_out)
        export_layout.addWidget(self.btn_export)
        export_layout.addWidget(self.btn_cancel)
        layout.addLayout(export_layout)

        # 進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

    def set_data(self, meta: VideoMetadata | None, candidates: list[AdCandidate]) -> None:
        self.meta = meta
        self.candidates = candidates
        self.timeline_bar.set_data(meta, candidates)
        self.refresh_summary()

    def refresh_summary(self) -> None:
        if not self.meta:
            self.lbl_summary.setText("尚未載入影片")
            return

        confirmed = collect_confirmed(self.candidates)
        try:
            keeps = compute_keep_segments(confirmed, self.meta.frame_count, strict=False)
            total_dur = sum(self.meta.segment_duration(k.start_frame, k.end_frame) for k in keeps)
            del_count = len(confirmed)
            self.lbl_summary.setText(
                f"已確認刪除廣告: {del_count} 段 | 保留正片: {len(keeps)} 段 | "
                f"預估輸出時長: {format_timestamp(total_dur)} (原片 {format_timestamp(self.meta.video_duration)})"
            )
        except Exception:
            self.lbl_summary.setText("無效的保留區間")

    def _select_output_file(self) -> None:
        if not self.meta:
            return
        default_out = str(default_output_path(self.meta.path))
        file_path, _ = QFileDialog.getSaveFileName(self, "儲存剪輯影片", default_out, "MP4 Video (*.mp4)")
        if file_path:
            self.custom_output_path = Path(file_path)

    def _on_export_clicked(self) -> None:
        if not self.meta:
            return
        out_path = self.custom_output_path or default_output_path(self.meta.path)
        self.export_requested.emit(out_path, self.spin_crf.value())

    def update_progress(self, progress: ExportProgress) -> None:
        pct = int(progress.fraction * 100)
        self.progress_bar.setValue(pct)
        eta_str = f"{progress.eta_seconds:.1f}s" if progress.eta_seconds is not None else "-"
        self.progress_bar.setFormat(f"轉檔中 {pct}% (ETA: {eta_str})")

    def set_exporting_state(self, exporting: bool) -> None:
        self.btn_export.setEnabled(not exporting)
        self.btn_select_out.setEnabled(not exporting)
        self.btn_cancel.setEnabled(exporting)
        if not exporting:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")
