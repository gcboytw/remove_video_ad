"""[模組3] 主視窗（Main Window）：整合五大版面區塊與執行緒訊號連接。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from video_ad_remover.core.exporter import (
    ExportProgress,
    ExportResult,
    ExportSettings,
    plan_export,
)
from video_ad_remover.core.frame_extractor import FrameExtractor
from video_ad_remover.core.interval_calc import collect_confirmed
from video_ad_remover.core.prober import probe_video
from video_ad_remover.gui.widgets.boundary_view import BoundaryView
from video_ad_remover.gui.widgets.candidate_list import CandidateListWidget
from video_ad_remover.gui.widgets.player_panel import PlayerPanel
from video_ad_remover.gui.widgets.timeline_export import TimelineExportPanel
from video_ad_remover.gui.workers import ExportWorker, ScanWorker
from video_ad_remover.models.interval import AdCandidate
from video_ad_remover.models.video_meta import VideoMetadata


class MainWindow(QMainWindow):
    """應用程式主視窗。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Ad Detector & Remover (v2.1)")
        self.resize(1180, 780)

        self.meta: VideoMetadata | None = None
        self.extractor: FrameExtractor | None = None
        self.candidates: list[AdCandidate] = []

        self.scan_worker: ScanWorker | None = None
        self.export_worker: ExportWorker | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # 1. 頂部工具列：載入影片、掃描按鈕與資訊提示
        top_bar = QHBoxLayout()
        btn_open = QPushButton("📂 開啟影片")
        btn_open.clicked.connect(self._open_file_dialog)
        top_bar.addWidget(btn_open)

        self.btn_scan = QPushButton("🔍 開始全片掃描")
        self.btn_scan.setEnabled(False)
        self.btn_scan.clicked.connect(self._start_scan)
        top_bar.addWidget(self.btn_scan)

        self.lbl_info = QLabel("尚未載入影片")
        self.lbl_info.setStyleSheet("color: #aaaaaa; margin-left: 10px;")
        top_bar.addWidget(self.lbl_info)
        top_bar.addStretch()

        main_layout.addLayout(top_bar)

        # 2. 中間工作區：左右分割（左側預覽與微調，右側候選清單）
        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左側垂直佈局（上：主播放器，下：四格對比視圖）
        v_left_widget = QWidget()
        v_left_layout = QVBoxLayout(v_left_widget)
        v_left_layout.setContentsMargins(0, 0, 0, 0)
        v_left_layout.setSpacing(6)

        self.player_panel = PlayerPanel()
        self.boundary_view = BoundaryView()
        self.boundary_view.interval_changed.connect(self._on_interval_modified)

        v_left_layout.addWidget(self.player_panel, stretch=3)
        v_left_layout.addWidget(self.boundary_view, stretch=2)
        h_splitter.addWidget(v_left_widget)

        # 右側：候選清單
        self.candidate_list = CandidateListWidget()
        self.player_panel.frame_changed.connect(self.candidate_list.update_player_frame)
        self.candidate_list.candidate_selected.connect(self._on_candidate_selected)
        self.candidate_list.candidates_updated.connect(self._on_candidates_updated)
        h_splitter.addWidget(self.candidate_list)

        h_splitter.setStretchFactor(0, 3)
        h_splitter.setStretchFactor(1, 2)
        main_layout.addWidget(h_splitter, stretch=1)

        # 3. 底部：視覺化時間軸與輸出面板
        self.timeline_panel = TimelineExportPanel()
        self.timeline_panel.export_requested.connect(self._start_export)
        self.timeline_panel.cancel_requested.connect(self._cancel_export)
        main_layout.addWidget(self.timeline_panel)

        # 啟用拖曳檔案支援
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                ext = Path(url.toLocalFile()).suffix.lower()
                if ext in {".mp4", ".mkv", ".mov", ".m4v", ".avi"}:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            file_path = Path(url.toLocalFile())
            if file_path.is_file():
                self.load_video(file_path)
                event.acceptProposedAction()
                return

    def _open_file_dialog(self) -> None:
        init_dir = str(Path.cwd())
        filters = "Video Files (*.mp4 *.mkv *.mov *.m4v *.avi);;All Files (*.*)"

        # 在 macOS 上直接使用 Qt 內建對話框，避開系統 NSOpenPanel 卡住或需點兩次的問題
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇影片檔案",
            init_dir,
            filters,
            options=QFileDialog.Option.DontUseNativeDialog,
        )

        if file_path:
            self.load_video(Path(file_path))

    def load_video(self, path: Path) -> None:
        try:
            if self.extractor:
                self.extractor.close()

            self.meta = probe_video(path)
            self.extractor = FrameExtractor(path, self.meta)

            # 更新介面
            summary = self.meta.summary()
            self.lbl_info.setText(
                f"{summary['檔案名稱']} | {summary['解析度']} | {summary['影格率']} | 時長: {summary['總時長']}"
            )
            self.btn_scan.setEnabled(True)

            self.player_panel.set_context(self.meta, self.extractor)
            self.boundary_view.set_context(self.meta, self.extractor)
            self.candidate_list.set_metadata(self.meta)
            self.candidate_list.set_candidates([])

            self.candidates = []
            self.timeline_panel.set_data(self.meta, self.candidates)

        except Exception as exc:
            QMessageBox.critical(self, "載入失敗", f"無法解析影片資訊：\n{exc}")

    def _start_scan(self) -> None:
        if not self.meta:
            return

        self.btn_scan.setEnabled(False)
        self.scan_worker = ScanWorker(self.meta.path, self.meta)
        self.scan_worker.progress_updated.connect(
            lambda stage, frac: self.lbl_info.setText(f"{stage} ({int(frac*100)}%)")
        )
        self.scan_worker.finished_success.connect(self._on_scan_success)
        self.scan_worker.finished_error.connect(self._on_scan_error)
        self.scan_worker.start()

    def _on_scan_success(self, candidates: list[AdCandidate]) -> None:
        self.btn_scan.setEnabled(True)
        self.candidates = candidates
        self.candidate_list.set_candidates(candidates)
        self.timeline_panel.set_data(self.meta, candidates)
        self.lbl_info.setText(f"掃描完成：找到 {len(candidates)} 個候選廣告區間")

        if candidates:
            # 預設選取第一個候選
            self.candidate_list.table.selectRow(0)

    def _on_scan_error(self, err_msg: str) -> None:
        self.btn_scan.setEnabled(True)
        self.lbl_info.setText("掃描中止")
        QMessageBox.warning(self, "掃描提示", err_msg)

    def _on_candidate_selected(self, cand: AdCandidate | None) -> None:
        self.boundary_view.load_candidate(cand)
        if cand:
            self.player_panel.seek_frame(cand.start_frame)

    def _on_interval_modified(self, cand: AdCandidate) -> None:
        self.candidate_list.refresh()
        self.timeline_panel.set_data(self.meta, self.candidates)

    def _on_candidates_updated(self) -> None:
        self.candidates = self.candidate_list.get_candidates()
        self.timeline_panel.set_data(self.meta, self.candidates)

    def _start_export(self, out_path: Path, crf: int) -> None:
        if not self.meta:
            return

        # 直接向 candidate_list 取得最新狀態
        self.candidates = self.candidate_list.get_candidates()
        confirmed = collect_confirmed(self.candidates)
        if not confirmed:
            QMessageBox.warning(self, "無法輸出", "目前沒有任何「已確認為廣告」的區間需要刪除。")
            return

        settings = ExportSettings(crf=crf, overwrite=True)
        try:
            plan = plan_export(self.meta, confirmed, out_path, settings=settings)
        except Exception as exc:
            QMessageBox.critical(self, "計畫建立失敗", f"無法建立剪輯計畫：\n{exc}")
            return

        self.timeline_panel.set_exporting_state(True)
        self.export_worker = ExportWorker(plan)
        self.export_worker.progress_updated.connect(self._on_export_progress)
        self.export_worker.finished_success.connect(self._on_export_success)
        self.export_worker.finished_error.connect(self._on_export_error)
        self.export_worker.start()

    def _on_export_progress(self, progress: ExportProgress) -> None:
        self.timeline_panel.update_progress(progress)

    def _on_export_success(self, result: ExportResult) -> None:
        self.timeline_panel.set_exporting_state(False)
        QMessageBox.information(
            self, "輸出完成", f"影片剪輯完成並已通過品質驗證！\n儲存至：{result.output}\n耗時：{result.elapsed:.1f} 秒"
        )

    def _on_export_error(self, err_msg: str) -> None:
        self.timeline_panel.set_exporting_state(False)
        QMessageBox.critical(self, "轉檔失敗", f"輸出過程發生錯誤：\n{err_msg}")

    def _cancel_export(self) -> None:
        if self.export_worker:
            self.export_worker.cancel()

    def closeEvent(self, event) -> None:
        if self.extractor:
            self.extractor.close()
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.cancel()
            self.export_worker.wait(2000)
        super().closeEvent(event)
