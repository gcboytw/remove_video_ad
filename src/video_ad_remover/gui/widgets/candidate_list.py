"""[模組3] 候選廣告清單元件（Candidate List）。

設計規範（FR-005 / FR-006 / FR-008）：
- 顯示區間編號、起訖時間、起訖影格、持續秒數、優先級評分。
- 三態確認下拉選單：未確認 (UNCONFIRMED) / 已確認為廣告 (CONFIRMED_AD) / 已排除 (EXCLUDED)。
- 手動新增區間對話框（可精確自訂起訖格與秒數）。
- 提供「只顯示 Top 10 候選」過濾切換，過濾雜訊。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from video_ad_remover.models.interval import (
    AdCandidate,
    DetectionSource,
    FrameInterval,
    IntervalStatus,
)
from video_ad_remover.models.video_meta import VideoMetadata
from video_ad_remover.utils.time_conv import format_timestamp


class AddIntervalDialog(QDialog):
    """手動新增廣告區間對話框。"""

    def __init__(
        self,
        meta: VideoMetadata,
        default_start: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("手動新增廣告區間")
        self.resize(360, 200)
        self.meta = meta

        fps = round(meta.nominal_fps if meta.nominal_fps > 0 else 30.0)
        default_end = min(meta.frame_count, default_start + 25 * fps)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, meta.frame_count - 1)
        self.spin_start.setValue(default_start)

        self.spin_end = QSpinBox()
        self.spin_end.setRange(1, meta.frame_count)
        self.spin_end.setValue(default_end)

        self.lbl_time_info = QLabel()
        self._update_time_label()

        self.spin_start.valueChanged.connect(self._update_time_label)
        self.spin_end.valueChanged.connect(self._update_time_label)

        form.addRow("起始影格 (0-based):", self.spin_start)
        form.addRow("結束影格 (0-based):", self.spin_end)
        form.addRow("時長摘要:", self.lbl_time_info)
        layout.addLayout(form)

        # 確認 / 取消按鈕
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._validate_and_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _update_time_label(self) -> None:
        s = self.spin_start.value()
        e = self.spin_end.value()
        if s < e and s < self.meta.frame_count:
            dur = self.meta.segment_duration(s, e)
            t_s = format_timestamp(self.meta.time_of(s))
            t_e = format_timestamp(self.meta.time_of(e))
            self.lbl_time_info.setText(f"{t_s} ~ {t_e} (共 {dur:.2f} 秒, {e - s} 格)")
        else:
            self.lbl_time_info.setText("無效區間 (起點必須小於終點)")

    def _validate_and_accept(self) -> None:
        if self.spin_start.value() >= self.spin_end.value():
            QMessageBox.warning(self, "無效區間", "起始影格必須小於結束影格！")
            return
        self.accept()

    def get_interval(self) -> tuple[int, int]:
        return self.spin_start.value(), self.spin_end.value()


class CandidateListWidget(QWidget):
    """候選廣告清單與狀態管理元件。"""

    candidate_selected = Signal(object)  # AdCandidate | None
    candidates_updated = Signal()  # 清單狀態有任何更新

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.meta: VideoMetadata | None = None
        self.candidates: list[AdCandidate] = []
        self._block_signals = False
        self.current_player_frame: int = 0

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # 頂部過濾控制列
        top_ctrl = QHBoxLayout()
        self.chk_top_only = QCheckBox("只顯示 Top 10 候選")
        self.chk_top_only.setChecked(True)
        self.chk_top_only.toggled.connect(self.refresh)

        self.lbl_count = QLabel("共 0 個候選")
        self.lbl_count.setStyleSheet("color: #888; font-size: 11px;")

        top_ctrl.addWidget(self.chk_top_only)
        top_ctrl.addStretch()
        top_ctrl.addWidget(self.lbl_count)
        layout.addLayout(top_ctrl)

        # 表格元件
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["#", "起訖時間", "起訖影格", "秒數", "評分", "確認狀態"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        layout.addWidget(self.table)

        # 底部按鈕
        btn_layout = QHBoxLayout()
        btn_add = QPushButton("➕ 手動新增區間")
        btn_del = QPushButton("🗑 刪除選取區間")
        btn_add.clicked.connect(self._add_manual_interval)
        btn_del.clicked.connect(self._delete_selected)
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_del)
        layout.addLayout(btn_layout)

    def set_metadata(self, meta: VideoMetadata) -> None:
        self.meta = meta

    def update_player_frame(self, frame_idx: int) -> None:
        self.current_player_frame = frame_idx

    def set_candidates(self, candidates: list[AdCandidate]) -> None:
        self.candidates = list(candidates)
        self.refresh()

    def _get_visible_candidates(self) -> list[AdCandidate]:
        if self.chk_top_only.isChecked():
            # 優先保留前 10 名或已確認者
            top10 = self.candidates[:10]
            confirmed = [c for c in self.candidates[10:] if c.status == IntervalStatus.CONFIRMED_AD]
            return top10 + confirmed
        return self.candidates

    def refresh(self) -> None:
        self._block_signals = True
        self.table.setRowCount(0)
        total_meta = self.meta

        visible = self._get_visible_candidates()
        self.lbl_count.setText(f"顯示 {len(visible)} / 共 {len(self.candidates)} 個候選")

        for row, cand in enumerate(visible):
            self.table.insertRow(row)

            # 1. 編號
            item_no = QTableWidgetItem(str(row + 1))
            item_no.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, item_no)

            # 2. 起訖時間
            if total_meta:
                t_str = f"{format_timestamp(total_meta.time_of(cand.start_frame))} ~ {format_timestamp(total_meta.time_of(cand.end_frame))}"
                dur_str = f"{total_meta.segment_duration(cand.start_frame, cand.end_frame):.1f}s"
            else:
                t_str = "-"
                dur_str = "-"
            self.table.setItem(row, 1, QTableWidgetItem(t_str))

            # 3. 起訖影格
            item_frames = QTableWidgetItem(f"{cand.start_frame} ~ {cand.end_frame}")
            item_frames.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, item_frames)

            # 4. 持續秒數
            item_dur = QTableWidgetItem(dur_str)
            item_dur.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 3, item_dur)

            # 5. 評分
            item_score = QTableWidgetItem(f"{cand.score:.2f}")
            item_score.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 4, item_score)

            # 6. 確認狀態下拉選單
            combo = QComboBox()
            combo.addItem(IntervalStatus.UNCONFIRMED.label, IntervalStatus.UNCONFIRMED)
            combo.addItem(IntervalStatus.CONFIRMED_AD.label, IntervalStatus.CONFIRMED_AD)
            combo.addItem(IntervalStatus.EXCLUDED.label, IntervalStatus.EXCLUDED)

            idx = combo.findData(cand.status)
            if idx >= 0:
                combo.setCurrentIndex(idx)

            # 使用明確的 slot 避免 lambda 閉包抓錯物件
            def make_handler(target_cand: AdCandidate, target_combo: QComboBox):
                def handler(new_idx: int):
                    if self._block_signals:
                        return
                    new_val = target_combo.itemData(new_idx)
                    if new_val:
                        self._on_status_changed(target_cand, new_val)
                return handler

            combo.currentIndexChanged.connect(make_handler(cand, combo))
            self.table.setCellWidget(row, 5, combo)

        self._block_signals = False

    def get_candidates(self) -> list[AdCandidate]:
        """取得全部候選清單最新狀態。"""
        return self.candidates

    def get_confirmed_candidates(self) -> list[AdCandidate]:
        """取得目前所有已確認為廣告的候選。"""
        return [c for c in self.candidates if c.status == IntervalStatus.CONFIRMED_AD]

    def _on_row_selected(self) -> None:
        if self._block_signals:
            return
        row = self.table.currentRow()
        visible = self._get_visible_candidates()
        if 0 <= row < len(visible):
            self.candidate_selected.emit(visible[row])
        else:
            self.candidate_selected.emit(None)

    def _on_status_changed(self, cand: AdCandidate, new_status: object) -> None:
        try:
            cand.status = IntervalStatus(str(new_status))
        except ValueError:
            cand.status = IntervalStatus.UNCONFIRMED
        self.candidates_updated.emit()

    def _add_manual_interval(self) -> None:
        if not self.meta:
            return

        dlg = AddIntervalDialog(self.meta, default_start=self.current_player_frame, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            s, e = dlg.get_interval()
            new_cand = AdCandidate(
                interval=FrameInterval(s, e),
                status=IntervalStatus.CONFIRMED_AD,
                source=DetectionSource.MANUAL,
                score=1.0,
            )
            # 將手動新增的區間置頂
            self.candidates.insert(0, new_cand)
            self.refresh()
            self.table.selectRow(0)
            self.candidates_updated.emit()

    def _delete_selected(self) -> None:
        row = self.table.currentRow()
        visible = self._get_visible_candidates()
        if 0 <= row < len(visible):
            target = visible[row]
            if target in self.candidates:
                self.candidates.remove(target)
            self.refresh()
            self.candidate_selected.emit(None)
            self.candidates_updated.emit()
