"""測試 GUI 視窗元件載入與訊號綁定（使用 pytest-qt 或無頭 QApplication）。"""

from __future__ import annotations

import os

from PySide6.QtWidgets import QApplication

from video_ad_remover.gui.main_window import MainWindow

# 避免在無桌面環境噴錯
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def test_main_window_instantiation():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.windowTitle() == "Video Ad Detector & Remover (v2.1)"
    assert window.player_panel is not None
    assert window.boundary_view is not None
    assert window.candidate_list is not None
    assert window.timeline_panel is not None
    window.close()
