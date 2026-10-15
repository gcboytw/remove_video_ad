"""應用程式啟動入口：檢查環境並啟動 PySide6 GUI。"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from video_ad_remover.core.env_checker import ToolNotFoundError, check_environment
from video_ad_remover.gui.main_window import MainWindow


def main() -> int:
    try:
        check_environment()
    except ToolNotFoundError as exc:
        print(f"[環境檢查失敗] {exc}", file=sys.stderr)
        return 1

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
