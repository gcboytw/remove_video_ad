"""[模組3] QThread 背景工作執行緒：全片偵測掃描與 FFmpeg 轉檔輸出。

設計規範（FR-012）：
- 耗時任務脫離主執行緒，確保介面不卡死。
- 提供進度回報（階段、百分比、ETA）。
- 提供安全取消機制。
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from video_ad_remover.core.detector.pipeline import PipelineConfig, detect_ads
from video_ad_remover.core.exporter import (
    ExportPlan,
    ExportProgress,
    ExportResult,
    run_export,
)
from video_ad_remover.models.interval import AdCandidate
from video_ad_remover.models.video_meta import VideoMetadata


class ScanWorker(QThread):
    """背景執行三階段廣告偵測掃描。"""

    progress_updated = Signal(str, float)  # stage_name, fraction (0.0~1.0)
    finished_success = Signal(list)  # list[AdCandidate]
    finished_error = Signal(str)  # error_message

    def __init__(
        self,
        path: Path,
        meta: VideoMetadata,
        config: PipelineConfig | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.path = path
        self.meta = meta
        self.config = config or PipelineConfig()
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            candidates = detect_ads(
                self.path,
                self.meta,
                config=self.config,
                stage_cb=lambda stage, frac: self.progress_updated.emit(stage, frac),
                cancel_event=self.cancel_event,
            )
            if self.cancel_event.is_set():
                self.finished_error.emit("使用者已取消掃描")
            else:
                self.finished_success.emit(candidates)
        except Exception as exc:
            self.finished_error.emit(str(exc))


class ExportWorker(QThread):
    """背景執行 FFmpeg 精確剪輯與轉檔。"""

    progress_updated = Signal(object)  # ExportProgress
    finished_success = Signal(object)  # ExportResult
    finished_error = Signal(str)  # error_message

    def __init__(self, plan: ExportPlan, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.plan = plan
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            result = run_export(
                self.plan,
                progress_cb=lambda p: self.progress_updated.emit(p),
                cancel_event=self.cancel_event,
            )
            self.finished_success.emit(result)
        except Exception as exc:
            self.finished_error.emit(str(exc))
