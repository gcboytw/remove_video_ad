"""檢查本機 FFmpeg / FFprobe 是否可用，並提供跨平台子程序參數。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache

# 可用環境變數覆寫執行檔位置（例如 Windows 未加入 PATH 時）
ENV_OVERRIDES = {"ffmpeg": "VAR_FFMPEG_PATH", "ffprobe": "VAR_FFPROBE_PATH"}


class ToolNotFoundError(RuntimeError):
    """找不到 FFmpeg / FFprobe 執行檔或無法執行。"""


@dataclass(frozen=True, slots=True)
class ToolInfo:
    name: str
    path: str
    version: str


def hidden_window_kwargs() -> dict:
    """Windows 上避免子程序跳出黑色主控台視窗；其他平台回傳空 dict。"""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}  # type: ignore[attr-defined]
    return {}


@lru_cache(maxsize=None)
def resolve_tool(name: str) -> str:
    """回傳執行檔完整路徑（環境變數覆寫優先，其次 PATH）。"""
    override = os.environ.get(ENV_OVERRIDES.get(name, ""), "")
    if override:
        if shutil.which(override) or os.path.isfile(override):
            return override
        raise ToolNotFoundError(f"環境變數 {ENV_OVERRIDES[name]} 指定的 {name} 不存在：{override}")
    found = shutil.which(name)
    if not found:
        raise ToolNotFoundError(
            f"找不到 {name}。請安裝 FFmpeg 並加入 PATH，"
            f"或設定環境變數 {ENV_OVERRIDES.get(name, name.upper())}。"
        )
    return found


def get_tool_info(name: str) -> ToolInfo:
    path = resolve_tool(name)
    try:
        proc = subprocess.run(
            [path, "-hide_banner", "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            **hidden_window_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolNotFoundError(f"無法執行 {name}（{path}）：{exc}") from exc
    if proc.returncode != 0:
        raise ToolNotFoundError(f"{name} -version 失敗：{proc.stderr.strip()[:300]}")
    first_line = (proc.stdout.splitlines() or [""])[0]
    return ToolInfo(name=name, path=path, version=first_line.strip())


def check_environment() -> dict[str, ToolInfo]:
    """啟動時呼叫：確認 ffmpeg 與 ffprobe 都可執行，否則拋出 ToolNotFoundError。"""
    return {name: get_tool_info(name) for name in ("ffmpeg", "ffprobe")}
