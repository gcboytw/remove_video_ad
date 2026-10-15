"""時間格式轉換工具：秒數 <-> HH:MM:SS.mmm、FFprobe 分數字串解析。"""

from __future__ import annotations

from fractions import Fraction


def format_timestamp(seconds: float) -> str:
    """將秒數格式化為 ``HH:MM:SS.mmm``（負數會加上 ``-`` 前綴）。"""
    sign = "-" if seconds < 0 else ""
    total_ms = round(abs(seconds) * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{sign}{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def parse_timestamp(text: str) -> float:
    """解析 ``HH:MM:SS.mmm`` / ``MM:SS.mmm`` / ``SS.mmm`` 為秒數。"""
    text = text.strip()
    if not text:
        raise ValueError("時間字串為空")
    sign = -1.0 if text.startswith("-") else 1.0
    parts = text.lstrip("+-").split(":")
    if len(parts) > 3:
        raise ValueError(f"無法解析時間字串：{text!r}")
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"無法解析時間字串：{text!r}") from exc
    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    return sign * seconds


def parse_fraction(text: str | None) -> Fraction | None:
    """解析 FFprobe 的分數字串（如 ``30000/1001``）；``0/0`` 或無效值回傳 ``None``。"""
    if not text or text in {"N/A", "0/0"}:
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def parse_float(text: object, default: float | None = None) -> float | None:
    """寬鬆解析浮點數；``N/A`` 或空值回傳 ``default``。"""
    if text is None or text == "" or text == "N/A":
        return default
    try:
        return float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
