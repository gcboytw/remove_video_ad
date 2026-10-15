# Video Ad Detector & Remover

影格級精確的影片中插廣告偵測與移除工具（SRS v2.1）。

## 環境需求

- Python 3.12+（使用 `uv` 管理）
- FFmpeg / FFprobe（需在 PATH 中，或以環境變數 `VAR_FFMPEG_PATH` / `VAR_FFPROBE_PATH` 指定）

```bash
uv sync
```

## Step 1 + 2：命令列驗證精確剪輯

```bash
# 產生合成測試片（每格帶 16-bit 影格編號條碼 + 每秒嗶聲）並剪掉兩段 25 秒廣告，自動驗證
uv run python tests/run_cut_check.py

# VFR 來源驗證
uv run python tests/run_cut_check.py --vfr

# 對真實影片剪輯（只做 FFprobe 驗證，不做逐格條碼比對）
uv run python tests/run_cut_check.py --source "影片.mp4" --cut 10000:10750 --cut 30000:30750

# 單元測試 + 精確剪輯驗收測試（AT-008 / AT-009 / AT-010 / AT-012）
uv run pytest
```
