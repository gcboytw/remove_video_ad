---
trigger: always_on
---

# CRITICAL SYSTEM RESTRICTION: TOOL USAGE FOR IMAGES

1. **NEVER CALL FILE VIEW TOOLS ON IMAGES**:
   - You are STRICTLY FORBIDDEN from using `view_file`, `read_file`, `open_file`, or any visual preview tools on files ending in:
     `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.gif`.
   - Reason: The IDE converts these files into oversized base64 API payloads that immediately crash the session (`Agent execution terminated due to error`).

2. **HANDLING IMAGE PATHS IN TERMINAL**:
   - If a Python script outputs a path to an image file, DO NOT attempt to inspect it.
   - Simply treat it as an opaque artifact for human review and report the script's numerical terminal output.


# 影格與影像檔案分析規範（No Binary Image Reading Rule）

本規則適用於本專案（`remove_video_ad`）。

## 核心限制
- **嚴禁 Agent 直接讀取二進位圖片**：
  Agent 在任何情況下，**絕對不可使用 `view_file` 或其他工具直接開啟、讀取、檢視或傳輸任何影像／圖片檔案**（包含 `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp` 等二進位媒體檔）。
  - *原因*：影格檔案或高解析度圖片傳送至模型時，極易引發 API Payload 過大、連線超時中斷，造成 `Agent execution terminated due to error` 當機。

## 影像分析的標準作業方式
若需要分析影格特徵、黑白畫面、邊界跳轉或廣告片段比對：
1. **純程式碼與數值萃取**：
   - 撰寫 Python 腳本（優先使用 `uv run` 搭配 OpenCV、NumPy、scikit-image 等函式庫）在本地環境運算。
   - 僅將計算結果以**文字指標、統計數據（如均值、標準差、直方圖差異、MSE/SSIM、場景變化率）或 JSON/CSV 報告**輸出至終端。
2. **人工檢視流程**：
   - 若必須肉眼檢視影格，腳本僅負責將影格另存至本機資料夾（例如 `tests/` 或暫存目錄）。
   - 由使用者直接在作業系統檔案總管中開啟查看，Agent 不可嘗試主動讀入對話上下文。
