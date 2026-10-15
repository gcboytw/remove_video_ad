# 方案 A：全片重新編碼實作指南（Frame-Accurate 精準剪輯）

## 1. 背景與問題核心

在進行影片廣告區間切除時，常面臨兩個核心矛盾：

1. **一閃而逝的廣告殘影**：
   - **成因**：H.264 / H.265 等格式採用 GOP（Group of Pictures）架構，僅有 **I 影格（關鍵影格）** 具備完整獨立影像。
   - 若使用直接串流複製（`-c copy`），FFmpeg 只能在關鍵影格上下刀，若指定的時間點落在 P/B 影格，會自動「吸附」至最近的關鍵影格，導致切除點殘留 0.1 ~ 2 秒的廣告畫面。
2. **轉檔後容量異常膨脹**：
   - **成因**：若腳本使用了重新編碼，但未正確指定品質參數（CRF / 碼率限制），或音訊未指定編碼器而被轉為未壓縮的 PCM（WAV），檔案體積便會大幅翻倍。

---

## 2. 方案 A 核心目標

- **精準度**：達到**影格級（Frame-accurate）** 精準切斷，廣告區間完全乾淨切除、不留任何一格殘影。
- **體積控制**：透過現代編碼器品質控制參數（CRF / Global Quality / q:v），維持與原片接近或更小的檔案大小。
- **執行效率**：充分利用 **Intel 258V (Quick Sync)** 與 **Mac mini M2 Pro (VideoToolbox)** 的硬體加速，將 40 分鐘影片的剪輯轉檔時間壓在 **2 ～ 4 分鐘** 內。

---

## 3. 使用者硬體專用參數配置

### 設備 1：Intel Core Ultra 7 258V (Windows / Lunar Lake)
> **推薦編碼器**：`h264_qsv`（Intel Quick Sync Video）

- **視訊參數**：`-c:v h264_qsv -global_quality 24 -preset fast`
  - `-global_quality 24`：類似 x264 的 CRF，數值在 23~25 間，能兼顧極高畫質與合理容量。
  - `-preset fast`：硬體編碼預設配置，速度與品質平衡點。
- **音訊參數**：`-c:a aac -b:a 128k`（防止音訊未壓縮膨脹）。
- **預估耗時**：40 分鐘 1080p 影片約 **2 ～ 4 分鐘**。

---

### 設備 2：Mac mini M2 Pro (macOS / Apple Silicon)
> **推薦編碼器**：`h264_videotoolbox`（Apple Media Engine 硬體加速）

- **視訊參數**：`-c:v h264_videotoolbox -q:v 60`
  - `-q:v 60`：Apple 專用品質級距（1~100），推薦設在 55~65 之間。
- **音訊參數**：`-c:a aac -b:a 128k`
- **預估耗時**：40 分鐘 1080p 影片約 **1.5 ～ 3 分鐘**，完全靜音且極低功耗。

---

### 備用方案：純 CPU 軟體編碼（跨平台無顯卡環境）
> **備用編碼器**：`libx264`

- **視訊參數**：`-c:v libx264 -crf 24 -preset veryfast`
- **音訊參數**：`-c:a aac -b:a 128k`
- **預估耗時**：約 **15 ～ 25 分鐘**。

---

## 4. FFmpeg 影格精準剪輯實作方式

要去除多段廣告，本質上是**保留所有非廣告的有效片段（Keep Segments），並無縫拼接**。

### 推薦方式：單一 Filtergraph（無需生成中間暫存檔）

假設一部 40 分鐘影片，識別出一段廣告在 `00:10:00.000` 到 `00:12:00.000`，代表要保留：
- 片段 1：`00:00:00.000` ~ `00:10:00.000`（0 ~ 600 秒）
- 片段 2：`00:12:00.000` ~ 結束（720 秒 ~ 結束）

#### Intel 258V (Windows QSV) 範例指令：
```bash
ffmpeg -hwaccel qsv -i "input.mp4" -filter_complex \
"[0:v]trim=start=0:end=600,setpts=PTS-STARTPTS[v1]; \
 [0:a]atrim=start=0:end=600,asetpts=PTS-STARTPTS[a1]; \
 [0:v]trim=start=720,setpts=PTS-STARTPTS[v2]; \
 [0:a]atrim=start=720,asetpts=PTS-STARTPTS[a2]; \
 [v1][a1][v2][a2]concat=n=2:v=1:a=1[outv][outa]" \
-map "[outv]" -map "[outa]" \
-c:v h264_qsv -global_quality 24 -preset fast \
-c:a aac -b:a 128k \
-y "output_ad_removed.mp4"
```

#### Mac mini M2 Pro (VideoToolbox) 範例指令：
```bash
ffmpeg -i "input.mp4" -filter_complex \
"[0:v]trim=start=0:end=600,setpts=PTS-STARTPTS[v1]; \
 [0:a]atrim=start=0:end=600,asetpts=PTS-STARTPTS[a1]; \
 [0:v]trim=start=720,setpts=PTS-STARTPTS[v2]; \
 [0:a]atrim=start=720,asetpts=PTS-STARTPTS[a2]; \
 [v1][a1][v2][a2]concat=n=2:v=1:a=1[outv][outa]" \
-map "[outv]" -map "[outa]" \
-c:v h264_videotoolbox -q:v 60 \
-c:a aac -b:a 128k \
-y "output_ad_removed.mp4"
```

---

## 5. Python 腳本動態組裝邏輯範例

在 Python 中，若已辨識出多個廣告區間 `ad_intervals = [(start_sec, end_sec), ...]`，可反向計算出要保留的 `keep_intervals`，並動態組裝指令：

```python
import platform
import subprocess

def get_encoder_settings():
    """依作業系統與硬體自動選擇最佳編碼器參數"""
    system_name = platform.system()
    if system_name == "Darwin":  # macOS M2 Pro
        return ["-c:v", "h264_videotoolbox", "-q:v", "60"]
    else:  # Windows (Intel 258V QSV)
        # 建議搭配 -hwaccel qsv
        return ["-hwaccel", "qsv", "-c:v", "h264_qsv", "-global_quality", "24", "-preset", "fast"]

def build_filter_complex(keep_intervals):
    v_filters = []
    a_filters = []
    concat_inputs = []

    for idx, (start, end) in enumerate(keep_intervals):
        v_label = f"v{idx}"
        a_label = f"a{idx}"
        
        # 組裝 trim 條件
        trim_v = f"trim=start={start}" + (f":end={end}" if end is not None else "")
        trim_a = f"atrim=start={start}" + (f":end={end}" if end is not None else "")
        
        v_filters.append(f"[0:v]{trim_v},setpts=PTS-STARTPTS[{v_label}]")
        a_filters.append(f"[0:a]{trim_a},asetpts=PTS-STARTPTS[{a_label}]")
        concat_inputs.append(f"[{v_label}][{a_label}]")

    filter_str = "; ".join(v_filters + a_filters)
    filter_str += f"; {''.join(concat_inputs)}concat=n={len(keep_intervals)}:v=1:a=1[outv][outa]"
    return filter_str
```

---

## 6. 優缺點與綜合評估

- **優點**：
  1. **零殘影**：每一影格完全按照時間點精確切齊。
  2. **影音同步**：透過 `setpts=PTS-STARTPTS` 與 `asetpts` 重設時間戳，避免切除後聲音與畫面不同步。
  3. **容量受控**：固定品質模式下，輸出大小正常（約 800MB ~ 1.5GB / 40 分鐘）。
  4. **速度極快**：在使用者現有的 M2 Pro 或 258V 晶片加持下，耗時僅數分鐘。
- **注意事項**：
  - 確保系統安裝的 FFmpeg 支援該硬體編碼器（Windows 下可用 `ffmpeg -encoders | findstr qsv`，Mac 下可用 `ffmpeg -encoders | grep videotoolbox` 驗證）。
