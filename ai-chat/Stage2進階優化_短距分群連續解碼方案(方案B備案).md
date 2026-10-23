# Stage 2 進階優化：短距分群連續解碼方案 (方案 B 備案)

---

## 1. 方案背景與物理瓶頸分析

在 Stage 2 候選區間配對完成 Step 1~6（特徵快取 + 不可逆超標早停）優化後，Stage 2 總耗時已從 591 秒大幅降至 321 秒（提速 1.84 倍）。然而，細部 Profiling 指標顯示：**影像解碼（`decode_time`）仍佔總耗時的 97.8%（314 秒）**。

### 1.1 隨機 Seek 的物理限制
* 現行 `extractor.get_frame(f_idx)` 內部使用 OpenCV 的 `self._cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)`。
* 在 H.264 / H.265 具備複雜 GOP（B-frames / P-frames）的影片中，每次 `set()` 隨機跳轉都必須回溯最近的 Keyframe（I-frame）並依序前向解碼至目標格，平均單次耗時高達 **16~19 毫秒**。

### 1.2 目標影格的時間軸分佈特徵（實測統計）
針對 42 分鐘真實影片 `video-22.mp4`（共 64,213 影格，940 個 Stage 1 切點），我們對所有進入時長窗（18~32s）的候選影格進行分佈統計：
* **需解碼的目標唯一影格總數**：**17,100 格**。
* **相鄰目標影格距離 $\Delta F = F_{i+1} - F_i$**：
  * 中位數（Median）：**僅 2.0 格**。
  * 平均數（Mean）：**僅 3.7 格**。
  * **$\Delta F \le 30$ 格（約 1 個 GOP / 1 秒內）比例高達 99.4%**！
  * **$\Delta F \le 60$ 格比例高達 99.9%**！

> **結論**：目前程式為了在時間軸上前進平均 3.7 格，每次都執行一次耗時 17ms 的隨機 Seek。如果改用前向連續 `read()`，將產生極為顯著的效能突破。

---

## 2. 實機 A/B 對決數據與效能預估

在相同硬體與 `video-22.mp4` 上進行 500 格真實解碼對比：

| 解碼機制 | 500 格實測耗時 | 平均單格耗時 | Seek 次數 | 實測加速比 |
| :--- | :--- | :--- | :--- | :--- |
| **方法 A（現行單格隨機 Seek）** | 8.34 秒 | 16.7 ms / 格 | 500 次 | 基準線 (1.0x) |
| **方法 B（短距分群前向讀取，$\Delta F \le 30$）** | **1.95 秒** | **3.9 ms / 格** | **僅 2 次** | **⚡ 4.28x 加速！** |

### 2.1 全片 17,100 格實測外推
* 收集全片 17,100 個目標影格索引並排序：**耗時僅 0.017 秒**。
* 實測前 2,000 格連續解碼 + 等比例縮圖 + 特徵抽取：**耗時 14.31 秒**。
* **全片 17,100 格推算總耗時：僅約 122 秒（約 2 分鐘）**！

### 2.2 全片各階段效能推演

```
[原始 Baseline]            591 秒 (約 9.8 分鐘)  ■■■■■■■■■■ (1.00x)
   ↓ (方案 A: LRU 快取 + 不可逆早停)
[方案 A 現行狀態]          321 秒 (約 5.3 分鐘)  ■■■■■■ (1.84x)
   ↓ (方案 B: 兩階段分群連續解碼)
[方案 B 預估成果]          ~130 秒 (約 2.1 分鐘)  ■■ (4.50x 加速)
```

---

## 3. 方案 B 架構設計：兩階段預填管線 (Two-Pass Pipeline)

### 3.1 處理流程
```
[Pass 1: 便宜的純數字先驗]
   ↓ 收集所有符合時長窗 (18-32s) 與強度先驗的 target_frame_indices
[時間軸去重與排序]
   ↓ sorted_frames = sorted(set(needed_frames)) (耗時 0.017s)
[Pass 2: 串流分群前向解碼 + 特徵預填]
   ↓ 若相鄰幀距 <= 30: 連續 read() 前進 (免 seek)
   ↓ 若相鄰幀距 > 30: 執行專門 seek()
   ↓ 預先萃取特徵並灌入 feat_cache (17,100 筆約 68MB)
[雙重迴圈配對與打分]
   ↓ 所有 _get_cached_feature(f) 皆 100% 記憶體命中 (0ms)
   ↓ 結合不可逆早停與 IoU 去重輸出候選
```

### 3.2 完整實作參考程式碼 (Ready-to-use)

```python
def prefill_features_grouped(
    video_path: Path,
    sorted_frames: list[int],
    target_size: tuple[int, int],
    max_forward_gap: int = 30,
) -> OrderedDict[int, tuple[np.ndarray, np.ndarray]]:
    """以短距分群前向連續解碼，預先提取所有目標影格之特徵。"""
    cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
    cap = cv2.VideoCapture(str(video_path))
    curr_pos = -9999

    try:
        for f_idx in sorted_frames:
            # 若與當前播放頭距離在門檻內，以前向循序讀取推進 (省去 GOP 重複解碼)
            if 0 <= f_idx - curr_pos <= max_forward_gap:
                while curr_pos < f_idx:
                    ret, frame = cap.read()
                    curr_pos += 1
            else:
                # 跨度過大，才執行專門隨機 Seek
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                curr_pos = f_idx + 1

            if ret and frame is not None:
                # 等比例縮圖並計算特徵
                frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                cache[f_idx] = compute_frame_features(frame_resized)
    finally:
        cap.release()

    return cache
```

---

## 4. 權衡與後續啟用指引

### 4.1 記憶體佔用評估
* 每個影格特徵（HSV 直方圖 + 邊緣純量）約佔 **4 KB**。
* 全片 17,100 筆特徵常駐記憶體約：$17,100 \times 4\text{ KB} \approx \mathbf{68.4\text{ MB}}$。
* 函式結束後立即隨區域變數釋放，對現代作業系統與桌面端應用完全在安全負荷內。

### 4.2 何時啟用方案 B？
1. **處理極長影片（時長 > 1 小時）**：當單片總處理時間仍需縮減至 2~3 分鐘內時。
2. **批次自動化流水線（Batch Pipeline）**：需要高吞吐量（Throughput）、伺服器後台批次轉檔時。
3. **欲追求絕對物理確定性**：單向解碼完全不受 OpenCV 非關鍵影格隨機 Seek 漂移干擾。
