# Stage 1 影格時間校正與 Stage 2 效能優化：整合實施計畫

## 1. 背景與核心痛點綜述

在目前影片去廣告（`remove_video_ad`）的處理管線中，前端廣告偵測分為兩大關鍵階段：
- **Stage 1（全片快速掃描）**：每秒抽樣約 3 格，透過色彩直方圖與邊緣變化，粗略篩選全片的劇烈切換點集合（Cut Points $T_{cuts}$）。
- **Stage 2（廣告區間配對與評分）**：將切點組合配對為候選廣告區間，檢驗時長、前後正片呼應度、廣告內部色彩穩定度（圖卡特徵）與反差度，最後以 IoU 去重。

在近期實測中，浮現了兩個不同維度的核心挑戰：
1. **準確度痛點（抓不到候選區間 / 嚴重影格漂移）**：
   以測試樣本 `video-22.mp4` 為例（解析度 1080×606 / 實際 1077×606、H.264、編碼 FPS: 25），前端辨識完全抓不到廣告候選區間，且存在嚴重的時間與影格漂移。
2. **效能瓶頸痛點（Stage 2 執行過慢）**：
   在長片或切點較多的影片中，Stage 2 頻繁進行隨機跳轉解碼（Random Seek I/O）且缺乏快取，導致大量重複解碼與計算，成為整體流程的主要耗時瓶頸。

---

## 2. 兩大計畫的定位與協同關係

這兩項優化分別處於管線的上下游，**彼此完全不衝突，而是「準確率」與「速度」的相輔相成**：

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1：影格時間校正與抗漂移（方案 B）                       │
│ 目的：保證「原料正確」── 消除時間軸漂移，確保切點時間精準    │
│ 成果：杜絕因時間失真被 Stage 2 誤殺，解決「抓不到候選區間」 │
└──────────────────────────────┬──────────────────────────────┘
                               │ 精確切點資料 (CutPoints)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2：效能瓶頸突破（快取 + 數學預篩 + 安全早停）          │
│ 目的：保證「處理飛快」── 消除無效 I/O 解碼，加速 3~6 倍      │
│ 成果：依賴 Stage 1 正確時間，先驗剪枝才能極度精準不誤殺    │
└─────────────────────────────────────────────────────────────┘
```

* **方案 B 是 Stage 2 效能優化的先決基石**：若 Stage 1 回傳的時間本身就飄移，Stage 2 任何基於時長的先驗剪枝（Pre-filter）都會變成「垃圾進、垃圾出」，甚至誤殺真正廣告；Stage 1 校準後，Stage 2 的純數值先驗過濾才能發揮最大效益。
* **Stage 2 優化提供效能護城河**：即使放寬時長容許區間以包容特殊廣告，Stage 2 的快取與早停機制依然能保證極致的執行速度。

---

## 3. Stage 1 影格時間校正方案（方案 B 詳細設計）

### 3.1 根因分析：為何特定影片會發生嚴重漂移？
1. **奇數寬度（1077px）與 SPS Cropping**：
   H.264 標準通常以偶數或 16 的倍數對齊，1077 是奇數。在 YUV420p 色彩格式中，色彩平面寬度為 538.5。OpenCV 底層解碼器在循序解碼時，容易因記憶體 Stride 錯位或解碼邊界問題產生丟格（Drop Frame）或跳格。
2. **`curr_frame` 計數假設失效**：
   目前 Stage 1 在 `read_sampled_frames` 中自己維護 `curr_frame += 1`，並以此查表 `meta.time_of(frame_idx)`。一旦底層解碼過程丟失哪怕 1 格，後續全片的時間戳記便如同扣錯第一顆鈕扣般整排走鐘，越後面漂移越嚴重。
3. **B-frame 重排與時間原點位移**：
   MP4 容器的 Presentation Time（PTS）與解碼順序（DTS）不同，且部分影片片頭帶有 Edit List 或微小的 PTS Offset。單靠影格序號推算時間，極易脫鉤。

### 3.2 具體改造方案
在 `FrameExtractor.read_sampled_frames` 與 `stage1_scan.py` 中，**改以解碼器回報的實際毫秒時間戳記（PTS）作為時間基準**：

```python
# 修改 frame_extractor.py 產生器
def read_sampled_frames(self, step: int = 1, target_long_edge: int = 400):
    self.open()
    assert self._cap is not None
    # ...（解析度縮放設定維持不變）...

    curr_frame = 0
    while True:
        ret, frame = self._cap.read()
        if not ret or frame is None:
            break

        if curr_frame % step == 0:
            # 方案 B 核心：向解碼器直接取得當前畫面的實際時間（毫秒）
            pts_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            real_time = pts_ms / 1000.0 if pts_ms >= 0 else None

            if target_size is not None:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

            # 同時產出真實時間戳記，避免下游單純依賴 curr_frame 推算
            yield curr_frame, real_time, frame

        curr_frame += 1
```

在 `stage1_scan.py` 中：
```python
for frame_idx, real_time, frame in extractor.read_sampled_frames(step=step, target_long_edge=target_long_edge):
    # 若有解碼器回報之真實時間，優先採用；否則降級回 meta.time_of(frame_idx)
    t = real_time if real_time is not None and real_time > 0 else meta.time_of(frame_idx)
    cut_points.append(CutPoint(frame_index=frame_idx, timestamp=t, diff_score=diff))
```

### 3.3 零效能損耗驗證
* `cap.get(cv2.CAP_PROP_POS_MSEC)` 僅為 C++ 內部結構體的記憶體數值讀取，耗時在微秒等級（約 $0.000001$ 秒）。
* 僅在抽樣命中時讀取（每秒 3 次），1 小時影片全片調用約 10,800 次，**累計耗時小於 0.01 秒，對 Stage 1 總耗時完全零影響**。

---

## 4. Stage 2 效能突破與安全剪枝方案

### 4.1 三大效能優化手段

#### 步驟 1：前後正片特徵快取池（消除重複解碼）
同一切點的前後正片影格，在與不同對手切點配對時會被反覆使用。建立局部快取，保證全片同一個影格只解碼一次、特徵只計算一次：
```python
feat_cache: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}

def get_cached_feature(f_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
    if f_idx not in feat_cache:
        img = extractor.get_frame(f_idx, target_size=(160, 90))
        feat_cache[f_idx] = compute_frame_features(img) if img is not None else None
    return feat_cache[f_idx]
```

#### 步驟 2：純數值先驗過濾（Cheap Math Pre-filter）
在進行任何耗時的畫面解碼前，先利用已知的時長與切點強度進行篩選：
```python
score_dur = _duration_score(dur, cfg.target_duration, cfg.duration_sigma)
score_cut = 0.5 * (cp_in.diff_score + cp_out.diff_score)

# 【協同安全調優】：門檻採保守設定，避免誤殺非標準時長或低對比切點的廣告
if score_dur < 0.10 and score_cut < 0.40:
    continue
```

#### 步驟 3：內部穩定度採樣之「延遲加載與保守早停」
將一次性解碼 4 格改為逐格檢查：
```python
internal_samples = [
    int(s_frame + (e_frame - s_frame) * r) for r in (0.2, 0.4, 0.6, 0.8)
]

internal_feats = []
aborted = False
for f_idx in internal_samples:
    feat = get_cached_feature(f_idx)
    if feat is not None:
        if internal_feats:
            step_diff = compare_frames(internal_feats[-1], feat)
            # 單步差異劇烈（明顯為一般劇情鏡頭切換），立即放棄該候選
            # 容許字幕替換與短暫閃爍（設定為 1.4 倍門檻）
            if step_diff > cfg.max_internal_variance * 1.4:
                aborted = True
                break
        internal_feats.append(feat)

if aborted or len(internal_feats) < 2:
    continue
```

### 4.2 時長窗規範與容錯擴充
* **SRS 規格對齊**：SRS 規格書定義廣告時長窗為 **15～35 秒**，目前代碼中的 `PairConfig` 預設為 **18～32 秒**。
* **調整策略**：將 `min_duration` 調寬至 **15 秒**，`max_duration` 調寬至 **35 秒**。寬度增加帶來的候選數增加，由步驟 2 的數學預篩與步驟 3 早停無痛吸收，徹底解決邊緣時長廣告被直接淘汰的問題。

---

## 5. 整合處理流程全貌

```
[輸入影片檔 (如 video-22.mp4)]
       │
       ▼
【Stage 1 全片掃描】
 ├─ 抽樣解碼 (Step = 8)
 ├─ 方案 B 時間校正：讀取 CAP_PROP_POS_MSEC (零累計漂移)
 └─ 產出精確 CutPoints (timestamp 100% 貼合真實時間)
       │
       ▼
【Stage 2 廣告區間配對】
 ├─ 候選區間枚舉 (15s <= dur <= 35s)
 ├─ [第一道防線] 純數值先驗過濾 (時長分數 + 切點強度，零 I/O 剔除 30%~50%)
 ├─ [第二道防線] 命中特徵快取池 (讀取 f_before / f_after，無重複解碼)
 ├─ [第三道防線] 內部採樣 Early-Exit (劇烈變動鏡頭即刻中斷)
 └─ 綜合評分與 IoU 最佳化去重
       │
       ▼
【Stage 3 局部逐格精確化 & GUI 預覽】
```

---

## 6. 綜合效益預估

| 評估指標 | 現況（優化前） | 整合優化後（預期） | 效益總結 |
| :--- | :--- | :--- | :--- |
| **特殊編碼 / 奇數寬度影片偵測** | 嚴重漂移、抓不到候選區間 | **精確鎖定切點與廣告時間** | 徹底解決誤殺與漏抓痛點 |
| **Stage 1 額外運算開銷** | 基準線 | **+0.00 秒（微秒級，體感 0 增加）** | 極致輕量 |
| **Stage 2 隨機 Seek 解碼次數** | 300 ~ 600 次 | **60 ~ 120 次** | 減少 **70% ~ 80%** I/O 負擔 |
| **Stage 2 重複特徵計算** | 普遍存在 | **完全消除（0% 重複）** | 快取池全域複用 |
| **Stage 2 執行耗時** | 30 ~ 60 秒 | **5 ~ 10 秒** | **速度提升 3 ~ 6 倍** |
| **時長適應彈性** | 18s ~ 32s（容易誤切） | **15s ~ 35s（寬容且精準）** | 提高邊緣樣本召回率 |

---

## 7. 建議實施步驟

1. **第一步（落實 Stage 1 方案 B）**：
   - 修改 `src/video_ad_remover/core/frame_extractor.py` 的 `read_sampled_frames`。
   - 修改 `src/video_ad_remover/core/detector/stage1_scan.py` 接收實際時間。
   - 使用 `video-22.mp4` 驗證 Stage 1 切點時間是否精準對齊播放器畫面。
2. **第二步（落實 Stage 2 效能三部曲）**：
   - 在 `src/video_ad_remover/core/detector/stage2_pair.py` 中引入 `feat_cache`、保守的純數值 Pre-filter，以及內部穩定度 Early-Exit。
   - 將時長窗對齊 SRS（15s ~ 35s）。
3. **第三步（回歸測試）**：
   - 執行 `uv run pytest tests/` 確認既有測試通過。
   - 驗證原測試集與 `video-22.mp4`，確保辨識結果既準又快。
