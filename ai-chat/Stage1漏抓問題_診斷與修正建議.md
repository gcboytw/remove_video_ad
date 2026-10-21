# Stage 1 漏抓問題：診斷與修正建議

## 結論與原則

建議不要直接以 OpenCV 的 `CAP_PROP_POS_MSEC` 取代現有時間計算。專案已經透過 FFprobe 建立逐格 PTS 索引，`VideoMetadata.time_of(frame_index)` 應保持為唯一的權威時間軸。

正確的推進順序是：**先定位漏抓發生在哪一個階段，再修正時間與影格的一致性，接著提高 Stage 1 召回率，最後才進行 Stage 2 的效能剪枝。**

---

## 1. 先建立診斷報表，不要預設根因是時間漂移

對 `video-22.mp4` 及其他可重現樣本，應輸出以下資訊：

- FFprobe 建立的 `meta.frame_count`、`meta.end_time` 與 FPS / VFR 判定。
- OpenCV 實際循序解碼的影格數，以及讀取失敗位置。
- Stage 1 所有 cut point：`frame_index`、`meta.time_of(frame_index)`、`diff_score`。
- Stage 2 的候選數量與每個候選被淘汰的原因：時長窗、pre-filter、解碼失敗、內部穩定度、去重等。
- Stage 3 精確化前後的區間變化。

這可區分下列情況：

1. Stage 1 沒有抓到廣告進／出切點：應調整取樣策略或切點偵測。
2. Stage 1 有抓到切點，但 Stage 2 的時長窗排除：應調整候選配對容忍度。
3. Stage 2 產生候選後被評分規則淘汰：應調整特徵、穩定度或 context 的規則。
4. Stage 3 把正確候選校正到錯誤位置：應檢查局部搜尋與 frame mapping。

未完成這個診斷前，不應將「抓不到候選」直接歸因於 OpenCV PTS 漂移。

---

## 2. 統一時間軸：FFprobe PTS 是 canonical source

`VideoMetadata` 已保存 FFprobe 讀出的呈現順序 PTS 索引；`meta.time_of(frame_index)` 與 `meta.frame_at_time(seconds)` 應是偵測管線所有時間／影格轉換的唯一依據。

`CutPoint` 必須維持以下不變式：

```python
cut.timestamp == meta.time_of(cut.frame_index)
```

不要把 OpenCV 的 `CAP_PROP_POS_MSEC` 和自行遞增的 `curr_frame` 混用。若 OpenCV 解碼時真的出現讀取影格數與 FFprobe index 不一致，會出現「timestamp 是正確秒數、frame_index 卻指向另一張畫面」的情形，Stage 2 雖可能算出正確時長，卻會抽到錯誤的 context 或內部樣本。

若診斷確認 OpenCV 解碼序列無法對應 PTS index，應改採能讀取每張 decoded frame 原始 PTS 的解碼路徑（例如 PyAV 或 FFmpeg），再透過下列轉換取得專案統一的影格索引：

```python
frame_index = meta.frame_at_time(decoded_pts)
```

此變更必須連同「解碼影格數、PTS 單調性與 index 對應」測試一起導入；不能只替換 Stage 1 的 timestamp 欄位。

---

## 3. 先修正 Stage 2 的 VFR 安全取樣

目前 Stage 2 使用 `frame_index ± nominal_fps` 代表前後一秒，並以影格 index 線性插值產生內部樣本。這對 VFR、非標準 FPS、PTS 不均勻來源均不精確。

應改為先在時間軸取樣，再映射回 frame index：

```python
before_idx = meta.frame_at_time(max(0.0, cp_in.timestamp - 1.0))
after_idx = meta.frame_at_time(
    min(meta.end_time, cp_out.timestamp + 1.0)
)

sample_times = [
    cp_in.timestamp + (cp_out.timestamp - cp_in.timestamp) * ratio
    for ratio in (0.2, 0.4, 0.6, 0.8)
]
internal_samples = [meta.frame_at_time(t) for t in sample_times]
```

這項調整不依賴 OpenCV 的 `POS_MSEC`，且能讓 Stage 2 的 context 與穩定度特徵始終對應正確的實際播放時間。

---

## 4. 提高 Stage 1 召回率，而不是盲目降低全域門檻

固定 3 fps 抽樣和單一 `diff_threshold` 可能漏掉淡入淡出、低對比切換或取樣點剛好落在轉換後穩定畫面的廣告邊界。建議採取分層策略：

1. 維持 3 fps 全片掃描，控制長片成本。
2. 對接近門檻、差異持續上升，或 Stage 2 可能配對成功的時間區域，局部提高至 8～12 fps。
3. 對相鄰的切點使用 non-maximum suppression（NMS）合併，避免加密掃描造成候選爆量。
4. 讓 `diff_threshold` 成為設定值；必要時用局部差異分布的 median / MAD 建立自適應門檻。
5. 保留「弱但可信」切點給 Stage 2 排序，而非在 Stage 1 過早淘汰。

低門檻本身不是解法；若沒有 NMS、候選上限與 Stage 2 預篩，切點數增加會讓配對成本快速上升。

---

## 5. 時長窗須容納 Stage 1 的抽樣誤差

3 fps 掃描的單端切點誤差可接近約 0.33 秒，廣告起訖兩端的誤差可能疊加。若直接以 SRS 的 15～35 秒作為硬門檻，剛好落在邊界的廣告可能被排除。

建議分為兩層：

- **候選枚舉窗**：例如 14.25～35.75 秒，容納取樣與時間映射誤差。
- **評分偏好窗**：仍以 15～35 秒為 SRS 規格，25 秒為最高分；候選窗外緣僅降低分數，不立刻淘汰。

這比直接擴大到 12～38 秒更可控，也降低正片切換組合大量增加的風險。

注意：若 `target_duration=25`、`duration_sigma=3.5`，15 秒或 35 秒的高斯時長分數約為 0.017。若要支援完整 SRS 時長窗，必須重新校準 `duration_sigma`、評分權重與 pre-filter，不能只修改 `min_duration` / `max_duration`。

---

## 6. Stage 2 優化應在正確性確認後逐步導入

建議實作順序：

1. 加入 Stage 1～3 診斷報表與 `video-22.mp4` 回歸測試。
2. 保留 FFprobe PTS 為 canonical source，修正 Stage 2 的時間取樣與 `frame_at_time` 映射。
3. 實作 Stage 2 的 `frame_index -> feature | None` 快取，並明確處理解碼失敗／樣本不足。
4. 導入保守且可設定的 duration / cut-strength pre-filter。
5. 彙整候選影格、去重、時間排序與分群，改用連續解碼減少 random seek。
6. 最後才啟用 Early-Exit；採「連續兩次高差異」或「最終平均必然超標」的條件，不能單次超標即淘汰。

---

## 7. 影像尺寸與降噪的建議

奇數寬度不應在沒有量測證據下被視為解碼漂移根因。OpenCV 可處理奇數尺寸；將縮圖尺寸轉為偶數可作為輸出尺寸一致化，但不會修復既有解碼錯位，且需避免不必要的長寬比變形。

Stage 2 改為依原始比例計算縮圖尺寸是合理的：

```python
def calc_target_size(width: int, height: int, long_edge: int = 160) -> tuple[int, int]:
    scale = min(1.0, long_edge / max(width, height))
    target_w = max(2, int(width * scale) // 2 * 2)
    target_h = max(2, int(height * scale) // 2 * 2)
    return target_w, target_h
```

高斯模糊應為可設定、可 A/B 驗證的選項。它可降低壓縮噪聲，但也可能降低真實切點、字幕與圖卡細節的訊號；啟用後需要重新校準 Stage 1 的 `diff_threshold` 與 Stage 2 的穩定度門檻。

---

## 8. 驗收標準與測試案例

除既有合成測試外，新增以下測試與檢查：

- `video-22.mp4` 的解碼影格數與 FFprobe PTS index 長度比對。
- sampled frame 時間戳記單調遞增，且每個 cut point 滿足 `timestamp == meta.time_of(frame_index)`。
- 已知廣告進出點附近存在 Stage 1 cut point；若無，診斷報表能說明其 `diff_score` 與門檻差距。
- Stage 2 每個淘汰理由均可被統計，且真值候選不會被 pre-filter 或 Early-Exit 錯誤排除。
- CFR、VFR、直式／超寬比例、低碼率、字幕／局部動畫影片皆有回歸案例。
- 每次優化均比較候選召回率、真值候選排名、解碼呼叫數與總耗時；「零效能損耗」與「3～6 倍加速」只能作為待驗證目標，不能視為保證。
