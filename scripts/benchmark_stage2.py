"""Stage 2 效能基準測試與真值比對腳本 (Step 1 Baseline Tooling)。

用途：
1. 針對指定影片（預設 video-22.mp4）測量 Stage 2 的真實效能指標（Wall-clock time, Seek 次數, Decode 耗時, 特徵計算耗時）。
2. 可將候選區間與分數固化存檔為 Baseline JSON。
3. 支援後續優化步驟（Step 2 快取、Step 6 早停等）與 Baseline 進行自動比對（驗證 candidates 100% 一致與加速幅度）。

用法：
    # 執行基準測試並輸出指標
    uv run python scripts/benchmark_stage2.py

    # 執行基準測試並將結果儲存為 baseline
    uv run python scripts/benchmark_stage2.py --save-baseline tests/baseline_stage2_video22.json

    # 與既有 baseline 比對一致性與加速比
    uv run python scripts/benchmark_stage2.py --compare tests/baseline_stage2_video22.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

# 加入專案路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from video_ad_remover.core.detector.pipeline import PipelineConfig
from video_ad_remover.core.detector.stage1_scan import scan_cut_points
from video_ad_remover.core.detector.stage2_pair import PairConfig, Stage2Stats, pair_and_score
from video_ad_remover.core.prober import probe_video
from video_ad_remover.utils.time_conv import format_timestamp


def run_benchmark(
    video_path: Path,
    config: PipelineConfig | None = None,
    save_baseline: Path | None = None,
    compare_baseline: Path | None = None,
) -> int:
    cfg = config or PipelineConfig()
    if not video_path.exists():
        print(f"[錯誤] 找不到測試影片: {video_path}")
        return 1

    print("=" * 65)
    print(f"🎬 Stage 2 效能基準測試: {video_path.name}")
    print("=" * 65)

    meta = probe_video(video_path)
    print(f"  - 影片時長: {meta.duration:.2f}s, 總影格數: {meta.frame_count}, FPS: {meta.nominal_fps:.2f}")

    # Stage 1: 掃描切點（或快取切點）
    cuts = None
    cuts_cache_file = project_root / "tests" / f".cuts_cache_{video_path.stem}.json"
    if cuts_cache_file.exists() and cuts_cache_file.stat().st_size > 0:
        try:
            print(f"  - 載入既有 Stage 1 切點快取: {cuts_cache_file.name}")
            with open(cuts_cache_file, encoding="utf-8") as f:
                raw_cuts = json.load(f)
            from video_ad_remover.core.detector.stage1_scan import CutPoint
            cuts = [CutPoint(**c) for c in raw_cuts]
        except Exception:
            cuts = None

    if cuts is None:
        print(f"  - 執行 Stage 1 快速切點掃描 (sample_fps={cfg.sample_fps})...")
        t1_start = time.perf_counter()
        cuts = scan_cut_points(
            video_path,
            meta,
            sample_fps=cfg.sample_fps,
            diff_threshold=cfg.diff_threshold,
        )
        t1_elapsed = time.perf_counter() - t1_start
        print(f"    Stage 1 完成: 耗時 {t1_elapsed:.2f}s, 偵測到 {len(cuts)} 個切點")
        # 寫入快取加速未來測試
        try:
            with open(cuts_cache_file, "w", encoding="utf-8") as f:
                json.dump([
                    {"frame_index": c.frame_index, "timestamp": c.timestamp, "diff_score": c.diff_score}
                    for c in cuts
                ], f, indent=2)
        except Exception:
            pass

    print(f"  - Stage 1 切點總數: {len(cuts)}")
    if len(cuts) < 2:
        print("  [警告] 切點數不足，無法執行 Stage 2 配對！")
        return 1

    # Stage 2: 執行配對並搜集 Profiling 指標
    stats = Stage2Stats()
    print("\n⏱️ 正在執行 Stage 2 配對與特徵分析...")
    candidates = pair_and_score(cuts, video_path, meta, config=cfg.pair_config, stats=stats)

    print("\n" + "-" * 65)
    print("📊 Stage 2 Profiling 指標統計")
    print("-" * 65)
    print(f"  - 總配對檢查數 (Total Checked)    : {stats.total_pairs_checked}")
    print(f"  - 時長窗直接篩除 (Duration Filter): {stats.duration_filtered}")
    print(f"  - 實際進入特徵評估配對數 (Evaluated): {stats.evaluated_pairs}")
    print(f"  - 不可逆超標早停配對數 (Early Stopped): {stats.early_stopped_pairs}")
    print(f"  - 影像解碼次數 (get_frame Calls)  : {stats.get_frame_calls}")
    if stats.cache_hits + stats.cache_misses > 0:
        total_cache = stats.cache_hits + stats.cache_misses
        hit_rate = stats.cache_hits / total_cache * 100
        print(f"  - 特徵快取命中率 (Cache Hit Rate): {hit_rate:.1f}% ({stats.cache_hits}/{total_cache})")
    print(f"  - 影像解碼累計耗時 (Decode Time)  : {stats.decode_time:.3f}s")
    if stats.get_frame_calls > 0:
        print(f"    (平均每次 get_frame 耗時       : {stats.decode_time / stats.get_frame_calls * 1000:.1f}ms)")
    print(f"  - 特徵計算累計耗時 (Compute Time) : {stats.feature_compute_time:.3f}s")
    print(f"  - Stage 2 總執行時間 (Wall Time)  : {stats.wall_time:.3f}s")
    print(f"  - 最終去重後存活候選數 (Surviving): {stats.surviving_candidates}")

    # 候選清單
    print("\n🎯 候選區間清單 (Top Candidates):")
    cand_records = []
    for idx, c in enumerate(candidates):
        t_start = meta.time_of(c.start_frame)
        t_end = meta.time_of(c.end_frame)
        dur = t_end - t_start
        print(f"  [{idx+1}] Frame [{c.start_frame:5d} - {c.end_frame:5d}] | {format_timestamp(t_start)} - {format_timestamp(t_end)} ({dur:5.1f}s) | Score: {c.score:.3f}")
        cand_records.append({
            "start_frame": c.start_frame,
            "end_frame": c.end_frame,
            "start_time": round(t_start, 3),
            "end_time": round(t_end, 3),
            "duration": round(dur, 3),
            "score": c.score,
            "score_detail": c.score_detail,
        })

    snapshot_data = {
        "video": video_path.name,
        "cut_points_count": len(cuts),
        "stats": {
            "total_pairs_checked": stats.total_pairs_checked,
            "duration_filtered": stats.duration_filtered,
            "evaluated_pairs": stats.evaluated_pairs,
            "early_stopped_pairs": stats.early_stopped_pairs,
            "get_frame_calls": stats.get_frame_calls,
            "decode_time": round(stats.decode_time, 4),
            "feature_compute_time": round(stats.feature_compute_time, 4),
            "wall_time": round(stats.wall_time, 4),
            "surviving_candidates": stats.surviving_candidates,
        },
        "candidates": cand_records,
    }

    # 存檔 Baseline
    if save_baseline:
        save_baseline.parent.mkdir(parents=True, exist_ok=True)
        with open(save_baseline, "w", encoding="utf-8") as f:
            json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 已成功將 Baseline 快照儲存至: {save_baseline}")

    # 比對 Baseline
    if compare_baseline:
        if not compare_baseline.exists():
            print(f"\n[錯誤] 找不到欲比對的 Baseline 檔案: {compare_baseline}")
            return 1

        with open(compare_baseline, encoding="utf-8") as f:
            base_data = json.load(f)

        print("\n" + "=" * 65)
        print("🔍 與 Baseline 快照比對結果")
        print("=" * 65)

        base_stats = base_data["stats"]
        curr_stats = snapshot_data["stats"]

        # 1. 候選名單與分數完全一致性檢驗
        base_cands = base_data["candidates"]
        curr_cands = snapshot_data["candidates"]

        match_ok = True
        if len(base_cands) != len(curr_cands):
            match_ok = False
            print(f"❌ 候選數量不一致: Baseline={len(base_cands)}, Current={len(curr_cands)}")
        else:
            for i, (b, c) in enumerate(zip(base_cands, curr_cands)):
                if b["start_frame"] != c["start_frame"] or b["end_frame"] != c["end_frame"]:
                    match_ok = False
                    print(f"❌ 候選 #{i+1} 區間不吻合: Base={b['start_frame']}-{b['end_frame']} vs Curr={c['start_frame']}-{c['end_frame']}")
                if abs(b["score"] - c["score"]) > 1e-4:
                    match_ok = False
                    print(f"❌ 候選 #{i+1} 分數不吻合: Base={b['score']} vs Curr={c['score']}")

        if match_ok:
            print("✅ 候選區間與各項評分: 100% Byte-for-byte 完全吻合！")

        # 2. 效能對比
        base_calls = base_stats["get_frame_calls"]
        curr_calls = curr_stats["get_frame_calls"]
        call_diff = curr_calls - base_calls
        call_pct = (call_diff / max(1, base_calls)) * 100

        base_wall = base_stats["wall_time"]
        curr_wall = curr_stats["wall_time"]
        speedup = base_wall / max(0.0001, curr_wall)

        print("\n📈 效能變化對比:")
        print(f"  - 解碼次數 (get_frame) : {base_calls} -> {curr_calls} ({call_diff:+d}, {call_pct:+.1f}%)")
        print(f"  - 解碼耗時 (Decode)    : {base_stats['decode_time']:.3f}s -> {curr_stats['decode_time']:.3f}s")
        print(f"  - 總執行耗時 (Wall)    : {base_wall:.3f}s -> {curr_wall:.3f}s (加速比: {speedup:.2f}x)")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Stage 2 基準測試與回歸比對")
    parser.add_argument(
        "--video",
        type=Path,
        default=project_root / "video-22.mp4",
        help="測試影片路徑 (預設: video-22.mp4)",
    )
    parser.add_argument(
        "--save-baseline",
        type=Path,
        help="儲存本次執行為基準快照 JSON",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="與指定的基準快照 JSON 比對",
    )
    args = parser.parse_args()
    sys.exit(run_benchmark(args.video, save_baseline=args.save_baseline, compare_baseline=args.compare))


if __name__ == "__main__":
    main()
