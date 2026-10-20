"""診斷腳本：針對 Stage 1 / Stage 2 / Stage 3 廣告偵測各階段輸出診斷報表。

依據 ai-chat/Stage1漏抓問題_診斷與修正建議.md 規格：
1. FFprobe 建立的 meta.frame_count、meta.end_time 與 FPS / VFR 判定。
2. OpenCV 實際循序解碼的影格數，以及讀取失敗位置。
3. Stage 1 所有 cut point：frame_index、meta.time_of(frame_index)、diff_score。
4. Stage 2 候選數量與淘汰原因統計（時長窗、內部穩定度、解碼失敗、去重等）。
5. Stage 3 精確化前後的區間變化。

用法：
    uv run python scripts/diagnose_ad_detection.py
    uv run python scripts/diagnose_ad_detection.py --video video-22.mp4
    uv run python scripts/diagnose_ad_detection.py --video video-22.mp4 --output-report diag_report.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import cv2

# 加入專案路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from video_ad_remover.core.detector.pipeline import PipelineConfig
from video_ad_remover.core.detector.stage1_scan import (
    CutPoint,
    compare_frames,
    compute_frame_features,
    scan_cut_points,
)
from video_ad_remover.core.detector.stage2_pair import PairConfig, _duration_score
from video_ad_remover.core.detector.stage3_refiner import refine_candidate_boundaries
from video_ad_remover.core.frame_extractor import FrameExtractor
from video_ad_remover.core.interval_calc import dedupe_candidates
from video_ad_remover.core.prober import probe_video
from video_ad_remover.models.interval import (
    AdCandidate,
    DetectionSource,
    FrameInterval,
    IntervalStatus,
)
from video_ad_remover.models.video_meta import VideoMetadata
from video_ad_remover.utils.time_conv import format_timestamp


@dataclass
class CandidateElimination:
    in_idx: int
    out_idx: int
    in_frame: int
    out_frame: int
    in_time: float
    out_time: float
    duration: float
    reason: str
    detail: dict[str, float] | None = None


def run_diagnostics(
    video_path: Path,
    config: PipelineConfig | None = None,
    output_report: Path | None = None,
) -> str:
    cfg = config or PipelineConfig()
    lines: list[str] = []

    def log(msg: str = ""):
        print(msg)
        lines.append(msg)

    log("=" * 80)
    log(f" 影片廣告偵測診斷報表: {video_path.name}")
    log("=" * 80)

    # --------------------------------------------------------------------------
    # 1. FFprobe 視訊中繼資料與時間軸檢驗
    # --------------------------------------------------------------------------
    log("\n[1] FFprobe 視訊中繼資料 (Canonical Timeline)")
    t0 = time.perf_counter()
    meta = probe_video(video_path)
    t_probe = time.perf_counter() - t0

    log(f"  - 檔案路徑: {meta.path}")
    log(f"  - 檔案大小: {meta.file_size / (1024 * 1024):.2f} MB")
    log(f"  - 解析度: {meta.width}x{meta.height} (寬度奇偶性: {'偶數' if meta.width % 2 == 0 else '奇數!'})")
    log(f"  - 容器時長 (container duration): {meta.duration:.3f}s")
    log(f"  - 視訊涵蓋時長 (video duration): {meta.video_duration:.3f}s (起始: {meta.time_of(0):.3f}s, 結束: {meta.end_time:.3f}s)")
    log(f"  - FFprobe 影格總數 (meta.frame_count): {meta.frame_count}")
    log(f"  - 容器宣告 nb_frames: {meta.reported_frame_count}")
    log(f"  - 影格率模式: {meta.frame_mode} (is_vfr={meta.is_vfr})")
    log(f"  - 標稱 FPS: {meta.nominal_fps:.3f} (avg={meta.avg_frame_rate}, r={meta.r_frame_rate})")
    log(f"  - Time Base: {meta.time_base}")
    log(f"  - 視訊編碼格式: {meta.video_codec} ({meta.pix_fmt})")
    log(f"  - 音訊串流: {'有' if meta.has_audio else '無'}")
    log(f"  - FFprobe 解析耗時: {t_probe:.2f}s")

    # --------------------------------------------------------------------------
    # 2. OpenCV 實際循序解碼影格數與中斷位置檢驗
    # --------------------------------------------------------------------------
    log("\n[2] OpenCV 循序解碼一致性檢驗")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log("  [錯誤] OpenCV 無法開啟此影片！")
        return "\n".join(lines)

    cv_fps = cap.get(cv2.CAP_PROP_FPS)
    cv_reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cv_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log(f"  - OpenCV Header 資訊: {cv_width}x{cv_height}, FPS={cv_fps:.3f}, CAP_PROP_FRAME_COUNT={cv_reported_frames}")

    log("  - 正在測試 OpenCV 循序逐格解碼...")
    decoded_count = 0
    first_fail_idx = -1
    t_decode_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            first_fail_idx = decoded_count
            break
        decoded_count += 1
        if decoded_count % 1000 == 0:
            print(f"\r    已解碼 {decoded_count} 格...", end="", flush=True)

    print()
    cap.release()
    t_decode = time.perf_counter() - t_decode_start

    log(f"  - OpenCV 實際成功解碼總影格數: {decoded_count}")
    log(f"  - OpenCV 首次讀取中斷位置 (Frame Index): {first_fail_idx}")
    log(f"  - 解碼耗時: {t_decode:.2f}s (約 {decoded_count / max(0.001, t_decode):.1f} fps)")

    frame_diff = decoded_count - meta.frame_count
    if frame_diff == 0:
        log("  - 一致性結果: OpenCV 解碼影格數與 FFprobe PTS 總數完全吻合 (落差 0 格) [OK]")
    else:
        log(f"  - 一致性結果: [警告] OpenCV 解碼影格數與 FFprobe 落差 {frame_diff:+d} 格！ (OpenCV: {decoded_count}, FFprobe: {meta.frame_count})")

    # --------------------------------------------------------------------------
    # 3. Stage 1 切點偵測掃描與輸出
    # --------------------------------------------------------------------------
    log(f"\n[3] Stage 1 全片切點掃描 (sample_fps={cfg.sample_fps}, diff_threshold={cfg.diff_threshold})")
    t1_start = time.perf_counter()
    cuts = scan_cut_points(
        video_path,
        meta,
        sample_fps=cfg.sample_fps,
        diff_threshold=cfg.diff_threshold,
    )
    t1_elapsed = time.perf_counter() - t1_start

    log(f"  - Stage 1 耗時: {t1_elapsed:.2f}s")
    log(f"  - 偵測到切點總數: {len(cuts)}")
    if cuts:
        log("\n  Cut Points 列表:")
        log(f"  {'#':<4} {'Frame':<8} {'Timestamp (s)':<14} {'Formatted Time':<16} {'Diff Score':<10}")
        log("  " + "-" * 56)
        for idx, cp in enumerate(cuts):
            t_str = format_timestamp(cp.timestamp)
            log(f"  {idx + 1:<4} {cp.frame_index:<8} {cp.timestamp:<14.3f} {t_str:<16} {cp.diff_score:<10.3f}")
    else:
        log("  [警告] Stage 1 偵測到 0 個切點！所有後續階段將無法產生候選。")

    # --------------------------------------------------------------------------
    # 4. Stage 2 配對與候選淘汰細部追蹤
    # --------------------------------------------------------------------------
    log(f"\n[4] Stage 2 配對與淘汰細部追蹤 (PairConfig: min={cfg.pair_config.min_duration}s, max={cfg.pair_config.max_duration}s, target={cfg.pair_config.target_duration}s, max_var={cfg.pair_config.max_internal_variance})")
    pair_cfg = cfg.pair_config

    eliminations: list[CandidateElimination] = []
    surviving_raw: list[AdCandidate] = []

    if len(cuts) < 2:
        log(f"  - 切點數不足 2 個 ({len(cuts)})，無法進行 Stage 2 配對。")
    else:
        extractor = FrameExtractor(video_path, meta)
        fps = meta.nominal_fps if meta.nominal_fps > 0 else 30.0
        n = len(cuts)
        total_possible_pairs = n * (n - 1) // 2
        too_short_count = 0
        too_long_count = 0

        with extractor:
            for i in range(n):
                cp_in = cuts[i]
                for j in range(i + 1, n):
                    cp_out = cuts[j]
                    dur = cp_out.timestamp - cp_in.timestamp

                    if dur < pair_cfg.min_duration:
                        too_short_count += 1
                        continue
                    if dur > pair_cfg.max_duration:
                        too_long_count += 1
                        break  # 切點已照時間排序，後面只會更長

                    # 時長落在範圍內，進入特徵檢驗
                    score_dur = _duration_score(dur, pair_cfg.target_duration, pair_cfg.duration_sigma)
                    score_cut = 0.5 * (cp_in.diff_score + cp_out.diff_score)

                    # 前後正片採樣
                    f_before = max(0, int(cp_in.frame_index - fps))
                    f_after = min(meta.frame_count - 1, int(cp_out.frame_index + fps))
                    img_before = extractor.get_frame(f_before, target_size=(160, 90))
                    img_after = extractor.get_frame(f_after, target_size=(160, 90))

                    score_context = 0.5
                    feat_b = None
                    feat_a = None
                    if img_before is not None and img_after is not None:
                        feat_b = compute_frame_features(img_before)
                        feat_a = compute_frame_features(img_after)
                        diff_context = compare_frames(feat_b, feat_a)
                        score_context = 1.0 - diff_context

                    # 內部穩定度採樣
                    s_frame = cp_in.frame_index
                    e_frame = cp_out.frame_index
                    internal_samples = [
                        int(s_frame + (e_frame - s_frame) * r) for r in (0.2, 0.4, 0.6, 0.8)
                    ]
                    internal_feats = []
                    for f_idx in internal_samples:
                        img = extractor.get_frame(f_idx, target_size=(160, 90))
                        if img is not None:
                            internal_feats.append(compute_frame_features(img))

                    internal_diffs = []
                    for k in range(len(internal_feats) - 1):
                        internal_diffs.append(
                            compare_frames(internal_feats[k], internal_feats[k + 1])
                        )
                    avg_internal_diff = (
                        sum(internal_diffs) / len(internal_diffs) if internal_diffs else 0.5
                    )

                    score_contrast = 0.5
                    if internal_feats and feat_b and feat_a:
                        mid_feat = internal_feats[len(internal_feats) // 2]
                        contrast_b = compare_frames(mid_feat, feat_b)
                        contrast_a = compare_frames(mid_feat, feat_a)
                        score_contrast = 0.5 * (contrast_b + contrast_a)

                    if avg_internal_diff > pair_cfg.max_internal_variance:
                        eliminations.append(
                            CandidateElimination(
                                in_idx=i + 1,
                                out_idx=j + 1,
                                in_frame=s_frame,
                                out_frame=e_frame,
                                in_time=cp_in.timestamp,
                                out_time=cp_out.timestamp,
                                duration=dur,
                                reason=f"內部劇烈變動 (avg_diff={avg_internal_diff:.3f} > 門檻 {pair_cfg.max_internal_variance})",
                                detail={"avg_internal_diff": avg_internal_diff},
                            )
                        )
                        continue

                    score_stability = max(0.0, 1.0 - (avg_internal_diff / pair_cfg.max_internal_variance))
                    total_score = (
                        0.30 * score_dur
                        + 0.25 * score_stability
                        + 0.20 * score_cut
                        + 0.15 * score_context
                        + 0.10 * score_contrast
                    )

                    score_detail = {
                        "duration": round(score_dur, 3),
                        "stability": round(score_stability, 3),
                        "cut_strength": round(score_cut, 3),
                        "context": round(score_context, 3),
                        "contrast": round(score_contrast, 3),
                        "avg_internal_diff": round(avg_internal_diff, 3),
                    }

                    cand = AdCandidate(
                        interval=FrameInterval(s_frame, e_frame),
                        status=IntervalStatus.UNCONFIRMED,
                        source=DetectionSource.AUTO_SCAN,
                        score=round(total_score, 3),
                        score_detail=score_detail,
                    )
                    surviving_raw.append(cand)

        log(f"  - 切點配對總組合數: {total_possible_pairs}")
        log(f"  - 時長過短淘汰 (dur < {pair_cfg.min_duration}s): {too_short_count}")
        log(f"  - 時長過長略過 (dur > {pair_cfg.max_duration}s): {too_long_count}")
        log(f"  - 進入時長窗的配對數: {len(eliminations) + len(surviving_raw)}")
        log(f"  - 因內部畫面變動 (max_internal_variance) 遭淘汰數: {len(eliminations)}")
        if eliminations:
            for el in eliminations:
                log(f"    * 淘汰配對 #{el.in_idx}->#{el.out_idx}: [{format_timestamp(el.in_time)} ~ {format_timestamp(el.out_time)}, {el.duration:.2f}s] 原因: {el.reason}")

        log(f"  - 去重前通過評分之候選數: {len(surviving_raw)}")
        for idx, sc in enumerate(surviving_raw):
            t_s = format_timestamp(meta.time_of(sc.start_frame))
            t_e = format_timestamp(meta.time_of(sc.end_frame))
            dur = meta.segment_duration(sc.start_frame, sc.end_frame)
            log(f"    * 候選 {idx + 1}: [{t_s} ~ {t_e}, {dur:.2f}s] 分數: {sc.score} 詳細: {sc.score_detail}")

        # 去重
        stage2_final = dedupe_candidates(surviving_raw, iou_threshold=pair_cfg.iou_threshold)
        log(f"  - IoU 去重後最終 Stage 2 候選數: {len(stage2_final)} (過濾掉 {len(surviving_raw) - len(stage2_final)} 個重複區間)")

    # --------------------------------------------------------------------------
    # 5. Stage 3 精確化前後變化
    # --------------------------------------------------------------------------
    log(f"\n[5] Stage 3 局部邊界逐格精確化 (window_sec={cfg.refine_window_sec})")
    if not surviving_raw:
        log("  - 無 Stage 2 候選，略過 Stage 3。")
    else:
        stage3_refined = refine_candidate_boundaries(
            stage2_final,
            video_path,
            meta,
            search_window_sec=cfg.refine_window_sec,
        )
        for idx, (orig, ref) in enumerate(zip(stage2_final, stage3_refined)):
            orig_ts = meta.time_of(orig.start_frame)
            orig_te = meta.time_of(orig.end_frame)
            ref_ts = meta.time_of(ref.start_frame)
            ref_te = meta.time_of(ref.end_frame)

            d_sf = ref.start_frame - orig.start_frame
            d_ef = ref.end_frame - orig.end_frame
            d_st = ref_ts - orig_ts
            d_et = ref_te - orig_te

            log(f"  - 候選 #{idx + 1}:")
            log(f"    起點: Frame {orig.start_frame} -> {ref.start_frame} ({d_sf:+d} 格, {d_st:+.3f}s)")
            log(f"    終點: Frame {orig.end_frame} -> {ref.end_frame} ({d_ef:+d} 格, {d_et:+.3f}s)")
            log(f"    精確化後區間: [{format_timestamp(ref_ts)} ~ {format_timestamp(ref_te)}] ({ref_te - ref_ts:.3f}s)")

    log("\n" + "=" * 80)
    log(" 診斷總結歸因")
    log("=" * 80)
    if not cuts:
        log("=> [根因分析] 漏抓發生在 Stage 1：完全沒有產出切點。請調整取樣頻率 (sample_fps) 或降低 diff_threshold。")
    elif not surviving_raw:
        if too_short_count + too_long_count == total_possible_pairs:
            log("=> [根因分析] 漏抓發生在 Stage 2 時長窗：所有切點配對的時長均不在 [min_duration, max_duration] 範圍內。")
        elif eliminations:
            log("=> [根因分析] 漏抓發生在 Stage 2 評分淘汰：有切點落入時長窗，但因 max_internal_variance 等內部特徵檢查被剔除。")
        else:
            log("=> [根因分析] 漏抓發生在 Stage 2 配對階段。")
    else:
        log("=> [診斷正常] 成功抓出廣告候選區間！")

    report_text = "\n".join(lines)
    if output_report:
        output_report.write_text(report_text, encoding="utf-8")
        print(f"\n[報告已存檔] {output_report.resolve()}")

    return report_text


def main():
    parser = argparse.ArgumentParser(description="影片去廣告偵測各階段診斷工具")
    parser.add_argument("--video", type=str, default="video-22.mp4", help="要診斷的影片路徑 (預設: video-22.mp4)")
    parser.add_argument("--sample-fps", type=float, default=3.0, help="Stage 1 取樣頻率 (預設: 3.0)")
    parser.add_argument("--diff-threshold", type=float, default=0.40, help="Stage 1 切點門檻 (預設: 0.40)")
    parser.add_argument("--output-report", type=str, default="", help="輸出報表檔名 (可選)")
    args = parser.parse_args()

    v_path = Path(args.video)
    if not v_path.is_absolute():
        v_path = project_root / v_path

    if not v_path.exists():
        print(f"[錯誤] 找不到影片檔案: {v_path}")
        sys.exit(1)

    cfg = PipelineConfig(
        sample_fps=args.sample_fps,
        diff_threshold=args.diff_threshold,
    )

    out_file = Path(args.output_report) if args.output_report else None
    run_diagnostics(v_path, config=cfg, output_report=out_file)


if __name__ == "__main__":
    main()
