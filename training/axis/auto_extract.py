"""高手 / UP 视频自动轴提取入口。

同一遍视频扫描同时保留三条互不污染的数据线：
- local signature（±1 analysis sample）只用于 visual clustering / recurrence；
- semantic signature（-60ms → +40..220ms）只用于 telemetry prototype 分类；
- swap 先找 party HUD transition，再以 recurring outgoing cluster + local gap + 三槽 HUD
  change profile 跨 loop 对齐；40s rotation phase 只做粗诊断，不再承担 strict timing。

语义证据缺失或标签退化时仍保存 analysis/review/loops/swaps，但 auto_axis_timeline
fail-closed 写空。swap_windows 只描述“相对局部视觉锚点的成功切人窗口”，不声称恢复
UP 主隐藏的物理按键时刻。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import auto_extract_core as core
from auto_events import (
    local_event_signatures,
    semantic_event_signatures,
    semantic_timeline_guard,
)
from auto_swap_anchors import build_swap_payloads


def _json_dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_timeline(
    path: Path,
    video: Path,
    sampled: core.SampledVideo,
    events: list[dict],
    start_slot: int,
    semantic_threshold: float,
    *,
    blocked_reason: str | None = None,
) -> int:
    note = (
        f"auto_extract timeline blocked: {blocked_reason}"
        if blocked_reason
        else "auto_extract high-confidence semantic events; verify review/swaps before compiling"
    )
    timeline = core.Timeline(
        video=video.name,
        fps=sampled.source_fps,
        start_slot=start_slot,
        source_note=note,
    )
    if blocked_reason:
        timeline.save(path)
        return 0

    current_slot = start_slot
    count = 0
    for row in sorted(events, key=lambda item: int(item["ms"])):
        token = row.get("semantic")
        confidence = float(row.get("semantic_confidence", 0.0))
        if token not in core.SEMANTIC_TO_TOKEN.values() or confidence < semantic_threshold:
            continue
        timeline.events.append(core.AxisVideoEvent(
            frame=int(row["frame"]),
            ms=int(row["ms"]),
            action=str(token),
            slot=current_slot,
            note=f"auto conf={confidence:.2f} cluster={row['cluster']}",
        ))
        count += 1
        if token in core.SWITCH_TOKENS:
            current_slot = core.SWITCH_TOKENS[str(token)]
    timeline.save(path)
    return count


def _build_review(events: list[dict], semantic_threshold: float, max_review: int) -> list[dict]:
    review_items = []
    for event in events:
        reasons = []
        priority = 0.0
        if event["swap_candidate"] and not str(event.get("semantic") or "").startswith("s"):
            reasons.append("swap target unresolved")
            priority = max(priority, 1.0)
        if int(event["recurrence_support"]) >= 2 and not event.get("semantic"):
            reasons.append("recurring event semantic unknown")
            priority = max(priority, 0.85)
        if (
            event.get("semantic")
            and float(event.get("semantic_confidence", 0.0)) < semantic_threshold
        ):
            reasons.append("semantic confidence below timeline threshold")
            priority = max(priority, 0.70)
        if reasons:
            review_items.append({
                "start_ms": max(0, int(event["ms"]) - 280),
                "end_ms": int(event["ms"]) + 320,
                "priority": priority,
                "reasons": reasons,
                "event_ms": int(event["ms"]),
            })
    return core._merge_review_windows(
        review_items, max_items=max(1, int(max_review))
    )


def _print_swap_matching_diagnostics(swaps_payload: dict) -> None:
    diagnostics = swaps_payload.get("matching_diagnostics") or {}
    if not diagnostics:
        return

    if diagnostics.get("matching_basis", "").startswith("recurring-outgoing-cluster"):
        print(
            "  local-anchor matching: "
            f"anchored={diagnostics.get('anchored_candidate_count', 0)}/"
            f"{diagnostics.get('raw_candidate_count', 0)} "
            f"anchors={diagnostics.get('anchor_cluster_count', 0)} "
            f"anchors>=support={diagnostics.get('anchor_clusters_with_required_loop_support', 0)}"
        )
        print(
            "  cross-loop local pairs: "
            f"same-anchor={diagnostics.get('same_anchor_pairs', 0)} "
            f"gap-near={diagnostics.get('local_gap_near_pairs', 0)} "
            f"slot-profile-compatible={diagnostics.get('slot_profile_compatible_pairs', 0)} "
            f"both={diagnostics.get('local_gap_and_profile_pairs', 0)}"
        )
        print(
            f"  local gap gate≈{float(diagnostics.get('local_gap_tolerance_ms', 0.0)):.1f}ms; "
            f"slot profile sim>={float(diagnostics.get('slot_profile_similarity_min', 0.0)):.2f}; "
            f"coarse rotation gate={float(diagnostics.get('coarse_rotation_phase_tolerance', 0.0)):.3f}"
        )
        gap_fail = diagnostics.get("failed_same_anchor_gap_distance_ms")
        if gap_fail:
            print(
                "  failed same-anchor gap drift p10/median/p90≈"
                f"{gap_fail['p10']:.1f}/{gap_fail['median']:.1f}/{gap_fail['p90']:.1f}ms"
            )
        print(f"  diagnostic hint={diagnostics.get('hint')}")
        return

    print(
        "  swap matching: "
        f"cross-loop pairs={diagnostics.get('cross_loop_pairs', 0)} "
        f"phase-near={diagnostics.get('phase_near_pairs', 0)} "
        f"identity-compatible={diagnostics.get('identity_compatible_pairs', 0)} "
        f"both={diagnostics.get('phase_and_identity_pairs', 0)}"
    )
    tolerance_ms = diagnostics.get("phase_tolerance_ms")
    drift = (
        diagnostics.get("identity_compatible_phase_distance", {})
        .get("nearest_other_loop_ms")
    )
    if tolerance_ms is not None:
        if drift:
            print(
                f"  phase gate≈{float(tolerance_ms):.1f}ms; nearest identity-compatible drift "
                f"p10/median/p90≈{drift['p10']:.1f}/{drift['median']:.1f}/{drift['p90']:.1f}ms"
            )
        else:
            print(
                f"  phase gate≈{float(tolerance_ms):.1f}ms; "
                "no identity-compatible cross-loop neighbor found"
            )
    failures = diagnostics.get("phase_near_identity_gate_failures") or {}
    if diagnostics.get("phase_near_pairs", 0):
        print(
            "  phase-near HUD gate failures: "
            f"signature={failures.get('signature', 0)} "
            f"pre={failures.get('pre_state', 0)} "
            f"post={failures.get('post_state', 0)} "
            f"any={failures.get('any_identity_gate', 0)}"
        )
    print(f"  diagnostic hint={diagnostics.get('hint')}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatically extract recurring axis structure and swap windows from an UP video"
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument(
        "--analysis-fps",
        type=float,
        default=0.0,
        help="0=use source FPS up to 60; event timing keeps original frame granularity",
    )
    parser.add_argument("--start-slot", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--min-loop-s", type=float, default=8.0)
    parser.add_argument("--max-loop-s", type=float, default=45.0)
    parser.add_argument(
        "--prototype-root",
        action="append",
        default=None,
        help="training_data/motion root; may be repeated",
    )
    parser.add_argument("--no-self-prototypes", action="store_true")
    parser.add_argument("--semantic-threshold", type=float, default=0.66)
    parser.add_argument("--max-review", type=int, default=18)
    parser.add_argument(
        "--swap-phase-tolerance",
        type=float,
        default=0.04,
        help="coarse whole-rotation guard only; strict swap timing uses local recurring anchors",
    )
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    prefix = (
        Path(args.out_prefix).expanduser().resolve()
        if args.out_prefix
        else video.with_suffix("")
    )

    print(f"[1/6] scan video: {video}")
    sampled = core._sample_video(video, args.analysis_fps)
    print(
        f"  source={sampled.source_fps:.3f}fps analysis={sampled.analysis_fps:.3f}fps "
        f"samples={len(sampled.frame_indexes)}"
    )

    body_delta = core._delta_norms(sampled.body)
    party_delta = core._delta_norms(sampled.party)
    ability_delta = core._delta_norms(sampled.ability)
    raw_activity = 0.48 * body_delta + 0.22 * party_delta + 0.30 * ability_delta
    smooth_radius = max(1, int(round(sampled.analysis_fps * 0.025)))
    activity = core.smooth_signal(raw_activity, radius=smooth_radius)
    activity_z = core.robust_zscore(activity)
    party_z = core.robust_zscore(party_delta)

    print("[2/6] discover repeated rotation + recurring visual events")
    period, boundary_rows, _loop_stride, loop_fps = core._loop_detection(
        sampled, args.min_loop_s, args.max_loop_s
    )
    boundaries = [int(index) for index, _ in boundary_rows]
    if period and len(boundaries) >= 3:
        print(
            f"  loop≈{period.lag / loop_fps:.3f}s score={period.score:.3f} "
            f"boundaries={len(boundaries)}"
        )
        analysis_left, analysis_right = boundaries[0], boundaries[-1]
    else:
        print("  no stable repeated team rotation found; keep structural event extraction only")
        analysis_left, analysis_right = 0, len(sampled.body) - 1

    min_distance = max(2, int(round(sampled.analysis_fps * 0.10)))
    peak_indexes = core.detect_peaks(
        activity,
        min_distance=min_distance,
        threshold_z=1.9,
        max_peaks=max(32, int((sampled.times_ms[-1] / 1000.0) * 5.0)),
    )
    peak_indexes = [
        index for index in peak_indexes if analysis_left <= index <= analysis_right
    ]

    visual_signatures = local_event_signatures(
        sampled.body, sampled.party, sampled.ability, peak_indexes
    )
    clusters = core.cluster_signatures(
        visual_signatures, similarity_threshold=0.72
    )
    recurrence = core.loop_support(
        np.asarray(peak_indexes, dtype=np.int32),
        clusters.labels,
        boundaries,
        phase_tolerance=0.06,
    )
    print(
        f"  event candidates={len(peak_indexes)} "
        f"visual clusters={len(clusters.counts)}"
    )
    print(
        "  signature scopes: visual=local ±1 sample; "
        "semantic=-60ms -> +40..220ms; "
        "swap=recurring local anchor + 3-slot HUD change profile"
    )

    print("[3/6] build optional self-telemetry semantic prototypes")
    if args.no_self_prototypes:
        bank = core.PrototypeBank(samples={}, source_files=0, source_events=0)
        roots = []
    else:
        roots = core._prototype_roots(args.prototype_root)
        bank = core._build_prototype_bank(roots)
    print(
        f"  prototype roots={len(roots)} files={bank.source_files} "
        f"events={bank.source_events} counts={bank.counts()}"
    )
    if bank.samples and len(bank.samples) < 2:
        only = ", ".join(sorted(bank.samples))
        print(
            f"  警告：原型库只有 1 个语义类别（{only}），无法做区分性分类；"
            "本轮 semantic 标注自动停用，但 video-only swap 检测继续运行。"
        )

    semantic_signatures = semantic_event_signatures(
        sampled.body,
        sampled.party,
        sampled.ability,
        peak_indexes,
        sampled.analysis_fps,
    )

    events: list[dict] = []
    for pos, sample_index in enumerate(peak_indexes):
        token, semantic_conf, scores = core._classify_signature(
            semantic_signatures[pos], bank
        )
        swap_candidate = bool(
            party_z[sample_index] >= 2.0
            and party_delta[sample_index] >= ability_delta[sample_index] * 1.08
        )
        events.append({
            "sample_index": int(sample_index),
            "frame": int(sampled.frame_indexes[sample_index]),
            "ms": int(sampled.times_ms[sample_index]),
            "cluster": int(clusters.labels[pos]),
            "cluster_size": int(clusters.counts[int(clusters.labels[pos])]),
            "activity_z": round(float(activity_z[sample_index]), 4),
            "body_delta": round(float(body_delta[sample_index]), 5),
            "party_delta": round(float(party_delta[sample_index]), 5),
            "ability_delta": round(float(ability_delta[sample_index]), 5),
            "recurrence_support": int(recurrence[pos]),
            "swap_candidate": swap_candidate,
            "semantic": token,
            "semantic_confidence": round(float(semantic_conf), 4),
            "semantic_source": "self-telemetry-prototype" if token else None,
            "prototype_scores": {
                key: round(value, 4) for key, value in scores.items()
            },
        })

    core._propagate_cluster_semantics(events)
    for event in events:
        event["confidence"] = round(
            core.confidence_from_signals(
                float(event["activity_z"]),
                int(event["recurrence_support"]),
                float(event.get("semantic_confidence") or 0.0),
            ),
            4,
        )

    labeled_tokens = [
        str(event["semantic"]) for event in events if event.get("semantic")
    ]
    timeline_blocked_reason = semantic_timeline_guard(labeled_tokens)
    if timeline_blocked_reason:
        print(f"  警告：{timeline_blocked_reason}")

    template = core._loop_template(events, boundaries)

    print("[4/6] detect local-anchor recurring swap transitions")
    swaps_payload, swap_windows_payload = build_swap_payloads(
        sampled.party,
        sampled.ability,
        sampled.analysis_fps,
        boundaries,
        sampled.times_ms,
        sampled.frame_indexes,
        events,
        source_fps=sampled.source_fps,
        analysis_left=analysis_left,
        analysis_right=analysis_right,
        phase_tolerance=args.swap_phase_tolerance,
    )
    print(
        f"  party-HUD candidates={swaps_payload['candidate_count']} "
        f"anchored={swaps_payload.get('anchored_candidate_count', 0)} "
        f"recurring transitions={swaps_payload['recurring_transition_count']} "
        f"strong windows={swaps_payload['strong_transition_count']}"
    )
    _print_swap_matching_diagnostics(swaps_payload)
    if swaps_payload["strong_transition_count"]:
        for row in swaps_payload["transitions"]:
            if row["status"] != "strong":
                continue
            gap = row.get("outgoing_gap_ms") or {}
            print(
                f"    swap#{row['transition']} anchor_cluster={row['outgoing_cluster']} "
                f"gap≈{float(gap.get('median', 0.0)):.1f}ms "
                f"support={row['support_loops']}/{row['loop_count']} "
                f"local_spread≈{row['local_visual_spread_ms']:.1f}ms "
                f"rotation_spread≈{row['rotation_spread_ms']:.1f}ms "
                f"slot_sim={row['slot_profile_consistency']:.2f} "
                f"conf={row['confidence']:.2f}"
            )

    print("[5/6] write analysis / review / timeline")
    review = _build_review(
        events, args.semantic_threshold, args.max_review
    )

    analysis_path = Path(str(prefix) + ".auto_analysis.json")
    loops_path = Path(str(prefix) + ".loops.json")
    review_path = Path(str(prefix) + ".review.json")
    timeline_path = Path(str(prefix) + ".auto_axis_timeline.json")
    swaps_path = Path(str(prefix) + ".swaps.json")
    swap_windows_path = Path(str(prefix) + ".swap_windows.json")
    report_path = Path(str(prefix) + ".analysis.txt")

    loop_payload = {
        "period_found": bool(period and len(boundaries) >= 3),
        "period_ms": round(period.lag / loop_fps * 1000.0) if period else None,
        "period_score": round(float(period.score), 5) if period else None,
        "boundaries": [
            {
                "sample_index": int(index),
                "frame": int(sampled.frame_indexes[index]),
                "ms": int(sampled.times_ms[index]),
                "score": round(float(score), 5),
            }
            for index, score in boundary_rows
        ],
        "template": template,
    }
    _json_dump(loops_path, loop_payload)
    _json_dump(review_path, {"video": str(video), "items": review})
    _json_dump(swaps_path, swaps_payload)
    _json_dump(swap_windows_path, swap_windows_payload)

    timeline_count = _write_timeline(
        timeline_path,
        video,
        sampled,
        events,
        args.start_slot,
        args.semantic_threshold,
        blocked_reason=timeline_blocked_reason,
    )

    analysis_payload = {
        "schema": 5,
        "video": str(video),
        "source_fps": sampled.source_fps,
        "analysis_fps": sampled.analysis_fps,
        "frame_time_ms": round(1000.0 / sampled.source_fps, 4),
        "prototype_roots": [str(path) for path in roots],
        "prototype_counts": bank.counts(),
        "signature_scopes": {
            "visual_cluster": "local ±1 analysis sample",
            "semantic_classifier": "-60ms before, best +40..220ms after",
            "swap_detector": "recurring outgoing cluster + anchor gap + three-slot party-HUD change profile",
        },
        "loop": loop_payload,
        "swap_summary": {
            "candidate_count": swaps_payload["candidate_count"],
            "anchored_candidate_count": swaps_payload.get("anchored_candidate_count", 0),
            "recurring_transition_count": swaps_payload["recurring_transition_count"],
            "strong_transition_count": swaps_payload["strong_transition_count"],
            "swap_windows_execution_ready": False,
            "matching_diagnostics": swaps_payload.get("matching_diagnostics"),
        },
        "events": events,
        "review_count": len(review),
        "timeline_event_count": timeline_count,
        "timeline_blocked": bool(timeline_blocked_reason),
        "timeline_blocked_reason": timeline_blocked_reason,
        "notes": [
            "30fps source has about 33.3ms visual-frame quantization; the extractor does not invent missing source frames.",
            "Self telemetry prototypes teach visual semantics only; expert timing comes from the UP video.",
            "Strict swap matching now uses a recurring local outgoing cluster and anchor-to-swap gap; whole-rotation phase is diagnostic only.",
            "The party descriptor is split into three vertical HUD bands to compare which slot regions change, without assuming a target slot.",
            "swap_windows are visual-success windows, not hidden key-down timestamps and not yet semantic outgoing-action phase.",
            "Missing or degenerate semantic labels fail closed: analysis is preserved but the compileable timeline is written empty.",
        ],
        "warnings": [
            line for line in (timeline_blocked_reason,) if line
        ],
    }
    _json_dump(analysis_path, analysis_payload)

    high_recurrence = sum(
        1 for row in events if int(row["recurrence_support"]) >= 2
    )
    semantic_events = sum(1 for row in events if row.get("semantic"))
    report_lines = [
        f"video: {video}",
        f"source_fps: {sampled.source_fps:.3f} (frame≈{1000.0 / sampled.source_fps:.2f}ms)",
        f"analysis_fps: {sampled.analysis_fps:.3f}",
        f"event_candidates: {len(events)}",
        f"recurring_candidates: {high_recurrence}",
        f"semantic_candidates: {semantic_events}",
        f"swap_hud_candidates: {swaps_payload['candidate_count']}",
        f"swap_anchored_candidates: {swaps_payload.get('anchored_candidate_count', 0)}",
        f"recurring_swap_transitions: {swaps_payload['recurring_transition_count']}",
        f"strong_swap_windows: {swaps_payload['strong_transition_count']}",
        "swap_windows_execution_ready: no",
        f"timeline_high_confidence_events: {timeline_count}",
        f"timeline_blocked: {'yes' if timeline_blocked_reason else 'no'}",
        f"review_windows: {len(review)}",
        f"loop_period_ms: {loop_payload['period_ms']}",
        f"loop_score: {loop_payload['period_score']}",
        "",
        "outputs:",
        f"  {analysis_path}",
        f"  {loops_path}",
        f"  {swaps_path}",
        f"  {swap_windows_path}",
        f"  {review_path}",
        f"  {timeline_path}",
        "",
        "swap_windows 是 video-only 的局部锚点→成功切人视觉窗口；当前只用于离线研究，不直接驱动 launcher。",
    ]
    if timeline_blocked_reason:
        report_lines.extend([
            f"警告：{timeline_blocked_reason}",
            "auto_axis_timeline 已按 fail-closed 写为空 timeline；其它分析产物仍保留。",
        ])
    else:
        report_lines.extend([
            "重要：auto_axis_timeline 只包含高置信度且已能映射到现有轴 token 的事件。",
            "先看 swaps/swap_windows/analysis；不要因为 timeline 文件存在就直接当成最终实机轴。",
        ])

    report = "\n".join(report_lines)
    report_path.write_text(report + "\n", encoding="utf-8")

    print("[6/6] done")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
