"""基于局部 recurring anchor 的 video-only swap 对齐。

旧版只用 40s 左右 team-rotation phase 对齐切人，容易受到每轮累计节奏漂移影响。
本模块改成：

1. 仍用高 recall party-HUD transition 找候选；
2. 把整条 party descriptor 按垂直方向拆成 3 个槽位 band，构造三槽 change profile；
3. 给每个候选绑定最近的 recurring visual cluster（outgoing local anchor）；
4. 跨 loop 只匹配“同 anchor cluster + 相近 anchor→swap gap + 相似三槽 change profile”；
5. rotation phase 只保留做粗诊断，不再作为 strict swap 的主时间轴。

输出的是“相对局部视觉锚点的成功切人窗口”，仍不是隐藏的物理 key-down 时间。
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from auto_swaps import circular_phase_distance, detect_party_transitions


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if norm < 1e-6:
        return np.zeros_like(value, dtype=np.float32)
    return value / norm


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return value / norms


def _distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    data = np.asarray(values, dtype=np.float32)
    p10, p50, p90 = np.percentile(data, [10, 50, 90])
    return {
        "p10": round(float(p10), 3),
        "median": round(float(p50), 3),
        "p90": round(float(p90), 3),
    }


def _loop_for_sample(sample_index: int, boundaries: list[int]) -> tuple[int, float] | None:
    for loop_index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        if int(left) <= sample_index < int(right):
            phase = (sample_index - int(left)) / max(1, int(right) - int(left))
            return int(loop_index), float(phase)
    return None


def _party_slot_views(party: np.ndarray) -> np.ndarray:
    """把原 14x24 party feature 的垂直结构拆成三个连续 band。

    当前 core 的 party feature 是 24(row) x 14(col) flatten，因此 336 维正好可按
    三个 8-row band 切分。若未来 feature 维度变化但仍能三等分，也保持兼容。
    """

    value = np.asarray(party, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] < 3 or value.shape[1] % 3 != 0:
        raise ValueError(
            "party feature dimension must be divisible by 3 for slot-band matching"
        )
    return value.reshape(len(value), 3, value.shape[1] // 3)


def _slot_transition_profile(
    slot_views: np.ndarray,
    sample_index: int,
    analysis_fps: float,
) -> dict:
    count = len(slot_views)
    far = max(2, int(round(0.30 * analysis_fps)))
    near = max(1, int(round(0.07 * analysis_fps)))
    left0 = max(0, sample_index - far)
    left1 = max(left0 + 1, sample_index - near + 1)
    right0 = min(count - 1, sample_index + near)
    right1 = min(count, sample_index + far + 1)

    pre_raw = slot_views[left0:left1].mean(axis=0)
    post_raw = slot_views[right0:right1].mean(axis=0)
    pre = _normalize_rows(pre_raw)
    post = _normalize_rows(post_raw)

    cosines = np.sum(pre * post, axis=1)
    slot_change = np.clip(1.0 - cosines, 0.0, 2.0).astype(np.float32)
    profile = _normalize(slot_change)
    order = np.argsort(slot_change)[::-1]
    dominant = int(order[0]) + 1
    top = float(slot_change[order[0]])
    second = float(slot_change[order[1]]) if len(order) > 1 else 0.0
    second_ratio = second / max(top, 1e-6)

    # 整块 descriptor 已经全局归一化；这里保留每个 band 的相对能量，作为诊断，
    # 不直接把“变亮/变暗”解释成目标槽位，避免对 UI 主题/画质做假设。
    pre_energy = np.linalg.norm(pre_raw, axis=1)
    post_energy = np.linalg.norm(post_raw, axis=1)
    energy_delta = (post_energy - pre_energy).astype(np.float32)

    return {
        "profile": profile,
        "slot_change": slot_change,
        "dominant_slot_change": dominant,
        "second_change_ratio": float(second_ratio),
        "energy_delta": energy_delta,
    }


def _nearest_recurring_anchor(
    event_rows: list[dict],
    swap_ms: int,
    *,
    max_gap_ms: int = 4200,
    min_recurrence: int = 2,
) -> dict | None:
    candidates = [
        row
        for row in event_rows
        if int(row.get("ms", -1)) < swap_ms
        and swap_ms - int(row["ms"]) <= max_gap_ms
        and int(row.get("recurrence_support", 0)) >= min_recurrence
    ]
    if not candidates:
        return None
    # 以最近的 recurring event 作为 local outgoing anchor；同距离时优先支持度高者。
    return max(
        candidates,
        key=lambda row: (int(row["ms"]), int(row.get("recurrence_support", 0))),
    )


def _decorate_candidates(
    candidates,
    party: np.ndarray,
    analysis_fps: float,
    boundaries: list[int],
    times_ms: np.ndarray,
    frame_indexes: np.ndarray,
    event_rows: list[dict],
) -> list[dict]:
    slot_views = _party_slot_views(party)
    rows = []
    for candidate in candidates:
        mapped = _loop_for_sample(candidate.sample_index, boundaries)
        if mapped is None:
            continue
        loop_index, phase = mapped
        ms = int(times_ms[candidate.sample_index])
        profile = _slot_transition_profile(
            slot_views, candidate.sample_index, analysis_fps
        )
        anchor = _nearest_recurring_anchor(event_rows, ms)
        rows.append({
            "candidate": candidate,
            "loop": loop_index,
            "phase": phase,
            "ms": ms,
            "frame": int(frame_indexes[candidate.sample_index]),
            "anchor": anchor,
            "anchor_cluster": int(anchor["cluster"]) if anchor is not None else None,
            "anchor_ms": int(anchor["ms"]) if anchor is not None else None,
            "anchor_gap_ms": ms - int(anchor["ms"]) if anchor is not None else None,
            **profile,
        })
    return rows


def _profile_similarity(left: dict, right: dict) -> float:
    return float(np.dot(left["profile"], right["profile"]))


def _candidate_compatible(
    seed: dict,
    other: dict,
    *,
    local_gap_tolerance_ms: float,
    profile_similarity_min: float,
    coarse_phase_tolerance: float,
) -> tuple[bool, dict]:
    same_anchor = (
        seed["anchor_cluster"] is not None
        and seed["anchor_cluster"] == other["anchor_cluster"]
    )
    gap_distance = (
        abs(float(seed["anchor_gap_ms"]) - float(other["anchor_gap_ms"]))
        if same_anchor
        else float("inf")
    )
    profile_similarity = _profile_similarity(seed, other)
    phase_distance = circular_phase_distance(seed["phase"], other["phase"])
    ok = bool(
        same_anchor
        and gap_distance <= local_gap_tolerance_ms
        and profile_similarity >= profile_similarity_min
        and phase_distance <= coarse_phase_tolerance
    )
    return ok, {
        "same_anchor": same_anchor,
        "gap_distance_ms": gap_distance,
        "profile_similarity": profile_similarity,
        "phase_distance": phase_distance,
    }


def _pairwise_profile_consistency(rows: list[dict]) -> float:
    if len(rows) < 2:
        return 1.0
    values = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            values.append(_profile_similarity(rows[left], rows[right]))
    return float(np.median(np.asarray(values, dtype=np.float32))) if values else 1.0


def _group_candidates(
    rows: list[dict],
    loop_count: int,
    *,
    local_gap_tolerance_ms: float,
    profile_similarity_min: float,
    coarse_phase_tolerance: float,
) -> list[dict]:
    minimum_support = 3 if loop_count >= 4 else 2
    seeds = []

    for seed in rows:
        if seed["anchor_cluster"] is None:
            continue
        chosen = [seed]
        for loop_index in range(loop_count):
            if loop_index == seed["loop"]:
                continue
            options = []
            for other in rows:
                if other["loop"] != loop_index:
                    continue
                ok, metrics = _candidate_compatible(
                    seed,
                    other,
                    local_gap_tolerance_ms=local_gap_tolerance_ms,
                    profile_similarity_min=profile_similarity_min,
                    coarse_phase_tolerance=coarse_phase_tolerance,
                )
                if not ok:
                    continue
                rank = (
                    metrics["gap_distance_ms"] / max(local_gap_tolerance_ms, 1.0)
                    + 0.20 * metrics["phase_distance"] / max(coarse_phase_tolerance, 1e-6)
                    - 0.30 * metrics["profile_similarity"]
                    - 0.08 * float(other["candidate"].score)
                )
                options.append((rank, other))
            if options:
                options.sort(key=lambda item: item[0])
                chosen.append(options[0][1])

        if len(chosen) < minimum_support:
            continue

        gap_values = [float(row["anchor_gap_ms"]) for row in chosen]
        phase_values = [float(row["phase"]) for row in chosen]
        gap_p10, gap_p50, gap_p90 = np.percentile(
            np.asarray(gap_values, dtype=np.float32), [10, 50, 90]
        )
        gap_spread = float(gap_p90 - gap_p10)
        profile_consistency = _pairwise_profile_consistency(chosen)
        support_ratio = len(chosen) / max(1, loop_count)
        gap_stability = float(
            np.clip(
                1.0 - gap_spread / max(local_gap_tolerance_ms * 1.5, 1.0),
                0.0,
                1.0,
            )
        )
        candidate_quality = float(
            np.median([float(row["candidate"].score) for row in chosen])
        )
        confidence = (
            0.40 * support_ratio
            + 0.30 * gap_stability
            + 0.20 * np.clip((profile_consistency - 0.55) / 0.45, 0.0, 1.0)
            + 0.10 * candidate_quality
        )
        seeds.append({
            "anchor_cluster": int(seed["anchor_cluster"]),
            "anchor_gap_median": float(gap_p50),
            "anchor_gap_p10": float(gap_p10),
            "anchor_gap_p90": float(gap_p90),
            "anchor_gap_spread": gap_spread,
            "phase_median": float(np.median(np.asarray(phase_values, dtype=np.float32))),
            "profile_consistency": profile_consistency,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "occurrences": chosen,
        })

    # 同一个 local anchor 后几乎相同 gap/profile 的 seed 只保留最强的一组。
    seeds.sort(
        key=lambda row: (
            -len(row["occurrences"]),
            -float(row["confidence"]),
            float(row["anchor_gap_spread"]),
        )
    )
    accepted = []
    for row in seeds:
        duplicate = False
        for existing in accepted:
            if row["anchor_cluster"] != existing["anchor_cluster"]:
                continue
            if abs(row["anchor_gap_median"] - existing["anchor_gap_median"]) > local_gap_tolerance_ms * 0.5:
                continue
            if _profile_similarity(row["occurrences"][0], existing["occurrences"][0]) < profile_similarity_min:
                continue
            duplicate = True
            break
        if not duplicate:
            accepted.append(row)

    return sorted(
        accepted,
        key=lambda row: (float(row["phase_median"]), int(row["anchor_cluster"])),
    )


def _matching_diagnostics(
    rows: list[dict],
    loop_count: int,
    *,
    local_gap_tolerance_ms: float,
    profile_similarity_min: float,
    coarse_phase_tolerance: float,
) -> dict:
    anchored = [row for row in rows if row["anchor_cluster"] is not None]
    anchor_loop_support = defaultdict(set)
    for row in anchored:
        anchor_loop_support[int(row["anchor_cluster"])].add(int(row["loop"]))
    anchor_clusters_3plus = sum(
        len(loops) >= (3 if loop_count >= 4 else 2)
        for loops in anchor_loop_support.values()
    )

    cross_loop_pairs = 0
    same_anchor_pairs = 0
    local_gap_near_pairs = 0
    profile_compatible_pairs = 0
    both_pairs = 0
    gap_failures: list[float] = []
    profile_failures: list[float] = []

    for left in range(len(anchored)):
        for right in range(left + 1, len(anchored)):
            a, b = anchored[left], anchored[right]
            if a["loop"] == b["loop"]:
                continue
            cross_loop_pairs += 1
            if a["anchor_cluster"] != b["anchor_cluster"]:
                continue
            same_anchor_pairs += 1
            gap_distance = abs(float(a["anchor_gap_ms"]) - float(b["anchor_gap_ms"]))
            profile_similarity = _profile_similarity(a, b)
            phase_distance = circular_phase_distance(a["phase"], b["phase"])
            gap_ok = gap_distance <= local_gap_tolerance_ms
            profile_ok = profile_similarity >= profile_similarity_min
            coarse_ok = phase_distance <= coarse_phase_tolerance
            if gap_ok:
                local_gap_near_pairs += 1
            else:
                gap_failures.append(gap_distance)
            if profile_ok:
                profile_compatible_pairs += 1
            else:
                profile_failures.append(profile_similarity)
            if gap_ok and profile_ok and coarse_ok:
                both_pairs += 1

    if not anchored:
        hint = "insufficient_recurring_local_anchors"
    elif anchor_clusters_3plus == 0:
        hint = "recurring_event_anchor_coverage_too_low"
    elif same_anchor_pairs == 0:
        hint = "anchors_exist_but_not_shared_across_loops"
    elif local_gap_near_pairs == 0:
        hint = "local_gap_drift_or_wrong_outgoing_anchor"
    elif profile_compatible_pairs == 0:
        hint = "three_slot_hud_profile_unstable"
    elif both_pairs == 0:
        hint = "coarse_rotation_guard_rejects_local_matches"
    else:
        hint = "local_anchor_and_slot_profile_overlap_exists"

    return {
        "matching_basis": "recurring-outgoing-cluster + local-gap + three-slot-change-profile",
        "raw_candidate_count": len(rows),
        "anchored_candidate_count": len(anchored),
        "anchor_cluster_count": len(anchor_loop_support),
        "anchor_clusters_with_required_loop_support": int(anchor_clusters_3plus),
        "cross_loop_anchored_pairs": int(cross_loop_pairs),
        "same_anchor_pairs": int(same_anchor_pairs),
        "local_gap_near_pairs": int(local_gap_near_pairs),
        "slot_profile_compatible_pairs": int(profile_compatible_pairs),
        "local_gap_and_profile_pairs": int(both_pairs),
        "local_gap_tolerance_ms": round(float(local_gap_tolerance_ms), 2),
        "slot_profile_similarity_min": round(float(profile_similarity_min), 3),
        "coarse_rotation_phase_tolerance": round(float(coarse_phase_tolerance), 4),
        "failed_same_anchor_gap_distance_ms": _distribution(gap_failures),
        "failed_slot_profile_similarity": _distribution(profile_failures),
        "hint": hint,
    }


def build_swap_payloads(
    party: np.ndarray,
    ability: np.ndarray,
    analysis_fps: float,
    boundaries: list[int],
    times_ms: np.ndarray,
    frame_indexes: np.ndarray,
    event_rows: list[dict],
    *,
    source_fps: float,
    analysis_left: int = 0,
    analysis_right: int | None = None,
    phase_tolerance: float = 0.04,
) -> tuple[dict, dict]:
    """生成 local-anchor 版 swaps / swap_windows payload。"""

    loop_count = max(0, len(boundaries) - 1)
    frame_ms = 1000.0 / max(float(source_fps), 1e-6)
    # 严格 local gap 允许约 8 帧匹配，strong spread 再收紧到约 5 帧。
    local_gap_tolerance_ms = max(240.0, frame_ms * 8.0)
    strong_spread_limit_ms = max(150.0, frame_ms * 5.0)
    profile_similarity_min = 0.78
    # 全局 rotation 只做防止“同 cluster 在完全不同 rotation 区域重复”的粗门。
    coarse_phase_tolerance = max(0.08, float(phase_tolerance) * 2.0)

    candidates = detect_party_transitions(
        party,
        ability,
        analysis_fps,
        left=analysis_left,
        right=analysis_right,
    )
    rows = _decorate_candidates(
        candidates,
        party,
        analysis_fps,
        boundaries,
        times_ms,
        frame_indexes,
        event_rows,
    )
    groups = _group_candidates(
        rows,
        loop_count,
        local_gap_tolerance_ms=local_gap_tolerance_ms,
        profile_similarity_min=profile_similarity_min,
        coarse_phase_tolerance=coarse_phase_tolerance,
    )
    diagnostics = _matching_diagnostics(
        rows,
        loop_count,
        local_gap_tolerance_ms=local_gap_tolerance_ms,
        profile_similarity_min=profile_similarity_min,
        coarse_phase_tolerance=coarse_phase_tolerance,
    )

    transitions = []
    windows = []
    minimum_support = 3 if loop_count >= 4 else 2

    for transition_index, group in enumerate(groups, start=1):
        occurrences = []
        phases = []
        dominant_changes = []
        for row in group["occurrences"]:
            candidate = row["candidate"]
            phases.append(float(row["phase"]))
            dominant_changes.append(int(row["dominant_slot_change"]))
            occurrences.append({
                "loop": int(row["loop"]),
                "sample_index": int(candidate.sample_index),
                "frame": int(row["frame"]),
                "ms": int(row["ms"]),
                "rotation_phase": round(float(row["phase"]), 6),
                "candidate_score": round(float(candidate.score), 4),
                "party_z": round(float(candidate.party_z), 4),
                "anchor_cluster": int(row["anchor_cluster"]),
                "anchor_ms": int(row["anchor_ms"]),
                "anchor_gap_ms": int(row["anchor_gap_ms"]),
                "slot_change": [round(float(x), 5) for x in row["slot_change"]],
                "slot_change_profile": [round(float(x), 5) for x in row["profile"]],
                "dominant_slot_change": int(row["dominant_slot_change"]),
                "second_change_ratio": round(float(row["second_change_ratio"]), 4),
            })

        phase_p10, phase_median, phase_p90 = np.percentile(
            np.asarray(phases, dtype=np.float32), [10, 50, 90]
        )
        rotation_spread_ms = float(
            (phase_p90 - phase_p10)
            * np.median([
                float(times_ms[right] - times_ms[left])
                for left, right in zip(boundaries, boundaries[1:])
                if 0 <= left < len(times_ms) and 0 <= right < len(times_ms)
            ])
        ) if loop_count else 0.0
        local_visual_spread_ms = max(frame_ms, float(group["anchor_gap_spread"]))
        support_loops = len({item["loop"] for item in occurrences})
        dominant_mode = Counter(dominant_changes).most_common(1)[0][0] if dominant_changes else None

        strong = bool(
            support_loops >= minimum_support
            and local_visual_spread_ms <= strong_spread_limit_ms
            and float(group["profile_consistency"]) >= 0.84
            and float(group["confidence"]) >= 0.70
        )
        status = "strong" if strong else "anchored_recurring"
        gap_summary = {
            "p10": round(float(group["anchor_gap_p10"]), 3),
            "median": round(float(group["anchor_gap_median"]), 3),
            "p90": round(float(group["anchor_gap_p90"]), 3),
        }
        transition = {
            "transition": transition_index,
            "status": status,
            "matching_basis": "outgoing_cluster + anchor_gap + three_slot_change_profile",
            "from_state": None,
            "to_state": None,
            "target_slot": None,
            "support_loops": support_loops,
            "loop_count": loop_count,
            "outgoing_cluster": int(group["anchor_cluster"]),
            "outgoing_gap_ms": gap_summary,
            "local_visual_spread_ms": round(local_visual_spread_ms, 2),
            "rotation_phase_p10": round(float(phase_p10), 6),
            "rotation_phase_median": round(float(phase_median), 6),
            "rotation_phase_p90": round(float(phase_p90), 6),
            "rotation_spread_ms": round(max(frame_ms, rotation_spread_ms), 2),
            "slot_profile_consistency": round(float(group["profile_consistency"]), 4),
            "dominant_slot_change_mode": dominant_mode,
            "confidence": round(float(group["confidence"]), 4),
            "frame_quantization_ms": round(frame_ms, 3),
            "occurrences": occurrences,
        }
        transitions.append(transition)

        if strong:
            windows.append({
                "transition": transition_index,
                "support_loops": support_loops,
                "outgoing_cluster": int(group["anchor_cluster"]),
                "anchor_gap_window_ms": gap_summary,
                "local_visual_spread_ms": transition["local_visual_spread_ms"],
                "rotation_phase_median": transition["rotation_phase_median"],
                "rotation_spread_ms": transition["rotation_spread_ms"],
                "slot_profile_consistency": transition["slot_profile_consistency"],
                "dominant_slot_change_mode": dominant_mode,
                "target_slot": None,
                "confidence": transition["confidence"],
                "execution_ready": False,
                "reason": (
                    "video-only local-anchor visual success window; not the hidden key-down "
                    "time and not yet a semantic outgoing-action phase/cancel model"
                ),
            })

    swaps_payload = {
        "schema": 4,
        "detector": "video-only local-anchor + three-slot party-HUD transition",
        "loop_count": loop_count,
        "candidate_count": len(candidates),
        "anchored_candidate_count": diagnostics["anchored_candidate_count"],
        "recurring_transition_count": len(transitions),
        "strong_transition_count": sum(row["status"] == "strong" for row in transitions),
        "matching_diagnostics": diagnostics,
        "transitions": transitions,
        "notes": [
            "Expert swap timing is matched relative to a recurring local outgoing cluster, not primarily to whole-rotation phase.",
            "Three vertical party-HUD bands provide a coarse slot-change identity without assuming which band is the target slot.",
            "target_slot remains null; dominant_slot_change is only a diagnostic and is not interpreted as s1/s2/s3.",
        ],
    }
    windows_payload = {
        "schema": 2,
        "matching_basis": "recurring local visual anchor + anchor-to-swap gap + three-slot HUD change profile",
        "execution_ready": False,
        "windows": windows,
        "notes": [
            "anchor_gap_window_ms is the repeated visual gap from outgoing recurring cluster to successful swap appearance.",
            "rotation phase is diagnostic only and may drift across human-executed loops.",
            "These windows are not physical input timestamps and must not directly drive launcher input.",
        ],
    }
    return swaps_payload, windows_payload
