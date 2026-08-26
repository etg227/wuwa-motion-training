"""只依赖高手视频自身的 recurring swap / window 提取。

不尝试恢复 UP 主真实按键时刻。这里输出的是：
- 右侧队伍 HUD 的稳定转场候选；
- 不同 rotation 中重复出现、且 HUD 转场形状一致的 visual swap phase；
- 最近的 recurring visual cluster 作为 outgoing anchor；
- visual success window（P10/P50/P90），供后续 action-phase/cancel 建模。

重要：phase 匹配容差按“几帧”自适应，而不是直接把 rotation 的 4% 当窗口。
39.8s rotation 的 0.04 相当于约 1.6s，会把大量普通 HUD 变化错误拼成切人。
30fps 源的基本时间量化仍是约 33.3ms；本模块不会插值伪造输入时刻。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if norm < 1e-6:
        return np.zeros_like(value, dtype=np.float32)
    return value / norm


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float32)
    if len(value) == 0:
        return value.copy()
    median = float(np.median(value))
    mad = float(np.median(np.abs(value - median)))
    scale = max(1e-6, 1.4826 * mad)
    return (value - median) / scale


def circular_phase_distance(left: float, right: float) -> float:
    delta = abs(float(left) - float(right)) % 1.0
    return min(delta, 1.0 - delta)


def _unwrap_phase(phase: float, center: float) -> float:
    value = float(phase)
    while value - center > 0.5:
        value -= 1.0
    while value - center < -0.5:
        value += 1.0
    return value


def _pairwise_median(rows: list[np.ndarray]) -> float:
    if len(rows) < 2:
        return 1.0
    matrix = np.stack(rows).astype(np.float32)
    similarities = matrix @ matrix.T
    upper = similarities[np.triu_indices(len(matrix), 1)]
    return float(np.median(upper)) if len(upper) else 1.0


@dataclass(frozen=True)
class SwapCandidate:
    sample_index: int
    score: float
    party_z: float
    state_change: float
    persistence: float
    hud_ratio: float
    signature: np.ndarray
    pre_state: np.ndarray
    post_state: np.ndarray


def detect_party_transitions(
    party: np.ndarray,
    ability: np.ndarray,
    analysis_fps: float,
    *,
    left: int = 0,
    right: int | None = None,
    z_threshold: float = 2.0,
    min_gap_s: float = 0.35,
) -> list[SwapCandidate]:
    """找队伍 HUD 的 step-like 转场。

    这一层故意保持较高 recall：先找 frame-to-frame party HUD 尖峰，再用转场
    前后的稳定窗口验证“旧状态 -> 新状态”持续存在。真正的 precision 主要在
    cross-loop recurring grouping：phase、transition signature、pre/post state 都要一致。
    """

    if analysis_fps <= 0:
        raise ValueError("analysis_fps must be > 0")

    party = np.asarray(party, dtype=np.float32)
    ability = np.asarray(ability, dtype=np.float32)
    if party.ndim != 2 or ability.ndim != 2 or len(party) != len(ability):
        raise ValueError("party/ability must be 2D arrays with equal sample count")
    if len(party) < 8:
        return []

    count = len(party)
    right = count - 1 if right is None else min(count - 1, int(right))
    left = max(0, int(left))
    if right <= left:
        return []

    party_delta = np.zeros(count, dtype=np.float32)
    ability_delta = np.zeros(count, dtype=np.float32)
    party_delta[1:] = np.linalg.norm(party[1:] - party[:-1], axis=1)
    ability_delta[1:] = np.linalg.norm(ability[1:] - ability[:-1], axis=1)
    party_z = _robust_zscore(party_delta)

    far = max(2, int(round(0.30 * analysis_fps)))
    near = max(1, int(round(0.07 * analysis_fps)))
    first = max(left, far)
    last = min(right, count - 1 - far)
    if last < first:
        return []

    raw: list[SwapCandidate] = []
    for index in range(first, last + 1):
        if float(party_z[index]) < z_threshold:
            continue
        if party_delta[index] < party_delta[max(left, index - 1)]:
            continue
        if party_delta[index] < party_delta[min(right, index + 1)]:
            continue

        pre_rows = party[index - far:index - near + 1]
        post_rows = party[index + near:index + far + 1]
        if len(pre_rows) < 2 or len(post_rows) < 2:
            continue

        pre_state = _normalize(pre_rows.mean(axis=0))
        post_state = _normalize(post_rows.mean(axis=0))
        state_change = float(np.clip(1.0 - np.dot(pre_state, post_state), 0.0, 2.0))

        pre_similarity = float(np.median(pre_rows @ pre_state))
        post_similarity = float(np.median(post_rows @ post_state))
        persistence = float(
            np.clip(((pre_similarity + post_similarity) * 0.5 - 0.55) / 0.40, 0.0, 1.0)
        )

        ratio = float(party_delta[index] / max(float(ability_delta[index]), 1e-3))
        hud_term = float(np.clip((ratio - 0.70) / 1.50, 0.0, 1.0))
        z_term = float(np.clip((float(party_z[index]) - 1.8) / 4.0, 0.0, 1.0))
        separation_term = float(np.clip(state_change / 0.25, 0.0, 1.0))
        score = (
            0.35 * z_term
            + 0.35 * separation_term
            + 0.20 * persistence
            + 0.10 * hud_term
        )

        raw.append(SwapCandidate(
            sample_index=index,
            score=float(np.clip(score, 0.0, 1.0)),
            party_z=float(party_z[index]),
            state_change=state_change,
            persistence=persistence,
            hud_ratio=ratio,
            signature=_normalize(post_state - pre_state),
            pre_state=pre_state,
            post_state=post_state,
        ))

    minimum_gap = max(1, int(round(min_gap_s * analysis_fps)))
    raw.sort(key=lambda row: row.score, reverse=True)
    kept: list[SwapCandidate] = []
    for candidate in raw:
        if all(
            abs(candidate.sample_index - existing.sample_index) > minimum_gap
            for existing in kept
        ):
            kept.append(candidate)
    return sorted(kept, key=lambda row: row.sample_index)


def _map_to_loops(
    candidates: list[SwapCandidate],
    boundaries: list[int],
) -> list[tuple[SwapCandidate, int, float]]:
    rows = []
    for candidate in candidates:
        for loop_index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
            if left <= candidate.sample_index < right:
                phase = (candidate.sample_index - left) / max(1, right - left)
                rows.append((candidate, loop_index, float(phase)))
                break
    return rows


def adaptive_phase_tolerance(
    boundaries: list[int],
    requested: float,
    *,
    frame_budget: float = 4.0,
    floor: float = 0.0025,
) -> float:
    """把 phase 容差限制到约几帧，而不是 rotation 的固定百分比。

    例如 39.8s / 30fps rotation 约 1195 个分析采样：4 帧约等于 phase 0.00335。
    即使 CLI 仍传 0.04，也只把它当“最大上限”，不会再允许 ±1.6s 配对。
    """

    if requested <= 0:
        raise ValueError("phase tolerance must be > 0")
    lengths = [
        int(right) - int(left)
        for left, right in zip(boundaries, boundaries[1:])
        if int(right) > int(left)
    ]
    if not lengths:
        return float(min(requested, max(floor, 0.01)))
    median_samples = float(np.median(np.asarray(lengths, dtype=np.float32)))
    frame_phase = frame_budget / max(median_samples, 1.0)
    return float(min(requested, max(floor, frame_phase)))


def _transition_compatibility(
    left: SwapCandidate,
    right: SwapCandidate,
) -> tuple[float, float, float, float]:
    signature_similarity = float(np.dot(left.signature, right.signature))
    pre_similarity = float(np.dot(left.pre_state, right.pre_state))
    post_similarity = float(np.dot(left.post_state, right.post_state))

    # Directional transition 必须大体一致；同时要求转场前/后 HUD 状态各自相似。
    # 阈值故意不是极高，保留压缩/特效噪声，但会排除“只是 phase 接近”的随机变化。
    if signature_similarity < 0.20 or pre_similarity < 0.52 or post_similarity < 0.52:
        return -1.0, signature_similarity, pre_similarity, post_similarity

    compatibility = (
        0.50 * np.clip((signature_similarity + 0.10) / 1.10, 0.0, 1.0)
        + 0.25 * np.clip((pre_similarity - 0.40) / 0.60, 0.0, 1.0)
        + 0.25 * np.clip((post_similarity - 0.40) / 0.60, 0.0, 1.0)
    )
    return float(compatibility), signature_similarity, pre_similarity, post_similarity


def recurring_swap_groups(
    candidates: list[SwapCandidate],
    boundaries: list[int],
    *,
    phase_tolerance: float = 0.04,
    min_support: int | None = None,
) -> list[dict]:
    """把不同 rotation 中相近 phase、且 HUD 转场身份一致的事件合成 recurring swap。"""

    loop_count = max(0, len(boundaries) - 1)
    if loop_count < 2 or not candidates:
        return []
    if min_support is None:
        min_support = 3 if loop_count >= 4 else 2
    min_support = max(2, min(int(min_support), loop_count))

    effective_tolerance = adaptive_phase_tolerance(boundaries, phase_tolerance)
    mapped = _map_to_loops(candidates, boundaries)
    seeds: list[dict] = []

    for seed_candidate, seed_loop, seed_phase in mapped:
        chosen = [(seed_candidate, seed_loop, seed_phase)]
        for loop_index in range(loop_count):
            if loop_index == seed_loop:
                continue
            options = []
            for row in mapped:
                candidate, candidate_loop, candidate_phase = row
                if candidate_loop != loop_index:
                    continue
                phase_distance = circular_phase_distance(candidate_phase, seed_phase)
                if phase_distance > effective_tolerance:
                    continue
                compatibility, sig_sim, pre_sim, post_sim = _transition_compatibility(
                    seed_candidate, candidate
                )
                if compatibility < 0:
                    continue
                # 先按 phase，再用 HUD transition identity / candidate quality 破平。
                rank = (
                    phase_distance / max(effective_tolerance, 1e-6)
                    - 0.30 * compatibility
                    - 0.08 * candidate.score
                )
                options.append((rank, row, sig_sim, pre_sim, post_sim))
            if options:
                options.sort(key=lambda item: item[0])
                chosen.append(options[0][1])

        if len(chosen) < min_support:
            continue

        unwrapped = np.asarray(
            [_unwrap_phase(phase, seed_phase) for _, _, phase in chosen],
            dtype=np.float32,
        )
        center = float(np.median(unwrapped)) % 1.0
        p10, p50, p90 = np.percentile(unwrapped, [10, 50, 90]).tolist()
        spread = float(p90 - p10)

        signature_consistency = _pairwise_median(
            [row[0].signature for row in chosen]
        )
        pre_state_consistency = _pairwise_median(
            [row[0].pre_state for row in chosen]
        )
        post_state_consistency = _pairwise_median(
            [row[0].post_state for row in chosen]
        )
        state_consistency = min(pre_state_consistency, post_state_consistency)

        support_ratio = len(chosen) / loop_count
        phase_stability = float(
            np.clip(1.0 - spread / max(effective_tolerance * 1.25, 1e-6), 0.0, 1.0)
        )
        candidate_quality = float(np.median([row[0].score for row in chosen]))
        transition_term = float(np.clip((signature_consistency - 0.10) / 0.90, 0.0, 1.0))
        state_term = float(np.clip((state_consistency - 0.45) / 0.55, 0.0, 1.0))
        confidence = (
            0.35 * support_ratio
            + 0.25 * phase_stability
            + 0.18 * transition_term
            + 0.12 * state_term
            + 0.10 * candidate_quality
        )

        seeds.append({
            "phase": center,
            "phase_p10_unwrapped": float(p10),
            "phase_p50_unwrapped": float(p50),
            "phase_p90_unwrapped": float(p90),
            "phase_spread": spread,
            "effective_phase_tolerance": effective_tolerance,
            "signature_consistency": signature_consistency,
            "pre_state_consistency": pre_state_consistency,
            "post_state_consistency": post_state_consistency,
            "state_consistency": state_consistency,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "occurrences": chosen,
        })

    seeds.sort(
        key=lambda row: (
            -len(row["occurrences"]),
            -float(row["confidence"]),
        )
    )
    result = []
    accepted: list[dict] = []
    for row in seeds:
        duplicate = False
        seed_candidate = row["occurrences"][0][0]
        for existing in accepted:
            if circular_phase_distance(float(row["phase"]), float(existing["phase"])) >= effective_tolerance * 0.75:
                continue
            existing_candidate = existing["occurrences"][0][0]
            compatibility, *_ = _transition_compatibility(seed_candidate, existing_candidate)
            if compatibility >= 0:
                duplicate = True
                break
        if duplicate:
            continue
        accepted.append(row)
        result.append(row)

    return sorted(result, key=lambda row: float(row["phase"]))


def _percentile_summary(values: list[float]) -> dict | None:
    if not values:
        return None
    p10, p50, p90 = np.percentile(np.asarray(values, dtype=np.float32), [10, 50, 90])
    return {
        "p10": round(float(p10), 3),
        "median": round(float(p50), 3),
        "p90": round(float(p90), 3),
    }


def _nearest_outgoing_event(
    event_rows: list[dict],
    swap_ms: int,
    *,
    max_gap_ms: int = 2500,
) -> dict | None:
    candidates = [
        row for row in event_rows
        if int(row.get("ms", -1)) < swap_ms
        and swap_ms - int(row["ms"]) <= max_gap_ms
        and int(row.get("recurrence_support", 0)) >= 2
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: int(row["ms"]))


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
    """生成 swaps.json / swap_windows.json 的可序列化 payload。

    swaps.json 保留 recurring 候选；swap_windows.json 只收真正窄且 HUD identity 一致
    的 strong visual-success windows。这样宽到几百毫秒/秒级的 recurring HUD 变化
    不会再被冒充成严格切人窗口。
    """

    loop_count = max(0, len(boundaries) - 1)
    period_ms = None
    period_samples = None
    if loop_count >= 1:
        loop_durations = [
            float(times_ms[right] - times_ms[left])
            for left, right in zip(boundaries, boundaries[1:])
            if 0 <= left < len(times_ms) and 0 <= right < len(times_ms)
        ]
        sample_lengths = [
            int(right) - int(left)
            for left, right in zip(boundaries, boundaries[1:])
            if int(right) > int(left)
        ]
        if loop_durations:
            period_ms = float(np.median(loop_durations))
        if sample_lengths:
            period_samples = float(np.median(np.asarray(sample_lengths, dtype=np.float32)))

    effective_tolerance = adaptive_phase_tolerance(boundaries, phase_tolerance)
    candidates = detect_party_transitions(
        party,
        ability,
        analysis_fps,
        left=analysis_left,
        right=analysis_right,
    )
    groups = recurring_swap_groups(
        candidates,
        boundaries,
        phase_tolerance=phase_tolerance,
    )

    swaps = []
    windows = []
    frame_ms = 1000.0 / max(float(source_fps), 1e-6)
    # strong 的时间离散必须是“几帧级”：30fps 时 6 帧≈200ms；60fps 时至少 140ms。
    strong_spread_limit_ms = max(frame_ms * 6.0, 140.0)

    for transition_index, group in enumerate(groups, start=1):
        occurrences = []
        anchor_clusters: list[int] = []
        anchor_gaps_by_cluster: dict[int, list[float]] = {}

        unwrapped_center = float(group["phase_p50_unwrapped"])
        phase_values = []
        for candidate, loop_index, phase in group["occurrences"]:
            sample_index = candidate.sample_index
            ms = int(times_ms[sample_index])
            frame = int(frame_indexes[sample_index])
            phase_unwrapped = _unwrap_phase(float(phase), unwrapped_center)
            phase_values.append(phase_unwrapped)

            anchor = _nearest_outgoing_event(event_rows, ms)
            anchor_cluster = None
            anchor_gap_ms = None
            if anchor is not None:
                anchor_cluster = int(anchor["cluster"])
                anchor_gap_ms = ms - int(anchor["ms"])
                anchor_clusters.append(anchor_cluster)
                anchor_gaps_by_cluster.setdefault(anchor_cluster, []).append(
                    float(anchor_gap_ms)
                )

            occurrences.append({
                "loop": int(loop_index),
                "sample_index": int(sample_index),
                "frame": frame,
                "ms": ms,
                "rotation_phase": round(float(phase % 1.0), 6),
                "candidate_score": round(float(candidate.score), 4),
                "party_z": round(float(candidate.party_z), 4),
                "state_change": round(float(candidate.state_change), 5),
                "persistence": round(float(candidate.persistence), 4),
                "outgoing_cluster": anchor_cluster,
                "outgoing_gap_ms": anchor_gap_ms,
            })

        support_loops = len({row["loop"] for row in occurrences})
        anchor_cluster = None
        anchor_support = 0
        anchor_gap_summary = None
        if anchor_clusters:
            unique, counts = np.unique(
                np.asarray(anchor_clusters, dtype=np.int32),
                return_counts=True,
            )
            best_pos = int(np.argmax(counts))
            best_cluster = int(unique[best_pos])
            best_count = int(counts[best_pos])
            if best_count >= max(2, int(np.ceil(support_loops * 0.60))):
                anchor_cluster = best_cluster
                anchor_support = best_count
                anchor_gap_summary = _percentile_summary(
                    anchor_gaps_by_cluster[best_cluster]
                )

        phase_p10, phase_median, phase_p90 = np.percentile(
            np.asarray(phase_values, dtype=np.float32),
            [10, 50, 90],
        )
        phase_spread = float(phase_p90 - phase_p10)
        observed_spread_ms = (
            phase_spread * period_ms
            if period_ms is not None
            else phase_spread / max(analysis_fps, 1e-6) * 1000.0
        )
        visual_spread_ms = max(frame_ms, float(observed_spread_ms))

        minimum_support = 3 if loop_count >= 4 else 2
        strong = bool(
            support_loops >= minimum_support
            and visual_spread_ms <= strong_spread_limit_ms
            and float(group["signature_consistency"]) >= 0.30
            and float(group["state_consistency"]) >= 0.58
            and float(group["confidence"]) >= 0.68
        )
        status = "strong" if strong else "recurring"

        row = {
            "transition": transition_index,
            "status": status,
            "from_state": None,
            "to_state": None,
            "target_slot": None,
            "support_loops": support_loops,
            "loop_count": loop_count,
            "rotation_phase_p10": round(float(phase_p10 % 1.0), 6),
            "rotation_phase_median": round(float(phase_median % 1.0), 6),
            "rotation_phase_p90": round(float(phase_p90 % 1.0), 6),
            "rotation_phase_wraps": bool((phase_p10 % 1.0) > (phase_p90 % 1.0)),
            "phase_offset_p10": round(float(phase_p10 - phase_median), 6),
            "phase_offset_p90": round(float(phase_p90 - phase_median), 6),
            "phase_spread": round(phase_spread, 6),
            "visual_spread_ms": round(float(visual_spread_ms), 2),
            "frame_quantization_ms": round(frame_ms, 3),
            "effective_phase_tolerance": round(float(group["effective_phase_tolerance"]), 6),
            "signature_consistency": round(float(group["signature_consistency"]), 4),
            "pre_state_consistency": round(float(group["pre_state_consistency"]), 4),
            "post_state_consistency": round(float(group["post_state_consistency"]), 4),
            "state_consistency": round(float(group["state_consistency"]), 4),
            "confidence": round(float(group["confidence"]), 4),
            "outgoing_cluster": anchor_cluster,
            "outgoing_cluster_support": anchor_support,
            "outgoing_gap_ms": anchor_gap_summary,
            "occurrences": occurrences,
        }
        swaps.append(row)

        if strong:
            windows.append({
                "transition": transition_index,
                "support_loops": support_loops,
                "rotation_phase_window": {
                    "p10": row["rotation_phase_p10"],
                    "median": row["rotation_phase_median"],
                    "p90": row["rotation_phase_p90"],
                    "wraps": row["rotation_phase_wraps"],
                    "offset_p10": row["phase_offset_p10"],
                    "offset_p90": row["phase_offset_p90"],
                },
                "visual_spread_ms": row["visual_spread_ms"],
                "signature_consistency": row["signature_consistency"],
                "state_consistency": row["state_consistency"],
                "outgoing_cluster": anchor_cluster,
                "outgoing_gap_ms": anchor_gap_summary,
                "confidence": row["confidence"],
                "execution_ready": False,
                "reason": (
                    "narrow recurring video-only visual success window; not the hidden key-down time "
                    "and not yet an outgoing-action phase model"
                ),
            })

    swaps_payload = {
        "schema": 2,
        "detector": "video-only recurring party-HUD transition",
        "loop_count": loop_count,
        "loop_period_ms": round(period_ms, 2) if period_ms is not None else None,
        "loop_period_samples": round(period_samples, 2) if period_samples is not None else None,
        "requested_phase_tolerance": float(phase_tolerance),
        "effective_phase_tolerance": round(effective_tolerance, 6),
        "strong_spread_limit_ms": round(strong_spread_limit_ms, 2),
        "candidate_count": len(candidates),
        "recurring_transition_count": len(swaps),
        "strong_transition_count": sum(row["status"] == "strong" for row in swaps),
        "transitions": swaps,
        "notes": [
            "No self telemetry is used to decide swap timing.",
            "requested phase tolerance is an upper bound; actual cross-loop matching is reduced to a few-frame adaptive tolerance.",
            "Cross-loop matches must agree in transition signature and in both pre/post HUD state, not only in rotation phase.",
            "Wide recurring HUD changes stay in swaps.json but do not enter swap_windows.json.",
            "Anonymous HUD transitions are not mapped to s1/s2/s3 until target slot evidence is reliable.",
        ],
    }
    windows_payload = {
        "schema": 2,
        "execution_ready": False,
        "strong_spread_limit_ms": round(strong_spread_limit_ms, 2),
        "windows": windows,
        "notes": [
            "These are narrow recurring visual-success windows, not physical input timestamps.",
            "Use outgoing_cluster/gap as an offline anchor; a later action-phase model must convert it into a cancel/request window.",
            "30fps source keeps roughly 33.3ms visual quantization.",
        ],
    }
    return swaps_payload, windows_payload
