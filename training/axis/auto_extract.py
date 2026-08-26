"""自动从高手/UP 视频提取重复轴结构与高置信度语义事件。

这不是“直接看一条视频就凭空知道所有按键”的黑箱分类器。它同时利用：
1. UP 视频自身的重复循环，自动学习 recurring visual event vocabulary；
2. 本机已有 video + telemetry（如果存在），只学习 A/E/Q/R/切人的视觉响应原型；
3. 右侧队伍 HUD 与技能 HUD 的变化，给切人候选与动作边界提供额外证据。

输出不会发送任何输入，也不会自动改 launcher。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

AXIS_DIR = Path(__file__).resolve().parent
MOTION_DIR = AXIS_DIR.parent / "motion"
ROOT = AXIS_DIR.parents[1]
sys.path.insert(0, str(AXIS_DIR))
sys.path.insert(0, str(MOTION_DIR))

from auto_analysis import (
    align_period_boundaries,
    cluster_signatures,
    confidence_from_signals,
    detect_peaks,
    estimate_repetition_period,
    loop_support,
    normalize_rows,
    normalize_vector,
    robust_zscore,
    smooth_signal,
)
from semantic_inputs import load_semantic_events
from timeline import AxisVideoEvent, Timeline


SEMANTIC_TO_TOKEN = {
    "ATTACK": "a",
    "SKILL_E": "e",
    "ECHO_Q": "q",
    "LIBERATION_R": "r",
    "SWAP_1": "s1",
    "SWAP_2": "s2",
    "SWAP_3": "s3",
}
SWITCH_TOKENS = {"s1": 1, "s2": 2, "s3": 3}

# Normalized ROIs. They deliberately avoid relying on OCR/template assets.
BODY_ROI = (0.06, 0.05, 0.82, 0.92)
PARTY_ROI = (0.80, 0.12, 1.00, 0.78)
ABILITY_ROI = (0.62, 0.67, 1.00, 1.00)


@dataclass
class SampledVideo:
    source_fps: float
    analysis_fps: float
    total_frames: int
    frame_indexes: np.ndarray
    times_ms: np.ndarray
    body: np.ndarray
    party: np.ndarray
    ability: np.ndarray


@dataclass
class PrototypeBank:
    samples: dict[str, np.ndarray]
    source_files: int
    source_events: int

    def counts(self) -> dict[str, int]:
        return {token: int(len(rows)) for token, rows in self.samples.items()}


def _json_dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _crop_feature(frame: np.ndarray, roi, size: tuple[int, int]) -> np.ndarray:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = roi
    left = int(np.clip(round(x0 * width), 0, max(0, width - 1)))
    right = int(np.clip(round(x1 * width), left + 1, width))
    top = int(np.clip(round(y0 * height), 0, max(0, height - 1)))
    bottom = int(np.clip(round(y1 * height), top + 1, height))
    crop = frame[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    value = gray.astype(np.float32).reshape(-1) / 255.0
    value -= float(value.mean())
    return normalize_vector(value)


def _descriptors(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body = _crop_feature(frame, BODY_ROI, (24, 14))
    party = _crop_feature(frame, PARTY_ROI, (14, 24))
    ability = _crop_feature(frame, ABILITY_ROI, (24, 14))
    return body, party, ability


def _signature_from_descriptors(
    before: tuple[np.ndarray, np.ndarray, np.ndarray],
    after: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    parts = [np.abs(after[index] - before[index]) for index in range(3)]
    return normalize_vector(np.concatenate(parts))


def _sample_video(video: Path, requested_fps: float | None) -> SampledVideo:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if source_fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(f"invalid video metadata: fps={source_fps} frames={total_frames}")

    analysis_fps = source_fps if not requested_fps or requested_fps <= 0 else min(source_fps, requested_fps)
    analysis_fps = min(60.0, max(4.0, analysis_fps))
    step = source_fps / analysis_fps

    frame_indexes: list[int] = []
    bodies: list[np.ndarray] = []
    parties: list[np.ndarray] = []
    abilities: list[np.ndarray] = []
    frame_index = 0
    next_sample = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index + 1e-6 >= next_sample:
            body, party, ability = _descriptors(frame)
            frame_indexes.append(frame_index)
            bodies.append(body)
            parties.append(party)
            abilities.append(ability)
            next_sample += step
        frame_index += 1
        if frame_index % max(1, int(source_fps * 20)) == 0:
            print(f"  video scan: {frame_index / source_fps:.0f}s / {total_frames / source_fps:.0f}s")

    cap.release()
    if len(frame_indexes) < 20:
        raise RuntimeError("video is too short for automatic analysis")

    frames = np.asarray(frame_indexes, dtype=np.int32)
    return SampledVideo(
        source_fps=source_fps,
        analysis_fps=analysis_fps,
        total_frames=total_frames,
        frame_indexes=frames,
        times_ms=np.rint(frames / source_fps * 1000.0).astype(np.int64),
        body=np.stack(bodies).astype(np.float32),
        party=np.stack(parties).astype(np.float32),
        ability=np.stack(abilities).astype(np.float32),
    )


def _delta_norms(matrix: np.ndarray) -> np.ndarray:
    result = np.zeros(len(matrix), dtype=np.float32)
    if len(matrix) > 1:
        result[1:] = np.linalg.norm(matrix[1:] - matrix[:-1], axis=1)
    return result


def _event_signatures(sampled: SampledVideo, indexes: list[int]) -> np.ndarray:
    rows = []
    for index in indexes:
        before = max(0, index - 1)
        after = min(len(sampled.body) - 1, index + 1)
        rows.append(normalize_vector(np.concatenate([
            np.abs(sampled.body[after] - sampled.body[before]),
            np.abs(sampled.party[after] - sampled.party[before]),
            np.abs(sampled.ability[after] - sampled.ability[before]),
        ])))
    if not rows:
        dimension = sampled.body.shape[1] + sampled.party.shape[1] + sampled.ability.shape[1]
        return np.empty((0, dimension), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


def _loop_detection(sampled: SampledVideo, min_loop_s: float, max_loop_s: float):
    stride = max(1, int(round(sampled.analysis_fps / 6.0)))
    body = sampled.body[::stride]
    party = sampled.party[::stride]
    ability = sampled.ability[::stride]
    loop_features = normalize_rows(np.concatenate((0.65 * body, 1.10 * party, 0.85 * ability), axis=1))
    loop_fps = sampled.analysis_fps / stride
    max_loop_s = min(max_loop_s, len(loop_features) / loop_fps / 2.05)
    if max_loop_s <= min_loop_s:
        return None, [], stride, loop_fps

    estimate = estimate_repetition_period(
        loop_features,
        min_lag=max(2, int(round(min_loop_s * loop_fps))),
        max_lag=max(2, int(round(max_loop_s * loop_fps))),
    )
    if estimate is None:
        return None, [], stride, loop_fps

    boundaries = align_period_boundaries(
        loop_features,
        estimate.lag,
        estimate.anchor,
        minimum_similarity=max(0.08, estimate.score - 0.35),
    )
    sample_boundaries = [min(len(sampled.body) - 1, index * stride) for index, _ in boundaries]
    return estimate, list(zip(sample_boundaries, [score for _, score in boundaries])), stride, loop_fps


def _read_frame_at(cap, source_fps: float, t_ms: float):
    frame = max(0, int(round(t_ms / 1000.0 * source_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    return image if ok else None


def _prototype_roots(explicit: list[str] | None) -> list[Path]:
    if explicit:
        return [Path(value).expanduser().resolve() for value in explicit]
    candidates = [
        ROOT / "training_data" / "motion",
        ROOT.parent / "wuwa-yg-launcher" / "training_data" / "motion",
    ]
    return [path for path in candidates if path.exists()]


def _build_prototype_bank(roots: list[Path], max_per_token: int = 36) -> PrototypeBank:
    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    source_files = 0
    source_events = 0

    telemetry_files: list[Path] = []
    for root in roots:
        if root.exists():
            telemetry_files.extend(root.rglob("*.inputs.jsonl"))

    for telemetry in sorted(set(telemetry_files)):
        video_name = telemetry.name.removesuffix(".inputs.jsonl") + ".mp4"
        video = telemetry.with_name(video_name)
        if not video.is_file():
            continue
        events = load_semantic_events(telemetry, edge="down")
        useful = [event for event in events if event.get("semantic") in SEMANTIC_TO_TOKEN]
        if not useful:
            continue

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            continue
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            cap.release()
            continue

        used_file = False
        for event in useful:
            token = SEMANTIC_TO_TOKEN[str(event["semantic"])]
            if len(rows[token]) >= max_per_token:
                continue
            t_ms = float(event.get("t_ms", 0.0))
            before_frame = _read_frame_at(cap, fps, max(0.0, t_ms - 60.0))
            if before_frame is None:
                continue
            before = _descriptors(before_frame)

            best_signature = None
            best_change = -1.0
            for offset in (40.0, 70.0, 100.0, 135.0, 175.0, 220.0):
                after_frame = _read_frame_at(cap, fps, t_ms + offset)
                if after_frame is None:
                    continue
                after = _descriptors(after_frame)
                signature = _signature_from_descriptors(before, after)
                change = float(np.linalg.norm(signature))
                raw_change = sum(float(np.linalg.norm(after[i] - before[i])) for i in range(3))
                if change > 0 and raw_change > best_change:
                    best_change = raw_change
                    best_signature = signature
            if best_signature is not None:
                rows[token].append(best_signature)
                source_events += 1
                used_file = True

        cap.release()
        if used_file:
            source_files += 1

    samples = {
        token: normalize_rows(np.stack(values)).astype(np.float32)
        for token, values in rows.items()
        if values
    }
    return PrototypeBank(samples=samples, source_files=source_files, source_events=source_events)


def _classify_signature(signature: np.ndarray, bank: PrototypeBank):
    if not bank.samples:
        return None, 0.0, {}
    scores: dict[str, float] = {}
    for token, samples in bank.samples.items():
        similarities = samples @ signature
        top = np.sort(similarities)[-min(3, len(similarities)):]
        scores[token] = float(np.mean(top))
    ordered = sorted(scores.items(), key=lambda row: row[1], reverse=True)
    best_token, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = best_score - second_score
    score_term = float(np.clip((best_score - 0.42) / 0.30, 0.0, 1.0))
    margin_term = float(np.clip((margin - 0.015) / 0.12, 0.0, 1.0))
    confidence = 0.68 * score_term + 0.32 * margin_term
    if best_score < 0.48 or margin < 0.025:
        return None, confidence, scores
    return best_token, float(np.clip(confidence, 0.0, 1.0)), scores


def _propagate_cluster_semantics(events: list[dict]) -> None:
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        by_cluster[int(event["cluster"])].append(event)

    for rows in by_cluster.values():
        known = [row for row in rows if row.get("semantic") and row.get("semantic_confidence", 0.0) >= 0.62]
        if not known:
            continue
        counts = Counter(str(row["semantic"]) for row in known)
        token, count = counts.most_common(1)[0]
        if count / len(known) < 0.75:
            continue
        base_conf = float(np.median([row["semantic_confidence"] for row in known if row["semantic"] == token]))
        for row in rows:
            if row.get("semantic") is None and int(row.get("recurrence_support", 1)) >= 2:
                row["semantic"] = token
                row["semantic_confidence"] = round(min(0.82, base_conf * 0.82), 4)
                row["semantic_source"] = "recurrence-propagated"


def _loop_template(events: list[dict], boundaries: list[int]) -> list[dict]:
    if len(boundaries) < 3:
        return []
    rows: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for event in events:
        sample_index = int(event["sample_index"])
        for loop_index in range(len(boundaries) - 1):
            left, right = boundaries[loop_index], boundaries[loop_index + 1]
            if left <= sample_index < right:
                phase = (sample_index - left) / max(1, right - left)
                phase_bin = int(round(phase / 0.035))
                rows[(int(event["cluster"]), phase_bin)].append({**event, "loop": loop_index, "phase": phase})
                break

    template = []
    loop_count = len(boundaries) - 1
    for (cluster, _), occurrences in rows.items():
        distinct_loops = sorted({int(row["loop"]) for row in occurrences})
        if len(distinct_loops) < 2:
            continue
        phases = [float(row["phase"]) for row in occurrences]
        semantics = [str(row["semantic"]) for row in occurrences if row.get("semantic")]
        semantic = Counter(semantics).most_common(1)[0][0] if semantics else None
        template.append({
            "cluster": cluster,
            "phase_median": round(float(np.median(phases)), 5),
            "support_loops": len(distinct_loops),
            "loop_count": loop_count,
            "semantic": semantic,
            "occurrence_ms": [int(row["ms"]) for row in occurrences],
        })
    template.sort(key=lambda row: row["phase_median"])
    return template


def _merge_review_windows(items: list[dict], max_items: int) -> list[dict]:
    if not items:
        return []
    items.sort(key=lambda row: (int(row["start_ms"]), int(row["end_ms"])))
    merged: list[dict] = []
    for item in items:
        if merged and int(item["start_ms"]) <= int(merged[-1]["end_ms"]) + 100:
            merged[-1]["end_ms"] = max(int(merged[-1]["end_ms"]), int(item["end_ms"]))
            merged[-1]["reasons"] = sorted(set(merged[-1]["reasons"] + item["reasons"]))
            merged[-1]["priority"] = max(float(merged[-1]["priority"]), float(item["priority"]))
        else:
            merged.append(dict(item))
    merged.sort(key=lambda row: float(row["priority"]), reverse=True)
    return merged[:max_items]


def _write_timeline(
    path: Path,
    video: Path,
    sampled: SampledVideo,
    events: list[dict],
    start_slot: int,
    semantic_threshold: float,
) -> int:
    timeline = Timeline(
        video=video.name,
        fps=sampled.source_fps,
        start_slot=start_slot,
        source_note="auto_extract high-confidence semantic events; verify review.json before compiling",
    )
    current_slot = start_slot
    count = 0
    for row in sorted(events, key=lambda item: int(item["ms"])):
        token = row.get("semantic")
        confidence = float(row.get("semantic_confidence", 0.0))
        if token not in SEMANTIC_TO_TOKEN.values() or confidence < semantic_threshold:
            continue
        timeline.events.append(AxisVideoEvent(
            frame=int(row["frame"]),
            ms=int(row["ms"]),
            action=str(token),
            slot=current_slot,
            note=f"auto conf={confidence:.2f} cluster={row['cluster']}",
        ))
        count += 1
        if token in SWITCH_TOKENS:
            current_slot = SWITCH_TOKENS[str(token)]
    timeline.save(path)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Automatically extract recurring axis structure from an UP video")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--analysis-fps", type=float, default=0.0,
                        help="0=use source FPS up to 60; event timing keeps original frame granularity")
    parser.add_argument("--start-slot", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--min-loop-s", type=float, default=8.0)
    parser.add_argument("--max-loop-s", type=float, default=45.0)
    parser.add_argument("--prototype-root", action="append", default=None,
                        help="training_data/motion root; may be repeated")
    parser.add_argument("--no-self-prototypes", action="store_true")
    parser.add_argument("--semantic-threshold", type=float, default=0.66)
    parser.add_argument("--max-review", type=int, default=18)
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    prefix = Path(args.out_prefix).expanduser().resolve() if args.out_prefix else video.with_suffix("")

    print(f"[1/5] scan video: {video}")
    sampled = _sample_video(video, args.analysis_fps)
    print(
        f"  source={sampled.source_fps:.3f}fps analysis={sampled.analysis_fps:.3f}fps "
        f"samples={len(sampled.frame_indexes)}"
    )

    body_delta = _delta_norms(sampled.body)
    party_delta = _delta_norms(sampled.party)
    ability_delta = _delta_norms(sampled.ability)
    raw_activity = 0.48 * body_delta + 0.22 * party_delta + 0.30 * ability_delta
    smooth_radius = max(1, int(round(sampled.analysis_fps * 0.025)))
    activity = smooth_signal(raw_activity, radius=smooth_radius)
    activity_z = robust_zscore(activity)
    party_z = robust_zscore(party_delta)

    print("[2/5] discover repeated rotation")
    period, boundary_rows, _loop_stride, loop_fps = _loop_detection(
        sampled, args.min_loop_s, args.max_loop_s)
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
    peak_indexes = detect_peaks(
        activity,
        min_distance=min_distance,
        threshold_z=1.9,
        max_peaks=max(32, int((sampled.times_ms[-1] / 1000.0) * 5.0)),
    )
    peak_indexes = [index for index in peak_indexes if analysis_left <= index <= analysis_right]
    signatures = _event_signatures(sampled, peak_indexes)
    clusters = cluster_signatures(signatures, similarity_threshold=0.72)
    recurrence = loop_support(
        np.asarray(peak_indexes, dtype=np.int32),
        clusters.labels,
        boundaries,
        phase_tolerance=0.06,
    )
    print(f"  event candidates={len(peak_indexes)} visual clusters={len(clusters.counts)}")

    print("[3/5] build optional self-telemetry visual prototypes")
    if args.no_self_prototypes:
        bank = PrototypeBank(samples={}, source_files=0, source_events=0)
        roots = []
    else:
        roots = _prototype_roots(args.prototype_root)
        bank = _build_prototype_bank(roots)
    print(f"  prototype roots={len(roots)} files={bank.source_files} events={bank.source_events} counts={bank.counts()}")

    events: list[dict] = []
    for pos, sample_index in enumerate(peak_indexes):
        token, semantic_conf, scores = _classify_signature(signatures[pos], bank)
        swap_candidate = bool(
            party_z[sample_index] >= 2.0
            and party_delta[sample_index] >= ability_delta[sample_index] * 1.08
        )
        event = {
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
            "prototype_scores": {key: round(value, 4) for key, value in scores.items()},
        }
        events.append(event)

    _propagate_cluster_semantics(events)
    for event in events:
        event["confidence"] = round(confidence_from_signals(
            float(event["activity_z"]),
            int(event["recurrence_support"]),
            float(event.get("semantic_confidence") or 0.0),
        ), 4)

    template = _loop_template(events, boundaries)

    print("[4/5] create low-touch review list and timeline")
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
        if event.get("semantic") and float(event.get("semantic_confidence", 0.0)) < args.semantic_threshold:
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
    review = _merge_review_windows(review_items, max_items=max(1, args.max_review))

    analysis_path = Path(str(prefix) + ".auto_analysis.json")
    loops_path = Path(str(prefix) + ".loops.json")
    review_path = Path(str(prefix) + ".review.json")
    timeline_path = Path(str(prefix) + ".auto_axis_timeline.json")
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

    timeline_count = _write_timeline(
        timeline_path,
        video,
        sampled,
        events,
        args.start_slot,
        args.semantic_threshold,
    )

    analysis_payload = {
        "schema": 1,
        "video": str(video),
        "source_fps": sampled.source_fps,
        "analysis_fps": sampled.analysis_fps,
        "frame_time_ms": round(1000.0 / sampled.source_fps, 4),
        "prototype_roots": [str(path) for path in roots],
        "prototype_counts": bank.counts(),
        "loop": loop_payload,
        "events": events,
        "review_count": len(review),
        "timeline_event_count": timeline_count,
        "notes": [
            "30fps source has about 33.3ms visual-frame quantization; the extractor does not invent missing source frames.",
            "Self telemetry prototypes teach visual semantics only; expert timing comes from the UP video and recurrence alignment.",
            "Unknown recurring clusters are intentionally kept as unknown instead of being forced into A/E/Q/R labels.",
        ],
    }
    _json_dump(analysis_path, analysis_payload)

    high_recurrence = sum(1 for row in events if int(row["recurrence_support"]) >= 2)
    semantic_events = sum(1 for row in events if row.get("semantic"))
    report = "\n".join([
        f"video: {video}",
        f"source_fps: {sampled.source_fps:.3f} (frame≈{1000.0 / sampled.source_fps:.2f}ms)",
        f"analysis_fps: {sampled.analysis_fps:.3f}",
        f"event_candidates: {len(events)}",
        f"recurring_candidates: {high_recurrence}",
        f"semantic_candidates: {semantic_events}",
        f"timeline_high_confidence_events: {timeline_count}",
        f"review_windows: {len(review)}",
        f"loop_period_ms: {loop_payload['period_ms']}",
        f"loop_score: {loop_payload['period_score']}",
        "",
        "outputs:",
        f"  {analysis_path}",
        f"  {loops_path}",
        f"  {review_path}",
        f"  {timeline_path}",
        "",
        "重要：auto_axis_timeline 只包含高置信度且已能映射到现有轴 token 的事件。",
        "先看 analysis/review；不要因为 timeline 文件存在就直接当成最终实机轴。",
    ])
    report_path.write_text(report + "\n", encoding="utf-8")

    print("[5/5] done")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
