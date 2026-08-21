from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import phase_training_legacy as core


DEFAULT_OUTLIER_SCORE_THRESHOLD = 2.20
DEFAULT_OUTLIER_MIN_IMPROVEMENT = 0.008
DEFAULT_OUTLIER_PROBE_EPOCHS = 8
DEFAULT_OUTLIER_MAX_FRACTION = 0.35
DEFAULT_OUTLIER_MAX_PROBES = 6


def quarantine_path(root: Path) -> Path:
    return root / "quarantine" / "cycles.jsonl"


def load_quarantine(root: Path) -> dict[str, dict]:
    path = quarantine_path(root)
    records: dict[str, dict] = {}
    if not path.is_file():
        return records
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"quarantine metadata damaged at line {line_no}: {exc}") from exc
        cycle = row.get("cycle")
        if isinstance(cycle, str) and cycle:
            records[cycle] = row
    return records


def split_existing_quarantine(root: Path, items, reconsider: bool):
    records = load_quarantine(root)
    if reconsider or not records:
        return list(items), [], records

    active = []
    excluded = []
    for item in items:
        cycle = str(item["cycle"])
        if cycle in records:
            excluded.append(item)
        else:
            active.append(item)
    return active, excluded, records


def save_quarantine(root: Path, records: dict[str, dict]) -> Path:
    path = quarantine_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for cycle in sorted(records):
            stream.write(json.dumps(records[cycle], ensure_ascii=False) + "\n")
    return path


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


def _frame_vector(frame: np.ndarray, size: int = 20) -> np.ndarray:
    height, width = frame.shape[:2]
    y0, y1 = int(height * 0.08), int(height * 0.94)
    x0, x1 = int(width * 0.08), int(width * 0.92)
    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    gray_u8 = np.asarray(gray, dtype=np.uint8)
    gray_f = gray_u8.astype(np.float32) / 255.0
    gray_f = (gray_f - float(gray_f.mean())) / max(float(gray_f.std()), 1e-4)
    edge = cv2.Canny(gray_u8, 60, 150).astype(np.float32) / 255.0
    return np.concatenate((gray_f.reshape(-1) * 0.45, edge.reshape(-1) * 0.55))


def _phase_sensitive_descriptor(root: Path, item, phase_bins: int = 24):
    with np.load(root / item["cycle"]) as data:
        frames = data["frames"].copy()
    if len(frames) < 8:
        raise RuntimeError(f"cycle too short for quality analysis: {item['cycle']}")

    positions = np.linspace(0, len(frames), phase_bins, endpoint=False)
    indexes = np.clip(np.floor(positions).astype(np.int32), 0, len(frames) - 1)
    sequence = np.stack([_frame_vector(frames[index]) for index in indexes])

    # 以运动变化为主，避免固定背景主导相似度；不做 FFT，因此保留 cycle 起点相位。
    motion = np.diff(sequence, axis=0, prepend=sequence[:1])
    descriptor = _normalize(motion.reshape(-1))

    first = _normalize(np.mean(sequence[:2], axis=0))
    last = _normalize(np.mean(sequence[-2:], axis=0))
    seam_distance = float(np.clip(1.0 - np.dot(first, last), 0.0, 2.0))
    return descriptor, seam_distance


def _pairwise_distance(values: np.ndarray) -> np.ndarray:
    values = np.stack([_normalize(row) for row in values])
    return np.clip(1.0 - values @ values.T, 0.0, 2.0)


def _robust_positive_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) <= 2:
        return np.zeros_like(values)
    median = float(np.median(values))
    absolute = np.abs(values - median)
    mad = float(np.median(absolute))
    if mad < 1e-6:
        q75, q25 = np.percentile(values, [75, 25])
        scale = max(float(q75 - q25) / 1.349, 1e-4)
    else:
        scale = max(1.4826 * mad, 1e-4)
    return np.clip((values - median) / scale, 0.0, 12.0)


def cycle_quality_scores(root: Path, members):
    if not members:
        return []

    invariant = np.stack([core._cycle_motion_descriptor(root, item) for item in members])
    phase_rows = []
    seams = []
    for item in members:
        descriptor, seam = _phase_sensitive_descriptor(root, item)
        phase_rows.append(descriptor)
        seams.append(seam)
    phase = np.stack(phase_rows)

    invariant_distance = _pairwise_distance(invariant)
    phase_distance = _pairwise_distance(phase)
    if len(members) == 1:
        invariant_med = np.zeros(1, dtype=np.float32)
        phase_med = np.zeros(1, dtype=np.float32)
    else:
        invariant_med = np.asarray([
            np.median(np.delete(invariant_distance[index], index))
            for index in range(len(members))
        ], dtype=np.float32)
        phase_med = np.asarray([
            np.median(np.delete(phase_distance[index], index))
            for index in range(len(members))
        ], dtype=np.float32)

    durations = np.asarray(
        [max(1e-4, float(item.get("duration_s", 0.0))) for item in members],
        dtype=np.float32,
    )
    median_duration = max(float(np.median(durations)), 1e-4)
    duration_deviation = np.abs(np.log(durations / median_duration))
    seams = np.asarray(seams, dtype=np.float32)

    z_invariant = _robust_positive_z(invariant_med)
    z_phase = _robust_positive_z(phase_med)
    z_duration = _robust_positive_z(duration_deviation)
    z_seam = _robust_positive_z(seams)

    # phase-sensitive trajectory is most useful for catching a boundary shifted cycle.
    combined = (
        0.45 * z_phase
        + 0.25 * z_invariant
        + 0.20 * z_duration
        + 0.10 * z_seam
    )

    rows = []
    for index, item in enumerate(members):
        rows.append({
            "index": index,
            "cycle": str(item["cycle"]),
            "video": str(item.get("video", "")),
            "score": float(combined[index]),
            "phase_distance": float(phase_med[index]),
            "invariant_distance": float(invariant_med[index]),
            "duration_deviation": float(duration_deviation[index]),
            "seam_distance": float(seams[index]),
        })
    return rows


def _hash_order(value: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _fixed_probe_split(members, remove_indexes: set[int], seed: int):
    eligible = [index for index in range(len(members)) if index not in remove_indexes]
    if len(eligible) < 3:
        return None

    ordered = sorted(
        eligible,
        key=lambda index: _hash_order(str(members[index]["cycle"]), seed),
    )
    val_count = max(1, round(len(eligible) * 0.20))
    val_count = min(val_count, max(1, len(eligible) - 2))
    val_indexes = set(ordered[:val_count])

    baseline_train = [
        index for index in range(len(members))
        if index not in val_indexes
    ]
    removed_train = [
        index for index in baseline_train
        if index not in remove_indexes
    ]
    if len(removed_train) < 2:
        return None
    return baseline_train, removed_train, sorted(val_indexes)


def _probe_error(root: Path, train_items, val_items, args, device, seed: int, epochs: int,
                 dataset_cls=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 与正式训练共用同一个 dataset_cls（例如相位对齐数据集），保证 leave-out
    # 质检和正式训练看到一致的目标标签；显式传参取代旧版的模块级猴补丁。
    dataset_cls = dataset_cls or core.CycleWindowDataset
    stride = max(3, int(args.stride))
    train_ds = dataset_cls(
        root, train_items, args.window, stride, augment=False
    )
    val_ds = dataset_cls(
        root, val_items, args.window, stride, augment=False
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        return float("nan")

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = core.PhaseNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for _ in range(max(2, epochs)):
        model.train()
        for clips, phases in train_loader:
            clips, phases = clips.to(device), phases.to(device)
            target = core.phase_vector(phases)
            predicted = model(clips)
            loss = loss_fn(predicted, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return core.evaluate_phase(model, val_loader, device)


def _probe_removal(root: Path, members, removal: set[int], args, device, seed: int,
                   dataset_cls=None):
    split = _fixed_probe_split(members, removal, seed)
    if split is None:
        return None
    baseline_indexes, removed_indexes, val_indexes = split
    baseline_items = [members[index] for index in baseline_indexes]
    removed_items = [members[index] for index in removed_indexes]
    val_items = [members[index] for index in val_indexes]

    baseline = _probe_error(
        root,
        baseline_items,
        val_items,
        args,
        device,
        seed,
        args.outlier_probe_epochs,
        dataset_cls=dataset_cls,
    )
    removed = _probe_error(
        root,
        removed_items,
        val_items,
        args,
        device,
        seed,
        args.outlier_probe_epochs,
        dataset_cls=dataset_cls,
    )
    if not np.isfinite(baseline) or not np.isfinite(removed):
        return None
    return {
        "baseline_error": float(baseline),
        "removed_error": float(removed),
        "improvement": float(baseline - removed),
        "validation_cycles": [str(members[index]["cycle"]) for index in val_indexes],
    }


def _candidate_groups(members, scores, max_remove: int, max_probes: int, score_threshold: float):
    by_video: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(members):
        video = str(item.get("video", ""))
        if video:
            by_video[video].append(index)

    candidates = []
    for video, indexes in by_video.items():
        if 2 <= len(indexes) <= max_remove:
            median_score = float(np.median([scores[index]["score"] for index in indexes]))
            candidates.append({
                "kind": "source_video",
                "indexes": frozenset(indexes),
                "label": Path(video).name,
                "score": median_score,
            })

    ranked = sorted(scores, key=lambda row: row["score"], reverse=True)
    for row in ranked:
        if row["score"] < score_threshold * 0.75 and candidates:
            continue
        candidates.append({
            "kind": "single_cycle",
            "indexes": frozenset({int(row["index"])}),
            "label": str(row["cycle"]),
            "score": float(row["score"]),
        })

    # Keep unique removal sets and prefer source-video candidates on ties.
    unique = {}
    for candidate in candidates:
        key = tuple(sorted(candidate["indexes"]))
        current = unique.get(key)
        if current is None or (
            candidate["kind"] == "source_video"
            and current["kind"] != "source_video"
        ):
            unique[key] = candidate

    ordered = sorted(
        unique.values(),
        key=lambda row: (
            0 if row["kind"] == "source_video" else 1,
            -row["score"],
            -len(row["indexes"]),
        ),
    )
    return ordered[:max_probes]


def filter_mode_cycles(root: Path, members, args, device, mode_name: str, dataset_cls=None):
    if len(members) < 5:
        return list(members), [], {
            "mode": mode_name,
            "candidate_count": 0,
            "reason": "fewer than 5 cycles; skip automatic quarantine",
        }

    max_remove = min(
        len(members) - 3,
        max(1, int(math.floor(len(members) * args.outlier_max_fraction))),
    )
    if max_remove <= 0:
        return list(members), [], {
            "mode": mode_name,
            "candidate_count": 0,
            "reason": "not enough spare cycles for quarantine",
        }

    scores = cycle_quality_scores(root, members)
    candidates = _candidate_groups(
        members,
        scores,
        max_remove=max_remove,
        max_probes=args.outlier_max_probes,
        score_threshold=args.outlier_score_threshold,
    )

    print(
        f"[{mode_name}] cycle quality scan: cycles={len(members)} "
        f"probe_candidates={len(candidates)} max_quarantine={max_remove}"
    )
    score_by_index = {int(row["index"]): row for row in scores}
    probe_rows = []

    for probe_index, candidate in enumerate(candidates):
        indexes = set(int(value) for value in candidate["indexes"])
        if len(indexes) > max_remove:
            continue
        seed = args.seed + 5000 + probe_index * 97 + len(members)
        result = _probe_removal(root, members, indexes, args, device, seed,
                                dataset_cls=dataset_cls)
        if result is None:
            continue
        median_score = float(np.median([
            score_by_index[index]["score"] for index in indexes
        ]))
        score_support = median_score >= args.outlier_score_threshold * 0.55
        required_improvement = args.outlier_min_improvement
        if not score_support:
            required_improvement *= 1.50
        accepted = bool(result["improvement"] >= required_improvement)
        row = {
            **candidate,
            **result,
            "median_outlier_score": median_score,
            "required_improvement": required_improvement,
            "accepted": accepted,
        }
        probe_rows.append(row)
        print(
            f"[{mode_name}] leave-out {candidate['kind']} {candidate['label']}: "
            f"baseline={result['baseline_error']:.4f} "
            f"without={result['removed_error']:.4f} "
            f"improvement={result['improvement']:+.4f} "
            f"score={median_score:.2f} "
            f"{'ACCEPT' if accepted else 'keep'}"
        )

    accepted = sorted(
        [row for row in probe_rows if row["accepted"]],
        key=lambda row: (
            row["improvement"],
            row["median_outlier_score"],
            len(row["indexes"]),
        ),
        reverse=True,
    )

    selected: set[int] = set()
    selected_rows = []
    for row in accepted:
        indexes = set(int(value) for value in row["indexes"])
        if selected.intersection(indexes):
            continue
        if len(selected | indexes) > max_remove:
            continue
        selected.update(indexes)
        selected_rows.append(row)

    if not selected:
        return list(members), [], {
            "mode": mode_name,
            "candidate_count": len(candidates),
            "probes": _jsonable_probe_rows(probe_rows),
            "quarantined_count": 0,
        }

    kept = [item for index, item in enumerate(members) if index not in selected]
    quarantine_records = []
    for index in sorted(selected):
        quality = score_by_index[index]
        supporting = [
            row for row in selected_rows if index in set(int(v) for v in row["indexes"])
        ]
        best = max(supporting, key=lambda row: row["improvement"])
        record = {
            "cycle": str(members[index]["cycle"]),
            "video": str(members[index].get("video", "")),
            "annotation": str(members[index].get("annotation", "")),
            "mode_at_quarantine": mode_name,
            "reason": "leave-out-improves-phase-quality",
            "outlier_score": float(quality["score"]),
            "phase_distance": float(quality["phase_distance"]),
            "invariant_distance": float(quality["invariant_distance"]),
            "duration_deviation": float(quality["duration_deviation"]),
            "seam_distance": float(quality["seam_distance"]),
            "probe_kind": best["kind"],
            "probe_label": best["label"],
            "baseline_probe_error": float(best["baseline_error"]),
            "removed_probe_error": float(best["removed_error"]),
            "probe_improvement": float(best["improvement"]),
        }
        quarantine_records.append(record)
        print(
            f"[{mode_name}] QUARANTINE {record['cycle']} "
            f"score={record['outlier_score']:.2f} "
            f"probe_improvement={record['probe_improvement']:+.4f}"
        )

    return kept, quarantine_records, {
        "mode": mode_name,
        "candidate_count": len(candidates),
        "probes": _jsonable_probe_rows(probe_rows),
        "quarantined_count": len(quarantine_records),
        "kept_count": len(kept),
    }


def _jsonable_probe_rows(rows):
    output = []
    for row in rows:
        output.append({
            "kind": row["kind"],
            "label": row["label"],
            "indexes": sorted(int(value) for value in row["indexes"]),
            "baseline_error": float(row["baseline_error"]),
            "removed_error": float(row["removed_error"]),
            "improvement": float(row["improvement"]),
            "median_outlier_score": float(row["median_outlier_score"]),
            "required_improvement": float(row["required_improvement"]),
            "accepted": bool(row["accepted"]),
        })
    return output


def merge_quarantine_records(
    root: Path,
    existing: dict[str, dict],
    new_records: list[dict],
    reconsider: bool,
):
    merged = {} if reconsider else dict(existing)
    for record in new_records:
        merged[str(record["cycle"])] = record
    path = save_quarantine(root, merged)
    return merged, path
