from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import phase_training_legacy as core
from common import character_root, write_json


DEFAULT_MAX_PHASE_ERROR = 0.08
DEFAULT_MIN_CLASSIFIER_ACCURACY = 0.85
TRUSTED_HISTORY_MIN_CYCLES = 5
NEAR_DURATION_RATIO = 0.065


def _load_previous_index(root: Path) -> dict | None:
    path = root / "modes" / "index.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _history_maps(previous_index: dict | None):
    assignments = {}
    modes = {}
    if not previous_index:
        return assignments, modes
    for row in previous_index.get("assignments", []):
        cycle = row.get("cycle")
        mode = row.get("mode")
        if isinstance(cycle, str) and isinstance(mode, int):
            assignments[cycle] = mode
    for mode in previous_index.get("modes", []):
        if isinstance(mode, dict) and isinstance(mode.get("id"), int):
            modes[int(mode["id"])] = mode
    return assignments, modes


def _trusted_history(previous_index: dict | None, max_phase_error: float):
    assignments, modes = _history_maps(previous_index)
    trusted = set()
    for mode_id, mode in modes.items():
        try:
            error = float(mode.get("val_circular_error"))
        except (TypeError, ValueError):
            continue
        if int(mode.get("cycle_count", 0) or 0) >= TRUSTED_HISTORY_MIN_CYCLES and error <= max_phase_error:
            trusted.add(mode_id)
    return assignments, modes, trusted


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


def _compact(labels: np.ndarray) -> np.ndarray:
    unique = sorted(set(int(value) for value in labels))
    remap = {old: new for new, old in enumerate(unique)}
    return np.asarray([remap[int(value)] for value in labels], dtype=np.int32)


def _cluster_stats(items, labels: np.ndarray, descriptors: np.ndarray):
    stats = {}
    for cluster in sorted(set(int(value) for value in labels)):
        indexes = np.where(labels == cluster)[0]
        members = descriptors[indexes]
        center = _normalize(members.mean(axis=0))
        distances = 1.0 - members @ center
        durations = [float(items[int(index)].get("duration_s", 0.0)) for index in indexes]
        stats[cluster] = {
            "indexes": indexes,
            "count": int(len(indexes)),
            "center": center,
            "spread": float(np.median(distances)) if len(distances) else 0.0,
            "median_duration": float(np.median(durations)) if durations else 0.0,
        }
    return stats


def _trusted_anchor(items, indexes, assignments, trusted_modes):
    votes = Counter()
    for index in indexes:
        old_mode = assignments.get(str(items[int(index)]["cycle"]))
        if old_mode in trusted_modes:
            votes[int(old_mode)] += 1
    if not votes:
        return None
    mode_id, count = votes.most_common(1)[0]
    required = max(2, math.ceil(len(indexes) * 0.4))
    return mode_id if count >= required else None


def _merge_near_duplicate_modes(
    items,
    labels: np.ndarray,
    descriptors: np.ndarray,
    previous_index: dict | None,
    max_phase_error: float,
):
    assignments, _modes, trusted_modes = _trusted_history(previous_index, max_phase_error)
    labels = _compact(labels)
    decisions = []

    while len(set(int(value) for value in labels)) > 1:
        stats = _cluster_stats(items, labels, descriptors)
        best = None
        clusters = sorted(stats)
        for pos, left in enumerate(clusters):
            for right in clusters[pos + 1:]:
                a, b = stats[left], stats[right]
                anchor_a = _trusted_anchor(items, a["indexes"], assignments, trusted_modes)
                anchor_b = _trusted_anchor(items, b["indexes"], assignments, trusted_modes)
                if anchor_a is not None and anchor_b is not None and anchor_a != anchor_b:
                    continue

                d1, d2 = a["median_duration"], b["median_duration"]
                duration_gap = abs(d1 - d2) / max(d1, d2, 1e-6)
                if duration_gap > NEAR_DURATION_RATIO:
                    continue

                center_distance = float(1.0 - np.dot(a["center"], b["center"]))
                small = min(a["count"], b["count"]) <= 3
                base_limit = 0.18 if small else 0.11
                adaptive = 2.2 * max(a["spread"], b["spread"]) + 0.035
                limit = min(0.22, max(base_limit, adaptive))
                if center_distance > limit:
                    continue

                score = center_distance + duration_gap * 0.35
                row = (score, left, right, duration_gap, center_distance, limit, anchor_a, anchor_b)
                if best is None or row[0] < best[0]:
                    best = row

        if best is None:
            break

        _, left, right, duration_gap, center_distance, limit, anchor_a, anchor_b = best
        left_count = int(np.sum(labels == left))
        right_count = int(np.sum(labels == right))
        if anchor_b is not None and anchor_a is None:
            keep, remove = right, left
        elif anchor_a is not None and anchor_b is None:
            keep, remove = left, right
        elif right_count > left_count:
            keep, remove = right, left
        else:
            keep, remove = left, right
        labels[labels == remove] = keep
        labels = _compact(labels)
        decisions.append({
            "duration_gap_ratio": duration_gap,
            "descriptor_distance": center_distance,
            "distance_limit": limit,
            "counts_before": [left_count, right_count],
            "trusted_anchors": [anchor_a, anchor_b],
        })

    return labels, decisions


def _sort_modes_by_duration(items, labels: np.ndarray) -> np.ndarray:
    stats = _cluster_stats(items, labels, np.zeros((len(items), 1), dtype=np.float32))
    order = sorted(
        stats,
        key=lambda cluster: (
            stats[cluster]["median_duration"],
            min(str(items[int(index)]["cycle"]) for index in stats[cluster]["indexes"]),
        ),
    )
    remap = {old: new for new, old in enumerate(order)}
    return np.asarray([remap[int(value)] for value in labels], dtype=np.int32)


def _stable_ids(items, labels: np.ndarray, previous_index: dict | None):
    assignments, previous_modes = _history_maps(previous_index)
    stable_ids = []
    used = set()
    for mode in range(len(set(int(value) for value in labels))):
        indexes = np.where(labels == mode)[0]
        votes = Counter()
        for index in indexes:
            old_mode = assignments.get(str(items[int(index)]["cycle"]))
            old_meta = previous_modes.get(int(old_mode), {}) if old_mode is not None else {}
            stable_id = old_meta.get("stable_id")
            if isinstance(stable_id, str) and stable_id:
                votes[stable_id] += 1
        stable_id = votes.most_common(1)[0][0] if votes else None
        if not stable_id or stable_id in used:
            earliest = min(str(items[int(index)]["cycle"]) for index in indexes)
            stable_id = "motion_" + hashlib.sha1(earliest.encode("utf-8")).hexdigest()[:10]
        used.add(stable_id)
        stable_ids.append(stable_id)
    return stable_ids


def _augment_mode_index(mode_index: dict, stable_ids: list[str], clustering: dict):
    mode_index["schema"] = 2
    mode_index["clustering"] = clustering
    for mode in mode_index.get("modes", []):
        mode_id = int(mode["id"])
        mode["stable_id"] = stable_ids[mode_id]
        mode["candidate_phase_model"] = str(
            Path("modes") / f"mode_{mode_id}" / "phase_model.candidate.pt"
        )
    for row in mode_index.get("assignments", []):
        mode_id = int(row["mode"])
        row["stable_id"] = stable_ids[mode_id]


def _promote(candidate: Path, formal: Path):
    formal.parent.mkdir(parents=True, exist_ok=True)
    os.replace(candidate, formal)


def _quarantine(path: Path):
    if not path.is_file():
        return
    target = path.with_name(f"{path.stem}.previous_unverified{path.suffix}")
    try:
        if target.exists():
            target.unlink()
        os.replace(path, target)
        print(f"quarantined previous unverified model: {target}")
    except OSError as exc:
        print(f"WARN failed to quarantine {path}: {exc}")


def _save_router(root: Path, args, mode_index: dict, router_ready: bool):
    payload = {
        "architecture": "MultiModePhaseRouter-v2-stable",
        "character": args.character,
        "router_ready": router_ready,
        "mode_index": "modes/index.json",
        "mode_classifier": mode_index.get("classifier"),
        "modes": [
            {
                "id": mode["id"],
                "stable_id": mode.get("stable_id"),
                "phase_model": mode.get("phase_model"),
                "candidate_phase_model": mode.get("candidate_phase_model"),
                "cycle_count": mode["cycle_count"],
                "quality": mode.get("quality"),
                "val_circular_error": mode.get("val_circular_error"),
            }
            for mode in mode_index["modes"]
        ],
    }
    candidate = root / "models" / "phase_model.candidate.pt"
    formal = root / "models" / "phase_model.pt"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, candidate)
    if router_ready:
        _promote(candidate, formal)
        print(f"router READY -> {formal}")
    else:
        _quarantine(formal)
        print(f"router NOT READY; candidate kept at {candidate}")


def main() -> int:
    parser = core.argparse.ArgumentParser(
        description="Stable multimode phase training with duplicate-mode merge and quality gates"
    )
    parser.add_argument("--character", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=227)
    parser.add_argument("--max-modes", type=int, default=4)
    parser.add_argument("--max-phase-error", type=float, default=DEFAULT_MAX_PHASE_ERROR)
    parser.add_argument("--min-classifier-accuracy", type=float, default=DEFAULT_MIN_CLASSIFIER_ACCURACY)
    args = parser.parse_args()

    core.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = character_root(args.character)
    items = core.load_manifest(root)
    if len(items) < 3:
        raise SystemExit("至少准备 3 个完整平A cycle。")

    previous_index = _load_previous_index(root)
    labels, clustering = core.discover_motion_modes(
        root, items, seed=args.seed, max_modes=max(1, args.max_modes)
    )
    initial_mode_count = int(clustering.get("mode_count", 1))
    descriptors = np.stack([core._cycle_motion_descriptor(root, item) for item in items])
    labels, merge_decisions = _merge_near_duplicate_modes(
        items, labels, descriptors, previous_index, args.max_phase_error
    )
    labels = _sort_modes_by_duration(items, labels)
    final_mode_count = len(set(int(value) for value in labels))
    clustering = dict(clustering)
    clustering["initial_mode_count"] = initial_mode_count
    clustering["mode_count"] = final_mode_count
    clustering["counts"] = [int(np.sum(labels == mode)) for mode in range(final_mode_count)]
    clustering["merge_decisions"] = merge_decisions
    clustering["stability_layer"] = "near-duration+motion+trusted-history-v1"

    stable_ids = _stable_ids(items, labels, previous_index)
    mode_items, mode_index = core.build_mode_manifests(root, items, labels, clustering)
    _augment_mode_index(mode_index, stable_ids, clustering)

    print(
        f"stable motion modes={final_mode_count} initial={initial_mode_count} "
        f"counts={[mode['cycle_count'] for mode in mode_index['modes']]}"
    )
    for decision in merge_decisions:
        print(
            "mode merge: near-duplicate -> "
            f"duration_gap={decision['duration_gap_ratio']:.3f} "
            f"descriptor_distance={decision['descriptor_distance']:.3f} "
            f"limit={decision['distance_limit']:.3f}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_modes_ready = True
    for mode, members in enumerate(mode_items):
        mode_dir = root / "modes" / f"mode_{mode}"
        candidate = mode_dir / "phase_model.candidate.pt"
        formal = mode_dir / "phase_model.pt"
        if len(mode_items) == 1:
            candidate = root / "models" / "phase_model.candidate.pt"
            formal = root / "models" / "phase_model.pt"

        phase_error = core.train_phase_model(
            root, members, args, device, candidate, f"mode_{mode}"
        )
        ready = bool(np.isfinite(phase_error) and phase_error <= args.max_phase_error)
        meta = mode_index["modes"][mode]
        meta["val_circular_error"] = phase_error
        meta["quality_threshold"] = args.max_phase_error
        meta["quality"] = "ready" if ready else "low_phase_confidence"
        if ready:
            _promote(candidate, formal)
            meta["phase_model"] = str(formal.relative_to(root))
            print(f"[mode_{mode}] quality PASS {phase_error:.4f}")
        else:
            all_modes_ready = False
            meta["phase_model"] = None
            print(
                f"[mode_{mode}] quality REJECT {phase_error:.4f} > {args.max_phase_error:.4f}; "
                "candidate kept, formal model not promoted"
            )

    classifier_ready = True
    if len(mode_items) > 1 and all_modes_ready:
        classifier = core.train_mode_classifier(root, mode_items, args, device)
        accuracy = float(classifier["val_accuracy"]) if classifier else -1.0
        classifier_ready = accuracy >= args.min_classifier_accuracy
        if not classifier_ready:
            _quarantine(root / "modes" / "mode_classifier.pt")
        mode_index["classifier"] = dict(classifier or {})
        mode_index["classifier"]["quality"] = "ready" if classifier_ready else "low_confidence"
        mode_index["classifier"]["quality_threshold"] = args.min_classifier_accuracy
    elif len(mode_items) > 1:
        classifier_ready = False
        _quarantine(root / "modes" / "mode_classifier.pt")
        mode_index["classifier"] = {
            "path": None,
            "val_accuracy": None,
            "quality": "blocked_by_phase_quality",
            "quality_threshold": args.min_classifier_accuracy,
        }
    else:
        mode_index["classifier"] = None

    router_ready = bool(all_modes_ready and classifier_ready)
    if len(mode_items) > 1:
        _save_router(root, args, mode_index, router_ready)
    elif not router_ready:
        _quarantine(root / "models" / "phase_model.pt")

    mode_index["router_ready"] = router_ready
    mode_index["quality_gate"] = {
        "max_phase_error": args.max_phase_error,
        "min_classifier_accuracy": args.min_classifier_accuracy,
        "all_modes_ready": all_modes_ready,
        "classifier_ready": classifier_ready,
    }
    write_json(root / "modes" / "index.json", mode_index)
    write_json(
        root / "models" / "router_status.json",
        {
            "character": args.character,
            "router_ready": router_ready,
            "mode_count": len(mode_items),
            "quality_gate": mode_index["quality_gate"],
        },
    )

    print("stable multimode training complete:")
    for mode in mode_index["modes"]:
        print(
            f"  {mode['name']}: cycles={mode['cycle_count']} "
            f"median_duration={mode['median_duration_s']:.3f}s "
            f"phase_error={mode.get('val_circular_error')} quality={mode.get('quality')} "
            f"stable_id={mode.get('stable_id')}"
        )
    if mode_index.get("classifier"):
        print(
            f"  mode classifier accuracy={mode_index['classifier'].get('val_accuracy')} "
            f"quality={mode_index['classifier'].get('quality')}"
        )
    print(f"  router_ready={router_ready}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
