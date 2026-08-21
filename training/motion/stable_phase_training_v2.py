from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import cycle_quality
import phase_training_legacy as core
import stable_phase_training as stable
from common import character_root, write_json


def _reset_training_seed(seed: int) -> None:
    core.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rewrite_filtered_manifests(
    root: Path,
    mode_items,
    mode_index: dict,
    quarantine_records: list[dict],
    reports: list[dict],
):
    quarantined = {str(row["cycle"]): row for row in quarantine_records}

    for mode, members in enumerate(mode_items):
        mode_dir = root / "modes" / f"mode_{mode}"
        manifest_path = mode_dir / "manifest.jsonl"
        core._write_manifest(manifest_path, members)

        meta = mode_index["modes"][mode]
        previous_count = int(meta.get("cycle_count", len(members)))
        meta["raw_cycle_count"] = previous_count
        meta["cycle_count"] = len(members)
        meta["quarantined_cycle_count"] = max(0, previous_count - len(members))
        durations = [float(item.get("duration_s", 0.0)) for item in members]
        meta["median_duration_s"] = float(np.median(durations)) if durations else 0.0
        meta["quality_screen"] = reports[mode]

    for assignment in mode_index.get("assignments", []):
        cycle = str(assignment.get("cycle", ""))
        record = quarantined.get(cycle)
        if record:
            assignment["quarantined"] = True
            assignment["quarantine_reason"] = record.get("reason")
        else:
            assignment["quarantined"] = False


def main() -> int:
    parser = core.argparse.ArgumentParser(
        description=(
            "Stable multimode phase training with mode merge, persistent cycle quarantine, "
            "leave-out quality probes, and formal-model gates"
        )
    )
    parser.add_argument("--character", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=227)
    parser.add_argument("--max-modes", type=int, default=4)
    parser.add_argument(
        "--max-phase-error",
        type=float,
        default=stable.DEFAULT_MAX_PHASE_ERROR,
    )
    parser.add_argument(
        "--min-classifier-accuracy",
        type=float,
        default=stable.DEFAULT_MIN_CLASSIFIER_ACCURACY,
    )
    parser.add_argument(
        "--outlier-score-threshold",
        type=float,
        default=cycle_quality.DEFAULT_OUTLIER_SCORE_THRESHOLD,
    )
    parser.add_argument(
        "--outlier-min-improvement",
        type=float,
        default=cycle_quality.DEFAULT_OUTLIER_MIN_IMPROVEMENT,
    )
    parser.add_argument(
        "--outlier-probe-epochs",
        type=int,
        default=cycle_quality.DEFAULT_OUTLIER_PROBE_EPOCHS,
    )
    parser.add_argument(
        "--outlier-max-fraction",
        type=float,
        default=cycle_quality.DEFAULT_OUTLIER_MAX_FRACTION,
    )
    parser.add_argument(
        "--outlier-max-probes",
        type=int,
        default=cycle_quality.DEFAULT_OUTLIER_MAX_PROBES,
    )
    parser.add_argument(
        "--reconsider-quarantine",
        action="store_true",
        help="temporarily re-admit previously quarantined cycles and rebuild quarantine decisions",
    )
    args = parser.parse_args()

    args.outlier_probe_epochs = max(2, int(args.outlier_probe_epochs))
    args.outlier_max_fraction = min(0.45, max(0.0, float(args.outlier_max_fraction)))
    args.outlier_max_probes = max(1, int(args.outlier_max_probes))
    args.outlier_min_improvement = max(0.0, float(args.outlier_min_improvement))
    args.outlier_score_threshold = max(0.5, float(args.outlier_score_threshold))

    core.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = character_root(args.character)
    all_items = core.load_manifest(root)
    if len(all_items) < 3:
        raise SystemExit("至少准备 3 个完整平A cycle。")

    active_items, preexcluded_items, existing_quarantine = cycle_quality.split_existing_quarantine(
        root,
        all_items,
        reconsider=args.reconsider_quarantine,
    )
    if preexcluded_items:
        print(
            f"persistent quarantine: excluded {len(preexcluded_items)} previously confirmed "
            f"bad cycles; active={len(active_items)}"
        )
        for item in preexcluded_items:
            print(f"  QUARANTINED(previous) {item['cycle']}")
    if len(active_items) < 3:
        raise SystemExit(
            "持久 quarantine 后可用 cycle 少于 3 个。请补录数据，或使用 "
            "--reconsider-quarantine 重新审查。"
        )

    previous_index = stable._load_previous_index(root)
    labels, clustering = core.discover_motion_modes(
        root,
        active_items,
        seed=args.seed,
        max_modes=max(1, args.max_modes),
    )
    initial_mode_count = int(clustering.get("mode_count", 1))
    descriptors = np.stack([
        core._cycle_motion_descriptor(root, item)
        for item in active_items
    ])
    labels, merge_decisions = stable._merge_near_duplicate_modes(
        active_items,
        labels,
        descriptors,
        previous_index,
        args.max_phase_error,
    )
    labels = stable._sort_modes_by_duration(active_items, labels)
    final_mode_count = len(set(int(value) for value in labels))

    clustering = dict(clustering)
    clustering["initial_mode_count"] = initial_mode_count
    clustering["mode_count"] = final_mode_count
    clustering["counts"] = [
        int(np.sum(labels == mode))
        for mode in range(final_mode_count)
    ]
    clustering["merge_decisions"] = merge_decisions
    clustering["stability_layer"] = (
        "near-duration+motion+trusted-history+leave-out-quarantine-v2"
    )

    stable_ids = stable._stable_ids(active_items, labels, previous_index)
    mode_items, mode_index = core.build_mode_manifests(
        root,
        active_items,
        labels,
        clustering,
    )
    stable._augment_mode_index(mode_index, stable_ids, clustering)

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

    # Before expensive formal training, use deterministic short probes to find cycles
    # whose removal measurably improves the remaining mode's phase validation.
    all_new_quarantine: list[dict] = []
    screening_reports = []
    filtered_mode_items = []

    for mode, members in enumerate(mode_items):
        filtered, records, report = cycle_quality.filter_mode_cycles(
            root,
            members,
            args,
            device,
            f"mode_{mode}",
        )
        filtered_mode_items.append(filtered)
        all_new_quarantine.extend(records)
        screening_reports.append(report)

    mode_items = filtered_mode_items
    _rewrite_filtered_manifests(
        root,
        mode_items,
        mode_index,
        all_new_quarantine,
        screening_reports,
    )

    merged_quarantine, quarantine_file = cycle_quality.merge_quarantine_records(
        root,
        existing_quarantine,
        all_new_quarantine,
        reconsider=args.reconsider_quarantine,
    )
    print(
        f"cycle quarantine: new={len(all_new_quarantine)} "
        f"persistent_total={len(merged_quarantine)} -> {quarantine_file}"
    )

    all_modes_ready = True
    for mode, members in enumerate(mode_items):
        mode_dir = root / "modes" / f"mode_{mode}"
        candidate = mode_dir / "phase_model.candidate.pt"
        formal = mode_dir / "phase_model.pt"
        if len(mode_items) == 1:
            candidate = root / "models" / "phase_model.candidate.pt"
            formal = root / "models" / "phase_model.pt"

        _reset_training_seed(args.seed + 1009 * (mode + 1))
        phase_error = core.train_phase_model(
            root,
            members,
            args,
            device,
            candidate,
            f"mode_{mode}",
        )
        ready = bool(
            np.isfinite(phase_error)
            and phase_error <= args.max_phase_error
        )
        meta = mode_index["modes"][mode]
        meta["val_circular_error"] = phase_error
        meta["quality_threshold"] = args.max_phase_error
        meta["quality"] = "ready" if ready else "low_phase_confidence"

        if ready:
            stable._promote(candidate, formal)
            meta["phase_model"] = str(formal.relative_to(root))
            print(f"[mode_{mode}] quality PASS {phase_error:.4f}")
        else:
            all_modes_ready = False
            meta["phase_model"] = None
            print(
                f"[mode_{mode}] quality REJECT {phase_error:.4f} > "
                f"{args.max_phase_error:.4f}; candidate kept, formal model not promoted"
            )

    classifier_ready = True
    if len(mode_items) > 1 and all_modes_ready:
        _reset_training_seed(args.seed + 9001)
        classifier = core.train_mode_classifier(root, mode_items, args, device)
        accuracy = float(classifier["val_accuracy"]) if classifier else -1.0
        classifier_ready = accuracy >= args.min_classifier_accuracy
        if not classifier_ready:
            stable._quarantine(root / "modes" / "mode_classifier.pt")
        mode_index["classifier"] = dict(classifier or {})
        mode_index["classifier"]["quality"] = (
            "ready" if classifier_ready else "low_confidence"
        )
        mode_index["classifier"]["quality_threshold"] = (
            args.min_classifier_accuracy
        )
    elif len(mode_items) > 1:
        classifier_ready = False
        stable._quarantine(root / "modes" / "mode_classifier.pt")
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
        stable._save_router(root, args, mode_index, router_ready)
    elif not router_ready:
        stable._quarantine(root / "models" / "phase_model.pt")

    mode_index["router_ready"] = router_ready
    mode_index["quality_gate"] = {
        "max_phase_error": args.max_phase_error,
        "min_classifier_accuracy": args.min_classifier_accuracy,
        "all_modes_ready": all_modes_ready,
        "classifier_ready": classifier_ready,
    }
    mode_index["cycle_quality_gate"] = {
        "persistent_quarantine_count": len(merged_quarantine),
        "preexcluded_count": len(preexcluded_items),
        "new_quarantine_count": len(all_new_quarantine),
        "outlier_score_threshold": args.outlier_score_threshold,
        "outlier_min_improvement": args.outlier_min_improvement,
        "outlier_probe_epochs": args.outlier_probe_epochs,
        "outlier_max_fraction": args.outlier_max_fraction,
        "outlier_max_probes": args.outlier_max_probes,
        "reports": screening_reports,
    }

    write_json(root / "modes" / "index.json", mode_index)
    write_json(
        root / "models" / "router_status.json",
        {
            "character": args.character,
            "router_ready": router_ready,
            "mode_count": len(mode_items),
            "quality_gate": mode_index["quality_gate"],
            "cycle_quality_gate": {
                "persistent_quarantine_count": len(merged_quarantine),
                "new_quarantine_count": len(all_new_quarantine),
            },
        },
    )

    print("stable multimode training complete:")
    for mode in mode_index["modes"]:
        print(
            f"  {mode['name']}: cycles={mode['cycle_count']} "
            f"raw_cycles={mode.get('raw_cycle_count')} "
            f"quarantined={mode.get('quarantined_cycle_count', 0)} "
            f"median_duration={mode['median_duration_s']:.3f}s "
            f"phase_error={mode.get('val_circular_error')} "
            f"quality={mode.get('quality')} "
            f"stable_id={mode.get('stable_id')}"
        )
    if mode_index.get("classifier"):
        print(
            f"  mode classifier accuracy="
            f"{mode_index['classifier'].get('val_accuracy')} "
            f"quality={mode_index['classifier'].get('quality')}"
        )
    print(
        f"  cycle_quarantine total={len(merged_quarantine)} "
        f"new={len(all_new_quarantine)}"
    )
    print(f"  router_ready={router_ready}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
