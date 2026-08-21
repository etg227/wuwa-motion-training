from __future__ import annotations

import phase_alignment
import phase_training_legacy as core
import stable_phase_training_v2 as previous


def main() -> int:
    """Run the stable pipeline with cross-recording phase-zero alignment enabled."""
    original_dataset = core.CycleWindowDataset
    original_build = core.build_mode_manifests

    def build_aligned_mode_manifests(root, items, labels, clustering):
        mode_items, mode_index = original_build(root, items, labels, clustering)
        aligned_modes, reports = phase_alignment.align_all_modes(
            root,
            mode_items,
            mode_index,
        )
        print("phase-zero alignment:")
        for mode, report in enumerate(reports):
            print(
                f"  mode_{mode}: cycles={report['cycle_count']} "
                f"videos={report['video_group_count']} "
                f"median_score={report['median_score']:.3f} "
                f"min_score={report['minimum_score']:.3f} "
                f"reference={report['reference_cycle']}"
            )
            for group in report["video_groups"]:
                print(
                    f"    video offset={group['phase_offset']:.4f} "
                    f"cycles={group['cycle_count']} "
                    f"spread={group['within_video_phase_spread']:.4f} "
                    f"{group['video']}"
                )
        return aligned_modes, mode_index

    # cycle_quality and phase_training_legacy both hold references to the same
    # module object, so this patch makes formal training and leave-out probes use
    # the identical aligned target labels without rewriting original NPZ files.
    core.CycleWindowDataset = phase_alignment.AlignedCycleWindowDataset
    core.build_mode_manifests = build_aligned_mode_manifests
    try:
        return previous.main()
    finally:
        core.CycleWindowDataset = original_dataset
        core.build_mode_manifests = original_build


if __name__ == "__main__":
    raise SystemExit(main())
