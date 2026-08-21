from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import replay_validate as replay
from common import character_root, write_json
from ready_evidence import (
    DEFAULT_DECAY,
    DEFAULT_ENTER_THRESHOLD,
    DEFAULT_EXIT_FRAMES,
    DEFAULT_EXIT_THRESHOLD,
    DEFAULT_MEMORY_FRAMES,
    ReadyEvidenceTracker,
)


DEFAULT_LATENCY_FRAMES = 2


def _apply_evidence(predictions: list[dict], args) -> list[dict]:
    tracker = ReadyEvidenceTracker(
        memory_frames=args.memory_frames,
        decay=args.decay,
        enter_threshold=args.enter_threshold,
        exit_threshold=args.exit_threshold,
        exit_frames=args.exit_frames,
    )
    output = []
    for row in predictions:
        item = dict(row)
        raw_ready = float(item["chain_ready"])
        result = tracker.update(int(item["mode"]), raw_ready)
        item["raw_chain_ready"] = raw_ready
        item["evidence_score"] = result.evidence
        item["evidence_ready"] = bool(result.ready)
        item["evidence_entered"] = bool(result.entered)
        item["evidence_exited"] = bool(result.exited)
        item["evidence_low_streak"] = int(result.low_streak)
        output.append(item)
    return output


def _binary_stats(expected: list[bool], predicted: list[bool]) -> dict:
    if not expected or len(expected) != len(predicted):
        return {
            "count": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "false_positive_rate": None,
        }
    exp = np.asarray(expected, dtype=np.bool_)
    pred = np.asarray(predicted, dtype=np.bool_)
    tp = int(np.sum(exp & pred))
    fp = int(np.sum(~exp & pred))
    fn = int(np.sum(exp & ~pred))
    tn = int(np.sum(~exp & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "count": int(len(exp)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
    }


def _ratio(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float32)))


def _evidence_report(
    predictions: list[dict],
    rows: list[dict],
    accepted: set[int],
    ready_profiles: dict[int, dict],
    raw_threshold: float,
    latency_frames: int,
) -> tuple[dict, list[dict]]:
    frame_count = len(predictions)
    expected_mode, expected_phase = replay._reference_arrays(frame_count, rows)

    oracle_labels: list[bool] = []
    raw_labels: list[bool] = []
    evidence_labels: list[bool] = []
    accepted_raw: list[float] = []
    accepted_score: list[float] = []
    accepted_state: list[bool] = []
    accepted_expected: list[float] = []
    accepted_latency_hits: list[bool] = []
    accepted_latency_values: list[int] = []

    csv_rows: list[dict] = []
    for frame, prediction in enumerate(predictions):
        exp_mode = int(expected_mode[frame])
        exp_phase = (
            float(expected_phase[frame])
            if np.isfinite(expected_phase[frame])
            else None
        )
        expected_ready = None
        if exp_mode >= 0 and exp_phase is not None:
            expected_ready = float(
                replay._ready_probability(ready_profiles.get(exp_mode), exp_phase)
            )
            oracle_labels.append(expected_ready >= raw_threshold)
            raw_labels.append(float(prediction["raw_chain_ready"]) >= raw_threshold)
            evidence_labels.append(bool(prediction["evidence_ready"]))

        if frame in accepted:
            accepted_raw.append(float(prediction["raw_chain_ready"]))
            accepted_score.append(float(prediction["evidence_score"]))
            accepted_state.append(bool(prediction["evidence_ready"]))
            if expected_ready is not None:
                accepted_expected.append(expected_ready)

            hit = False
            first_latency = None
            for offset in range(0, latency_frames + 1):
                index = frame + offset
                if index >= frame_count:
                    break
                if bool(predictions[index]["evidence_ready"]):
                    hit = True
                    first_latency = offset
                    break
            accepted_latency_hits.append(hit)
            if first_latency is not None:
                accepted_latency_values.append(int(first_latency))

        csv_rows.append(
            {
                "frame": frame,
                "pred_mode": int(prediction["mode"]),
                "pred_phase": float(prediction["phase"]),
                "raw_chain_ready": float(prediction["raw_chain_ready"]),
                "raw_ready": int(float(prediction["raw_chain_ready"]) >= raw_threshold),
                "evidence_score": float(prediction["evidence_score"]),
                "evidence_ready": int(bool(prediction["evidence_ready"])),
                "evidence_entered": int(bool(prediction["evidence_entered"])),
                "evidence_exited": int(bool(prediction["evidence_exited"])),
                "expected_mode": "" if exp_mode < 0 else exp_mode,
                "expected_phase": "" if exp_phase is None else exp_phase,
                "expected_ready": "" if expected_ready is None else expected_ready,
                "expected_ready_label": (
                    "" if expected_ready is None else int(expected_ready >= raw_threshold)
                ),
                "accepted_candidate": int(frame in accepted),
            }
        )

    raw_stats = _binary_stats(oracle_labels, raw_labels)
    evidence_stats = _binary_stats(oracle_labels, evidence_labels)
    raw_accepted_ratio = (
        float(np.mean(np.asarray(accepted_raw) >= raw_threshold))
        if accepted_raw
        else None
    )
    expected_accepted_ratio = (
        float(np.mean(np.asarray(accepted_expected) >= raw_threshold))
        if accepted_expected
        else None
    )
    enter_count = sum(bool(row["evidence_entered"]) for row in predictions)
    exit_count = sum(bool(row["evidence_exited"]) for row in predictions)
    ready_frame_ratio = (
        float(np.mean([bool(row["evidence_ready"]) for row in predictions]))
        if predictions
        else 0.0
    )

    report = {
        "schema": 1,
        "frames": frame_count,
        "raw_ready_threshold": raw_threshold,
        "oracle_raw": raw_stats,
        "oracle_evidence": evidence_stats,
        "accepted_candidate_count": len(accepted),
        "accepted_raw_ready_median": _median(accepted_raw),
        "accepted_raw_ready_ge_threshold_ratio": raw_accepted_ratio,
        "accepted_evidence_score_median": _median(accepted_score),
        "accepted_evidence_state_ratio": _ratio(accepted_state),
        "accepted_expected_ready_median": _median(accepted_expected),
        "accepted_expected_ready_ge_threshold_ratio": expected_accepted_ratio,
        "accepted_within_latency_ready_ratio": _ratio(accepted_latency_hits),
        "accepted_latency_window_frames": int(latency_frames),
        "accepted_first_ready_latency_frames_median": _median(
            [float(value) for value in accepted_latency_values]
        ),
        "evidence_enter_count": int(enter_count),
        "evidence_exit_count": int(exit_count),
        "evidence_ready_frame_ratio": float(ready_frame_ratio),
    }
    return report, csv_rows


def validate_one(root: Path, video: Path, args, models, segments_by_video):
    (
        _mode_index,
        _modes,
        classifier,
        classifier_window,
        phase_models,
        phase_windows,
        ready_profiles,
        image_size,
    ) = models

    rows = segments_by_video.get(replay._path_key(video), [])
    roi = replay._roi_for_video(root, video, rows)
    print(f"validating {video}")
    print(f"  annotated cycles={len(rows)} roi={','.join(f'{v:.3f}' for v in roi)}")
    raw_predictions, fps, frame_count = replay._infer_video(
        video,
        roi,
        args.device,
        classifier,
        classifier_window,
        phase_models,
        phase_windows,
        ready_profiles,
        image_size,
        args.batch_size,
    )
    predictions = _apply_evidence(raw_predictions, args)

    accepted = {
        frame
        for frame in replay._accepted_frames(root, video)
        if 0 <= frame < frame_count
    }
    report, csv_rows = _evidence_report(
        predictions,
        rows,
        accepted,
        ready_profiles,
        args.ready_threshold,
        args.latency_frames,
    )
    report["character_root"] = str(root)
    report["video"] = str(video)
    report["fps"] = fps
    report["evidence_parameters"] = {
        "memory_frames": args.memory_frames,
        "decay": args.decay,
        "enter_threshold": args.enter_threshold,
        "exit_threshold": args.exit_threshold,
        "exit_frames": args.exit_frames,
        "latency_frames": args.latency_frames,
    }

    replay_dir = root / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem + ".ready_evidence"
    output_report = replay_dir / f"{stem}.report.json"
    output_csv = replay_dir / f"{stem}.csv"
    write_json(output_report, report)
    if csv_rows:
        with output_csv.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    raw = report["oracle_raw"]
    evidence = report["oracle_evidence"]
    print(f"  inference complete: frames={frame_count} fps={fps:.3f}")
    print(
        "  oracle READY raw -> evidence: "
        f"precision {raw['precision']:.3f} -> {evidence['precision']:.3f}; "
        f"recall {raw['recall']:.3f} -> {evidence['recall']:.3f}; "
        f"F1 {raw['f1']:.3f} -> {evidence['f1']:.3f}; "
        f"FPR {raw['false_positive_rate']:.3f} -> {evidence['false_positive_rate']:.3f}"
    )
    print(
        "  accepted READY raw -> evidence-state: "
        f"{report['accepted_raw_ready_ge_threshold_ratio']:.3f} -> "
        f"{report['accepted_evidence_state_ratio']:.3f}; "
        f"evidence_score_median={report['accepted_evidence_score_median']:.3f}"
    )
    print(
        f"  accepted <=+{args.latency_frames}f READY="
        f"{report['accepted_within_latency_ready_ratio']:.3f}; "
        f"first_ready_latency_median="
        f"{report['accepted_first_ready_latency_frames_median']}"
    )
    print(
        f"  evidence transitions: enter={report['evidence_enter_count']} "
        f"exit={report['evidence_exit_count']} "
        f"ready_frame_ratio={report['evidence_ready_frame_ratio']:.3f}"
    )
    print(f"  report -> {output_report}")
    print(f"  frame data -> {output_csv}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline A/B validator for causal CHAIN_READY evidence accumulation "
            "and hysteresis; phase itself is left untouched"
        )
    )
    parser.add_argument("--character", required=True)
    parser.add_argument("--video")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--batch-size", type=int, default=replay.DEFAULT_BATCH_SIZE)
    parser.add_argument("--ready-threshold", type=float, default=replay.DEFAULT_READY_THRESHOLD)
    parser.add_argument("--memory-frames", type=int, default=DEFAULT_MEMORY_FRAMES)
    parser.add_argument("--decay", type=float, default=DEFAULT_DECAY)
    parser.add_argument("--enter-threshold", type=float, default=DEFAULT_ENTER_THRESHOLD)
    parser.add_argument("--exit-threshold", type=float, default=DEFAULT_EXIT_THRESHOLD)
    parser.add_argument("--exit-frames", type=int, default=DEFAULT_EXIT_FRAMES)
    parser.add_argument("--latency-frames", type=int, default=DEFAULT_LATENCY_FRAMES)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    args.batch_size = max(1, int(args.batch_size))
    args.ready_threshold = float(np.clip(args.ready_threshold, 0.0, 1.0))
    args.memory_frames = max(1, int(args.memory_frames))
    args.decay = float(np.clip(args.decay, 0.0, 1.0))
    args.enter_threshold = float(np.clip(args.enter_threshold, 0.0, 1.0))
    args.exit_threshold = float(np.clip(args.exit_threshold, 0.0, 1.0))
    args.exit_frames = max(1, int(args.exit_frames))
    args.latency_frames = max(0, int(args.latency_frames))
    if args.exit_threshold >= args.enter_threshold:
        raise SystemExit("--exit-threshold must be lower than --enter-threshold")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    args.device = torch.device(device_name)

    root = character_root(args.character)
    models = replay._load_models(root, args.device)
    mode_index = models[0]
    segments_by_video = replay._load_segments(root, mode_index)
    candidates = replay._candidate_videos(root, segments_by_video)

    if args.all:
        videos = candidates
        if not videos:
            raise SystemExit("当前 mode manifests 中没有带真实 ATTACK telemetry 的录像。")
    else:
        selected = replay._resolve_video(root, args.video)
        if selected is None:
            if not candidates:
                raise SystemExit("没有找到带真实 ATTACK telemetry 的训练录像。")
            selected = candidates[-1]
        videos = [selected]

    print(
        f"device={args.device} videos={len(videos)} "
        f"raw_threshold={args.ready_threshold:.2f} "
        f"evidence(memory={args.memory_frames}, decay={args.decay:.2f}, "
        f"enter={args.enter_threshold:.2f}, exit={args.exit_threshold:.2f}, "
        f"exit_frames={args.exit_frames})"
    )
    for number, video in enumerate(videos, start=1):
        print(f"[{number}/{len(videos)}]")
        validate_one(root, video, args, models, segments_by_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
