from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

import replay_validate as replay
from common import character_root, write_json
from phase_tracker import CircularPhaseTracker


def _mode_durations(modes: list[dict]) -> dict[int, float]:
    result = {}
    for meta in modes:
        mode_id = int(meta["id"])
        duration = float(meta.get("median_duration_s", 0.0) or 0.0)
        if duration > 0:
            result[mode_id] = duration
    if not result:
        raise RuntimeError("modes/index.json 缺少 median_duration_s，无法建立 phase motion clock。")
    return result


def _apply_phase_tracker(
    predictions: list[dict],
    fps: float,
    mode_durations: dict[int, float],
    ready_profiles: dict[int, dict],
) -> list[dict]:
    tracker = CircularPhaseTracker(mode_durations)
    dt_s = 1.0 / max(float(fps), 1e-6)
    tracked = []

    for row in predictions:
        item = dict(row)
        mode_id = int(item["mode"])
        raw_phase = float(item["phase"])
        raw_ready = float(item["chain_ready"])
        result = tracker.update(
            mode_id,
            raw_phase,
            dt_s,
            mode_confidence=float(item.get("mode_confidence", 1.0)),
        )
        item["raw_phase"] = raw_phase
        item["raw_chain_ready"] = raw_ready
        item["phase"] = result.phase
        item["chain_ready"] = float(np.clip(
            replay._ready_probability(ready_profiles.get(mode_id), result.phase),
            0.0,
            1.0,
        ))
        item["tracker_predicted_phase"] = result.predicted_phase
        item["tracker_residual"] = result.residual
        item["tracker_observation_weight"] = result.observation_weight
        item["tracker_rejected"] = bool(result.rejected)
        item["tracker_reanchored"] = bool(result.reanchored)
        tracked.append(item)
    return tracked


def _metric(values, percentile: float | None = None):
    if not values:
        return None
    array = np.asarray(values, dtype=np.float32)
    if percentile is None:
        return float(np.median(array))
    return float(np.percentile(array, percentile))


def _ratio(values, threshold: float):
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float32) >= threshold))


def _tracking_report(
    predictions: list[dict],
    rows: list[dict],
    accepted: set[int],
    ready_profiles: dict[int, dict],
    ready_threshold: float,
) -> dict:
    expected_mode, expected_phase = replay._reference_arrays(len(predictions), rows)
    raw_errors = []
    tracked_errors = []
    accepted_raw_errors = []
    accepted_tracked_errors = []
    accepted_raw_ready = []
    accepted_tracked_ready = []
    accepted_expected_ready = []

    for frame, prediction in enumerate(predictions):
        exp_mode = int(expected_mode[frame])
        exp_phase = (
            float(expected_phase[frame])
            if np.isfinite(expected_phase[frame])
            else None
        )
        mode_match = exp_mode >= 0 and int(prediction["mode"]) == exp_mode
        if mode_match and exp_phase is not None:
            raw_error = replay._circular_error(float(prediction["raw_phase"]), exp_phase)
            tracked_error = replay._circular_error(float(prediction["phase"]), exp_phase)
            raw_errors.append(raw_error)
            tracked_errors.append(tracked_error)
            if frame in accepted:
                accepted_raw_errors.append(raw_error)
                accepted_tracked_errors.append(tracked_error)

        if frame in accepted:
            accepted_raw_ready.append(float(prediction["raw_chain_ready"]))
            accepted_tracked_ready.append(float(prediction["chain_ready"]))
            if exp_mode >= 0 and exp_phase is not None:
                accepted_expected_ready.append(float(
                    replay._ready_probability(ready_profiles.get(exp_mode), exp_phase)
                ))

    residuals = [abs(float(row.get("tracker_residual", 0.0))) for row in predictions]
    rejected = sum(bool(row.get("tracker_rejected")) for row in predictions)
    # The first frame/mode reset is intentionally a re-anchor; report later
    # re-anchors separately because those are actual recovery events.
    reanchors = sum(bool(row.get("tracker_reanchored")) for row in predictions)

    return {
        "architecture": "CircularPhaseTracker-v1",
        "raw_phase_error_median": _metric(raw_errors),
        "raw_phase_error_p90": _metric(raw_errors, 90),
        "tracked_phase_error_median": _metric(tracked_errors),
        "tracked_phase_error_p90": _metric(tracked_errors, 90),
        "accepted_raw_phase_error_median": _metric(accepted_raw_errors),
        "accepted_raw_phase_error_p90": _metric(accepted_raw_errors, 90),
        "accepted_tracked_phase_error_median": _metric(accepted_tracked_errors),
        "accepted_tracked_phase_error_p90": _metric(accepted_tracked_errors, 90),
        "accepted_raw_ready_median": _metric(accepted_raw_ready),
        "accepted_raw_ready_ge_threshold_ratio": _ratio(accepted_raw_ready, ready_threshold),
        "accepted_tracked_ready_median": _metric(accepted_tracked_ready),
        "accepted_tracked_ready_ge_threshold_ratio": _ratio(
            accepted_tracked_ready, ready_threshold
        ),
        "accepted_expected_ready_median": _metric(accepted_expected_ready),
        "accepted_expected_ready_ge_threshold_ratio": _ratio(
            accepted_expected_ready, ready_threshold
        ),
        "mean_abs_observation_residual": (
            float(np.mean(residuals)) if residuals else None
        ),
        "rejected_observation_count": int(rejected),
        "reanchor_count_including_initial": int(reanchors),
    }


def validate_one(root: Path, video: Path, args, models, segments_by_video):
    (
        _mode_index,
        modes,
        classifier,
        classifier_window,
        phase_models,
        phase_windows,
        ready_profiles,
        image_size,
    ) = models
    rows = segments_by_video.get(replay._path_key(video), [])
    roi = replay._roi_for_video(root, video, rows)
    durations = _mode_durations(modes)

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
    predictions = _apply_phase_tracker(
        raw_predictions,
        fps,
        durations,
        ready_profiles,
    )
    print(f"  inference+tracking complete: frames={frame_count} fps={fps:.3f}")

    output_video, report = replay._render(
        root,
        video,
        predictions,
        fps,
        rows,
        ready_profiles,
        args.ready_threshold,
        args.render_scale,
    )

    accepted = {
        frame
        for frame in replay._accepted_frames(root, video)
        if 0 <= frame < len(predictions)
    }
    tracking = _tracking_report(
        predictions,
        rows,
        accepted,
        ready_profiles,
        args.ready_threshold,
    )
    report["phase_tracker"] = tracking
    report["schema"] = max(3, int(report.get("schema", 1)))
    report_path = (
        root / "replays" / f"{video.stem}.ready_validation.report.json"
    )
    write_json(report_path, report)

    print(
        "  phase tracker raw -> tracked: "
        f"median {tracking['raw_phase_error_median']:.4f} -> "
        f"{tracking['tracked_phase_error_median']:.4f}; "
        f"p90 {tracking['raw_phase_error_p90']:.4f} -> "
        f"{tracking['tracked_phase_error_p90']:.4f}"
    )
    if tracking["accepted_raw_phase_error_median"] is not None:
        print(
            "  accepted phase raw -> tracked: "
            f"median {tracking['accepted_raw_phase_error_median']:.4f} -> "
            f"{tracking['accepted_tracked_phase_error_median']:.4f}; "
            f"p90 {tracking['accepted_raw_phase_error_p90']:.4f} -> "
            f"{tracking['accepted_tracked_phase_error_p90']:.4f}"
        )
    if tracking["accepted_raw_ready_median"] is not None:
        print(
            "  accepted READY raw -> tracked: "
            f"median {tracking['accepted_raw_ready_median']:.3f} -> "
            f"{tracking['accepted_tracked_ready_median']:.3f}; "
            f">={args.ready_threshold:.2f} "
            f"{tracking['accepted_raw_ready_ge_threshold_ratio']:.3f} -> "
            f"{tracking['accepted_tracked_ready_ge_threshold_ratio']:.3f}"
        )
    print(
        "  tracker observations: "
        f"rejected={tracking['rejected_observation_count']} "
        f"reanchors={tracking['reanchor_count_including_initial']} "
        f"mean_abs_residual={tracking['mean_abs_observation_residual']:.4f}"
    )
    return output_video, report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline replay validation with circular phase tracking"
    )
    parser.add_argument("--character", required=True)
    parser.add_argument("--video")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--batch-size", type=int, default=replay.DEFAULT_BATCH_SIZE)
    parser.add_argument("--ready-threshold", type=float, default=replay.DEFAULT_READY_THRESHOLD)
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    args.batch_size = max(1, int(args.batch_size))
    args.ready_threshold = float(np.clip(args.ready_threshold, 0.0, 1.0))
    args.render_scale = float(np.clip(args.render_scale, 0.25, 1.0))
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
        f"ready_threshold={args.ready_threshold:.2f} tracker=CircularPhaseTracker-v1"
    )
    last_output = None
    for number, video in enumerate(videos, start=1):
        print(f"[{number}/{len(videos)}]")
        last_output, _report = validate_one(
            root, video, args, models, segments_by_video
        )

    if args.open and last_output is not None and os.name == "nt":
        os.startfile(str(last_output))  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
