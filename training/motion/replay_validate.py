from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

import phase_training_legacy as core
from common import DEFAULT_ROI, character_root, crop_roi, read_json, write_json
from semantic_inputs import load_semantic_events, telemetry_path_for_video


DEFAULT_READY_THRESHOLD = 0.85
DEFAULT_BATCH_SIZE = 48


def _path_key(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _resolve_video(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    direct = Path(value)
    candidates = [direct]
    if not direct.is_absolute():
        candidates.extend((root / "videos" / value, root / "videos" / direct.name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"找不到录像：{value}")


def _mode_manifest(root: Path, meta: dict) -> Path:
    value = meta.get("manifest")
    if isinstance(value, str) and value:
        return root / value
    return root / "modes" / f"mode_{int(meta['id'])}" / "manifest.jsonl"


def _load_segments(root: Path, mode_index: dict) -> dict[str, list[dict]]:
    by_video: dict[str, list[dict]] = {}
    for meta in mode_index.get("modes", []):
        mode_id = int(meta["id"])
        for item in _load_jsonl(_mode_manifest(root, meta)):
            video = Path(item["video"])
            row = dict(item)
            row["expected_mode"] = mode_id
            by_video.setdefault(_path_key(video), []).append(row)
    for rows in by_video.values():
        rows.sort(key=lambda row: int(row.get("start_frame", 0)))
    return by_video


def _candidate_videos(root: Path, segments_by_video: dict[str, list[dict]]) -> list[Path]:
    videos = []
    seen = set()
    for rows in segments_by_video.values():
        for row in rows:
            video = Path(row["video"])
            key = _path_key(video)
            if key in seen or not video.is_file():
                continue
            events = load_semantic_events(
                telemetry_path_for_video(video), action="ATTACK", edge="down"
            )
            if not events:
                continue
            seen.add(key)
            videos.append(video.resolve())
    videos.sort(key=lambda path: path.stat().st_mtime)
    return videos


def _roi_for_video(root: Path, video: Path, rows: list[dict]) -> tuple[float, float, float, float]:
    for row in rows:
        annotation = row.get("annotation")
        if not isinstance(annotation, str):
            continue
        path = root / annotation
        if not path.is_file():
            continue
        try:
            value = read_json(path).get("roi")
            if isinstance(value, list) and len(value) == 4:
                return tuple(float(item) for item in value)
        except Exception:
            continue
    return DEFAULT_ROI


def _model_input(frame_bgr: np.ndarray, roi, image_size: int) -> np.ndarray:
    crop = crop_roi(frame_bgr, roi)
    crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return np.asarray(crop, dtype=np.uint8)


def _clip_tensor(clips: np.ndarray, window: int, device: torch.device) -> torch.Tensor:
    values = clips[:, -window:].astype(np.float32) / 255.0
    tensor = torch.from_numpy(values).permute(0, 1, 4, 2, 3)
    tensor = (tensor - 0.5) / 0.5
    return tensor.to(device, non_blocking=True)


def _load_models(root: Path, device: torch.device):
    index_path = root / "modes" / "index.json"
    if not index_path.is_file():
        raise RuntimeError(f"缺少 {index_path}；请先训练 phase 模型。")
    mode_index = read_json(index_path)
    if not bool(mode_index.get("router_ready")):
        raise RuntimeError("phase router 还不是 READY，暂不做回放验证。")

    modes = sorted(mode_index.get("modes", []), key=lambda row: int(row["id"]))
    if not modes:
        raise RuntimeError("modes/index.json 中没有动作 mode。")

    phase_models = {}
    phase_windows = {}
    image_size = 112
    for meta in modes:
        mode_id = int(meta["id"])
        relative = meta.get("phase_model")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError(f"mode_{mode_id} 没有正式 phase_model。")
        checkpoint = torch.load(root / relative, map_location=device, weights_only=False)
        model = core.PhaseNet().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        phase_models[mode_id] = model
        phase_windows[mode_id] = int(checkpoint.get("window", 12))

        manifest = _load_jsonl(_mode_manifest(root, meta))
        if manifest:
            image_size = int(manifest[0].get("image_size", image_size))

    classifier = None
    classifier_window = max(phase_windows.values())
    if len(modes) > 1:
        classifier_meta = mode_index.get("classifier")
        if not isinstance(classifier_meta, dict) or not classifier_meta.get("path"):
            raise RuntimeError("多 mode 模型缺少正式 mode classifier。")
        checkpoint = torch.load(
            root / classifier_meta["path"], map_location=device, weights_only=False
        )
        classifier = core.ModeNet(int(checkpoint.get("mode_count", len(modes)))).to(device)
        classifier.load_state_dict(checkpoint["state_dict"])
        classifier.eval()
        classifier_window = int(checkpoint.get("window", classifier_window))

    ready_path = root / "models" / "attack_ready.pt"
    if not ready_path.is_file():
        raise RuntimeError(f"缺少 {ready_path}；请先训练 ATTACK/CHAIN_READY。")
    ready_bundle = torch.load(ready_path, map_location="cpu", weights_only=False)
    if not bool(ready_bundle.get("ready_model_ready")):
        raise RuntimeError("attack_ready.pt 不是 READY 正式模型。")

    ready_profiles = {
        int(profile["id"]): profile
        for profile in ready_bundle.get("modes", [])
        if isinstance(profile, dict)
    }
    return (
        mode_index,
        modes,
        classifier,
        classifier_window,
        phase_models,
        phase_windows,
        ready_profiles,
        image_size,
    )


def _ready_probability(profile: dict | None, phase: float) -> float:
    if not profile:
        return 0.0
    values = profile.get("probabilities")
    if not isinstance(values, list) or not values:
        return 0.0
    count = len(values)
    position = (phase % 1.0) * count
    left = int(math.floor(position)) % count
    frac = position - math.floor(position)
    right = (left + 1) % count
    return float((1.0 - frac) * float(values[left]) + frac * float(values[right]))


def _infer_video(
    video: Path,
    roi,
    device: torch.device,
    classifier,
    classifier_window: int,
    phase_models: dict[int, torch.nn.Module],
    phase_windows: dict[int, int],
    ready_profiles: dict[int, dict],
    image_size: int,
    batch_size: int,
):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开录像：{video}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0 or fps <= 0:
        cap.release()
        raise RuntimeError(f"无效录像元数据：frames={frame_count} fps={fps}")

    max_window = max([classifier_window, *phase_windows.values()])
    history: deque[np.ndarray] = deque(maxlen=max_window)
    batch_clips: list[np.ndarray] = []
    outputs: list[dict] = []

    def flush():
        if not batch_clips:
            return
        clips = np.stack(batch_clips)
        with torch.inference_mode():
            if classifier is None:
                mode_ids = np.zeros(len(clips), dtype=np.int32)
                mode_conf = np.ones(len(clips), dtype=np.float32)
            else:
                logits = classifier(_clip_tensor(clips, classifier_window, device))
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted = torch.max(probabilities, dim=1)
                mode_ids = predicted.cpu().numpy().astype(np.int32)
                mode_conf = confidence.cpu().numpy().astype(np.float32)

            phases = np.zeros(len(clips), dtype=np.float32)
            for mode_id in sorted(set(int(value) for value in mode_ids)):
                indexes = np.where(mode_ids == mode_id)[0]
                if mode_id not in phase_models:
                    continue
                tensor = _clip_tensor(clips[indexes], phase_windows[mode_id], device)
                vector = phase_models[mode_id](tensor)
                phase = core.vector_phase(vector).cpu().numpy().astype(np.float32)
                phases[indexes] = phase

        for mode_id, confidence, phase in zip(mode_ids, mode_conf, phases):
            ready = _ready_probability(ready_profiles.get(int(mode_id)), float(phase))
            outputs.append(
                {
                    "mode": int(mode_id),
                    "mode_confidence": float(confidence),
                    "phase": float(phase % 1.0),
                    "chain_ready": float(np.clip(ready, 0.0, 1.0)),
                }
            )
        batch_clips.clear()

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = _model_input(frame, roi, image_size)
        if not history:
            for _ in range(max_window - 1):
                history.append(small)
        history.append(small)
        batch_clips.append(np.stack(history))
        if len(batch_clips) >= batch_size:
            flush()
        frame_index += 1
        if frame_index % 300 == 0:
            print(f"  inference {frame_index}/{frame_count}")

    flush()
    cap.release()
    if len(outputs) != frame_index:
        raise RuntimeError(f"推理帧数不一致：video={frame_index} predictions={len(outputs)}")
    return outputs, fps, frame_index


def _reference_arrays(frame_count: int, rows: list[dict]):
    expected_mode = np.full(frame_count, -1, dtype=np.int16)
    expected_phase = np.full(frame_count, np.nan, dtype=np.float32)
    for row in rows:
        start = max(0, int(row.get("start_frame", 0)))
        end = min(frame_count, int(row.get("end_frame", 0)))
        if end <= start:
            continue
        mode_id = int(row["expected_mode"])
        offset = float(row.get("phase_offset", 0.0))
        indexes = np.arange(start, end, dtype=np.float32)
        phases = ((indexes - start) / max(1, end - start) + offset) % 1.0
        expected_mode[start:end] = mode_id
        expected_phase[start:end] = phases
    return expected_mode, expected_phase


def _circular_error(a: float, b: float) -> float:
    delta = abs((a - b) % 1.0)
    return min(delta, 1.0 - delta)


def _accepted_frames(root: Path, video: Path) -> set[int]:
    path = root / "ready" / "accepted_attack_samples.jsonl"
    key = _path_key(video)
    return {
        int(row.get("accepted_frame", -1))
        for row in _load_jsonl(path)
        if _path_key(row.get("video", "")) == key
    }


def _put_text(frame, text: str, origin, scale=0.66, thickness=2):
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )


def _draw_bar(frame, x0, y0, width, height, value, label):
    value = float(np.clip(value, 0.0, 1.0))
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (90, 90, 90), 1)
    cv2.rectangle(
        frame,
        (x0 + 1, y0 + 1),
        (x0 + int((width - 2) * value), y0 + height - 1),
        (225, 225, 225),
        -1,
    )
    _put_text(frame, f"{label} {value*100:5.1f}%", (x0, y0 - 7), 0.54, 1)


def _draw_phase_profile(frame, profile: dict | None, phase: float, ready_threshold: float):
    height, width = frame.shape[:2]
    x0, x1 = 40, width - 40
    y0, y1 = height - 118, height - 72
    cv2.rectangle(frame, (x0, y0), (x1, y1), (40, 40, 40), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (180, 180, 180), 1)
    values = profile.get("probabilities", []) if profile else []
    if values:
        count = len(values)
        for i, value in enumerate(values):
            px = x0 + int(i / max(1, count - 1) * (x1 - x0))
            py = y1 - int(float(value) * (y1 - y0 - 3))
            if float(value) >= ready_threshold:
                cv2.line(frame, (px, y1 - 2), (px, py), (70, 220, 70), 2)
            else:
                cv2.line(frame, (px, y1 - 2), (px, py), (160, 160, 160), 1)
    marker = x0 + int((phase % 1.0) * (x1 - x0))
    cv2.line(frame, (marker, y0 - 3), (marker, y1 + 3), (0, 210, 255), 2)
    _put_text(frame, "phase 0.0", (x0, y0 - 8), 0.48, 1)
    _put_text(frame, "1.0", (x1 - 30, y0 - 8), 0.48, 1)


def _draw_timeline(
    frame,
    frame_index: int,
    frame_count: int,
    attack_frames: set[int],
    accepted: set[int],
):
    height, width = frame.shape[:2]
    x0, x1 = 40, width - 40
    y = height - 34
    cv2.line(frame, (x0, y), (x1, y), (190, 190, 190), 2)
    for value in attack_frames:
        px = x0 + int(value / max(1, frame_count - 1) * (x1 - x0))
        cv2.line(frame, (px, y - 7), (px, y + 7), (180, 180, 180), 1)
    for value in accepted:
        px = x0 + int(value / max(1, frame_count - 1) * (x1 - x0))
        cv2.line(frame, (px, y - 10), (px, y + 10), (0, 200, 255), 2)
    px = x0 + int(frame_index / max(1, frame_count - 1) * (x1 - x0))
    cv2.line(frame, (px, y - 14), (px, y + 14), (255, 255, 255), 2)


def _render(
    root: Path,
    video: Path,
    predictions: list[dict],
    fps: float,
    rows: list[dict],
    ready_profiles: dict[int, dict],
    ready_threshold: float,
    render_scale: float,
):
    frame_count = len(predictions)
    expected_mode, expected_phase = _reference_arrays(frame_count, rows)
    attacks = load_semantic_events(
        telemetry_path_for_video(video), action="ATTACK", edge="down"
    )
    attack_frames = {
        int(row.get("frame", -1))
        for row in attacks
        if 0 <= int(row.get("frame", -1)) < frame_count
    }
    accepted = {
        frame for frame in _accepted_frames(root, video) if 0 <= frame < frame_count
    }

    replay_dir = root / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem + ".ready_validation"
    output_video = replay_dir / f"{stem}.mp4"
    output_csv = replay_dir / f"{stem}.csv"
    output_report = replay_dir / f"{stem}.report.json"

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法重新打开录像：{video}")
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    width = max(2, int(round(source_width * render_scale)) // 2 * 2)
    height = max(2, int(round(source_height * render_scale)) // 2 * 2)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频：{output_video}")

    csv_rows = []
    mode_hits = 0
    annotated = 0
    phase_errors = []
    accepted_ready = []

    frame_index = 0
    while frame_index < frame_count:
        ok, frame = cap.read()
        if not ok:
            break
        if (width, height) != (source_width, source_height):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        prediction = predictions[frame_index]
        mode_id = int(prediction["mode"])
        mode_conf = float(prediction["mode_confidence"])
        phase = float(prediction["phase"])
        ready = float(prediction["chain_ready"])
        exp_mode = int(expected_mode[frame_index])
        exp_phase = (
            float(expected_phase[frame_index])
            if np.isfinite(expected_phase[frame_index])
            else None
        )
        phase_error = None

        if exp_mode >= 0:
            annotated += 1
            mode_hits += int(mode_id == exp_mode)
            if exp_phase is not None and mode_id == exp_mode:
                phase_error = _circular_error(phase, exp_phase)
                phase_errors.append(phase_error)

        if frame_index in accepted:
            accepted_ready.append(ready)

        panel = frame.copy()
        cv2.rectangle(panel, (18, 18), (610, 248), (10, 10, 10), -1)
        cv2.addWeighted(panel, 0.58, frame, 0.42, 0.0, frame)

        _put_text(
            frame,
            f"{video.name}  frame {frame_index}/{frame_count-1}  t={frame_index/fps:6.2f}s",
            (34, 48),
            0.58,
            1,
        )
        exp_mode_text = str(exp_mode) if exp_mode >= 0 else "-"
        _put_text(
            frame,
            f"MODE {mode_id}  conf={mode_conf*100:5.1f}%  expected={exp_mode_text}",
            (34, 80),
        )
        if exp_phase is None:
            _put_text(frame, f"PHASE {phase:0.3f}  expected=-", (34, 113))
        else:
            error_text = (
                f"{phase_error*100:4.1f}%" if phase_error is not None else "mode mismatch"
            )
            _put_text(
                frame,
                f"PHASE {phase:0.3f}  expected={exp_phase:0.3f}  err={error_text}",
                (34, 113),
            )
        _put_text(
            frame,
            f"CHAIN_READY {ready*100:5.1f}%  {'READY' if ready >= ready_threshold else 'WAIT'}",
            (34, 147),
        )
        _draw_bar(frame, 34, 177, 360, 20, ready, "ready")

        if frame_index in attack_frames:
            _put_text(frame, "HUMAN ATTACK", (34, 230), 0.68, 2)
        if frame_index in accepted:
            _put_text(frame, "ACCEPTED CANDIDATE", (260, 230), 0.68, 2)
        if exp_mode < 0:
            _put_text(frame, "OUTSIDE ANNOTATED CYCLE", (400, 147), 0.52, 1)

        _draw_phase_profile(frame, ready_profiles.get(mode_id), phase, ready_threshold)
        _draw_timeline(frame, frame_index, frame_count, attack_frames, accepted)
        writer.write(frame)

        csv_rows.append(
            {
                "frame": frame_index,
                "t_ms": round(frame_index / fps * 1000.0, 3),
                "pred_mode": mode_id,
                "mode_confidence": mode_conf,
                "pred_phase": phase,
                "chain_ready": ready,
                "ready": int(ready >= ready_threshold),
                "expected_mode": "" if exp_mode < 0 else exp_mode,
                "expected_phase": "" if exp_phase is None else exp_phase,
                "phase_error": "" if phase_error is None else phase_error,
                "human_attack": int(frame_index in attack_frames),
                "accepted_candidate": int(frame_index in accepted),
            }
        )
        frame_index += 1

    cap.release()
    writer.release()

    if not csv_rows:
        raise RuntimeError("录像没有可渲染帧。")
    with output_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer_csv = csv.DictWriter(stream, fieldnames=list(csv_rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)

    mode_accuracy = mode_hits / annotated if annotated else None
    report = {
        "schema": 1,
        "character_root": str(root),
        "video": str(video),
        "rendered_video": str(output_video),
        "csv": str(output_csv),
        "frames": frame_index,
        "fps": fps,
        "annotated_cycle_frames": annotated,
        "mode_accuracy": mode_accuracy,
        "median_phase_error": float(np.median(phase_errors)) if phase_errors else None,
        "p90_phase_error": float(np.percentile(phase_errors, 90)) if phase_errors else None,
        "human_attack_count": len(attack_frames),
        "accepted_candidate_count": len(accepted),
        "accepted_ready_median": float(np.median(accepted_ready)) if accepted_ready else None,
        "accepted_ready_ge_threshold_ratio": (
            float(np.mean(np.asarray(accepted_ready) >= ready_threshold))
            if accepted_ready
            else None
        ),
        "ready_threshold": ready_threshold,
    }
    write_json(output_report, report)

    print(f"  rendered -> {output_video}")
    print(f"  frame data -> {output_csv}")
    print(f"  report -> {output_report}")
    if mode_accuracy is not None:
        print(f"  annotated mode accuracy={mode_accuracy:.3f}")
    if phase_errors:
        print(
            f"  phase circular error median={np.median(phase_errors):.4f} "
            f"p90={np.percentile(phase_errors, 90):.4f}"
        )
    if accepted_ready:
        print(
            f"  accepted candidate READY median={np.median(accepted_ready):.3f} "
            f">={ready_threshold:.2f}: "
            f"{np.mean(np.asarray(accepted_ready) >= ready_threshold):.3f}"
        )
    return output_video, report


def validate_one(
    root: Path,
    video: Path,
    args,
    models,
    segments_by_video: dict[str, list[dict]],
):
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
    rows = segments_by_video.get(_path_key(video), [])
    roi = _roi_for_video(root, video, rows)
    print(f"validating {video}")
    print(f"  annotated cycles={len(rows)} roi={','.join(f'{v:.3f}' for v in roi)}")
    predictions, fps, frame_count = _infer_video(
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
    print(f"  inference complete: frames={frame_count} fps={fps:.3f}")
    return _render(
        root,
        video,
        predictions,
        fps,
        rows,
        ready_profiles,
        args.ready_threshold,
        args.render_scale,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline replay validator for mode / phase / ATTACK CHAIN_READY"
    )
    parser.add_argument("--character", required=True)
    parser.add_argument(
        "--video",
        help=(
            "MP4 path or file name. If omitted, use the newest training video "
            "with ATTACK telemetry."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="render every current training video that contains real ATTACK down telemetry",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--ready-threshold", type=float, default=DEFAULT_READY_THRESHOLD)
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="open the last rendered MP4 with the OS default player (Windows only)",
    )
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
    models = _load_models(root, args.device)
    mode_index = models[0]
    segments_by_video = _load_segments(root, mode_index)
    candidates = _candidate_videos(root, segments_by_video)

    if args.all:
        videos = candidates
        if not videos:
            raise SystemExit("当前 mode manifests 中没有带真实 ATTACK telemetry 的录像。")
    else:
        selected = _resolve_video(root, args.video)
        if selected is None:
            if not candidates:
                raise SystemExit("没有找到带真实 ATTACK telemetry 的训练录像。")
            selected = candidates[-1]
        videos = [selected]

    print(
        f"device={args.device} videos={len(videos)} "
        f"ready_threshold={args.ready_threshold:.2f}"
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
