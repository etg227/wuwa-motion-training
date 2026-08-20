from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import DEFAULT_ROI, character_root, crop_roi, parse_roi, video_info, write_json


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32, copy=False)
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.zeros_like(vector, dtype=np.float32)
    return vector / norm


def frame_feature(frame: np.ndarray, roi=DEFAULT_ROI, size: int = 48) -> np.ndarray:
    crop = crop_roi(frame, roi)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 70, 160)
    appearance = gray.astype(np.float32) / 255.0
    edge = edges.astype(np.float32) / 255.0
    return _normalize(np.concatenate((appearance.reshape(-1), edge.reshape(-1))))


def _sample_video(video: Path, roi, analysis_fps: float):
    info = video_info(video)
    source_fps = float(info["fps"])
    step = max(1.0, source_fps / max(1.0, analysis_fps))

    cap = cv2.VideoCapture(str(video))
    sample_frames: list[int] = []
    features: list[np.ndarray] = []
    frame_index = 0
    next_sample = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index + 1e-6 >= next_sample:
            sample_frames.append(frame_index)
            features.append(frame_feature(frame, roi))
            next_sample += step
        frame_index += 1

    cap.release()
    if len(features) < 24:
        raise RuntimeError("录像太短，至少需要能看到数轮连续平A。")
    return info, np.asarray(sample_frames, dtype=np.int32), np.stack(features)


def _motion_features(features: np.ndarray) -> np.ndarray:
    motion = np.zeros_like(features)
    motion[1:] = features[1:] - features[:-1]
    norms = np.linalg.norm(motion, axis=1, keepdims=True)
    return motion / np.maximum(norms, 1e-6)


def _trimmed_mean(values: np.ndarray, trim: float = 0.12) -> float:
    if len(values) == 0:
        return -1.0
    ordered = np.sort(values)
    cut = int(len(ordered) * trim)
    if cut > 0 and len(ordered) > cut * 2:
        ordered = ordered[cut:-cut]
    return float(ordered.mean())


def estimate_period(
    features: np.ndarray,
    effective_fps: float,
    min_period_s: float = 0.8,
    max_period_s: float = 8.0,
):
    count = len(features)
    motion = _motion_features(features)
    min_lag = max(3, int(round(min_period_s * effective_fps)))
    max_lag = min(int(round(max_period_s * effective_fps)), count // 2)
    if max_lag <= min_lag:
        raise RuntimeError("录像长度不足以估计完整平A循环。")

    lags = np.arange(min_lag, max_lag + 1, dtype=np.int32)
    scores = np.empty(len(lags), dtype=np.float32)

    for i, lag in enumerate(lags):
        appearance = np.sum(features[:-lag] * features[lag:], axis=1)
        motion_sim = np.sum(motion[:-lag] * motion[lag:], axis=1)
        scores[i] = 0.72 * _trimmed_mean(appearance) + 0.28 * _trimmed_mean(motion_sim)

    if len(scores) >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        smooth = np.convolve(scores, kernel, mode="same")
        smooth[:2] = scores[:2]
        smooth[-2:] = scores[-2:]
    else:
        smooth = scores

    peak_indexes = [
        i
        for i in range(1, len(smooth) - 1)
        if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]
    ]
    if not peak_indexes:
        peak_indexes = [int(np.argmax(smooth))]

    best_index = max(peak_indexes, key=lambda i: float(smooth[i]))
    best_score = float(smooth[best_index])

    # 两轮/三轮也会形成高相似峰；接近最佳值时优先取最短周期。
    strong = [i for i in peak_indexes if float(smooth[i]) >= best_score - 0.025]
    chosen = min(strong, key=lambda i: int(lags[i]))
    period = int(lags[chosen])
    score = float(smooth[chosen])

    if score < 0.22:
        raise RuntimeError(
            f"没有找到足够稳定的重复动作周期（period score={score:.3f}）。"
            "请录更长一点，并尽量保持连续平A。"
        )
    return period, score


def _choose_anchor(features: np.ndarray, motion: np.ndarray, period: int) -> int:
    usable = len(features) - period
    if usable <= 4:
        raise RuntimeError("可用循环不足。")

    left = max(1, int(usable * 0.12))
    right = max(left + 1, int(usable * 0.88))
    candidates = np.arange(left, right, dtype=np.int32)

    recurrence = np.sum(features[candidates] * features[candidates + period], axis=1)
    motion_recurrence = np.sum(motion[candidates] * motion[candidates + period], axis=1)
    local_motion = np.linalg.norm(motion[candidates], axis=1)
    quality = 0.78 * recurrence + 0.17 * motion_recurrence + 0.05 * np.clip(local_motion, 0, 2)
    return int(candidates[int(np.argmax(quality))])


def _reference_feature(features: np.ndarray, anchor: int, period: int) -> np.ndarray:
    refs = []
    for offset in range(-4, 5):
        index = anchor + offset * period
        if 0 <= index < len(features):
            refs.append(features[index])
    reference = np.mean(refs, axis=0)
    norm = np.linalg.norm(reference)
    return reference / max(float(norm), 1e-6)


def _walk_boundaries(
    features: np.ndarray,
    anchor: int,
    period: int,
    period_score: float,
):
    reference = _reference_feature(features, anchor, period)
    radius = max(2, int(round(period * 0.20)))
    threshold = max(0.34, period_score - 0.20)

    boundaries: list[tuple[int, float]] = [(anchor, 1.0)]

    previous = anchor
    while previous + period < len(features):
        expected = previous + period
        start = max(previous + max(2, int(period * 0.60)), expected - radius)
        end = min(len(features) - 1, expected + radius)
        if end <= start:
            break
        candidates = np.arange(start, end + 1, dtype=np.int32)
        reference_scores = features[candidates] @ reference
        recurrence_scores = np.sum(
            features[candidates] * features[np.maximum(0, candidates - period)], axis=1
        )
        scores = 0.82 * reference_scores + 0.18 * recurrence_scores
        best_pos = int(np.argmax(scores))
        candidate = int(candidates[best_pos])
        score = float(scores[best_pos])
        if score < threshold:
            break
        boundaries.append((candidate, score))
        previous = candidate

    previous = anchor
    backward: list[tuple[int, float]] = []
    while previous - period >= 0:
        expected = previous - period
        start = max(0, expected - radius)
        end = min(previous - max(2, int(period * 0.60)), expected + radius)
        if end <= start:
            break
        candidates = np.arange(start, end + 1, dtype=np.int32)
        reference_scores = features[candidates] @ reference
        forward_index = np.minimum(len(features) - 1, candidates + period)
        recurrence_scores = np.sum(features[candidates] * features[forward_index], axis=1)
        scores = 0.82 * reference_scores + 0.18 * recurrence_scores
        best_pos = int(np.argmax(scores))
        candidate = int(candidates[best_pos])
        score = float(scores[best_pos])
        if score < threshold:
            break
        backward.append((candidate, score))
        previous = candidate

    all_boundaries = list(reversed(backward)) + boundaries
    cleaned: list[tuple[int, float]] = []
    for item in all_boundaries:
        if not cleaned or item[0] > cleaned[-1][0]:
            cleaned.append(item)
    return cleaned


def discover_cycles(
    character: str,
    video: Path,
    roi=DEFAULT_ROI,
    analysis_fps: float = 30.0,
    min_period_s: float = 0.8,
    max_period_s: float = 8.0,
):
    video = Path(video).resolve()
    info, sampled_frames, features = _sample_video(video, roi, analysis_fps)
    effective_fps = len(sampled_frames) / max(float(info["duration_ms"]) / 1000.0, 1e-6)
    period, period_score = estimate_period(features, effective_fps, min_period_s, max_period_s)
    motion = _motion_features(features)
    anchor = _choose_anchor(features, motion, period)
    sample_boundaries = _walk_boundaries(features, anchor, period, period_score)

    if len(sample_boundaries) < 4:
        raise RuntimeError(
            f"只自动找到 {len(sample_boundaries)} 个重复姿态边界，"
            "至少需要 4 个边界（3 个完整 cycle）才开始训练。"
        )

    boundary_scores = [score for _, score in sample_boundaries[1:]]
    confidence = float(np.median(boundary_scores)) if boundary_scores else 0.0
    if confidence < 0.36:
        raise RuntimeError(
            f"重复姿态边界置信度太低（{confidence:.3f}），本次录像保存但不自动加入训练。"
        )

    boundaries = []
    for sample_index, score in sample_boundaries:
        frame = int(sampled_frames[min(sample_index, len(sampled_frames) - 1)])
        boundaries.append(
            {
                "frame": frame,
                "source": "auto-periodic-vision",
                "score": round(float(score), 5),
            }
        )

    annotation = {
        "schema": 2,
        "character": character,
        "video": str(video),
        "roi": list(roi),
        "video_info": info,
        "boundaries": boundaries,
        "auto_detection": {
            "analysis_fps": analysis_fps,
            "effective_fps": effective_fps,
            "period_sample_frames": period,
            "period_s": period / max(effective_fps, 1e-6),
            "period_score": period_score,
            "boundary_confidence": confidence,
            "cycle_count": len(boundaries) - 1,
        },
    }
    return annotation


def save_annotation(character: str, video: Path, annotation: dict) -> Path:
    root = character_root(character)
    target = root / "annotations" / f"{Path(video).stem}.cycles.json"
    write_json(target, annotation)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Automatically discover repeated full basic-attack cycles")
    parser.add_argument("--character", required=True)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--roi", default=None, help="normalized x,y,w,h")
    parser.add_argument("--analysis-fps", type=float, default=30.0)
    args = parser.parse_args()

    roi = parse_roi(args.roi)
    annotation = discover_cycles(args.character, args.video, roi=roi, analysis_fps=args.analysis_fps)
    target = save_annotation(args.character, args.video, annotation)
    auto = annotation["auto_detection"]
    print(
        f"auto cycles={auto['cycle_count']} period={auto['period_s']:.3f}s "
        f"confidence={auto['boundary_confidence']:.3f} -> {target}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
