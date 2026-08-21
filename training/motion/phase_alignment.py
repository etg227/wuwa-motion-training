from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

import phase_training_legacy as core


ALIGNMENT_BINS = 32
ALIGNMENT_IMAGE_SIZE = 20


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


def _frame_feature(frame: np.ndarray, size: int = ALIGNMENT_IMAGE_SIZE) -> np.ndarray:
    height, width = frame.shape[:2]
    y0, y1 = int(height * 0.08), int(height * 0.94)
    x0, x1 = int(width * 0.08), int(width * 0.92)
    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    gray_u8 = np.asarray(gray, dtype=np.uint8)

    # Normalize per frame so map brightness/background contrast contributes less.
    gray_f = gray_u8.astype(np.float32) / 255.0
    gray_f = (gray_f - float(gray_f.mean())) / max(float(gray_f.std()), 1e-4)
    edge = cv2.Canny(gray_u8, 55, 145).astype(np.float32) / 255.0
    return np.concatenate((gray_f.reshape(-1) * 0.35, edge.reshape(-1) * 0.65))


def _cycle_sequence(root: Path, item, bins: int = ALIGNMENT_BINS) -> np.ndarray:
    with np.load(root / item["cycle"]) as data:
        frames = data["frames"].copy()
    if len(frames) < 8:
        raise RuntimeError(f"cycle too short for phase alignment: {item['cycle']}")

    positions = np.linspace(0, len(frames), bins, endpoint=False)
    indexes = np.clip(np.floor(positions).astype(np.int32), 0, len(frames) - 1)
    appearance = np.stack([_frame_feature(frames[index]) for index in indexes])

    # Circular motion descriptor keeps the temporal ordering needed to discover
    # a phase shift while reducing dependence on a static background.
    motion = appearance - np.roll(appearance, 1, axis=0)

    # Normalize each temporal bin independently before combining appearance/motion.
    app_norm = np.stack([_normalize(row) for row in appearance])
    motion_norm = np.stack([_normalize(row) for row in motion])
    sequence = np.concatenate((app_norm * 0.25, motion_norm * 0.75), axis=1)
    return np.stack([_normalize(row) for row in sequence])


def _shift_scores(sequence: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Score offset s where local bin k corresponds to reference bin k+s."""
    scores = []
    for shift in range(len(sequence)):
        shifted_reference = np.roll(reference, -shift, axis=0)
        scores.append(float(np.mean(np.sum(sequence * shifted_reference, axis=1))))
    return np.asarray(scores, dtype=np.float32)


def _best_shift(sequence: np.ndarray, reference: np.ndarray):
    scores = _shift_scores(sequence, reference)
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])

    # Sub-bin parabolic interpolation around the circular maximum.
    left = float(scores[(best_index - 1) % len(scores)])
    center = best_score
    right = float(scores[(best_index + 1) % len(scores)])
    denominator = left - 2.0 * center + right
    delta = 0.0
    if abs(denominator) > 1e-6:
        delta = 0.5 * (left - right) / denominator
        delta = float(np.clip(delta, -0.5, 0.5))

    refined = (best_index + delta) % len(scores)
    offset = refined / len(scores)

    # Adjacent shifts are expected to be similar on smooth animations. Compare
    # the peak with the overall score distribution rather than only shift +/- 1.
    prominence = best_score - float(np.percentile(scores, 65))
    return offset, best_score, prominence, scores


def _pair_best_scores(sequences: list[np.ndarray]) -> np.ndarray:
    count = len(sequences)
    matrix = np.eye(count, dtype=np.float32)
    for left in range(count):
        for right in range(left + 1, count):
            _offset, score, _prominence, _scores = _best_shift(
                sequences[right], sequences[left]
            )
            matrix[left, right] = score
            matrix[right, left] = score
    return matrix


def _choose_reference(sequences: list[np.ndarray]) -> int:
    if len(sequences) <= 1:
        return 0
    pair_scores = _pair_best_scores(sequences)
    # Medoid under best circular alignment: choose the cycle that resembles the
    # largest number of peers, not simply the first recording.
    peer_score = []
    for index in range(len(sequences)):
        values = np.delete(pair_scores[index], index)
        peer_score.append(float(np.median(values)))
    return int(np.argmax(peer_score))


def _circular_mean(offsets, weights=None) -> float:
    if not offsets:
        return 0.0
    values = np.asarray(offsets, dtype=np.float64)
    if weights is None:
        weights_array = np.ones(len(values), dtype=np.float64)
    else:
        weights_array = np.maximum(np.asarray(weights, dtype=np.float64), 1e-4)
    vector = np.sum(weights_array * np.exp(1j * 2.0 * math.pi * values))
    if abs(vector) < 1e-8:
        return float(values[0] % 1.0)
    return float((np.angle(vector) / (2.0 * math.pi)) % 1.0)


def _circular_distance(a: float, b: float) -> float:
    delta = abs((a - b) % 1.0)
    return min(delta, 1.0 - delta)


def _build_prototype(sequences, offsets, weights):
    aligned = []
    for sequence, offset, weight in zip(sequences, offsets, weights):
        bins = len(sequence)
        integer_shift = int(round(offset * bins)) % bins
        # If local k == global k+offset, moving local samples forward by offset
        # places them into the common global reference phase.
        aligned_sequence = np.roll(sequence, integer_shift, axis=0)
        aligned.append(aligned_sequence * max(float(weight), 0.05))
    prototype = np.sum(aligned, axis=0)
    return np.stack([_normalize(row) for row in prototype])


def align_mode_items(root: Path, members, mode_name: str):
    """Return copies of mode items carrying a globally consistent phase_offset."""
    if not members:
        return [], {
            "mode": mode_name,
            "cycle_count": 0,
            "reference_cycle": None,
            "median_score": None,
        }

    items = [dict(item) for item in members]
    sequences = [_cycle_sequence(root, item) for item in items]
    reference_index = _choose_reference(sequences)
    reference = sequences[reference_index]

    raw_offsets = []
    raw_scores = []
    raw_prominence = []
    for sequence in sequences:
        offset, score, prominence, _scores = _best_shift(sequence, reference)
        raw_offsets.append(offset)
        raw_scores.append(score)
        raw_prominence.append(prominence)

    # One prototype refinement makes the reference less dependent on a single cycle.
    prototype = _build_prototype(
        sequences,
        raw_offsets,
        [max(0.01, score) for score in raw_scores],
    )
    refined_offsets = []
    refined_scores = []
    refined_prominence = []
    for sequence in sequences:
        offset, score, prominence, _scores = _best_shift(sequence, prototype)
        refined_offsets.append(offset)
        refined_scores.append(score)
        refined_prominence.append(prominence)

    # auto_cycle chooses one anchor per source video, so cycles from the same
    # recording should share one phase-zero offset. Enforce a weighted circular
    # consensus per video to suppress per-cycle cross-correlation jitter.
    by_video = defaultdict(list)
    for index, item in enumerate(items):
        video = str(item.get("video", "")) or f"__cycle__{index}"
        by_video[video].append(index)

    final_offsets = list(refined_offsets)
    video_reports = []
    for video, indexes in sorted(by_video.items()):
        offsets = [refined_offsets[index] for index in indexes]
        weights = [
            max(0.01, refined_scores[index]) * max(0.01, refined_prominence[index] + 0.05)
            for index in indexes
        ]
        consensus = _circular_mean(offsets, weights)
        spread = float(np.median([
            _circular_distance(value, consensus) for value in offsets
        ])) if offsets else 0.0
        for index in indexes:
            final_offsets[index] = consensus
        video_reports.append({
            "video": video,
            "cycle_count": len(indexes),
            "phase_offset": consensus,
            "within_video_phase_spread": spread,
            "median_alignment_score": float(np.median([
                refined_scores[index] for index in indexes
            ])),
        })

    cycle_reports = []
    for index, item in enumerate(items):
        item["phase_offset"] = float(final_offsets[index] % 1.0)
        item["phase_alignment_score"] = float(refined_scores[index])
        item["phase_alignment_prominence"] = float(refined_prominence[index])
        cycle_reports.append({
            "cycle": str(item["cycle"]),
            "video": str(item.get("video", "")),
            "phase_offset": item["phase_offset"],
            "score": item["phase_alignment_score"],
            "prominence": item["phase_alignment_prominence"],
        })

    report = {
        "schema": 1,
        "mode": mode_name,
        "method": "circular-cross-correlation-video-consensus-v1",
        "bins": len(sequences[0]),
        "cycle_count": len(items),
        "reference_cycle": str(items[reference_index]["cycle"]),
        "median_score": float(np.median(refined_scores)),
        "minimum_score": float(np.min(refined_scores)),
        "median_prominence": float(np.median(refined_prominence)),
        "video_group_count": len(video_reports),
        "video_groups": video_reports,
        "cycles": cycle_reports,
    }
    return items, report


def align_all_modes(root: Path, mode_items, mode_index: dict):
    aligned_modes = []
    reports = []
    assignment_map = {
        str(row.get("cycle", "")): row
        for row in mode_index.get("assignments", [])
        if isinstance(row, dict)
    }

    for mode, members in enumerate(mode_items):
        aligned, report = align_mode_items(root, members, f"mode_{mode}")
        aligned_modes.append(aligned)
        reports.append(report)

        mode_dir = root / "modes" / f"mode_{mode}"
        core._write_manifest(mode_dir / "manifest.jsonl", aligned)
        (mode_dir / "phase_alignment.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        meta = mode_index["modes"][mode]
        meta["phase_alignment"] = {
            "method": report["method"],
            "reference_cycle": report["reference_cycle"],
            "median_score": report["median_score"],
            "minimum_score": report["minimum_score"],
            "median_prominence": report["median_prominence"],
            "video_group_count": report["video_group_count"],
            "report": str((mode_dir / "phase_alignment.json").relative_to(root)),
        }

        for item in aligned:
            row = assignment_map.get(str(item["cycle"]))
            if row is not None:
                row["phase_offset"] = item["phase_offset"]
                row["phase_alignment_score"] = item["phase_alignment_score"]

    return aligned_modes, reports


class AlignedCycleWindowDataset(core.CycleWindowDataset):
    """Cycle dataset that applies an optional circular phase offset per cycle."""

    def __getitem__(self, index):
        cycle_index, end = self.samples[index]
        frames, phases = self._load(cycle_index)
        start = end - self.window + 1
        indexes = np.arange(start, end + 1)
        indexes = np.clip(indexes, 0, len(frames) - 1)
        clip = frames[indexes].astype(np.float32) / 255.0
        if self.augment and core.random.random() < 0.5:
            clip = clip[:, :, ::-1, :].copy()
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2)
        clip = (clip - 0.5) / 0.5

        offset = float(self.items[cycle_index].get("phase_offset", 0.0))
        phase_value = (float(phases[end]) + offset) % 1.0
        phase = torch.tensor(phase_value, dtype=torch.float32)
        return clip, phase
