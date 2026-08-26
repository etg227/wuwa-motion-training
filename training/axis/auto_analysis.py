from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm <= EPS:
        return np.zeros_like(value, dtype=np.float32)
    return value / norm


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError("matrix must be 2-D")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.maximum(norms, EPS)


def smooth_signal(values: np.ndarray, radius: int = 1) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float32).reshape(-1)
    radius = max(0, int(radius))
    if radius <= 0 or len(signal) <= 2:
        return signal.copy()
    width = radius * 2 + 1
    padded = np.pad(signal, (radius, radius), mode="edge")
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def robust_zscore(values: np.ndarray) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(signal) == 0:
        return signal.copy()
    median = float(np.median(signal))
    mad = float(np.median(np.abs(signal - median)))
    scale = max(1.4826 * mad, float(np.std(signal)) * 0.25, EPS)
    return ((signal - median) / scale).astype(np.float32)


def detect_peaks(
    values: np.ndarray,
    *,
    min_distance: int = 2,
    threshold_z: float = 2.0,
    max_peaks: int | None = None,
) -> list[int]:
    signal = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(signal) < 3:
        return []
    z = robust_zscore(signal)
    candidates = [
        index
        for index in range(1, len(signal) - 1)
        if z[index] >= threshold_z
        and signal[index] >= signal[index - 1]
        and signal[index] >= signal[index + 1]
    ]
    candidates.sort(key=lambda index: float(signal[index]), reverse=True)
    selected: list[int] = []
    min_distance = max(1, int(min_distance))
    for index in candidates:
        if all(abs(index - other) >= min_distance for other in selected):
            selected.append(index)
            if max_peaks is not None and len(selected) >= max_peaks:
                break
    selected.sort()
    return selected


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    centers: np.ndarray
    counts: np.ndarray


def cluster_signatures(
    signatures: np.ndarray,
    *,
    similarity_threshold: float = 0.74,
) -> ClusterResult:
    matrix = normalize_rows(signatures)
    if len(matrix) == 0:
        return ClusterResult(
            labels=np.empty(0, dtype=np.int32),
            centers=np.empty((0, matrix.shape[1]), dtype=np.float32),
            counts=np.empty(0, dtype=np.int32),
        )

    centers: list[np.ndarray] = []
    sums: list[np.ndarray] = []
    counts: list[int] = []
    labels = np.empty(len(matrix), dtype=np.int32)

    for row_index, row in enumerate(matrix):
        if not centers:
            sums.append(row.copy())
            counts.append(1)
            centers.append(row.copy())
            labels[row_index] = 0
            continue

        sims = np.asarray([float(np.dot(row, center)) for center in centers], dtype=np.float32)
        best = int(np.argmax(sims))
        if float(sims[best]) >= similarity_threshold:
            labels[row_index] = best
            sums[best] += row
            counts[best] += 1
            centers[best] = normalize_vector(sums[best])
        else:
            labels[row_index] = len(centers)
            sums.append(row.copy())
            counts.append(1)
            centers.append(row.copy())

    return ClusterResult(
        labels=labels,
        centers=np.stack(centers).astype(np.float32),
        counts=np.asarray(counts, dtype=np.int32),
    )


@dataclass(frozen=True)
class PeriodEstimate:
    lag: int
    score: float
    anchor: int
    scores: np.ndarray


def _rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float32)
    width = max(1, min(int(width), len(signal)))
    if width <= 1:
        return signal.copy()
    cumulative = np.concatenate(([0.0], np.cumsum(signal, dtype=np.float64)))
    return ((cumulative[width:] - cumulative[:-width]) / width).astype(np.float32)


def estimate_repetition_period(
    features: np.ndarray,
    *,
    min_lag: int,
    max_lag: int,
    near_best: float = 0.035,
) -> PeriodEstimate | None:
    matrix = normalize_rows(features)
    count = len(matrix)
    min_lag = max(2, int(min_lag))
    max_lag = min(int(max_lag), max(0, count // 2))
    if count < 12 or max_lag < min_lag:
        return None

    lags = np.arange(min_lag, max_lag + 1, dtype=np.int32)
    scores = np.full(len(lags), -1.0, dtype=np.float32)
    anchors = np.zeros(len(lags), dtype=np.int32)

    for pos, lag in enumerate(lags):
        similarity = np.sum(matrix[:-lag] * matrix[lag:], axis=1)
        if len(similarity) < max(6, lag // 2):
            continue
        window = max(6, min(len(similarity), int(round(lag * 0.85))))
        local = _rolling_mean(similarity, window)
        if len(local) == 0:
            continue
        best_local = int(np.argmax(local))
        scores[pos] = float(local[best_local])
        anchors[pos] = best_local + window // 2

    valid = np.flatnonzero(scores > -0.5)
    if len(valid) == 0:
        return None

    peak_positions: list[int] = []
    for pos in valid:
        left = scores[pos - 1] if pos > 0 else -np.inf
        right = scores[pos + 1] if pos + 1 < len(scores) else -np.inf
        if scores[pos] >= left and scores[pos] >= right:
            peak_positions.append(int(pos))
    if not peak_positions:
        peak_positions = [int(valid[np.argmax(scores[valid])])]

    best_score = max(float(scores[pos]) for pos in peak_positions)
    strong = [pos for pos in peak_positions if float(scores[pos]) >= best_score - near_best]
    chosen = min(strong, key=lambda pos: int(lags[pos]))
    if float(scores[chosen]) < 0.18:
        return None

    return PeriodEstimate(
        lag=int(lags[chosen]),
        score=float(scores[chosen]),
        anchor=int(anchors[chosen]),
        scores=scores,
    )


def align_period_boundaries(
    features: np.ndarray,
    period: int,
    anchor: int,
    *,
    search_ratio: float = 0.16,
    minimum_similarity: float = 0.10,
) -> list[tuple[int, float]]:
    matrix = normalize_rows(features)
    count = len(matrix)
    period = max(2, int(period))
    anchor = int(np.clip(anchor, 0, max(0, count - 1)))
    if count < period * 2:
        return []

    refs = []
    for offset in range(-4, 5):
        index = anchor + offset * period
        if 0 <= index < count:
            refs.append(matrix[index])
    if len(refs) < 2:
        return []
    reference = normalize_vector(np.mean(np.stack(refs), axis=0))
    radius = max(2, int(round(period * search_ratio)))

    found: list[tuple[int, float]] = [(anchor, float(np.dot(matrix[anchor], reference)))]

    previous = anchor
    while previous + int(period * 0.65) < count:
        expected = previous + period
        left = max(previous + max(2, int(period * 0.65)), expected - radius)
        right = min(count - 1, expected + radius)
        if right < left:
            break
        indexes = np.arange(left, right + 1, dtype=np.int32)
        similarities = matrix[indexes] @ reference
        best_pos = int(np.argmax(similarities))
        index = int(indexes[best_pos])
        score = float(similarities[best_pos])
        if score < minimum_similarity:
            break
        found.append((index, score))
        previous = index

    previous = anchor
    backward: list[tuple[int, float]] = []
    while previous - int(period * 0.65) >= 0:
        expected = previous - period
        left = max(0, expected - radius)
        right = min(previous - max(2, int(period * 0.65)), expected + radius)
        if right < left:
            break
        indexes = np.arange(left, right + 1, dtype=np.int32)
        similarities = matrix[indexes] @ reference
        best_pos = int(np.argmax(similarities))
        index = int(indexes[best_pos])
        score = float(similarities[best_pos])
        if score < minimum_similarity:
            break
        backward.append((index, score))
        previous = index

    combined = list(reversed(backward)) + found
    cleaned: list[tuple[int, float]] = []
    for index, score in combined:
        if not cleaned or index > cleaned[-1][0]:
            cleaned.append((index, score))
    return cleaned


def loop_support(
    event_indexes: np.ndarray,
    cluster_labels: np.ndarray,
    boundaries: list[int],
    *,
    phase_tolerance: float = 0.055,
) -> np.ndarray:
    events = np.asarray(event_indexes, dtype=np.int32)
    labels = np.asarray(cluster_labels, dtype=np.int32)
    support = np.ones(len(events), dtype=np.int32)
    if len(boundaries) < 3 or len(events) == 0:
        return support

    loop_rows: list[tuple[int, int, float]] = []
    for event_pos, sample_index in enumerate(events):
        for boundary_index in range(len(boundaries) - 1):
            left = boundaries[boundary_index]
            right = boundaries[boundary_index + 1]
            if left <= sample_index < right:
                phase = (sample_index - left) / max(1, right - left)
                loop_rows.append((event_pos, boundary_index, float(phase)))
                break

    for event_pos, loop, phase in loop_rows:
        label = int(labels[event_pos])
        matched_loops = {loop}
        for other_pos, other_loop, other_phase in loop_rows:
            if other_pos == event_pos or other_loop == loop:
                continue
            if int(labels[other_pos]) != label:
                continue
            delta = abs(other_phase - phase)
            delta = min(delta, 1.0 - delta)
            if delta <= phase_tolerance:
                matched_loops.add(other_loop)
        support[event_pos] = len(matched_loops)
    return support


def confidence_from_signals(
    activity_z: float,
    recurrence_support: int,
    semantic_confidence: float | None,
) -> float:
    activity_term = float(np.clip((activity_z - 1.0) / 4.0, 0.0, 1.0))
    recurrence_term = float(np.clip((recurrence_support - 1) / 2.0, 0.0, 1.0))
    semantic_term = float(np.clip(semantic_confidence or 0.0, 0.0, 1.0))
    score = 0.40 * activity_term + 0.35 * recurrence_term + 0.25 * semantic_term
    return float(np.clip(score, 0.0, 1.0))
