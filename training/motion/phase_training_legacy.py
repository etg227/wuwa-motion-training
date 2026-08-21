from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import character_root, write_json


def phase_vector(phase: torch.Tensor) -> torch.Tensor:
    angle = phase * (2 * math.pi)
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def vector_phase(vector: torch.Tensor) -> torch.Tensor:
    angle = torch.atan2(vector[..., 0], vector[..., 1])
    return torch.remainder(angle / (2 * math.pi), 1.0)


def circular_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    delta = torch.abs(a - b)
    return torch.minimum(delta, 1.0 - delta)


class CycleWindowDataset(Dataset):
    def __init__(self, root: Path, items, window: int, stride: int, augment: bool):
        self.root = root
        self.items = items
        self.window = window
        self.augment = augment
        self.samples = []
        for cycle_index, item in enumerate(items):
            path = root / item["cycle"]
            with np.load(path) as data:
                length = len(data["frames"])
            for end in range(0, length, max(1, stride)):
                self.samples.append((cycle_index, end))

    @lru_cache(maxsize=12)
    def _load(self, cycle_index):
        item = self.items[cycle_index]
        with np.load(self.root / item["cycle"]) as data:
            return data["frames"].copy(), data["phases"].copy()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cycle_index, end = self.samples[index]
        frames, phases = self._load(cycle_index)
        start = end - self.window + 1
        indexes = np.arange(start, end + 1)
        indexes = np.clip(indexes, 0, len(frames) - 1)
        clip = frames[indexes].astype(np.float32) / 255.0
        if self.augment and random.random() < 0.5:
            clip = clip[:, :, ::-1, :].copy()
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2)
        clip = (clip - 0.5) / 0.5
        phase = torch.tensor(float(phases[end]), dtype=torch.float32)
        return clip, phase


class ModeWindowDataset(Dataset):
    def __init__(self, root: Path, items, labels, window: int, stride: int, augment: bool):
        self.root = root
        self.items = items
        self.labels = labels
        self.window = window
        self.augment = augment
        self.samples = []
        for cycle_index, item in enumerate(items):
            with np.load(root / item["cycle"]) as data:
                length = len(data["frames"])
            effective_stride = max(2, stride)
            for end in range(0, length, effective_stride):
                self.samples.append((cycle_index, end))

    @lru_cache(maxsize=16)
    def _load_frames(self, cycle_index):
        with np.load(self.root / self.items[cycle_index]["cycle"]) as data:
            return data["frames"].copy()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cycle_index, end = self.samples[index]
        frames = self._load_frames(cycle_index)
        start = end - self.window + 1
        indexes = np.arange(start, end + 1)
        indexes = np.clip(indexes, 0, len(frames) - 1)
        clip = frames[indexes].astype(np.float32) / 255.0
        if self.augment and random.random() < 0.5:
            clip = clip[:, :, ::-1, :].copy()
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2)
        clip = (clip - 0.5) / 0.5
        return clip, torch.tensor(int(self.labels[cycle_index]), dtype=torch.long)


class FrameTemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.SiLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.temporal = nn.GRU(128, 128, batch_first=True)

    def forward_features(self, x):
        batch, steps, channels, height, width = x.shape
        features = self.encoder(x.reshape(batch * steps, channels, height, width)).flatten(1)
        features = features.reshape(batch, steps, -1)
        sequence, _ = self.temporal(features)
        return sequence[:, -1]


class PhaseNet(FrameTemporalEncoder):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(128, 2)

    def forward(self, x):
        vector = self.head(self.forward_features(x))
        return torch.nn.functional.normalize(vector, dim=-1)


class ModeNet(FrameTemporalEncoder):
    def __init__(self, mode_count: int):
        super().__init__()
        self.head = nn.Linear(128, mode_count)

    def forward(self, x):
        return self.head(self.forward_features(x))


def load_manifest(root: Path):
    path = root / "manifest.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path}; run build_dataset.py first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_manifest(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def evaluate_phase(model, loader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for clips, phases in loader:
            clips, phases = clips.to(device), phases.to(device)
            predicted = vector_phase(model(clips))
            errors.append(circular_error(predicted, phases).cpu())
    if not errors:
        return float("nan")
    return float(torch.cat(errors).mean())


def evaluate_mode(model, loader, device):
    model.eval()
    correct = 0
    count = 0
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            predicted = torch.argmax(model(clips), dim=1)
            correct += int((predicted == labels).sum().item())
            count += int(len(labels))
    return correct / max(1, count)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _cycle_motion_descriptor(root: Path, item, phase_bins: int = 24, size: int = 24) -> np.ndarray:
    """构建对循环起始相位较不敏感、以动作变化为主的描述子。

    每个 cycle 先统一采样为固定相位数，再使用中心区域灰度/边缘序列。
    时间维做 FFT 后丢弃 DC，只保留低频幅值；因此同一套动作即使自动边界
    选到不同起始姿势，也不会仅因为循环整体发生相位平移而被拆成不同 mode。
    """

    with np.load(root / item["cycle"]) as data:
        frames = data["frames"].copy()
    if len(frames) < 8:
        raise RuntimeError(f"cycle too short for mode descriptor: {item['cycle']}")

    sample_positions = np.linspace(0, len(frames), phase_bins, endpoint=False)
    indexes = np.clip(np.floor(sample_positions).astype(np.int32), 0, len(frames) - 1)
    sequence = []

    for frame in frames[indexes]:
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
        sequence.append(np.concatenate((gray_f.reshape(-1), edge.reshape(-1))))

    sequence = np.asarray(sequence, dtype=np.float32)
    sequence -= sequence.mean(axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(sequence, axis=0)).astype(np.float32)
    max_frequency = min(6, spectrum.shape[0] - 1)
    if max_frequency < 1:
        raise RuntimeError("not enough temporal frequencies for mode descriptor")
    descriptor = spectrum[1:max_frequency + 1].reshape(-1)

    duration = float(item.get("duration_s", 0.0))
    descriptor = np.concatenate((descriptor, np.asarray([duration * 0.04], dtype=np.float32)))
    norm = float(np.linalg.norm(descriptor))
    return descriptor / max(norm, 1e-8)


def _pairwise_cosine_distance(descriptors: np.ndarray) -> np.ndarray:
    normalized = _normalize_rows(descriptors.astype(np.float32, copy=False))
    return np.clip(1.0 - normalized @ normalized.T, 0.0, 2.0)


def _spherical_kmeans(descriptors: np.ndarray, k: int, seed: int, attempts: int = 16):
    rng = np.random.default_rng(seed)
    values = _normalize_rows(descriptors.astype(np.float32, copy=False))
    best = None

    for attempt in range(attempts):
        if attempt == 0:
            first = int(np.argmax(np.sum(_pairwise_cosine_distance(values), axis=1)))
            centers = [values[first]]
            while len(centers) < k:
                similarity = np.stack([values @ center for center in centers], axis=1)
                distance = 1.0 - np.max(similarity, axis=1)
                centers.append(values[int(np.argmax(distance))])
            centers = np.stack(centers)
        else:
            indexes = rng.choice(len(values), size=k, replace=False)
            centers = values[indexes].copy()

        labels = np.zeros(len(values), dtype=np.int32)
        valid = True
        for iteration in range(80):
            similarities = values @ centers.T
            new_labels = np.argmax(similarities, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels) and iteration > 0:
                break
            labels = new_labels
            new_centers = []
            for cluster in range(k):
                members = values[labels == cluster]
                if len(members) == 0:
                    valid = False
                    break
                center = members.mean(axis=0)
                center /= max(float(np.linalg.norm(center)), 1e-8)
                new_centers.append(center)
            if not valid:
                break
            centers = np.stack(new_centers)
        if not valid:
            continue

        assigned_similarity = np.sum(values * centers[labels], axis=1)
        inertia = float(np.mean(1.0 - assigned_similarity))
        if best is None or inertia < best[0]:
            best = (inertia, labels.copy(), centers.copy())

    if best is None:
        raise RuntimeError(f"unable to cluster cycles into {k} modes")
    return best


def _silhouette(distance: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    unique = sorted(set(int(value) for value in labels))
    for index, label in enumerate(labels):
        own = np.where(labels == label)[0]
        own = own[own != index]
        if len(own) == 0:
            return -1.0
        a = float(distance[index, own].mean())
        alternatives = []
        for other in unique:
            if other == int(label):
                continue
            members = np.where(labels == other)[0]
            if len(members):
                alternatives.append(float(distance[index, members].mean()))
        if not alternatives:
            return -1.0
        b = min(alternatives)
        scores.append((b - a) / max(a, b, 1e-8))
    return float(np.mean(scores)) if scores else -1.0


def discover_motion_modes(root: Path, items, seed: int, max_modes: int = 4):
    if len(items) < 6:
        labels = np.zeros(len(items), dtype=np.int32)
        return labels, {
            "mode_count": 1,
            "silhouette": None,
            "reason": "fewer than 6 cycles; keep one mode until each possible mode can have >=3 cycles",
        }

    descriptors = np.stack([_cycle_motion_descriptor(root, item) for item in items])
    distance = _pairwise_cosine_distance(descriptors)
    upper = min(max_modes, len(items) // 3)

    best_candidate = None
    for k in range(2, upper + 1):
        inertia, labels, centers = _spherical_kmeans(descriptors, k, seed + k)
        counts = [int(np.sum(labels == cluster)) for cluster in range(k)]
        if min(counts) < 3:
            continue
        silhouette = _silhouette(distance, labels)

        within = float(np.mean([
            1.0 - float(np.dot(descriptors[index], centers[int(labels[index])]))
            for index in range(len(descriptors))
        ]))
        center_distance = _pairwise_cosine_distance(centers)
        between_values = center_distance[np.triu_indices(k, 1)]
        minimum_between = float(np.min(between_values)) if len(between_values) else 0.0

        adjusted = silhouette - 0.035 * max(0, k - 2)
        candidate = {
            "k": k,
            "labels": labels,
            "centers": centers,
            "counts": counts,
            "silhouette": silhouette,
            "adjusted": adjusted,
            "within_distance": within,
            "minimum_center_distance": minimum_between,
            "inertia": inertia,
        }
        if best_candidate is None or adjusted > best_candidate["adjusted"]:
            best_candidate = candidate

    if best_candidate is None:
        labels = np.zeros(len(items), dtype=np.int32)
        return labels, {"mode_count": 1, "silhouette": None, "reason": "no valid >=3-cycle split"}

    separation_ok = best_candidate["minimum_center_distance"] >= max(
        0.055,
        best_candidate["within_distance"] * 1.45,
    )
    if best_candidate["silhouette"] < 0.16 or not separation_ok:
        labels = np.zeros(len(items), dtype=np.int32)
        return labels, {
            "mode_count": 1,
            "silhouette": best_candidate["silhouette"],
            "reason": "candidate split not sufficiently separated",
            "candidate_mode_count": best_candidate["k"],
            "candidate_counts": best_candidate["counts"],
            "within_distance": best_candidate["within_distance"],
            "minimum_center_distance": best_candidate["minimum_center_distance"],
        }

    labels = best_candidate["labels"].copy()
    order = sorted(
        range(best_candidate["k"]),
        key=lambda cluster: (
            float(np.median([
                float(items[index].get("duration_s", 0.0))
                for index in range(len(items))
                if int(labels[index]) == cluster
            ])),
            min(
                str(items[index]["cycle"])
                for index in range(len(items))
                if int(labels[index]) == cluster
            ),
        ),
    )
    remap = {old: new for new, old in enumerate(order)}
    labels = np.asarray([remap[int(label)] for label in labels], dtype=np.int32)

    return labels, {
        "mode_count": int(best_candidate["k"]),
        "silhouette": float(best_candidate["silhouette"]),
        "counts": [
            int(np.sum(labels == mode))
            for mode in range(best_candidate["k"])
        ],
        "within_distance": float(best_candidate["within_distance"]),
        "minimum_center_distance": float(best_candidate["minimum_center_distance"]),
        "descriptor": "phase-shift-invariant-motion-fft-v1",
    }


def build_mode_manifests(root: Path, items, labels: np.ndarray, clustering: dict):
    modes_root = root / "modes"
    if modes_root.exists():
        shutil.rmtree(modes_root)
    modes_root.mkdir(parents=True, exist_ok=True)

    mode_count = int(clustering["mode_count"])
    mode_items = []
    mode_summaries = []
    assignments = []

    for mode in range(mode_count):
        members = []
        for index, item in enumerate(items):
            if int(labels[index]) != mode:
                continue
            enriched = dict(item)
            enriched["motion_mode"] = mode
            members.append(enriched)
            assignments.append({"cycle": str(item["cycle"]), "mode": mode})

        mode_dir = modes_root / f"mode_{mode}"
        manifest_path = mode_dir / "manifest.jsonl"
        _write_manifest(manifest_path, members)
        durations = [float(item.get("duration_s", 0.0)) for item in members]
        mode_summaries.append({
            "id": mode,
            "name": f"mode_{mode}",
            "cycle_count": len(members),
            "median_duration_s": float(np.median(durations)) if durations else 0.0,
            "manifest": str(manifest_path.relative_to(root)),
            "phase_model": str((mode_dir / "phase_model.pt").relative_to(root)),
        })
        mode_items.append(members)

    index = {
        "schema": 1,
        "mode_count": mode_count,
        "automatic": True,
        "clustering": clustering,
        "modes": mode_summaries,
        "assignments": assignments,
    }
    write_json(modes_root / "index.json", index)
    return mode_items, index


def _split_phase_items(items, seed: int):
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * 0.2))
    val_count = min(val_count, len(shuffled) - 2)
    val_count = max(1, val_count)
    return shuffled[val_count:], shuffled[:val_count]


def train_phase_model(root: Path, items, args, device, model_path: Path, mode_name: str):
    if len(items) < 3:
        raise RuntimeError(f"{mode_name} 只有 {len(items)} 个 cycle，至少需要 3 个。")

    mode_seed = args.seed + sum((index + 1) * ord(char) for index, char in enumerate(mode_name))
    train_items, val_items = _split_phase_items(items, mode_seed)
    train_ds = CycleWindowDataset(root, train_items, args.window, args.stride, augment=True)
    val_ds = CycleWindowDataset(root, val_items, args.window, args.stride, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PhaseNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_error = float("inf")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[{mode_name}] device={device} train_cycles={len(train_items)} "
        f"val_cycles={len(val_items)} train_windows={len(train_ds)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for clips, phases in train_loader:
            clips, phases = clips.to(device), phases.to(device)
            target = phase_vector(phases)
            predicted = model(clips)
            loss = loss_fn(predicted, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(clips)
            total_count += len(clips)
        val_error = evaluate_phase(model, val_loader, device)
        print(
            f"[{mode_name}] epoch {epoch:03d} "
            f"loss={total_loss/max(1,total_count):.5f} val_circular_error={val_error:.4f}"
        )
        if val_error < best_error:
            best_error = val_error
            torch.save({
                "state_dict": model.state_dict(),
                "character": args.character,
                "motion_mode": mode_name,
                "window": args.window,
                "val_circular_error": val_error,
                "architecture": "PhaseNet-v2-multimode",
            }, model_path)

    print(
        f"[{mode_name}] best model: {model_path} circular_error={best_error:.4f} "
        f"({best_error*100:.1f}% of one full combo cycle)"
    )
    return best_error


def _stratified_mode_split(mode_items, seed: int):
    train_items = []
    train_labels = []
    val_items = []
    val_labels = []
    rng = random.Random(seed)

    for mode, items in enumerate(mode_items):
        indexes = list(range(len(items)))
        rng.shuffle(indexes)
        val_count = max(1, round(len(indexes) * 0.2))
        val_count = min(val_count, len(indexes) - 2)
        val_indexes = set(indexes[:max(1, val_count)])
        for index, item in enumerate(items):
            if index in val_indexes:
                val_items.append(item)
                val_labels.append(mode)
            else:
                train_items.append(item)
                train_labels.append(mode)

    return train_items, train_labels, val_items, val_labels


def train_mode_classifier(root: Path, mode_items, args, device):
    mode_count = len(mode_items)
    if mode_count <= 1:
        return None

    train_items, train_labels, val_items, val_labels = _stratified_mode_split(mode_items, args.seed)
    train_ds = ModeWindowDataset(
        root, train_items, train_labels, args.window, max(args.stride, 3), augment=True
    )
    val_ds = ModeWindowDataset(
        root, val_items, val_labels, args.window, max(args.stride, 3), augment=False
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    counts = np.bincount(np.asarray(train_labels, dtype=np.int32), minlength=mode_count)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()

    model = ModeNet(mode_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    best_accuracy = -1.0
    model_path = root / "modes" / "mode_classifier.pt"

    mode_epochs = max(12, min(args.epochs, 24))
    print(
        f"[mode-classifier] modes={mode_count} train_cycles={len(train_items)} "
        f"val_cycles={len(val_items)} train_windows={len(train_ds)}"
    )
    for epoch in range(1, mode_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(clips)
            total_count += len(clips)

        accuracy = evaluate_mode(model, val_loader, device)
        print(
            f"[mode-classifier] epoch {epoch:03d} "
            f"loss={total_loss/max(1,total_count):.5f} val_accuracy={accuracy:.3f}"
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                "state_dict": model.state_dict(),
                "character": args.character,
                "mode_count": mode_count,
                "window": args.window,
                "val_accuracy": accuracy,
                "architecture": "ModeNet-v1",
            }, model_path)

    print(f"[mode-classifier] best model: {model_path} val_accuracy={best_accuracy:.3f}")
    return {"path": str(model_path.relative_to(root)), "val_accuracy": best_accuracy}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatically separate character motion modes and train a phase model for each mode"
    )
    parser.add_argument("--character", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window", type=int, default=12, help="number of recent frames used for each prediction")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=227)
    parser.add_argument("--max-modes", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = character_root(args.character)
    items = load_manifest(root)
    if len(items) < 3:
        raise SystemExit("至少准备 3 个完整平A cycle；实际建议继续积累多个视频/场景。")

    labels, clustering = discover_motion_modes(
        root, items, seed=args.seed, max_modes=max(1, args.max_modes)
    )
    mode_items, mode_index = build_mode_manifests(root, items, labels, clustering)

    print(
        f"auto motion modes={mode_index['mode_count']} "
        f"counts={[mode['cycle_count'] for mode in mode_index['modes']]} "
        f"silhouette={clustering.get('silhouette')}"
    )
    if clustering.get("reason"):
        print(f"mode clustering kept single mode: {clustering['reason']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(mode_items) == 1:
        model_path = root / "models" / "phase_model.pt"
        phase_error = train_phase_model(
            root, mode_items[0], args, device, model_path, "mode_0"
        )
        mode_index["modes"][0]["phase_model"] = str(model_path.relative_to(root))
        mode_index["modes"][0]["val_circular_error"] = phase_error
        mode_index["classifier"] = None
    else:
        classifier = train_mode_classifier(root, mode_items, args, device)
        mode_index["classifier"] = classifier
        for mode, members in enumerate(mode_items):
            model_path = root / "modes" / f"mode_{mode}" / "phase_model.pt"
            phase_error = train_phase_model(
                root, members, args, device, model_path, f"mode_{mode}"
            )
            mode_index["modes"][mode]["val_circular_error"] = phase_error

    write_json(root / "modes" / "index.json", mode_index)

    if len(mode_items) > 1:
        router_path = root / "models" / "phase_model.pt"
        router_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "architecture": "MultiModePhaseRouter-v1",
            "character": args.character,
            "mode_index": "modes/index.json",
            "mode_classifier": mode_index.get("classifier"),
            "modes": [
                {
                    "id": mode["id"],
                    "name": mode["name"],
                    "phase_model": mode["phase_model"],
                    "cycle_count": mode["cycle_count"],
                }
                for mode in mode_index["modes"]
            ],
        }, router_path)
        print(f"multimode router bundle: {router_path}")

    print("multimode training complete:")
    for mode in mode_index["modes"]:
        print(
            f"  {mode['name']}: cycles={mode['cycle_count']} "
            f"median_duration={mode['median_duration_s']:.3f}s "
            f"phase_error={mode.get('val_circular_error')}"
        )
    if mode_index.get("classifier"):
        print(
            f"  mode classifier accuracy="
            f"{float(mode_index['classifier']['val_accuracy']):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
