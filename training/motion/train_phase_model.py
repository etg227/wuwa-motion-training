from __future__ import annotations

import argparse
import json
import math
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import character_root


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


class PhaseNet(nn.Module):
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
        self.head = nn.Linear(128, 2)

    def forward(self, x):
        batch, steps, channels, height, width = x.shape
        features = self.encoder(x.reshape(batch * steps, channels, height, width)).flatten(1)
        features = features.reshape(batch, steps, -1)
        sequence, _ = self.temporal(features)
        vector = self.head(sequence[:, -1])
        return torch.nn.functional.normalize(vector, dim=-1)


def load_manifest(root):
    path = root / "manifest.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path}; run build_dataset.py first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(model, loader, device):
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a stage-count-free basic-attack cycle phase model")
    parser.add_argument("--character", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window", type=int, default=12, help="number of recent frames used for each prediction")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=227)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = character_root(args.character)
    items = load_manifest(root)
    if len(items) < 3:
        raise SystemExit("至少准备 3 个完整平A cycle；实际建议 30+，并来自多个视频/场景。")
    shuffled = items[:]
    random.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * 0.2))
    val_items = shuffled[:val_count]
    train_items = shuffled[val_count:]

    train_ds = CycleWindowDataset(root, train_items, args.window, args.stride, augment=True)
    val_ds = CycleWindowDataset(root, val_items, args.window, args.stride, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_error = float("inf")
    model_path = root / "models" / "phase_model.pt"

    print(f"device={device} train_cycles={len(train_items)} val_cycles={len(val_items)} train_windows={len(train_ds)}")
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
            total_loss += float(loss) * len(clips)
            total_count += len(clips)
        val_error = evaluate(model, val_loader, device)
        print(f"epoch {epoch:03d} loss={total_loss/max(1,total_count):.5f} val_circular_error={val_error:.4f}")
        if val_error < best_error:
            best_error = val_error
            torch.save({
                "state_dict": model.state_dict(),
                "character": args.character,
                "window": args.window,
                "val_circular_error": val_error,
                "architecture": "PhaseNet-v1",
            }, model_path)

    print(f"best model: {model_path} circular_error={best_error:.4f} ({best_error*100:.1f}% of one full combo cycle)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
