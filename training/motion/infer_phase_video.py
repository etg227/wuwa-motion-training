from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from common import crop_roi, parse_roi
from train_phase_model import PhaseNet, vector_phase


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize learned basic-attack cycle phase on a local video")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--roi", help="normalized x,y,w,h; should match dataset marking ROI")
    parser.add_argument("--size", type=int, default=112)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    window = int(checkpoint["window"])
    model = PhaseNet()
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    roi = parse_roi(args.roi)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    history = deque(maxlen=window)
    previous_phase = None
    cycle_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = crop_roi(frame, roi)
        image = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        history.append(image)
        while len(history) < window:
            history.appendleft(image)
        clip = np.stack(history)
        tensor = torch.from_numpy(clip).permute(0, 3, 1, 2).unsqueeze(0)
        tensor = ((tensor - 0.5) / 0.5).to(device)
        with torch.no_grad():
            phase = float(vector_phase(model(tensor))[0].cpu())
        if previous_phase is not None and previous_phase > 0.78 and phase < 0.22:
            cycle_count += 1
            print(f"cycle wrap #{cycle_count} phase {previous_phase:.3f}->{phase:.3f}")
        previous_phase = phase

        bar_left, bar_top, bar_width = 20, 36, 360
        cv2.rectangle(frame, (bar_left, bar_top), (bar_left + bar_width, bar_top + 18), (255, 255, 255), 1)
        cv2.rectangle(frame, (bar_left, bar_top), (bar_left + round(bar_width * phase), bar_top + 18), (255, 255, 255), -1)
        cv2.putText(frame, f"combo phase={phase*100:5.1f}% cycles={cycle_count}", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Motion phase inference", frame)
        key = cv2.waitKey(max(1, round(1000 / fps))) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
