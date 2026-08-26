import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "axis"))

from auto_swaps import (
    build_swap_payloads,
    circular_phase_distance,
    detect_party_transitions,
    recurring_swap_groups,
)


def _normalize(vector):
    value = np.asarray(vector, dtype=np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-6)


class TestAutoSwaps(unittest.TestCase):
    def _synthetic(self, loops=4, loop_samples=100, fps=30.0):
        rng = np.random.default_rng(12)
        dimension = 36
        states = [_normalize(rng.normal(size=dimension)) for _ in range(3)]
        party = []
        ability = []
        for _loop in range(loops):
            for index in range(loop_samples):
                phase = index / loop_samples
                if phase < 0.20:
                    state = states[0]
                elif phase < 0.50:
                    state = states[1]
                elif phase < 0.80:
                    state = states[2]
                else:
                    state = states[0]
                party.append(_normalize(
                    state + rng.normal(scale=0.015, size=dimension)
                ))
                ability.append(_normalize(rng.normal(size=dimension)))
        return (
            np.stack(party).astype(np.float32),
            np.stack(ability).astype(np.float32),
            [i * loop_samples for i in range(loops + 1)],
            fps,
        )

    def test_circular_phase_distance_wraps(self):
        self.assertAlmostEqual(circular_phase_distance(0.99, 0.01), 0.02)

    def test_detects_recurring_three_swap_pattern(self):
        party, ability, boundaries, fps = self._synthetic()
        candidates = detect_party_transitions(party, ability, fps)
        groups = recurring_swap_groups(candidates, boundaries)
        phases = [row["phase"] for row in groups]
        self.assertEqual(len(phases), 3)
        for actual, expected in zip(phases, (0.20, 0.50, 0.80)):
            self.assertLess(abs(actual - expected), 0.025)
        self.assertTrue(all(len(row["occurrences"]) == 4 for row in groups))

    def test_one_loop_spike_does_not_become_recurring_swap(self):
        party, ability, boundaries, fps = self._synthetic()
        rng = np.random.default_rng(99)
        transient = _normalize(rng.normal(size=party.shape[1]))
        party[34:40] = transient
        candidates = detect_party_transitions(party, ability, fps)
        groups = recurring_swap_groups(candidates, boundaries)
        self.assertTrue(
            all(circular_phase_distance(row["phase"], 0.35) > 0.04 for row in groups)
        )

    def test_build_payload_attaches_recurring_outgoing_cluster(self):
        party, ability, boundaries, fps = self._synthetic()
        count = len(party)
        times_ms = np.rint(np.arange(count) / fps * 1000.0).astype(np.int64)
        frames = np.arange(count, dtype=np.int32)
        events = []
        for loop_index in range(4):
            base = loop_index * 100
            for offset, cluster in ((18, 7), (48, 9), (78, 11)):
                sample = base + offset
                events.append({
                    "ms": int(times_ms[sample]),
                    "cluster": cluster,
                    "recurrence_support": 4,
                })

        swaps, windows = build_swap_payloads(
            party,
            ability,
            fps,
            boundaries,
            times_ms,
            frames,
            events,
            source_fps=fps,
            analysis_left=0,
            analysis_right=count - 1,
        )
        self.assertEqual(swaps["strong_transition_count"], 3)
        self.assertEqual(len(windows["windows"]), 3)
        self.assertEqual(
            [row["outgoing_cluster"] for row in swaps["transitions"]],
            [7, 9, 11],
        )
        self.assertTrue(all(row["execution_ready"] is False for row in windows["windows"]))
        self.assertTrue(all(row["visual_spread_ms"] >= 33.3 for row in swaps["transitions"]))

    def test_invalid_fps_rejected(self):
        party = np.zeros((12, 4), dtype=np.float32)
        ability = np.zeros((12, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            detect_party_transitions(party, ability, 0.0)


if __name__ == "__main__":
    unittest.main()
