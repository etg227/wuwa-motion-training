import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "axis"))

from auto_swaps import (
    SwapCandidate,
    adaptive_phase_tolerance,
    build_swap_payloads,
    circular_phase_distance,
    detect_party_transitions,
    recurring_swap_groups,
    swap_matching_diagnostics,
)


def _normalize(vector):
    value = np.asarray(vector, dtype=np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-6)


class TestAutoSwaps(unittest.TestCase):
    def _synthetic(self, loops=4, loop_samples=1200, fps=30.0):
        rng = np.random.default_rng(12)
        dimension = 24
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
                    state + rng.normal(scale=0.012, size=dimension)
                ))
                ability.append(_normalize(rng.normal(size=dimension)))
        return (
            np.stack(party).astype(np.float32),
            np.stack(ability).astype(np.float32),
            [i * loop_samples for i in range(loops + 1)],
            fps,
        )

    def _candidate(self, sample_index, signature, pre, post, score=0.9):
        return SwapCandidate(
            sample_index=sample_index,
            score=score,
            party_z=5.0,
            state_change=0.4,
            persistence=0.9,
            hud_ratio=2.0,
            signature=_normalize(signature),
            pre_state=_normalize(pre),
            post_state=_normalize(post),
        )

    def test_circular_phase_distance_wraps(self):
        self.assertAlmostEqual(circular_phase_distance(0.99, 0.01), 0.02)

    def test_adaptive_tolerance_turns_point04_into_few_frames(self):
        boundaries = [0, 1200, 2400, 3600, 4800]
        tolerance = adaptive_phase_tolerance(boundaries, 0.04)
        self.assertLess(tolerance, 0.004)
        self.assertGreater(tolerance, 0.003)

    def test_detects_recurring_three_swap_pattern(self):
        party, ability, boundaries, fps = self._synthetic()
        candidates = detect_party_transitions(party, ability, fps)
        groups = recurring_swap_groups(candidates, boundaries)
        phases = [row["phase"] for row in groups]
        self.assertEqual(len(phases), 3)
        for actual, expected in zip(phases, (0.20, 0.50, 0.80)):
            self.assertLess(abs(actual - expected), 0.006)
        self.assertTrue(all(len(row["occurrences"]) == 4 for row in groups))
        self.assertTrue(all(row["state_consistency"] > 0.9 for row in groups))

    def test_one_loop_spike_does_not_become_recurring_swap(self):
        party, ability, boundaries, fps = self._synthetic()
        rng = np.random.default_rng(99)
        transient = _normalize(rng.normal(size=party.shape[1]))
        party[410:430] = transient
        candidates = detect_party_transitions(party, ability, fps)
        groups = recurring_swap_groups(candidates, boundaries)
        self.assertTrue(
            all(circular_phase_distance(row["phase"], 0.35) > 0.01 for row in groups)
        )

    def test_phase_only_matches_more_than_few_frames_are_rejected(self):
        boundaries = [0, 1200, 2400, 3600, 4800]
        signature = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        pre = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        post = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        phases = [0.200, 0.220, 0.185, 0.215]
        candidates = [
            self._candidate(
                loop * 1200 + int(round(phase * 1200)),
                signature,
                pre,
                post,
            )
            for loop, phase in enumerate(phases)
        ]
        self.assertEqual(recurring_swap_groups(candidates, boundaries), [])

    def test_same_phase_but_different_hud_transition_identity_is_rejected(self):
        boundaries = [0, 1200, 2400, 3600, 4800]
        candidates = []
        for loop in range(4):
            basis = np.eye(12, dtype=np.float32)
            candidates.append(self._candidate(
                loop * 1200 + 600,
                basis[loop],
                basis[4 + loop],
                basis[8 + loop],
            ))
        self.assertEqual(recurring_swap_groups(candidates, boundaries), [])

    def test_diagnostics_separates_tight_phase_from_stable_identity(self):
        boundaries = [0, 1200, 2400, 3600, 4800]
        signature = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        pre = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        post = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        phases = [0.200, 0.210, 0.220, 0.230]
        candidates = [
            self._candidate(loop * 1200 + int(phase * 1200), signature, pre, post)
            for loop, phase in enumerate(phases)
        ]
        diagnostics = swap_matching_diagnostics(
            candidates, boundaries, period_ms=40000.0
        )
        self.assertEqual(diagnostics["cross_loop_pairs"], 6)
        self.assertEqual(diagnostics["phase_near_pairs"], 0)
        self.assertEqual(diagnostics["identity_compatible_pairs"], 6)
        self.assertEqual(diagnostics["phase_and_identity_pairs"], 0)
        self.assertEqual(
            diagnostics["hint"],
            "phase_gate_likely_too_tight_or_loop_boundaries_drift",
        )
        nearest = diagnostics["identity_compatible_phase_distance"]["nearest_other_loop_ms"]
        self.assertIsNotNone(nearest)
        self.assertGreater(nearest["median"], diagnostics["phase_tolerance_ms"])

    def test_diagnostics_separates_identity_failure_from_phase_overlap(self):
        boundaries = [0, 1200, 2400, 3600, 4800]
        basis = np.eye(12, dtype=np.float32)
        candidates = [
            self._candidate(
                loop * 1200 + 600,
                basis[loop],
                basis[4 + loop],
                basis[8 + loop],
            )
            for loop in range(4)
        ]
        diagnostics = swap_matching_diagnostics(candidates, boundaries)
        self.assertEqual(diagnostics["phase_near_pairs"], 6)
        self.assertEqual(diagnostics["identity_compatible_pairs"], 0)
        self.assertEqual(diagnostics["phase_and_identity_pairs"], 0)
        self.assertEqual(
            diagnostics["hint"], "hud_identity_gate_or_feature_instability"
        )
        failures = diagnostics["phase_near_identity_gate_failures"]
        self.assertEqual(failures["any_identity_gate"], 6)

    def test_build_payload_attaches_recurring_outgoing_cluster(self):
        party, ability, boundaries, fps = self._synthetic()
        count = len(party)
        times_ms = np.rint(np.arange(count) / fps * 1000.0).astype(np.int64)
        frames = np.arange(count, dtype=np.int32)
        events = []
        for loop_index in range(4):
            base = loop_index * 1200
            for offset, cluster in ((216, 7), (576, 9), (936, 11)):
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
        self.assertLess(swaps["effective_phase_tolerance"], 0.004)
        self.assertAlmostEqual(swaps["strong_spread_limit_ms"], 200.0, places=1)
        self.assertEqual(
            [row["outgoing_cluster"] for row in swaps["transitions"]],
            [7, 9, 11],
        )
        self.assertTrue(all(row["execution_ready"] is False for row in windows["windows"]))
        self.assertTrue(all(row["visual_spread_ms"] >= 33.3 for row in swaps["transitions"]))
        diagnostics = swaps["matching_diagnostics"]
        self.assertGreater(diagnostics["phase_and_identity_pairs"], 0)
        self.assertEqual(diagnostics["recurring_group_count"], 3)

    def test_invalid_fps_rejected(self):
        party = np.zeros((12, 4), dtype=np.float32)
        ability = np.zeros((12, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            detect_party_transitions(party, ability, 0.0)


if __name__ == "__main__":
    unittest.main()
