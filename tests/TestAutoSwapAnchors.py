import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "axis"))

from auto_swap_anchors import (
    _party_slot_views,
    _slot_transition_profile,
    build_swap_payloads,
)


def _normalize(vector):
    value = np.asarray(vector, dtype=np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-6)


class TestAutoSwapAnchors(unittest.TestCase):
    def test_party_descriptor_splits_into_three_vertical_bands(self):
        party = np.arange(24, dtype=np.float32).reshape(2, 12)
        slots = _party_slot_views(party)
        self.assertEqual(slots.shape, (2, 3, 4))
        np.testing.assert_array_equal(slots[0, 0], [0, 1, 2, 3])
        np.testing.assert_array_equal(slots[0, 2], [8, 9, 10, 11])

    def test_slot_profile_reports_dominant_changed_band(self):
        fps = 30.0
        party = np.zeros((40, 12), dtype=np.float32)
        before = np.concatenate((
            _normalize([1, 0, 0, 0]),
            _normalize([0, 1, 0, 0]),
            _normalize([0, 0, 1, 0]),
        ))
        after = np.concatenate((
            _normalize([0, 1, 0, 0]),
            _normalize([0, 1, 0, 0]),
            _normalize([0, 0, 1, 0]),
        ))
        party[:20] = before
        party[20:] = after
        profile = _slot_transition_profile(_party_slot_views(party), 20, fps)
        self.assertEqual(profile["dominant_slot_change"], 1)
        self.assertGreater(profile["slot_change"][0], profile["slot_change"][1])
        self.assertGreater(profile["profile"][0], 0.9)

    def _synthetic_drifting_loops(self):
        fps = 30.0
        loop_samples = 1200
        loops = 4
        total = loop_samples * loops
        party = np.zeros((total, 12), dtype=np.float32)
        ability = np.zeros((total, 6), dtype=np.float32)

        state_a = np.concatenate((
            _normalize([1.0, 0.1, 0.0, 0.0]),
            _normalize([0.0, 1.0, 0.1, 0.0]),
            _normalize([0.0, 0.0, 1.0, 0.1]),
        ))
        state_b = np.concatenate((
            _normalize([0.1, 1.0, 0.0, 0.0]),
            _normalize([1.0, 0.1, 0.0, 0.0]),
            _normalize([0.0, 0.0, 1.0, 0.1]),
        ))

        # 故意让全局 rotation phase 漂移 1 秒以上，但 local anchor→swap gap 固定 200ms。
        transition_offsets = [240, 276, 222, 264]
        anchors = []
        for loop, offset in enumerate(transition_offsets):
            base = loop * loop_samples
            party[base:base + offset] = state_a
            party[base + offset:base + loop_samples] = state_b
            anchor_sample = base + offset - 6
            anchors.append({
                "sample_index": anchor_sample,
                "ms": int(round(anchor_sample / fps * 1000.0)),
                "cluster": 7,
                "recurrence_support": 4,
            })

        # 很轻的 deterministic noise，避免完全理想化零方差。
        rng = np.random.default_rng(42)
        party += rng.normal(scale=0.002, size=party.shape).astype(np.float32)
        ability += rng.normal(scale=0.001, size=ability.shape).astype(np.float32)

        frames = np.arange(total, dtype=np.int32)
        times_ms = np.rint(frames / fps * 1000.0).astype(np.int64)
        boundaries = [i * loop_samples for i in range(loops + 1)]
        return party, ability, fps, boundaries, times_ms, frames, anchors

    def test_local_anchor_recovers_swap_despite_global_phase_drift(self):
        party, ability, fps, boundaries, times_ms, frames, anchors = self._synthetic_drifting_loops()
        swaps, windows = build_swap_payloads(
            party,
            ability,
            fps,
            boundaries,
            times_ms,
            frames,
            anchors,
            source_fps=fps,
            analysis_left=0,
            analysis_right=len(party) - 1,
        )
        self.assertGreaterEqual(swaps["recurring_transition_count"], 1)
        matching = [
            row for row in swaps["transitions"]
            if row["outgoing_cluster"] == 7
        ]
        self.assertTrue(matching)
        row = matching[0]
        self.assertEqual(row["support_loops"], 4)
        self.assertGreater(row["rotation_spread_ms"], 500.0)
        self.assertLessEqual(row["local_visual_spread_ms"], 40.0)
        self.assertEqual(row["outgoing_gap_ms"]["median"], 200.0)
        self.assertEqual(row["status"], "strong")
        self.assertTrue(any(window["outgoing_cluster"] == 7 for window in windows["windows"]))
        self.assertFalse(windows["execution_ready"])

    def test_without_recurring_anchor_no_strong_window_is_created(self):
        party, ability, fps, boundaries, times_ms, frames, _anchors = self._synthetic_drifting_loops()
        swaps, windows = build_swap_payloads(
            party,
            ability,
            fps,
            boundaries,
            times_ms,
            frames,
            [],
            source_fps=fps,
            analysis_left=0,
            analysis_right=len(party) - 1,
        )
        self.assertEqual(swaps["anchored_candidate_count"], 0)
        self.assertEqual(swaps["recurring_transition_count"], 0)
        self.assertEqual(windows["windows"], [])
        self.assertEqual(
            swaps["matching_diagnostics"]["hint"],
            "insufficient_recurring_local_anchors",
        )


if __name__ == "__main__":
    unittest.main()
