import unittest

from training.motion.ready_evidence import ReadyEvidenceTracker, combine_ready_evidence


class TestReadyEvidence(unittest.TestCase):
    def test_low_noise_does_not_accumulate_to_ready(self):
        tracker = ReadyEvidenceTracker()
        states = [tracker.update(0, 0.30) for _ in range(6)]
        self.assertFalse(any(row.ready for row in states))

    def test_two_adjacent_moderate_observations_can_accumulate(self):
        tracker = ReadyEvidenceTracker()
        first = tracker.update(0, 0.70)
        second = tracker.update(0, 0.70)
        self.assertFalse(first.ready)
        self.assertTrue(second.ready)
        self.assertGreaterEqual(second.evidence, 0.85)

    def test_single_strong_observation_enters_immediately(self):
        tracker = ReadyEvidenceTracker()
        result = tracker.update(0, 0.95)
        self.assertTrue(result.entered)
        self.assertTrue(result.ready)

    def test_hysteresis_survives_brief_drop_then_exits(self):
        tracker = ReadyEvidenceTracker()
        tracker.update(0, 0.95)
        brief = tracker.update(0, 0.0)
        self.assertTrue(brief.ready)
        result = brief
        for _ in range(8):
            result = tracker.update(0, 0.0)
            if not result.ready:
                break
        self.assertFalse(result.ready)
        self.assertTrue(result.exited)

    def test_mode_switch_clears_old_evidence(self):
        tracker = ReadyEvidenceTracker()
        self.assertTrue(tracker.update(0, 0.95).ready)
        switched = tracker.update(1, 0.0)
        self.assertFalse(switched.ready)
        self.assertLess(switched.evidence, 0.1)

    def test_combiner_is_causal_and_newest_has_full_weight(self):
        self.assertAlmostEqual(combine_ready_evidence([1.0]), 1.0, places=6)
        self.assertGreater(combine_ready_evidence([0.7, 0.7]), 0.85)


if __name__ == "__main__":
    unittest.main()
