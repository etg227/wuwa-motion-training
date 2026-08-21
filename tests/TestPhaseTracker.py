import unittest

from training.motion.phase_tracker import CircularPhaseTracker, signed_circular_delta


class TestPhaseTracker(unittest.TestCase):
    def test_signed_delta_wraps_across_zero(self):
        self.assertAlmostEqual(signed_circular_delta(0.02, 0.98), 0.04, places=6)
        self.assertAlmostEqual(signed_circular_delta(0.98, 0.02), -0.04, places=6)

    def test_tracker_progresses_through_wrap(self):
        tracker = CircularPhaseTracker({0: 1.0})
        first = tracker.update(0, 0.96, 0.0)
        second = tracker.update(0, 0.995, 0.04)
        third = tracker.update(0, 0.035, 0.04)
        self.assertAlmostEqual(first.phase, 0.96, places=6)
        self.assertLess(abs(signed_circular_delta(second.phase, 0.995)), 0.04)
        self.assertLess(abs(signed_circular_delta(third.phase, 0.035)), 0.04)

    def test_single_large_visual_outlier_is_rejected(self):
        tracker = CircularPhaseTracker({0: 1.0})
        tracker.update(0, 0.10, 0.0)
        normal = tracker.update(0, 0.14, 0.04)
        outlier = tracker.update(0, 0.72, 0.04)
        self.assertFalse(normal.rejected)
        self.assertTrue(outlier.rejected)
        # Motion clock should remain near 0.18 instead of following the 0.72 jump.
        self.assertLess(abs(signed_circular_delta(outlier.phase, 0.18)), 0.05)

    def test_repeated_disagreement_eventually_reanchors(self):
        tracker = CircularPhaseTracker({0: 1.0}, reanchor_after=3)
        tracker.update(0, 0.10, 0.0)
        a = tracker.update(0, 0.65, 0.04)
        b = tracker.update(0, 0.69, 0.04)
        c = tracker.update(0, 0.73, 0.04)
        self.assertTrue(a.rejected)
        self.assertTrue(b.rejected)
        self.assertTrue(c.reanchored)
        self.assertLess(abs(signed_circular_delta(c.phase, 0.73)), 0.05)

    def test_mode_change_resets_without_dragging_old_phase(self):
        tracker = CircularPhaseTracker({0: 1.0, 1: 2.0})
        tracker.update(0, 0.80, 0.0)
        switched = tracker.update(1, 0.25, 0.04)
        self.assertTrue(switched.reanchored)
        self.assertAlmostEqual(switched.phase, 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
