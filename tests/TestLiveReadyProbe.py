import unittest

from training.motion.live_probe_control import ReadyWindowGate


class TestLiveReadyProbeGate(unittest.TestCase):
    def test_high_island_triggers_only_once(self):
        gate = ReadyWindowGate()
        first = gate.update(0, 0.90, 1.0)
        second = gate.update(0, 0.95, 1.04)
        self.assertTrue(first.trigger)
        self.assertFalse(second.trigger)
        self.assertFalse(second.armed)

    def test_rearms_only_after_required_low_frames(self):
        gate = ReadyWindowGate(rearm_frames=2)
        self.assertTrue(gate.update(0, 0.90, 1.0).trigger)
        low1 = gate.update(0, 0.10, 1.1)
        self.assertFalse(low1.armed)
        low2 = gate.update(0, 0.10, 1.2)
        self.assertTrue(low2.rearmed)
        self.assertTrue(low2.armed)
        self.assertTrue(gate.update(0, 0.90, 1.3).trigger)

    def test_blocked_high_island_is_consumed_not_delayed(self):
        gate = ReadyWindowGate()
        blocked = gate.update(0, 0.92, 1.0, can_trigger=False)
        self.assertFalse(blocked.trigger)
        self.assertEqual(blocked.reason, "blocked_high_island")
        later = gate.update(0, 0.95, 1.1, can_trigger=True)
        self.assertFalse(later.trigger)

    def test_mode_change_requires_low_rearm_before_trigger(self):
        gate = ReadyWindowGate(rearm_frames=2)
        self.assertTrue(gate.update(0, 0.90, 1.0).trigger)
        gate.update(0, 0.05, 1.1)
        gate.update(0, 0.05, 1.2)
        switched = gate.update(1, 0.95, 1.3)
        self.assertFalse(switched.trigger)
        self.assertTrue(switched.mode_changed)
        gate.update(1, 0.05, 1.4)
        low2 = gate.update(1, 0.05, 1.5)
        self.assertTrue(low2.rearmed)
        self.assertTrue(gate.update(1, 0.90, 1.6).trigger)

    def test_cooldown_high_island_is_consumed(self):
        gate = ReadyWindowGate(min_trigger_interval_s=0.50, rearm_frames=1)
        self.assertTrue(gate.update(0, 0.90, 1.0).trigger)
        gate.update(0, 0.05, 1.1)
        too_soon = gate.update(0, 0.90, 1.2)
        self.assertFalse(too_soon.trigger)
        self.assertEqual(too_soon.reason, "cooldown_high_island")
        self.assertFalse(gate.update(0, 0.95, 1.6).trigger)
        gate.update(0, 0.05, 1.7)
        self.assertTrue(gate.update(0, 0.90, 1.8).trigger)

    def test_initial_low_then_high_can_trigger(self):
        gate = ReadyWindowGate()
        self.assertFalse(gate.update(1, 0.02, 0.0).trigger)
        self.assertTrue(gate.update(1, 0.91, 0.3).trigger)


if __name__ == "__main__":
    unittest.main()
