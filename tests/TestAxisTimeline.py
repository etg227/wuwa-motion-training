import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "axis"))
from timeline import (
    AxisVideoEvent,
    Timeline,
    compile_axis_nodes,
    compile_macro,
    estimate_press_schedule,
    frame_to_ms,
    macro_source,
    reconstruct_press_times,
)


def _timeline(events):
    timeline = Timeline(video="up.mp4", fps=60.0)
    timeline.events.extend(events)
    return timeline


class TestTimelineModel(unittest.TestCase):
    def test_frame_to_ms_rounds(self):
        self.assertEqual(frame_to_ms(30, 60.0), 500)
        self.assertEqual(frame_to_ms(1, 30.0), 33)

    def test_invalid_fps_rejected(self):
        with self.assertRaises(ValueError):
            frame_to_ms(1, 0.0)

    def test_unknown_action_rejected(self):
        with self.assertRaises(ValueError):
            AxisVideoEvent(frame=0, ms=0, action="dodge", slot=1)

    def test_save_load_roundtrip_sorts_events(self):
        timeline = _timeline([
            AxisVideoEvent(frame=60, ms=1000, action="e", slot=2),
            AxisVideoEvent(frame=0, ms=0, action="s2", slot=1),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            timeline.save(path)
            loaded = Timeline.load(path)
        self.assertEqual([e.action for e in loaded.sorted_events()], ["s2", "e"])
        self.assertEqual(loaded.fps, 60.0)


class TestInputReconstruction(unittest.TestCase):
    def test_no_lead_preserves_relative_visual_gaps(self):
        timeline = _timeline([
            AxisVideoEvent(frame=0, ms=0, action="s2", slot=1),
            AxisVideoEvent(frame=30, ms=500, action="a", slot=2),
            AxisVideoEvent(frame=60, ms=1000, action="e", slot=2),
        ])
        schedule, warnings = estimate_press_schedule(timeline)
        self.assertEqual(warnings, [])
        self.assertEqual([row.macro_ms for row in schedule], [0, 500, 1000])

    def test_uniform_lead_moves_source_origin_not_relative_gaps(self):
        timeline = _timeline([
            AxisVideoEvent(frame=1, ms=30, action="a", slot=1),
            AxisVideoEvent(frame=30, ms=500, action="e", slot=1),
        ])
        schedule, warnings = estimate_press_schedule(timeline, lead_ms=60)
        self.assertEqual([row.source_press_ms for row in schedule], [-30, 440])
        self.assertEqual([row.macro_ms for row in schedule], [0, 470])
        self.assertTrue(any("视频起点前" in line for line in warnings))

    def test_action_leads_change_relative_input_gap(self):
        timeline = _timeline([
            AxisVideoEvent(frame=60, ms=1000, action="a", slot=1),
            AxisVideoEvent(frame=90, ms=1500, action="e", slot=1),
        ])
        schedule, _ = estimate_press_schedule(
            timeline,
            action_leads={"a": 40, "e": 100},
        )
        self.assertEqual([row.source_press_ms for row in schedule], [960, 1400])
        self.assertEqual([row.macro_ms for row in schedule], [0, 440])

    def test_transition_lead_overrides_action_lead(self):
        timeline = _timeline([
            AxisVideoEvent(frame=60, ms=1000, action="e", slot=1),
            AxisVideoEvent(frame=90, ms=1500, action="s2", slot=1),
        ])
        schedule, _ = estimate_press_schedule(
            timeline,
            action_leads={"s2": 60},
            transition_leads={"e:s2": 120},
        )
        self.assertEqual(schedule[1].lead_ms, 120)
        self.assertEqual([row.macro_ms for row in schedule], [0, 380])

    def test_large_transition_lead_reorders_and_warns(self):
        timeline = _timeline([
            AxisVideoEvent(frame=60, ms=1000, action="e", slot=1),
            AxisVideoEvent(frame=63, ms=1050, action="s2", slot=1),
        ])
        schedule, warnings = estimate_press_schedule(
            timeline,
            transition_leads={"e:s2": 100},
        )
        self.assertEqual([row.event.action for row in schedule], ["s2", "e"])
        self.assertTrue(any("顺序" in line for line in warnings))


class TestMacroCompile(unittest.TestCase):
    def test_compiled_press_times_follow_estimated_schedule(self):
        timeline = _timeline([
            AxisVideoEvent(frame=60, ms=1000, action="a", slot=1),
            AxisVideoEvent(frame=90, ms=1500, action="e", slot=1),
        ])
        steps, warnings = compile_macro(
            timeline,
            action_leads={"a": 40, "e": 100},
        )
        self.assertEqual(warnings, [])
        self.assertEqual(reconstruct_press_times(steps), [0, 440])

    def test_hold_clamped_when_next_event_is_close(self):
        timeline = _timeline([
            AxisVideoEvent(frame=0, ms=0, action="z", slot=1),
            AxisVideoEvent(frame=6, ms=100, action="e", slot=1),
        ])
        steps, warnings = compile_macro(timeline)
        self.assertTrue(any("截短" in line for line in warnings))
        self.assertEqual(reconstruct_press_times(steps), [0, 100])
        self.assertLess(steps[0].delay_after_ms, 100)
        self.assertGreaterEqual(steps[1].delay_after_ms, 0)

    def test_tiny_gap_never_generates_negative_delay(self):
        timeline = _timeline([
            AxisVideoEvent(frame=0, ms=0, action="a", slot=1),
            AxisVideoEvent(frame=1, ms=5, action="e", slot=1),
        ])
        steps, warnings = compile_macro(timeline)
        self.assertTrue(any("截短" in line for line in warnings))
        self.assertEqual(reconstruct_press_times(steps), [0, 5])
        self.assertTrue(all(step.delay_after_ms >= 0 for step in steps))

    def test_marker_events_excluded_from_macro(self):
        timeline = _timeline([
            AxisVideoEvent(frame=0, ms=0, action="intro", slot=1),
            AxisVideoEvent(frame=30, ms=500, action="q", slot=1),
        ])
        steps, _ = compile_macro(timeline)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].code, "q")
        self.assertEqual(reconstruct_press_times(steps), [0])

    def test_macro_source_uses_helper_call_style(self):
        timeline = _timeline([
            AxisVideoEvent(frame=0, ms=0, action="s3", slot=1),
            AxisVideoEvent(frame=30, ms=500, action="a", slot=3),
        ])
        steps, _ = compile_macro(timeline)
        source = macro_source(steps, "TEST_MACRO")
        self.assertIn("TEST_MACRO = (", source)
        self.assertIn('*_key("3", 78,', source)
        self.assertIn("*_mouse(78, 0,", source)


class TestAxisNodes(unittest.TestCase):
    def test_nodes_split_on_switch_and_track_visual_gaps(self):
        timeline = _timeline([
            AxisVideoEvent(frame=0, ms=0, action="e", slot=1),
            AxisVideoEvent(frame=30, ms=500, action="s2", slot=1),
            AxisVideoEvent(frame=36, ms=600, action="a", slot=2),
            AxisVideoEvent(frame=48, ms=800, action="a", slot=2),
            AxisVideoEvent(frame=60, ms=1000, action="fall_a", slot=2),
        ])
        nodes = compile_axis_nodes(timeline)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0]["slot"], 1)
        self.assertEqual(nodes[0]["label"], "e")
        self.assertEqual(nodes[0]["swap_gap_ms"], 500)
        self.assertEqual(nodes[1]["slot"], 2)
        self.assertEqual(nodes[1]["label"], "aa下落a")
        self.assertEqual(
            [row["gap_ms"] for row in nodes[1]["events"]], [None, 200, 200]
        )


if __name__ == "__main__":
    unittest.main()
