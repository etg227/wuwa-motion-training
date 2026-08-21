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


class TestMacroCompile(unittest.TestCase):
    def test_press_times_match_annotation_without_lead(self):
        timeline = _timeline([
            AxisVideoEvent(frame=0, ms=0, action="s2", slot=1),
            AxisVideoEvent(frame=30, ms=500, action="a", slot=2),
            AxisVideoEvent(frame=60, ms=1000, action="e", slot=2),
        ])
        steps, warnings = compile_macro(timeline)
        self.assertEqual(warnings, [])
        self.assertEqual(reconstruct_press_times(steps), [0, 500, 1000])

    def test_lead_shifts_presses_earlier_and_clamps_first(self):
        timeline = _timeline([
            AxisVideoEvent(frame=1, ms=30, action="a", slot=1),
            AxisVideoEvent(frame=30, ms=500, action="e", slot=1),
        ])
        steps, warnings = compile_macro(timeline, lead_ms=60)
        # 首事件 30-60 < 0 被钳到 0 并告警；第二事件 500-60=440。
        self.assertEqual(reconstruct_press_times(steps), [0, 440])
        self.assertTrue(any("钳到 0" in line for line in warnings))

    def test_hold_clamped_when_next_event_is_close(self):
        timeline = _timeline([
            AxisVideoEvent(frame=0, ms=0, action="z", slot=1),   # 默认按住 550ms
            AxisVideoEvent(frame=6, ms=100, action="e", slot=1),
        ])
        steps, warnings = compile_macro(timeline)
        self.assertTrue(any("截短" in line for line in warnings))
        # 截短后仍保持绝对按下时刻不变。
        self.assertEqual(reconstruct_press_times(steps), [0, 100])
        z_down = steps[0]
        self.assertLess(z_down.delay_after_ms, 100)

    def test_marker_events_excluded_from_macro(self):
        timeline = _timeline([
            AxisVideoEvent(frame=0, ms=0, action="intro", slot=1),
            AxisVideoEvent(frame=30, ms=500, action="q", slot=1),
        ])
        steps, _ = compile_macro(timeline)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].code, "q")

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
    def test_nodes_split_on_switch_and_track_gaps(self):
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
            [row["gap_ms"] for row in nodes[1]["events"]], [None, 200, 200])


if __name__ == "__main__":
    unittest.main()
