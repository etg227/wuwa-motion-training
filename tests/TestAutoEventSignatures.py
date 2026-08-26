import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "axis"))

from auto_events import (
    local_event_signatures,
    semantic_event_signatures,
    semantic_timeline_guard,
)


class TestAutoEventSignatures(unittest.TestCase):
    def _empty_streams(self, frames=10, dims=2):
        return (
            np.zeros((frames, dims), dtype=np.float32),
            np.zeros((frames, dims), dtype=np.float32),
            np.zeros((frames, dims), dtype=np.float32),
        )

    def test_local_and_semantic_signatures_use_different_time_scopes(self):
        body, party, ability = self._empty_streams()
        body[4] = np.array([1.0, 0.0], dtype=np.float32)
        body[7] = np.array([0.0, 3.0], dtype=np.float32)

        local = local_event_signatures(body, party, ability, [3])
        semantic = semantic_event_signatures(
            body, party, ability, [3], analysis_fps=30.0
        )

        self.assertEqual(local.shape, semantic.shape)
        self.assertFalse(np.allclose(local, semantic))
        self.assertGreater(local[0, 0], 0.9)
        self.assertGreater(semantic[0, 1], 0.9)

    def test_empty_signatures_keep_expected_dimension(self):
        body, party, ability = self._empty_streams(frames=4, dims=3)
        local = local_event_signatures(body, party, ability, [])
        semantic = semantic_event_signatures(
            body, party, ability, [], analysis_fps=30.0
        )
        self.assertEqual(local.shape, (0, 9))
        self.assertEqual(semantic.shape, (0, 9))

    def test_semantic_signature_requires_valid_fps(self):
        body, party, ability = self._empty_streams()
        with self.assertRaises(ValueError):
            semantic_event_signatures(
                body, party, ability, [3], analysis_fps=0.0
            )


class TestSemanticTimelineGuard(unittest.TestCase):
    def test_no_labels_fail_closed(self):
        reason = semantic_timeline_guard([])
        self.assertIsNotNone(reason)
        self.assertIn("没有可用的语义事件", reason)
        self.assertIn("阻止", reason)

    def test_sparse_labels_do_not_block(self):
        self.assertIsNone(semantic_timeline_guard(["a"] * 7))

    def test_balanced_labels_do_not_block(self):
        self.assertIsNone(semantic_timeline_guard(["a"] * 4 + ["e"] * 4))

    def test_degenerate_labels_fail_closed(self):
        reason = semantic_timeline_guard(["a"] * 9 + ["e"])
        self.assertIsNotNone(reason)
        self.assertIn("90%", reason)
        self.assertIn("阻止", reason)

    def test_share_at_threshold_is_not_blocked(self):
        labels = ["a"] * 17 + ["e"] * 3
        self.assertIsNone(
            semantic_timeline_guard(labels, max_dominant_share=0.85)
        )


if __name__ == "__main__":
    unittest.main()
