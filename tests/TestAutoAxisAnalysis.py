import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "axis"))

from auto_analysis import (
    align_period_boundaries,
    cluster_signatures,
    detect_peaks,
    estimate_repetition_period,
    loop_support,
    normalize_rows,
)


class TestAutoAxisAnalysis(unittest.TestCase):
    def test_detect_peaks_keeps_separated_strong_events(self):
        signal = np.array([0, 0, 1, 8, 1, 0, 0, 6, 0, 0], dtype=np.float32)
        self.assertEqual(
            detect_peaks(signal, min_distance=2, threshold_z=1.0),
            [3, 7],
        )

    def test_cluster_signatures_groups_similar_vectors(self):
        signatures = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.98, 0.05, 0.0],
                [0.0, 1.0, 0.0],
                [0.02, 0.99, 0.0],
            ],
            dtype=np.float32,
        )
        result = cluster_signatures(signatures, similarity_threshold=0.9)
        self.assertEqual(result.labels[0], result.labels[1])
        self.assertEqual(result.labels[2], result.labels[3])
        self.assertNotEqual(result.labels[0], result.labels[2])

    def test_period_estimator_finds_repeated_rotation(self):
        rng = np.random.default_rng(4)
        period = 20
        base = normalize_rows(rng.normal(size=(period, 12)).astype(np.float32))
        features = np.vstack(
            [
                normalize_rows(base + rng.normal(scale=0.02, size=base.shape))
                for _ in range(5)
            ]
        )
        estimate = estimate_repetition_period(features, min_lag=14, max_lag=45)
        self.assertIsNotNone(estimate)
        self.assertLessEqual(abs(estimate.lag - period), 1)

    def test_align_boundaries_tracks_period(self):
        rng = np.random.default_rng(8)
        period = 18
        base = normalize_rows(rng.normal(size=(period, 10)))
        features = np.vstack([base for _ in range(5)])
        found = align_period_boundaries(features, period, 36)
        indexes = [row[0] for row in found]
        self.assertGreaterEqual(len(indexes), 4)
        self.assertTrue(np.all(np.abs(np.diff(indexes) - period) <= 2))

    def test_loop_support_counts_matching_cluster_phase(self):
        events = np.array([2, 8, 22, 28, 42, 48], dtype=np.int32)
        labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        support = loop_support(events, labels, [0, 20, 40, 60])
        self.assertEqual(support.tolist(), [3, 3, 3, 3, 3, 3])


if __name__ == "__main__":
    unittest.main()


class TestClassifySignature(unittest.TestCase):
    def _bank(self, tokens):
        # 每个 token 一组正交方向的原型
        base = np.eye(4, dtype=np.float32)
        return {token: np.stack([base[index]] * 3) for index, token in enumerate(tokens)}

    def test_single_class_bank_never_classifies(self):
        # 首次实测的故障：只有 'a' 原型时 margin 恒等于最高分，
        # 99 个事件全部被标成高置信度普攻。单类别必须拒绝分类。
        from auto_analysis import classify_signature

        bank = self._bank(["a"])
        signature = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        token, confidence, scores = classify_signature(signature, bank)
        self.assertIsNone(token)
        self.assertEqual(confidence, 0.0)
        self.assertIn("a", scores)  # 分数仍保留供诊断

    def test_two_class_bank_classifies_clear_match(self):
        from auto_analysis import classify_signature

        bank = self._bank(["a", "e"])
        signature = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        token, confidence, _ = classify_signature(signature, bank)
        self.assertEqual(token, "a")
        self.assertGreater(confidence, 0.5)

    def test_ambiguous_match_rejected_by_margin(self):
        from auto_analysis import classify_signature

        bank = self._bank(["a", "e"])
        # 与两类原型等距的方向：margin≈0，必须拒绝
        signature = np.array([0.7071, 0.7071, 0.0, 0.0], dtype=np.float32)
        token, _, _ = classify_signature(signature, bank)
        self.assertIsNone(token)

    def test_dominant_label_share(self):
        from auto_analysis import dominant_label_share

        token, share = dominant_label_share(["a"] * 9 + ["e"])
        self.assertEqual(token, "a")
        self.assertAlmostEqual(share, 0.9)
        self.assertEqual(dominant_label_share([]), (None, 0.0))
