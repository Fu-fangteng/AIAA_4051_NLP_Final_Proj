import unittest

import numpy as np

from task3_visualization import build_param_relevance_payload, top_abs_difference_layers


class Task3RelevanceUtilsTests(unittest.TestCase):
    def test_build_param_relevance_payload_normalizes_and_computes_delta(self):
        rel_a = np.array([2.0, 6.0])
        rel_b = np.array([1.0, 3.0])

        payload = build_param_relevance_payload(rel_a, rel_b, ["a"], ["b"])

        self.assertEqual(payload["layers"].tolist(), [0, 1])
        np.testing.assert_allclose(payload["rel_A_norm"], [0.25, 0.75])
        np.testing.assert_allclose(payload["rel_B_norm"], [0.25, 0.75])
        np.testing.assert_allclose(payload["diff_norm"], [0.0, 0.0], atol=1e-8)
        self.assertEqual(payload["test_texts_A"], ["a"])
        self.assertEqual(payload["test_texts_B"], ["b"])

    def test_top_abs_difference_layers_returns_largest_abs_delta_first(self):
        diff = np.array([0.01, -0.30, 0.20, -0.05])

        self.assertEqual(top_abs_difference_layers(diff, n=3).tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
