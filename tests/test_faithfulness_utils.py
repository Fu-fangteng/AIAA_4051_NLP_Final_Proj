import unittest

from faithfulness_utils import (
    build_cumulative_attention_masks,
    build_cumulative_mask_inputs,
    least_relevant_first_order,
    mask_percent_axis,
    relevance_first_order,
)


class FaithfulnessUtilsTests(unittest.TestCase):
    def test_cumulative_mask_inputs_start_with_unmasked_tokens(self):
        tokens = [10, 20, 30]
        masked = build_cumulative_mask_inputs(tokens, [2, 0, 1], mask_token_id=99)

        self.assertEqual(masked[0], [10, 20, 30])
        self.assertEqual(masked[1], [10, 20, 99])
        self.assertEqual(masked[2], [99, 20, 99])
        self.assertEqual(masked[3], [99, 99, 99])

    def test_cumulative_mask_inputs_validate_order_length(self):
        with self.assertRaises(ValueError):
            build_cumulative_mask_inputs([10, 20, 30], [0, 1], mask_token_id=99)

    def test_cumulative_attention_masks_start_fully_visible_then_delete_tokens(self):
        masks = build_cumulative_attention_masks(3, [2, 0, 1])

        self.assertEqual(masks[0], [1, 1, 1])
        self.assertEqual(masks[1], [1, 1, 0])
        self.assertEqual(masks[2], [0, 1, 0])
        self.assertEqual(masks[3], [0, 0, 0])

    def test_cumulative_attention_masks_validate_order_length(self):
        with self.assertRaises(ValueError):
            build_cumulative_attention_masks(3, [0, 1])

    def test_mask_percent_axis_includes_zero_and_full_mask(self):
        self.assertEqual(mask_percent_axis(4).tolist(), [0.0, 25.0, 50.0, 75.0, 100.0])

    def test_relevance_first_order_uses_descending_signed_relevance(self):
        self.assertEqual(relevance_first_order([0.1, -0.9, 0.5, 0.0]), [2, 0, 3, 1])

    def test_least_relevant_first_order_uses_smallest_absolute_relevance(self):
        self.assertEqual(least_relevant_first_order([0.1, -0.9, 0.5, 0.0]), [3, 0, 2, 1])


if __name__ == "__main__":
    unittest.main()
