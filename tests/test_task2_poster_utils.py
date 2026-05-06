import unittest

import numpy as np

from task2_poster_utils import aggregate_question_words


class Task2PosterUtilsTests(unittest.TestCase):
    def test_aggregate_question_words_merges_subwords_and_punctuation(self):
        tokens = [
            "Context", ":", "Ġirrelevant", "Ċ",
            "Question", ":", "ĠWho", "Ġwas", "Ġthe", "ĠN", "orse", "Ġleader", "?",
            "Ċ", "Answer", ":",
        ]
        relevance = np.arange(len(tokens), dtype=float)

        words, values = aggregate_question_words(tokens, relevance)

        self.assertEqual(words, ["Who", "was", "the", "Norse", "leader"])
        self.assertEqual(values.tolist(), [6.0, 7.0, 8.0, 19.0, 23.0])

    def test_aggregate_question_words_stops_before_answer_without_newline(self):
        tokens = ["Question", ":", "ĠWhat", "Ġreligion", "Ġwere", "ĠNormans", "Answer", ":"]
        relevance = np.ones(len(tokens), dtype=float)

        words, values = aggregate_question_words(tokens, relevance)

        self.assertEqual(words, ["What", "religion", "were", "Normans"])
        self.assertEqual(values.tolist(), [1.0, 1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
