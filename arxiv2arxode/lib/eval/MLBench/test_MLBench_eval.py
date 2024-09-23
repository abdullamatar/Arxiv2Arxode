import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
from transformers import AutoModel, AutoTokenizer

project_root = os.path.abspath("./")
print(project_root)
if project_root not in sys.path:
    sys.path.append(project_root)

from lib.eval.MLBench.MLBench_eval import evaluate_generated_code


class TestEvaluateGeneratedCode(unittest.TestCase):
    @patch("lib.eval.MLBench.MLBench_eval.compute_sentence_embeddings")
    @patch("lib.eval.MLBench.MLBench_eval.compute_cosine_similarity")
    def test_evaluate_generated_code(
        self, mock_cosine_similarity, mock_sentence_embeddings
    ):
        tokenizer = MagicMock(spec=AutoTokenizer)
        model = MagicMock(spec=AutoModel)

        mock_sentence_embeddings.side_effect = [
            torch.tensor([[0.1, 0.2, 0.3]]),  # candidate embedding
            torch.tensor([[0.4, 0.5, 0.6]]),  # ground truth embedding
        ]
        mock_cosine_similarity.return_value = 0.9

        runs = [
            {
                "task_idx": 1,
                "code_generations": ["print('Hello, world!')"],
                "exit_codes": [0],
            }
        ]
        test_set = [
            {"id": 1, "instruction": "Print Hello, world!", "output": "Hello, world!"}
        ]

        df, series = evaluate_generated_code(runs, test_set, tokenizer, model)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["task_idx"], 1)
        self.assertEqual(df.iloc[0]["task_description"], "Print Hello, world!")
        self.assertEqual(df.iloc[0]["attempt"], 1)
        self.assertEqual(df.iloc[0]["generated_code"], "print('Hello, world!')")
        self.assertEqual(df.iloc[0]["exit_code"], 0)
        self.assertEqual(df.iloc[0]["similarity"], 0.9)

        # Check the Series
        self.assertAlmostEqual(series["valuation"], 0.7 * 0.9 + 1)
        self.assertAlmostEqual(series["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
