"""
Unit tests for the ClinicalEvaluator module.
"""

import unittest
from med7_plus.evaluator import ClinicalEvaluator


class TestClinicalEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = ClinicalEvaluator()

    def test_perfect_strict_match(self):
        gold = [[{"start": 0, "end": 7, "label": "DRUG"}]]
        pred = [[{"start": 0, "end": 7, "label": "DRUG"}]]

        res = self.evaluator.evaluate(gold, pred)
        self.assertEqual(res["micro_strict"]["f1"], 1.0)
        self.assertEqual(res["micro_lenient"]["f1"], 1.0)

    def test_partial_overlap_lenient_vs_strict(self):
        # Gold covers "Metoprolol succinate" (0 to 20), pred covers "Metoprolol" (0 to 10)
        gold = [[{"start": 0, "end": 20, "label": "DRUG"}]]
        pred = [[{"start": 0, "end": 10, "label": "DRUG"}]]

        res = self.evaluator.evaluate(gold, pred)
        # Strict must fail (0.0 F1)
        self.assertEqual(res["micro_strict"]["f1"], 0.0)
        # Lenient must pass (1.0 F1)
        self.assertEqual(res["micro_lenient"]["f1"], 1.0)

    def test_mismatched_label(self):
        gold = [[{"start": 0, "end": 10, "label": "DRUG"}]]
        pred = [[{"start": 0, "end": 10, "label": "STRENGTH"}]]

        res = self.evaluator.evaluate(gold, pred)
        self.assertEqual(res["micro_strict"]["f1"], 0.0)
        self.assertEqual(res["micro_lenient"]["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
