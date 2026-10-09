"""
Unit tests for the ClinicalContextAnalyzer.
"""

import unittest
from med7_plus.context_analyzer import ClinicalContextAnalyzer


class TestClinicalContextAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = ClinicalContextAnalyzer()

    def test_negation_detection_discontinued(self):
        text = "Discontinued Lisinopril due to persistent cough."
        start = text.index("Lisinopril")
        end = start + len("Lisinopril")
        
        ctx = self.analyzer.analyze_entity_context(text, start, end)
        self.assertTrue(ctx["is_negated"])
        self.assertEqual(ctx["status"], "DISCONTINUED_OR_NEGATED")

    def test_negation_detection_denies(self):
        text = "Patient denies taking Warfarin at home."
        start = text.index("Warfarin")
        end = start + len("Warfarin")

        ctx = self.analyzer.analyze_entity_context(text, start, end)
        self.assertTrue(ctx["is_negated"])

    def test_active_medication_no_negation(self):
        text = "Patient prescribed Metformin 500 mg daily."
        start = text.index("Metformin")
        end = start + len("Metformin")

        ctx = self.analyzer.analyze_entity_context(text, start, end)
        self.assertFalse(ctx["is_negated"])
        self.assertEqual(ctx["status"], "ACTIVE")

    def test_conditional_certainty(self):
        text = "Take Morphine 2 mg IV as needed for pain."
        start = text.index("Morphine")
        end = start + len("Morphine")

        ctx = self.analyzer.analyze_entity_context(text, start, end)
        self.assertEqual(ctx["certainty"], "CONDITIONAL")


if __name__ == "__main__":
    unittest.main()
