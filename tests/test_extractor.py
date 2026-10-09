"""
Unit tests for the primary Med7ClinicalExtractor.
"""

import unittest
from med7_plus.extractor import Med7ClinicalExtractor


class TestMed7ClinicalExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = Med7ClinicalExtractor()

    def test_basic_medication_extraction(self):
        text = "Patient prescribed Metformin 500 mg PO BID for diabetes."
        res = self.extractor.extract(text)

        labels = [e["label"] for e in res["raw_entities"]]
        self.assertIn("DRUG", labels)
        self.assertIn("STRENGTH", labels)
        self.assertIn("ROUTE", labels)
        self.assertIn("FREQUENCY", labels)

    def test_multi_drug_linking(self):
        text = "Discharge on Furosemide 40 mg PO daily and Metoprolol 25 mg PO BID."
        res = self.extractor.extract(text)

        self.assertEqual(len(res["prescriptions"]), 2)
        drug_names = [p["drug"]["text"] for p in res["prescriptions"]]
        self.assertIn("Furosemide", drug_names)
        self.assertIn("Metoprolol", drug_names)

        # Check slot linking
        furo_presc = next(p for p in res["prescriptions"] if p["drug"]["text"] == "Furosemide")
        self.assertIsNotNone(furo_presc["strength"])
        self.assertEqual(furo_presc["strength"]["text"], "40 mg")

    def test_fhir_bundle_generation(self):
        text = "Patient takes Aspirin 81 mg oral daily."
        res = self.extractor.extract(text)

        bundle = res["fhir_bundle"]
        self.assertEqual(bundle["resourceType"], "Bundle")
        self.assertGreater(bundle["total"], 0)
        self.assertEqual(bundle["entry"][0]["resource"]["resourceType"], "MedicationStatement")


if __name__ == "__main__":
    unittest.main()
