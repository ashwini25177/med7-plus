"""
HL7 FHIR MedicationStatement JSON exporter.
Converts structured clinical extraction outputs into interoperable FHIR R4 JSON resources.
"""

from typing import Dict, List, Optional
import uuid
from datetime import datetime, timezone


class FHIRExporter:
    """
    Exports structured medication records into HL7 FHIR R4 MedicationStatement resource format.
    """

    @staticmethod
    def to_fhir_medication_statement(
        structured_prescription: Dict[str, any],
        patient_id: str = "PATIENT-DEMO-001"
    ) -> Dict[str, any]:
        """
        Convert a single linked medication record into a FHIR R4 MedicationStatement JSON resource.
        """
        drug_entity = structured_prescription.get("drug", {})
        drug_text = drug_entity.get("text", "Unknown Medication")
        norm = drug_entity.get("normalization", {})
        context = drug_entity.get("context", {})

        is_negated = context.get("is_negated", False)
        status = "not-taken" if is_negated else "active"

        # Build FHIR MedicationStatement resource
        fhir_resource = {
            "resourceType": "MedicationStatement",
            "id": f"med7-{uuid.uuid4().hex[:8]}",
            "status": status,
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "effectiveDateTime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "medicationCodeableConcept": {
                "text": drug_text,
                "coding": [
                    {
                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "code": norm.get("rxcui", "UNKNOWN"),
                        "display": norm.get("generic_name", drug_text)
                    }
                ]
            },
            "dosage": []
        }

        # Build dosage instruction structure
        dosage_instruction: Dict[str, any] = {}

        if structured_prescription.get("strength"):
            dosage_instruction["doseAndRate"] = [
                {
                    "type": {
                        "text": "ordered strength"
                    },
                    "doseQuantity": {
                        "value": structured_prescription["strength"]["text"]
                    }
                }
            ]

        if structured_prescription.get("route"):
            dosage_instruction["route"] = {
                "text": structured_prescription["route"]["text"]
            }

        if structured_prescription.get("frequency"):
            dosage_instruction["timing"] = {
                "code": {
                    "text": structured_prescription["frequency"]["text"]
                }
            }

        if structured_prescription.get("duration"):
            dosage_instruction["timing"] = dosage_instruction.get("timing", {})
            dosage_instruction["timing"]["repeat"] = {
                "boundsDuration": {
                    "text": structured_prescription["duration"]["text"]
                }
            }

        if dosage_instruction:
            fhir_resource["dosage"].append(dosage_instruction)

        # Append note if negated or conditional
        notes = []
        if is_negated:
            notes.append({"text": "Medication noted as withheld, discontinued, or denied in clinical narrative."})
        if context.get("certainty") == "CONDITIONAL":
            notes.append({"text": "Prescription indicated as PRN / Conditional."})

        if notes:
            fhir_resource["note"] = notes

        return fhir_resource

    @classmethod
    def export_bundle(
        cls,
        prescriptions: List[Dict[str, any]],
        patient_id: str = "PATIENT-DEMO-001"
    ) -> Dict[str, any]:
        """
        Wrap multiple medication statements into a FHIR collection Bundle.
        """
        entries = [
            {
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": cls.to_fhir_medication_statement(p, patient_id=patient_id)
            }
            for p in prescriptions
        ]

        return {
            "resourceType": "Bundle",
            "type": "collection",
            "total": len(entries),
            "entry": entries
        }
