"""
Clinical drug concept normalizer.
Maps unstructured drug entity strings to standardized RxNorm Concept Unique Identifiers (RxCUI),
generic brand equivalents, and Anatomical Therapeutic Chemical (ATC) classifications.
"""

from typing import Dict, Optional, Tuple

# Built-in clinical dictionary of common medications for instant offline lookup
STANDARD_RXNORM_LOOKUP: Dict[str, Dict[str, str]] = {
    "aspirin": {
        "rxcui": "1191",
        "generic_name": "Aspirin",
        "atc_code": "B01AC06",
        "category": "Antiplatelet / Analgesic"
    },
    "metformin": {
        "rxcui": "6809",
        "generic_name": "Metformin Hydrochloride",
        "atc_code": "A10BA02",
        "category": "Antidiabetic / Biguanide"
    },
    "lisinopril": {
        "rxcui": "29046",
        "generic_name": "Lisinopril",
        "atc_code": "C09AA03",
        "category": "Antihypertensive / ACE Inhibitor"
    },
    "atorvastatin": {
        "rxcui": "83367",
        "generic_name": "Atorvastatin Calcium",
        "atc_code": "C10AA05",
        "category": "Lipid-lowering / Statin"
    },
    "lipitor": {
        "rxcui": "153666",
        "generic_name": "Atorvastatin Calcium",
        "atc_code": "C10AA05",
        "category": "Lipid-lowering / Statin (Brand)"
    },
    "metoprolol": {
        "rxcui": "6918",
        "generic_name": "Metoprolol Tartrate",
        "atc_code": "C07AB02",
        "category": "Beta-blocker"
    },
    "metoprolol succinate": {
        "rxcui": "217010",
        "generic_name": "Metoprolol Succinate",
        "atc_code": "C07AB02",
        "category": "Beta-blocker"
    },
    "amoxicillin": {
        "rxcui": "723",
        "generic_name": "Amoxicillin",
        "atc_code": "J01CA04",
        "category": "Antibiotic / Penicillin"
    },
    "augmentin": {
        "rxcui": "151392",
        "generic_name": "Amoxicillin / Clavulanate",
        "atc_code": "J01CR02",
        "category": "Antibiotic / Penicillin combo"
    },
    "omeprazole": {
        "rxcui": "7646",
        "generic_name": "Omeprazole",
        "atc_code": "A02BC01",
        "category": "Gastrointestinal / Proton Pump Inhibitor"
    },
    "furosemide": {
        "rxcui": "4603",
        "generic_name": "Furosemide",
        "atc_code": "C03CA01",
        "category": "Loop Diuretic"
    },
    "lasix": {
        "rxcui": "202433",
        "generic_name": "Furosemide",
        "atc_code": "C03CA01",
        "category": "Loop Diuretic (Brand)"
    },
    "insulin glargine": {
        "rxcui": "274783",
        "generic_name": "Insulin Glargine",
        "atc_code": "A10AE04",
        "category": "Antidiabetic / Long-acting Insulin"
    },
    "lantus": {
        "rxcui": "285018",
        "generic_name": "Insulin Glargine",
        "atc_code": "A10AE04",
        "category": "Antidiabetic / Long-acting Insulin"
    },
    "heparin": {
        "rxcui": "5224",
        "generic_name": "Heparin Sodium",
        "atc_code": "B01AB01",
        "category": "Anticoagulant"
    },
    "warfarin": {
        "rxcui": "11289",
        "generic_name": "Warfarin Sodium",
        "atc_code": "B01AA03",
        "category": "Anticoagulant"
    },
    "coumadin": {
        "rxcui": "202421",
        "generic_name": "Warfarin Sodium",
        "atc_code": "B01AA03",
        "category": "Anticoagulant (Brand)"
    },
    "morphine": {
        "rxcui": "7052",
        "generic_name": "Morphine Sulfate",
        "atc_code": "N02AA01",
        "category": "Opioid Analgesic"
    },
    "vancomycin": {
        "rxcui": "11124",
        "generic_name": "Vancomycin",
        "atc_code": "J01XA01",
        "category": "Glycopeptide Antibacterial"
    },
    "tylenol": {
        "rxcui": "202433",
        "generic_name": "Acetaminophen",
        "atc_code": "N02BE01",
        "category": "Analgesic / Antipyretic"
    },
    "acetaminophen": {
        "rxcui": "161",
        "generic_name": "Acetaminophen",
        "atc_code": "N02BE01",
        "category": "Analgesic / Antipyretic"
    },
}


class DrugNormalizer:
    """
    Normalizes raw drug text tokens to standard clinical medical ontologies.
    """

    def __init__(self, custom_mapping: Optional[Dict[str, Dict[str, str]]] = None):
        self.knowledge_base = dict(STANDARD_RXNORM_LOOKUP)
        if custom_mapping:
            self.knowledge_base.update(custom_mapping)

    def normalize(self, drug_text: str) -> Dict[str, str]:
        """
        Normalize an extracted drug text into RxCUI, generic name, ATC code, and category.
        """
        clean_text = drug_text.strip().lower()
        
        # Exact match
        if clean_text in self.knowledge_base:
            return self.knowledge_base[clean_text]

        # Partial / prefix match
        for key, record in self.knowledge_base.items():
            if key in clean_text or clean_text in key:
                return record

        # Unknown / unmapped drug fallback
        return {
            "rxcui": "UNKNOWN",
            "generic_name": drug_text.capitalize(),
            "atc_code": "N/A",
            "category": "Unclassified Medication"
        }
