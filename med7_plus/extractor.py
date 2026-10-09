"""
Primary Clinical Entity and Medication Information Extraction Pipeline.
Supports both pattern/dictionary parsing and neural spaCy models (e.g. en_core_med7_lg or Bio_ClinicalBERT).
"""

import re
from typing import Dict, List, Optional, Union
from med7_plus.config import CORE_ENTITIES, ALL_ENTITIES, ENTITY_COLORS
from med7_plus.context_analyzer import ClinicalContextAnalyzer
from med7_plus.normalizer import DrugNormalizer
from med7_plus.relation_linker import MedicationRelationLinker
from med7_plus.fhir_exporter import FHIRExporter


class Med7ClinicalExtractor:
    """
    High-level clinical NLP pipeline for extracting and linking medication entities,
    evaluating clinical context, normalizing to RxNorm, and exporting to FHIR.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        use_rules_fallback: bool = True
    ):
        """
        Initialize the clinical extractor.

        Args:
            model_name_or_path: spaCy model name (e.g., 'en_core_med7_lg' or 'en_core_web_sm').
            use_rules_fallback: If True, uses the high-precision regex/lexicon engine when model is not loaded.
        """
        self.model_name = model_name_or_path
        self.use_rules_fallback = use_rules_fallback
        self.nlp = None

        # Try to load spaCy model if provided
        if model_name_or_path:
            try:
                import spacy
                self.nlp = spacy.load(model_name_or_path)
            except Exception as e:
                print(f"[Med7-Plus] Notice: Could not load spaCy model '{model_name_or_path}' ({e}). Falling back to clinical rules.")

        # Core pipeline components
        self.context_analyzer = ClinicalContextAnalyzer()
        self.normalizer = DrugNormalizer()
        self.relation_linker = MedicationRelationLinker()
        self.fhir_exporter = FHIRExporter()

        # Build clinical regex rules
        self._init_clinical_patterns()

    def _init_clinical_patterns(self):
        """Compile regex patterns for high-precision clinical entity extraction."""
        # Strength: 50mg, 500 mg, 0.5 mcg, 10 mg/ml, 5 %
        self.pat_strength = re.compile(
            r"\b(\d+(?:\.\d+)?\s*(?:mg|mcg|micrograms?|milligrams?|g|grams?|ml|meq|units?|%|mg/ml))\b",
            re.IGNORECASE
        )

        # Dosage: 1 tablet, 2 tabs, 1-2 puffs, 1 vial, 2 capsules
        self.pat_dosage = re.compile(
            r"\b(\d+(?:-\d+)?\s*(?:tablets?|tabs?|capsules?|caps?|puffs?|drops?|vials?|ampules?|teaspoons?|tsp|packets?))\b",
            re.IGNORECASE
        )

        # Form: tablet, solution, suspension, cream, ointment, inhaler
        self.pat_form = re.compile(
            r"\b(tablet|tablets|tab|capsule|capsules|cap|injection|solution|suspension|cream|ointment|inhaler|patch|suppository|syrup)\b",
            re.IGNORECASE
        )

        # Route: oral, PO, IV, intravenous, IM, subcutaneous, subQ, topical, PR, PRN
        self.pat_route = re.compile(
            r"\b(p\.?o\.?|oral(?:ly)?|i\.?v\.?|intravenous(?:ly)?|i\.?m\.?|subcutaneous(?:ly)?|subq|topical(?:ly)?|pr|p\.?r\.?|sublingual(?:ly)?|inhalation|nasal(?:ly)?)\b",
            re.IGNORECASE
        )

        # Frequency: daily, once daily, BID, TID, QID, q8h, every 12 hours, at bedtime
        self.pat_frequency = re.compile(
            r"\b(b\.?i\.?d\.?|t\.?i\.?d\.?|q\.?i\.?d\.?|q\.?d\.?|daily|once daily|twice daily|three times daily|four times daily|q\s*\d+\s*h(?:ours)?|every \d+ hours?|at bedtime|qhs|qam|prn|as needed)\b",
            re.IGNORECASE
        )

        # Duration: for 10 days, x 7 days, for 2 weeks, for 1 month
        self.pat_duration = re.compile(
            r"\b(?:for|x)\s*(\d+\s*(?:days?|weeks?|months?|years?|doses?))\b",
            re.IGNORECASE
        )

        # Indication: for hypertension, for diabetes, for pain relief, to treat infection
        self.pat_indication = re.compile(
            r"\b(?:for|to treat|indicated for)\s+([a-z\s]+(?:hypertension|diabetes|pain|infection|headache|nausea|fever|cough|insomnia|asthma|depression|anxiety|edema|heart failure))\b",
            re.IGNORECASE
        )

        # Common drugs dictionary from normalizer
        drug_names = list(self.normalizer.knowledge_base.keys())
        # Sort by length descending to match multi-word names first
        drug_names.sort(key=len, reverse=True)
        drug_pattern_str = r"\b(" + "|".join(re.escape(d) for d in drug_names) + r")\b"
        self.pat_drug = re.compile(drug_pattern_str, re.IGNORECASE)

    def extract_entities_regex(self, text: str) -> List[Dict[str, any]]:
        """Extract medication entities using curated clinical regular expressions."""
        entities = []

        patterns = [
            (self.pat_drug, "DRUG"),
            (self.pat_strength, "STRENGTH"),
            (self.pat_dosage, "DOSAGE"),
            (self.pat_form, "FORM"),
            (self.pat_route, "ROUTE"),
            (self.pat_frequency, "FREQUENCY"),
            (self.pat_duration, "DURATION"),
            (self.pat_indication, "INDICATION"),
        ]

        seen_spans = set()

        for pattern, label in patterns:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                # Check for overlap with existing spans
                overlaps = False
                for s, e in seen_spans:
                    if max(start, s) < min(end, e):
                        overlaps = True
                        break

                if not overlaps:
                    entities.append({
                        "text": match.group(0),
                        "label": label,
                        "start": start,
                        "end": end,
                        "confidence": 0.95
                    })
                    seen_spans.add((start, end))

        # Sort entities by start position
        entities.sort(key=lambda x: x["start"])
        return entities

    def extract_entities_spacy(self, text: str) -> List[Dict[str, any]]:
        """Extract medication entities using loaded neural spaCy pipeline."""
        if not self.nlp:
            return []

        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": getattr(ent, "confidence", 0.90)
            })
        return entities

    def extract(self, text: str) -> Dict[str, any]:
        """
        Execute full end-to-end extraction pipeline on a clinical narrative text.

        Returns:
            Dictionary containing:
            - "raw_entities": List of detected entity spans.
            - "prescriptions": Structured drug prescriptions with slots linked.
            - "fhir_bundle": HL7 FHIR MedicationStatement bundle.
            - "text": Original input text.
        """
        # Step 1: Detect entities
        if self.nlp:
            entities = self.extract_entities_spacy(text)
            if not entities and self.use_rules_fallback:
                entities = self.extract_entities_regex(text)
        else:
            entities = self.extract_entities_regex(text)

        # Step 2: Augment DRUG entities with Context & RxNorm Normalization
        for ent in entities:
            if ent["label"] == "DRUG":
                ent["context"] = self.context_analyzer.analyze_entity_context(
                    text, ent["start"], ent["end"]
                )
                ent["normalization"] = self.normalizer.normalize(ent["text"])

        # Step 3: Link modifier slots to parent drugs
        linked_prescriptions = self.relation_linker.link_entities(text, entities)

        # Step 4: Generate FHIR bundle
        fhir_bundle = self.fhir_exporter.export_bundle(linked_prescriptions)

        return {
            "text": text,
            "raw_entities": entities,
            "prescriptions": linked_prescriptions,
            "fhir_bundle": fhir_bundle,
            "metadata": {
                "entity_count": len(entities),
                "drug_count": len(linked_prescriptions),
                "engine": "spaCy-neural" if self.nlp else "clinical-rules-hybrid"
            }
        }
