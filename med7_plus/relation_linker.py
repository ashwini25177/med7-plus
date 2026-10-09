"""
Medication Slot Relation Linker.
Associates extracted non-drug entities (STRENGTH, DOSAGE, ROUTE, FREQUENCY, FORM, DURATION)
with their corresponding parent DRUG entity in clinical free text.
"""

from typing import Dict, List, Optional
import math


class MedicationRelationLinker:
    """
    Links clinical medication attributes to their corresponding parent drug entity
    using character distance, sentence-boundary constraints, and syntactic proximity.
    """

    def __init__(self, max_token_distance: int = 150):
        self.max_distance = max_token_distance

    def link_entities(self, text: str, entities: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Group extracted entities into structured medication events.

        Returns a list of structured medication records:
        [
            {
                "drug": {...},
                "strength": {...},
                "dosage": {...},
                "route": {...},
                "frequency": {...},
                "duration": {...},
                "form": {...},
                "indication": {...},
                "adverse_effect": {...}
            },
            ...
        ]
        """
        drugs = [e for e in entities if e["label"] == "DRUG"]
        modifiers = [e for e in entities if e["label"] != "DRUG"]

        if not drugs:
            # If no drug is explicitly detected, return modifiers as orphan entities
            return []

        # Initialize structured records for each drug
        structured_prescriptions: List[Dict[str, any]] = []
        for drug in drugs:
            structured_prescriptions.append({
                "drug": drug,
                "strength": None,
                "dosage": None,
                "route": None,
                "frequency": None,
                "duration": None,
                "form": None,
                "indication": None,
                "adverse_effect": None,
            })

        # Assign each modifier to the closest logical drug
        for mod in modifiers:
            mod_start = mod["start"]
            mod_end = mod["end"]
            mod_label = mod["label"].lower()

            best_drug_idx = -1
            min_dist = float("inf")

            for idx, drug in enumerate(drugs):
                drug_start = drug["start"]
                drug_end = drug["end"]

                # Calculate distance between drug span and modifier span
                if mod_start >= drug_end:
                    dist = mod_start - drug_end
                elif drug_start >= mod_end:
                    dist = drug_start - mod_end
                else:
                    dist = 0

                # Sentence boundary penalty: check if there's a period or newline between them
                intervening_text = text[min(drug_end, mod_start):max(drug_start, mod_end)]
                if "." in intervening_text or "\n" in intervening_text:
                    dist += 80  # Penalty for crossing sentence boundaries

                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    best_drug_idx = idx

            if best_drug_idx != -1:
                # If slot is already filled, append or only overwrite if closer
                current_val = structured_prescriptions[best_drug_idx].get(mod_label)
                if current_val is None:
                    structured_prescriptions[best_drug_idx][mod_label] = mod
                else:
                    # If this one is closer, replace or combine
                    existing_dist = abs(current_val["start"] - drugs[best_drug_idx]["start"])
                    if min_dist < existing_dist:
                        structured_prescriptions[best_drug_idx][mod_label] = mod

        return structured_prescriptions
