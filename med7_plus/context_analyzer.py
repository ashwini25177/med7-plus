"""
Context analysis module for clinical named entity recognition.
Implements rule-based NegEx and temporal/certainty heuristics inspired by
the 2022 n2c2 Track 1 (Contextualized Medication Event Dataset - CMED).
"""

import re
from typing import Dict, List, Optional
from med7_plus.config import NEGATION_TRIGGERS, CERTAINTY_TRIGGERS, TEMPORALITY_TRIGGERS


class ClinicalContextAnalyzer:
    """
    Analyzes the clinical context surrounding extracted medication entities,
    including negation (is the drug active or withheld?), temporality (past/current/future),
    and certainty (confirmed vs conditional/possible).
    """

    def __init__(self, window_size_tokens: int = 10):
        """
        Initialize the context analyzer.

        Args:
            window_size_tokens: Number of words before and after an entity to inspect for cues.
        """
        self.window_size = window_size_tokens
        self.negation_patterns = [
            re.compile(r"\b" + re.escape(trigger) + r"\b", re.IGNORECASE)
            for trigger in NEGATION_TRIGGERS
        ]

    def get_surrounding_window(self, full_text: str, start_char: int, end_char: int) -> str:
        """Extract a substring window around the entity bounds."""
        # Preceding context
        pre_text = full_text[:start_char].strip()
        pre_words = pre_text.split()[-self.window_size:] if pre_text else []
        
        # Following context
        post_text = full_text[end_char:].strip()
        post_words = post_text.split()[:self.window_size] if post_text else []

        window_text = " ".join(pre_words + [full_text[start_char:end_char]] + post_words)
        return window_text.lower()

    def is_negated(self, full_text: str, start_char: int, end_char: int) -> bool:
        """
        Determine if an entity is negated (e.g., 'patient denies aspirin', 'discontinued metoprolol').
        """
        # Look predominantly in the preceding window up to the nearest clause boundary (comma, period, semicolon)
        pre_text = full_text[:start_char]
        # Split on clause terminators
        clauses = re.split(r"[\.\;\n\,]", pre_text)
        current_clause = clauses[-1] if clauses else ""

        for pattern in self.negation_patterns:
            if pattern.search(current_clause):
                return True

        # Check immediate post-context for constructs like 'aspirin was stopped'
        post_text = full_text[end_char:]
        post_clause = re.split(r"[\.\;\n\,]", post_text)[0] if post_text else ""
        post_triggers = ["was stopped", "discontinued", "withheld", "ceased", "not tolerated"]
        for pt in post_triggers:
            if pt in post_clause.lower():
                return True

        return False

    def detect_certainty(self, full_text: str, start_char: int, end_char: int) -> str:
        """
        Classifies certainty into: 'CONFIRMED', 'POSSIBLE', or 'CONDITIONAL' (PRN).
        """
        window = self.get_surrounding_window(full_text, start_char, end_char)

        for trigger in CERTAINTY_TRIGGERS["CONDITIONAL"]:
            if trigger in window:
                return "CONDITIONAL"

        for trigger in CERTAINTY_TRIGGERS["POSSIBLE"]:
            if trigger in window:
                return "POSSIBLE"

        return "CONFIRMED"

    def detect_temporality(self, full_text: str, start_char: int, end_char: int) -> str:
        """
        Classifies temporality into: 'PAST', 'CURRENT', or 'FUTURE'.
        """
        window = self.get_surrounding_window(full_text, start_char, end_char)

        for trigger in TEMPORALITY_TRIGGERS["PAST"]:
            if trigger in window:
                return "PAST"

        for trigger in TEMPORALITY_TRIGGERS["FUTURE"]:
            if trigger in window:
                return "FUTURE"

        return "CURRENT"

    def analyze_entity_context(self, full_text: str, start_char: int, end_char: int) -> Dict[str, any]:
        """
        Produce a comprehensive context dictionary for an extracted clinical entity.
        """
        negated = self.is_negated(full_text, start_char, end_char)
        certainty = self.detect_certainty(full_text, start_char, end_char)
        temporality = self.detect_temporality(full_text, start_char, end_char)

        return {
            "is_negated": negated,
            "status": "DISCONTINUED_OR_NEGATED" if negated else "ACTIVE",
            "certainty": certainty,
            "temporality": temporality,
        }
