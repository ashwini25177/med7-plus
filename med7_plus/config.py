"""
Configuration and constants for Med7-Plus Clinical Information Extraction System.
Defines entity categories, visual color schemes, clinical regex lexicons, and defaults.
"""

from typing import Dict, List

# Core Med7 Entity Categories (Original paper: Kormilitzin et al., 2021)
CORE_ENTITIES: List[str] = [
    "DRUG",
    "STRENGTH",
    "DOSAGE",
    "DURATION",
    "FREQUENCY",
    "FORM",
    "ROUTE",
]

# Extended Modern Clinical Entities (n2c2 2022 & Modern EHR Extensions)
EXTENDED_ENTITIES: List[str] = [
    "INDICATION",       # Clinical rationale (e.g., 'for hypertension')
    "ADVERSE_EFFECT",   # Documented drug allergy or reaction
]

ALL_ENTITIES: List[str] = CORE_ENTITIES + EXTENDED_ENTITIES

# Semantic Color Palette for Clinical Visualizers (DisplaCy & Streamlit)
ENTITY_COLORS: Dict[str, str] = {
    "DRUG": "#70a1ff",          # Vibrant medical blue
    "STRENGTH": "#2ed573",      # Emerald green
    "DOSAGE": "#ffa502",        # Warm amber
    "DURATION": "#ff6b81",      # Coral rose
    "FREQUENCY": "#9b59b6",     # Amethyst purple
    "FORM": "#1abc9c",          # Turquoise
    "ROUTE": "#e67e22",         # Terracotta orange
    "INDICATION": "#00d2d3",    # Cyan
    "ADVERSE_EFFECT": "#ff4757" # Crimson red
}

# Context Classifications (from 2022 n2c2 Track 1: CMED)
NEGATION_TRIGGERS: List[str] = [
    "no", "not", "denies", "denied", "stopped", "discontinued", "hold", "held",
    "withheld", "without", "never", "refused", "intolerant to", "allergies to",
    "allergic to", "ceased", "tapered off", "avoid"
]

CERTAINTY_TRIGGERS: Dict[str, List[str]] = {
    "POSSIBLE": ["possible", "suspected", "evaluate for", "consider", "presumed", "questionable", "unclear if"],
    "CONFIRMED": ["confirmed", "definite", "diagnosed", "documented", "known"],
    "CONDITIONAL": ["if needed", "as needed", "prn", "as required", "in case of", "when necessary"]
}

TEMPORALITY_TRIGGERS: Dict[str, List[str]] = {
    "PAST": ["prior to", "previously", "history of", "in the past", "formerly", "used to take", "discontinued on"],
    "CURRENT": ["currently", "presently", "regimen consists of", "takes", "taking", "initiated on", "started on"],
    "FUTURE": ["will start", "to begin", "plan for", "discharge prescription", "follow-up", "to be administered"]
}

# Standardized Units & Formats
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_SPACY_MODEL = "en_core_web_sm"
RECOMMENDED_TRANSFORMER_BACKBONE = "emilyalsentzer/Bio_ClinicalBERT"
