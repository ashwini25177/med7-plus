"""
Med7-Plus: Clinical Information Extraction System in Python & spaCy
Next-generation clinical Named Entity Recognition, medication relation extraction,
and contextual event analysis for Electronic Health Records (EHR).
"""

from med7_plus.config import (
    CORE_ENTITIES,
    EXTENDED_ENTITIES,
    ALL_ENTITIES,
    ENTITY_COLORS
)
from med7_plus.extractor import Med7ClinicalExtractor
from med7_plus.context_analyzer import ClinicalContextAnalyzer
from med7_plus.normalizer import DrugNormalizer
from med7_plus.relation_linker import MedicationRelationLinker
from med7_plus.evaluator import ClinicalEvaluator
from med7_plus.fhir_exporter import FHIRExporter

__version__ = "2.0.0"
__author__ = "Med7-Plus Open Source Initiative"

__all__ = [
    "Med7ClinicalExtractor",
    "ClinicalContextAnalyzer",
    "DrugNormalizer",
    "MedationRelationLinker",
    "ClinicalEvaluator",
    "FHIRExporter",
    "CORE_ENTITIES",
    "EXTENDED_ENTITIES",
    "ALL_ENTITIES",
    "ENTITY_COLORS",
]
