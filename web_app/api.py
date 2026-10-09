"""
FastAPI Production Microservice for Med7-Plus.
Provides high-performance RESTful API endpoints for clinical information extraction,
entity recognition, medication relation linking, and FHIR export.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path so med7_plus is always importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from med7_plus.extractor import Med7ClinicalExtractor
from med7_plus.config import ALL_ENTITIES

app = FastAPI(
    title="Med7-Plus Clinical NLP API",
    description="Next-generation Clinical Named Entity Recognition & Medication Slot Extraction Service.",
    version="2.0.0"
)

# Initialize pipeline
extractor = Med7ClinicalExtractor()


class ExtractionRequest(BaseModel):
    text: str = Field(..., example="Patient prescribed Metformin 500 mg PO BID for diabetes. Discontinued Lisinopril.")
    patient_id: Optional[str] = Field("PATIENT-DEFAULT-001", example="PATIENT-12345")


class EntityResponse(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: Optional[float] = 0.95


class ExtractionResponse(BaseModel):
    text: str
    raw_entities: List[Dict[str, Any]]
    prescriptions: List[Dict[str, Any]]
    fhir_bundle: Dict[str, Any]
    metadata: Dict[str, Any]


@app.get("/health")
def health_check():
    """Health check probe."""
    return {"status": "healthy", "service": "med7-plus-clinical-nlp", "version": "2.0.0"}


@app.get("/labels")
def get_supported_labels():
    """Return all supported clinical entity labels."""
    return {"supported_entities": ALL_ENTITIES}


@app.post("/extract", response_model=ExtractionResponse)
def extract_clinical_info(request: ExtractionRequest):
    """
    Extract medication entities, link attributes, detect negation/context,
    and format into HL7 FHIR MedicationStatements.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    try:
        result = extractor.extract(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failure: {str(e)}")


@app.post("/extract/fhir")
def extract_fhir_only(request: ExtractionRequest):
    """
    Extract and directly return an HL7 FHIR R4 Bundle.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    result = extractor.extract(request.text)
    return result["fhir_bundle"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
