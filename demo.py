"""
Med7-Plus Quickstart Interactive CLI Demo.
Run with: python demo.py
"""

import sys
import json
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from med7_plus.extractor import Med7ClinicalExtractor
from med7_plus.evaluator import ClinicalEvaluator

# Force stdout UTF-8 encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_demo():
    print("=" * 75)
    print("[+] Med7-Plus: Clinical Information Extraction System (CLI Demo)")
    print("=" * 75)

    extractor = Med7ClinicalExtractor()

    sample_note = (
        "Patient admitted with acute decompensated heart failure. "
        "Discharge medications: Furosemide 40 mg tablet PO once daily for 14 days. "
        "Metoprolol succinate 25 mg PO daily. "
        "Discontinued Lisinopril due to persistent dry cough. "
        "Initiate Metformin 500 mg PO BID with meals. "
        "Patient denies taking Warfarin at home."
    )

    print("\n[*] Input Clinical Narrative:")
    print("-" * 75)
    print(sample_note)
    print("-" * 75)

    print("\n[*] Running Med7-Plus Pipeline...")
    result = extractor.extract(sample_note)

    print(f"\n[+] Total Raw Extracted Entities: {len(result['raw_entities'])}")
    for ent in result["raw_entities"]:
        print(f"  * [{ent['label']:<10}] {ent['text']:<22} (Span: {ent['start']:>3}:{ent['end']:<3})")

    print("\n[+] Resolved Medication Events (Slot Linking + Context Analysis):")
    print("=" * 75)
    for idx, p in enumerate(result["prescriptions"], 1):
        drug = p["drug"]
        norm = drug.get("normalization", {})
        ctx = drug.get("context", {})

        status_str = "[DISCONTINUED / NEGATED]" if ctx.get("is_negated") else "[ACTIVE]"
        print(f"\n[Medication #{idx}] {drug['text']}")
        print(f"  |-- RxNorm Generic:  {norm.get('generic_name')} (RxCUI: {norm.get('rxcui')})")
        print(f"  |-- ATC Category:    {norm.get('category')} ({norm.get('atc_code')})")
        print(f"  |-- Clinical Status: {status_str}")
        print(f"  |-- Certainty:       {ctx.get('certainty')}")
        print(f"  |-- Strength:        {p['strength']['text'] if p.get('strength') else '--'}")
        print(f"  |-- Route:           {p['route']['text'] if p.get('route') else '--'}")
        print(f"  |-- Frequency:       {p['frequency']['text'] if p.get('frequency') else '--'}")
        print(f"  \\-- Duration:        {p['duration']['text'] if p.get('duration') else '--'}")

    print("\n" + "=" * 75)
    print("[+] Sample HL7 FHIR R4 MedicationStatement (First Entry):")
    print("=" * 75)
    if result["fhir_bundle"]["entry"]:
        first_statement = result["fhir_bundle"]["entry"][0]["resource"]
        print(json.dumps(first_statement, indent=2))

    print("\n" + "=" * 75)
    print("[OK] Demo Complete! Run 'streamlit run web_app/app_streamlit.py' for the visual dashboard.")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()
