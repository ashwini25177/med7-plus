"""
MIMIC Data Preprocessing Adapter.
Converts free-text clinical discharge summaries from MIMIC-III (NOTEEVENTS.csv)
or MIMIC-IV-Note into standardized JSONL and spaCy training formats.
"""

import os
import csv
import json
import argparse
from typing import List, Dict


def extract_discharge_summaries_from_mimic(
    csv_path: str,
    output_json_path: str,
    max_records: int = 500
):
    """
    Parse MIMIC-III NOTEEVENTS.csv or MIMIC-IV discharge summaries and filter
    for discharge summary text records.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input MIMIC CSV file not found at: {csv_path}")

    extracted_records = []
    print(f"Reading MIMIC notes from {csv_path} (max: {max_records})...")

    with open(csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # Check MIMIC-III category column or MIMIC-IV note_type column
            category = row.get("CATEGORY", row.get("note_type", ""))
            
            # Filter for Discharge summaries
            if "Discharge summary" in category or "discharge" in category.lower() or not category:
                text = row.get("TEXT", row.get("text", "")).strip()
                if text:
                    record = {
                        "row_id": row.get("ROW_ID", row.get("note_id", str(idx))),
                        "subject_id": row.get("SUBJECT_ID", row.get("subject_id", "N/A")),
                        "hadm_id": row.get("HADM_ID", row.get("hadm_id", "N/A")),
                        "category": category,
                        "text": text
                    }
                    extracted_records.append(record)

            if len(extracted_records) >= max_records:
                break

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, mode="w", encoding="utf-8") as out_f:
        json.dump(extracted_records, out_f, indent=2)

    print(f"Successfully extracted {len(extracted_records)} records to {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract discharge notes from MIMIC-III or MIMIC-IV CSV.")
    parser.add_argument("--input", required=True, help="Path to NOTEEVENTS.csv or discharge.csv")
    parser.add_argument("--output", default="data/processed/mimic_notes.json", help="Output JSON path")
    parser.add_argument("--limit", type=int, default=500, help="Maximum number of records to extract")
    args = parser.parse_args()

    extract_discharge_summaries_from_mimic(args.input, args.output, args.limit)
