"""
n2c2 / i2b2 Annotation Converter.
Converts Brat (.txt and .ann) standoff clinical annotations from the 2018 n2c2 Track 2
and 2022 n2c2 Track 1 challenges into standard spaCy training examples.
"""

import os
import glob
import json
import argparse
from typing import List, Dict


def parse_brat_ann_file(ann_path: str) -> List[Dict[str, any]]:
    """
    Parse a Brat .ann standoff file into structured entity span records.
    Format: T1  Drug 45 53  Aspirin
    """
    entities = []
    with open(ann_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("T"):
                continue  # Skip relations or non-text-bound annotations for now
            parts = line.split("\t")
            if len(parts) >= 3:
                tag_info = parts[1].split()
                label = tag_info[0].upper()
                start = int(tag_info[1])
                end = int(tag_info[2])
                text = parts[2]
                entities.append({
                    "start": start,
                    "end": end,
                    "label": label,
                    "text": text
                })
    return entities


def convert_n2c2_directory(input_dir: str, output_json_path: str):
    """
    Find all pairs of .txt and .ann files in input_dir and convert to a unified JSON dataset.
    """
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    dataset = []
    for txt_file in txt_files:
        base_name = os.path.splitext(txt_file)[0]
        ann_file = base_name + ".ann"

        with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        entities = []
        if os.path.exists(ann_file):
            entities = parse_brat_ann_file(ann_file)

        dataset.append({
            "id": os.path.basename(base_name),
            "text": text,
            "entities": entities
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as out_f:
        json.dump(dataset, out_f, indent=2)

    print(f"Successfully converted {len(dataset)} documents to {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert n2c2 Brat .ann files to JSON dataset.")
    parser.add_argument("--input-dir", required=True, help="Directory containing .txt and .ann files")
    parser.add_argument("--output", default="data/processed/n2c2_dataset.json", help="Output JSON path")
    args = parser.parse_args()

    convert_n2c2_directory(args.input_dir, args.output)
