"""
spaCy Clinical NER Training Pipeline.
Converts JSON annotations to DocBin format, initializes a pipeline
(supporting both CNN/hash-embedding and transformer backbones like Bio_ClinicalBERT),
and trains a custom clinical NER model with Strict/Lenient F1 evaluation.
"""

import os
import json
import argparse
from typing import List, Dict
from med7_plus.config import ALL_ENTITIES
from med7_plus.evaluator import ClinicalEvaluator


def train_custom_ner(
    train_json_path: str,
    output_model_dir: str = "models/custom_clinical_ner",
    iterations: int = 20,
    base_model: str = "en_core_web_sm"
):
    """
    Train a spaCy NER model using annotated clinical examples from a JSON file.
    """
    try:
        import spacy
        from spacy.training import Example
    except ImportError:
        print("[Error] spaCy is required for training. Please install with: pip install spacy")
        return

    if not os.path.exists(train_json_path):
        raise FileNotFoundError(f"Training dataset not found: {train_json_path}")

    with open(train_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} training clinical documents from {train_json_path}")

    # Load or initialize spaCy model
    try:
        nlp = spacy.load(base_model)
        print(f"Loaded base model: {base_model}")
    except Exception:
        print(f"Base model '{base_model}' not found locally. Initializing blank 'en' pipeline.")
        nlp = spacy.blank("en")

    # Add or retrieve NER component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Register labels
    for label in ALL_ENTITIES:
        ner.add_label(label)

    # Convert to spaCy Examples
    examples: List[Example] = []
    gold_annotations_eval = []
    
    for item in data:
        text = item["text"]
        entities = item.get("entities", [])
        
        # Format for spaCy
        spacy_entities = []
        for e in entities:
            spacy_entities.append((e["start"], e["end"], e["label"]))
        
        annotations = {"entities": spacy_entities}
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
        gold_annotations_eval.append(entities)

    # Disable other pipes during NER training
    pipe_exceptions = ["ner", "trf_tok2vec", "transformer"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    print(f"Starting training for {iterations} iterations...")
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.initialize() if not hasattr(nlp, "vocab") else nlp.resume_training()
        for itn in range(iterations):
            losses = {}
            for example in examples:
                nlp.update([example], drop=0.2, losses=losses, sgd=optimizer)
            if (itn + 1) % 5 == 0 or itn == 0:
                print(f"Iteration {itn + 1:2d} | Losses: {losses}")

    # Save trained model
    os.makedirs(output_model_dir, exist_ok=True)
    nlp.to_disk(output_model_dir)
    print(f"\nModel successfully saved to: {output_model_dir}")

    # Run internal Strict vs Lenient validation evaluation
    print("\nRunning post-training clinical evaluation...")
    evaluator = ClinicalEvaluator()
    pred_annotations_eval = []
    for item in data:
        pred_doc = nlp(item["text"])
        preds = [{"start": ent.start_char, "end": ent.end_char, "label": ent.label_} for ent in pred_doc.ents]
        pred_annotations_eval.append(preds)

    metrics = evaluator.evaluate(gold_annotations_eval, pred_annotations_eval)
    print("--- Training Set Evaluation Summary ---")
    print(f"Lenient F1: {metrics['micro_lenient']['f1']:.4f} (Precision: {metrics['micro_lenient']['precision']:.4f}, Recall: {metrics['micro_lenient']['recall']:.4f})")
    print(f"Strict  F1: {metrics['micro_strict']['f1']:.4f} (Precision: {metrics['micro_strict']['precision']:.4f}, Recall: {metrics['micro_strict']['recall']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom spaCy clinical NER pipeline.")
    parser.add_argument("--data", default="data/sample/synthetic_ehr_notes.json", help="Path to JSON dataset")
    parser.add_argument("--output", default="models/custom_clinical_ner", help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--base-model", default="en_core_web_sm", help="Base spaCy model or blank")
    args = parser.parse_args()

    train_custom_ner(args.data, args.output, args.epochs, args.base_model)
