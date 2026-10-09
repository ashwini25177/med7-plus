"""
Clinical NER evaluation module implementing Strict and Lenient F1 scoring
according to the standards established in Kormilitzin et al. (2021) and the n2c2/i2b2 challenges.
"""

from typing import Dict, List, Tuple


class ClinicalEvaluator:
    """
    Computes precision, recall, and F1-score for clinical named entity recognition
    under both Strict and Lenient matching criteria.
    """

    @staticmethod
    def is_overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        """Check if two character intervals overlap."""
        return max(span1[0], span2[0]) < min(span1[1], span2[1])

    def evaluate(
        self,
        gold_annotations: List[List[Dict[str, any]]],
        pred_annotations: List[List[Dict[str, any]]]
    ) -> Dict[str, any]:
        """
        Evaluate predicted entities against gold annotations across a corpus of documents.

        Args:
            gold_annotations: List of documents, each having a list of gold entity dicts
                              with keys: "start", "end", "label".
            pred_annotations: List of documents, each having a list of predicted entity dicts.

        Returns:
            Dictionary with strict and lenient micro-averaged precision, recall, and F1 metrics,
            plus per-class breakdowns.
        """
        strict_tp = 0
        strict_fp = 0
        strict_fn = 0

        lenient_tp = 0
        lenient_fp = 0
        lenient_fn = 0

        per_label_counts: Dict[str, Dict[str, int]] = {}

        for gold_doc, pred_doc in zip(gold_annotations, pred_annotations):
            gold_matched_strict = set()
            pred_matched_strict = set()

            gold_matched_lenient = set()
            pred_matched_lenient = set()

            # Strict Matching (Exact span + exact label)
            for p_idx, p in enumerate(pred_doc):
                label = p["label"]
                if label not in per_label_counts:
                    per_label_counts[label] = {
                        "strict_tp": 0, "strict_fp": 0, "strict_fn": 0,
                        "lenient_tp": 0, "lenient_fp": 0, "lenient_fn": 0
                    }

                matched = False
                for g_idx, g in enumerate(gold_doc):
                    if g_idx in gold_matched_strict:
                        continue
                    if g["label"] == p["label"] and g["start"] == p["start"] and g["end"] == p["end"]:
                        strict_tp += 1
                        per_label_counts[label]["strict_tp"] += 1
                        gold_matched_strict.add(g_idx)
                        pred_matched_strict.add(p_idx)
                        matched = True
                        break
                if not matched:
                    strict_fp += 1
                    per_label_counts[label]["strict_fp"] += 1

            for g_idx, g in enumerate(gold_doc):
                if g_idx not in gold_matched_strict:
                    strict_fn += 1
                    lbl = g["label"]
                    if lbl not in per_label_counts:
                        per_label_counts[lbl] = {
                            "strict_tp": 0, "strict_fp": 0, "strict_fn": 0,
                            "lenient_tp": 0, "lenient_fp": 0, "lenient_fn": 0
                        }
                    per_label_counts[lbl]["strict_fn"] += 1

            # Lenient Matching (Overlapping span + exact label)
            for p_idx, p in enumerate(pred_doc):
                label = p["label"]
                matched = False
                for g_idx, g in enumerate(gold_doc):
                    if g_idx in gold_matched_lenient:
                        continue
                    if g["label"] == p["label"] and self.is_overlap((g["start"], g["end"]), (p["start"], p["end"])):
                        lenient_tp += 1
                        per_label_counts[label]["lenient_tp"] += 1
                        gold_matched_lenient.add(g_idx)
                        pred_matched_lenient.add(p_idx)
                        matched = True
                        break
                if not matched:
                    lenient_fp += 1
                    per_label_counts[label]["lenient_fp"] += 1

            for g_idx, g in enumerate(gold_doc):
                if g_idx not in gold_matched_lenient:
                    lenient_fn += 1
                    lbl = g["label"]
                    per_label_counts[lbl]["lenient_fn"] += 1

        def calc_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            return round(precision, 4), round(recall, 4), round(f1, 4)

        s_p, s_r, s_f1 = calc_prf(strict_tp, strict_fp, strict_fn)
        l_p, l_r, l_f1 = calc_prf(lenient_tp, lenient_fp, lenient_fn)

        per_label_summary = {}
        for lbl, stats in per_label_counts.items():
            sp, sr, sf = calc_prf(stats["strict_tp"], stats["strict_fp"], stats["strict_fn"])
            lp, lr, lf = calc_prf(stats["lenient_tp"], stats["lenient_fp"], stats["lenient_fn"])
            per_label_summary[lbl] = {
                "strict": {"precision": sp, "recall": sr, "f1": sf},
                "lenient": {"precision": lp, "recall": lr, "f1": lf}
            }

        return {
            "micro_strict": {
                "precision": s_p,
                "recall": s_r,
                "f1": s_f1,
                "tp": strict_tp,
                "fp": strict_fp,
                "fn": strict_fn
            },
            "micro_lenient": {
                "precision": l_p,
                "recall": l_r,
                "f1": l_f1,
                "tp": lenient_tp,
                "fp": lenient_fp,
                "fn": lenient_fn
            },
            "per_label": per_label_summary
        }
