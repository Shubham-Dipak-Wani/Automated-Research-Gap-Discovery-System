from sklearn.metrics import precision_score, recall_score, f1_score
from bert_score import score as bert_score
import numpy as np


def compute_claim_f1(predicted_claims, gold_claims, threshold=0.8):
    """
    Compute precision, recall, F1 for claim extraction using BERTScore
    for semantic matching (accounts for valid paraphrasing).
    """
    if not predicted_claims or not gold_claims:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # BERTScore: compare each predicted claim against all gold claims
    P, R, F1 = bert_score(
        predicted_claims, gold_claims,
        lang="en", rescale_with_baseline=True, verbose=False,
    )

    # A predicted claim "matches" if its best BERTScore F1 >= threshold
    pred_matched = (F1.max(dim=1).values >= threshold).float()
    precision = pred_matched.mean().item()

    # For recall: check each gold claim against all predicted claims
    P2, R2, F12 = bert_score(
        gold_claims, predicted_claims,
        lang="en", rescale_with_baseline=True, verbose=False,
    )
    gold_matched = (F12.max(dim=1).values >= threshold).float()
    recall = gold_matched.mean().item()

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_nli_f1(predictions, gold_labels):
    """
    Compute F1 for NLI contradiction detection.
    predictions: list of predicted labels ("contradiction", "entailment", "neutral")
    gold_labels: list of gold labels
    """
    label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}

    pred_ids = [label_map.get(p, 2) for p in predictions]
    gold_ids = [label_map.get(g, 2) for g in gold_labels]

    return {
        "f1_macro": f1_score(gold_ids, pred_ids, average="macro"),
        "f1_contradiction": f1_score(gold_ids, pred_ids, average="binary", pos_label=0),
        "precision_contradiction": precision_score(gold_ids, pred_ids, average="binary", pos_label=0),
        "recall_contradiction": recall_score(gold_ids, pred_ids, average="binary", pos_label=0),
    }
