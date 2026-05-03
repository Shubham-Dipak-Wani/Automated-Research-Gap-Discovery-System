"""Evaluation metrics.

BERTScore compares meaning rather than exact wording, so a correct paraphrase
of a gold claim still counts as a match.
"""
from sklearn.metrics import f1_score, precision_score, recall_score

NLI_LABELS = ("contradiction", "entailment", "neutral")


def _bert_scorer(cands, refs):
    from bert_score import score

    _, _, f1 = score(cands, refs, lang="en", rescale_with_baseline=True, verbose=False)
    return f1.tolist()


def pairwise_similarity(a: list, b: list, scorer=_bert_scorer) -> list:
    """len(a) x len(b) matrix of BERTScore F1.

    bert_score.score compares item i with item i only, so every (a, b) pair is
    expanded first. (Calling .max(dim=1) on its 1-D output, as the first
    version did, raises an error.)
    """
    if not a or not b:
        return [[0.0] * len(b) for _ in a]
    cands = [x for x in a for _ in b]
    refs = [y for _ in a for y in b]
    flat = scorer(cands, refs)
    return [flat[i * len(b):(i + 1) * len(b)] for i in range(len(a))]


def compute_claim_f1(predicted_claims, gold_claims, threshold=0.8, scorer=_bert_scorer):
    """Set-level precision/recall/F1: a claim matches if its best BERTScore >= threshold."""
    predicted = [c for c in predicted_claims if c and c.strip()]
    gold = [c for c in gold_claims if c and c.strip()]
    if not predicted or not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    sims = pairwise_similarity(predicted, gold, scorer)
    precision = sum(max(row) >= threshold for row in sims) / len(predicted)
    recall = sum(max(sims[i][j] for i in range(len(predicted))) >= threshold for j in range(len(gold))) / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_claim_recall(examples, threshold=0.8, scorer=_bert_scorer):
    """Share of gold claims recovered among the claims extracted from their evidence.

    examples: [(gold_claim, [extracted claims])].
    """
    if not examples:
        return {"recall": 0.0, "matched": 0, "total": 0}
    matched = 0
    for gold, extracted in examples:
        extracted = [c for c in extracted if c and c.strip()]
        if extracted and max(pairwise_similarity([gold], extracted, scorer)[0]) >= threshold:
            matched += 1
    return {"recall": matched / len(examples), "matched": matched, "total": len(examples)}


def compute_nli_f1(predictions, gold_labels):
    """Macro F1 over the three NLI labels, plus contradiction-only precision, recall and F1."""
    if not predictions or len(predictions) != len(gold_labels):
        raise ValueError("predictions and gold_labels must be non-empty and the same length")
    pred = [p if p in NLI_LABELS else "neutral" for p in predictions]
    gold = [g if g in NLI_LABELS else "neutral" for g in gold_labels]
    kw = {"labels": ["contradiction"], "average": "micro", "zero_division": 0}
    return {
        "f1_macro": f1_score(gold, pred, labels=list(NLI_LABELS), average="macro", zero_division=0),
        "f1_contradiction": f1_score(gold, pred, **kw),
        "precision_contradiction": precision_score(gold, pred, **kw),
        "recall_contradiction": recall_score(gold, pred, **kw),
        "n": len(gold),
    }
