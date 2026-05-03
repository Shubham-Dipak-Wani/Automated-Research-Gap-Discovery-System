"""Evaluate claim extraction and NLI against SciFact (allenai/scifact).

    python -m evaluation.evaluator --nli 300 --claims 100

SciFact pairs expert-written claims with evidence sentences from paper
abstracts, labeled SUPPORT or CONTRADICT. Two checks:

  NLI         premise = the evidence sentences, hypothesis = the claim.
              SUPPORT → entailment, CONTRADICT → contradiction. Claims with no
              evidence are paired with a cited abstract as neutral.
  Extraction  run the claim extractor on the evidence sentences of SUPPORT
              examples, and count how often the gold claim is recovered
              (BERTScore >= threshold). This measures recall only: an abstract
              holds many valid claims that SciFact doesn't list.
"""
import argparse
import json
import os

from evaluation.metrics import compute_claim_recall, compute_nli_f1

GOLD_TO_NLI = {"SUPPORT": "entailment", "CONTRADICT": "contradiction"}


def build_examples(claims_rows, corpus_rows) -> list:
    """Join SciFact claims with abstract text: [{claim, premise, label}]."""
    corpus = {int(r["doc_id"]): r["abstract"] for r in corpus_rows}
    examples = []
    for row in claims_rows:
        label = (row.get("evidence_label") or "").upper()
        doc_id = row.get("evidence_doc_id")
        if label in GOLD_TO_NLI and doc_id not in (None, ""):
            abstract = corpus.get(int(doc_id))
            sentence_ids = row.get("evidence_sentences") or []
            if not abstract or not sentence_ids:
                continue
            premise = " ".join(abstract[i] for i in sentence_ids if 0 <= i < len(abstract))
            examples.append({"claim": row["claim"], "premise": premise, "label": GOLD_TO_NLI[label]})
        elif not label and row.get("cited_doc_ids"):
            abstract = corpus.get(int(row["cited_doc_ids"][0]))
            if abstract:
                examples.append({"claim": row["claim"], "premise": " ".join(abstract[:3]), "label": "neutral"})
    return examples


def load_scifact(split="train"):
    from datasets import load_dataset

    claims = load_dataset("allenai/scifact", "claims")[split]
    corpus = load_dataset("allenai/scifact", "corpus")["train"]
    return list(claims), list(corpus)


def evaluate_nli(nli, examples) -> dict:
    predictions = [nli.predict(e["premise"], e["claim"])[0] for e in examples]
    return compute_nli_f1(predictions, [e["label"] for e in examples])


def evaluate_extraction(extractor, examples, threshold=0.8) -> dict:
    pairs = []
    for e in examples:
        if e["label"] != "entailment":
            continue
        extracted = extractor.extract_from_sentence(e["premise"], paper_title="SciFact", section="abstract")
        pairs.append((e["claim"], [c["text"] for c in extracted]))
    return compute_claim_recall(pairs, threshold=threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli", type=int, default=300, help="examples for the NLI check")
    ap.add_argument("--claims", type=int, default=100, help="SUPPORT examples for the extraction check (uses the Claude API)")
    ap.add_argument("--out", default="data/evaluation_results.json")
    args = ap.parse_args()

    examples = build_examples(*load_scifact("train"))
    print(f"{len(examples)} SciFact examples")

    from nli.nli_engine import NLIEngine
    results = {"nli": evaluate_nli(NLIEngine(), examples[: args.nli])}
    print("NLI:", results["nli"])

    if args.claims:
        from claims.claim_extractor import ClaimExtractor
        support = [e for e in examples if e["label"] == "entailment"][: args.claims]
        results["claim_extraction"] = evaluate_extraction(ClaimExtractor(), support)
        print("Claim extraction:", results["claim_extraction"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
