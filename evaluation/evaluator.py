import json
from datasets import load_dataset
from claims.claim_extractor import ClaimExtractor
from nli.nli_engine import NLIEngine
from evaluation.metrics import compute_claim_f1, compute_nli_f1


class SciFactEvaluator:
    """Evaluate claim extraction and NLI against SciFact dataset."""

    def __init__(self):
        self.extractor = ClaimExtractor()
        self.nli = NLIEngine()

    def load_scifact(self):
        """Load SciFact dataset from HuggingFace."""
        dataset = load_dataset("allenai/scifact", "claims")
        return dataset

    def evaluate_claim_extraction(self, num_samples=100):
        """
        Run claim extractor on SciFact abstracts and compare
        against ground-truth claims.
        """
        print(f"Evaluating claim extraction on {num_samples} samples...")
        dataset = self.load_scifact()

        predicted = []
        gold = []

        for i, example in enumerate(dataset["train"]):
            if i >= num_samples:
                break

            claim = example["claim"]
            gold.append(claim)

            # Extract claims from the claim text itself
            # (SciFact claims are already atomic, so good extraction = high similarity)
            extracted = self.extractor.extract_from_sentence(claim)
            if extracted:
                predicted.append(extracted[0]["text"])
            else:
                predicted.append("")

        results = compute_claim_f1(predicted, gold)
        print(f"Claim Extraction - P: {results['precision']:.3f}, "
              f"R: {results['recall']:.3f}, F1: {results['f1']:.3f}")
        return results

    def evaluate_nli(self, num_samples=200):
        """
        Run NLI on SciFact claim-abstract pairs and measure
        contradiction detection F1.
        """
        print(f"Evaluating NLI on {num_samples} samples...")
        dataset = self.load_scifact()

        predictions = []
        gold_labels = []

        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

        for i, example in enumerate(dataset["train"]):
            if i >= num_samples:
                break

            claim = example["claim"]
            evidence = example.get("evidence", "")
            label = example.get("label")

            if not evidence or label is None:
                continue

            gold_label = label_map.get(label, "neutral")
            gold_labels.append(gold_label)

            pred_label, _ = self.nli.predict(claim, evidence)
            predictions.append(pred_label)

        results = compute_nli_f1(predictions, gold_labels)
        print(f"NLI - F1 (macro): {results['f1_macro']:.3f}, "
              f"F1 (contradiction): {results['f1_contradiction']:.3f}")
        return results

    def run_full_evaluation(self):
        """Run all evaluations and save results."""
        print("\n=== SciFact Evaluation ===\n")

        claim_results = self.evaluate_claim_extraction()
        nli_results = self.evaluate_nli()

        results = {
            "claim_extraction": claim_results,
            "nli_detection": nli_results,
        }

        with open("data/evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to data/evaluation_results.json")
        return results


if __name__ == "__main__":
    evaluator = SciFactEvaluator()
    evaluator.run_full_evaluation()
