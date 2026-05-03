from config.settings import NLI_MODEL


class NLIEngine:
    """DeBERTa-v3 natural language inference: entailment, neutral or contradiction."""

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print("Loading NLI model...")
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)

        # Read the label order from the model config instead of hardcoding it,
        # so swapping checkpoints can't silently scramble the labels.
        self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]

    def predict(self, premise, hypothesis):
        """Return (label, confidence) for premise → hypothesis."""
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self._torch.softmax(logits, dim=1)[0]
        idx = self._torch.argmax(probs).item()
        return self.labels[idx].lower(), probs[idx].item()
