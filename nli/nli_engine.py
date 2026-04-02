from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config.settings import NLI_MODEL


class NLIEngine:
    def __init__(self):
        print("Loading NLI model...")

        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)

        # Read label order from model config instead of hardcoding
        self.labels = [
            self.model.config.id2label[i]
            for i in range(len(self.model.config.id2label))
        ]

    def predict(self, c1, c2):
        inputs = self.tokenizer(c1, c2, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]

        idx = torch.argmax(probs).item()
        return self.labels[idx], probs[idx].item()
