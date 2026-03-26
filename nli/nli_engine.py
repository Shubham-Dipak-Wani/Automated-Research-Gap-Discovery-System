from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class NLIEngine:
    def __init__(self):
        print("Loading NLI model...")

        model_name = "MoritzLaurer/deberta-v3-base-zeroshot-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def predict(self, c1, c2):
        inputs = self.tokenizer(c1, c2, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]
        labels = ["contradiction", "neutral", "entailment"]

        idx = torch.argmax(probs).item()
        return labels[idx], probs[idx].item()