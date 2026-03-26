from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class ClaimExtractor:
    def __init__(self):
        print("Using FLAN-T5-SMALL")

        model_name = "google/flan-t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def extract_from_sentence(self, sentence):
        prompt = f"""
Extract atomic scientific claims from the sentence.

Sentence: "{sentence}"

Claims:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return self._parse_output(text)

    def _parse_output(self, text):
        lines = text.split("\n")
        claims = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            low = line.lower()

            # remove metadata / garbage
            if any(x in low for x in [
                "acm", "workshop", "arxiv", "doi",
                "copyright", "permission", "manuscript",
                "author", "email", "@", "http", "www"
            ]):
                continue

            if len(line) < 30:
                continue

            if '"' in line or "p." in line:
                continue

            line = line.lstrip("- ").strip()

            if not line.endswith("."):
                line += "."

            claims.append(line)

        return claims