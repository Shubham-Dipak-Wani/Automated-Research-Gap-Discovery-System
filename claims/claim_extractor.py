import requests
from config.settings import OLLAMA_MODEL, OLLAMA_URL, MIN_CLAIM_LENGTH
from claims.prompts import CLAIM_EXTRACTION_PROMPT


class ClaimExtractor:
    def __init__(self):
        print(f"Using Ollama ({OLLAMA_MODEL}) for claim extraction")

    def extract_from_sentence(self, sentence, paper_title="", section=""):
        prompt = CLAIM_EXTRACTION_PROMPT.format(sentence=sentence)

        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256}
        })

        if response.status_code != 200:
            print(f"Ollama error: {response.status_code}")
            return []

        text = response.json().get("response", "")
        claims = self._parse_output(text)

        # Attach source metadata for citation tracing
        return [
            {
                "text": c,
                "paper_title": paper_title,
                "section": section,
                "source_sentence": sentence,
            }
            for c in claims
        ]

    def _parse_output(self, text):
        if "NONE" in text.strip().upper():
            return []

        lines = text.strip().split("\n")
        claims = []

        for line in lines:
            line = line.strip().lstrip("- ").lstrip("* ").strip()

            if not line:
                continue

            low = line.lower()

            # Skip metadata / garbage
            if any(x in low for x in [
                "acm", "workshop", "arxiv", "doi",
                "copyright", "permission", "manuscript",
                "author", "email", "@", "http", "www",
                "sentence:", "claims:", "none"
            ]):
                continue

            if len(line) < MIN_CLAIM_LENGTH:
                continue

            if not line.endswith("."):
                line += "."

            claims.append(line)

        return claims
