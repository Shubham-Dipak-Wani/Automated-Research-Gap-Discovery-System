import requests
from config.settings import (
    OLLAMA_MODEL, OLLAMA_URL,
    MIN_CLAIM_LENGTH_FOR_GAP, MAX_CLAIMS_PER_GAP
)
from gap_generation.prompts import GAP_GENERATION_PROMPT, CONTRADICTION_GAP_PROMPT


class GapGenerator:
    def __init__(self):
        print(f"Using Ollama ({OLLAMA_MODEL}) for gap generation")

    def generate_gap(self, cluster_claims, contradictions=None):
        if contradictions:
            return self._generate_contradiction_gap(contradictions)
        return self._generate_cluster_gap(cluster_claims)

    def _generate_cluster_gap(self, cluster_claims):
        clean = [c for c in cluster_claims if len(c["text"]) > MIN_CLAIM_LENGTH_FOR_GAP]
        clean = clean[:MAX_CLAIMS_PER_GAP]

        if not clean:
            return {
                "gap": "Insufficient claim data for gap generation.",
                "supporting_claims": [],
                "source_papers": [],
                "type": "open_question",
            }

        claims_text = "\n".join(
            f'- "{c["text"]}" (from: {c["paper_title"]}, section: {c["section"]})'
            for c in clean
        )

        prompt = GAP_GENERATION_PROMPT.format(claims_text=claims_text)
        gap_text = self._call_ollama(prompt)

        source_papers = list({c["paper_title"] for c in clean})

        return {
            "gap": gap_text,
            "supporting_claims": [c["text"] for c in clean],
            "source_papers": source_papers,
            "type": "open_question",
        }

    def _generate_contradiction_gap(self, contradictions):
        lines = []
        all_claims = []
        for c1, c2, conf in contradictions:
            lines.append(
                f'- "{c1["text"]}" (from: {c1["paper_title"]})\n'
                f'  vs. "{c2["text"]}" (from: {c2["paper_title"]})\n'
                f'  [confidence: {conf:.2f}]'
            )
            all_claims.extend([c1, c2])

        prompt = CONTRADICTION_GAP_PROMPT.format(contradictions_text="\n".join(lines))
        gap_text = self._call_ollama(prompt)

        source_papers = list({c["paper_title"] for c in all_claims})

        return {
            "gap": gap_text,
            "supporting_claims": [c["text"] for c in all_claims],
            "source_papers": source_papers,
            "type": "contradiction",
        }

    def _call_ollama(self, prompt):
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 300}
        })

        if response.status_code != 200:
            return "Error: Could not generate gap."

        text = response.json().get("response", "").strip()

        # Clean up: remove "Research Gap:" prefix if present
        if text.lower().startswith("research gap:"):
            text = text[len("research gap:"):].strip()

        return f"Research Gap: {text}" if text else "Research Gap: Unable to synthesize gap from provided claims."
