import anthropic
from config.settings import (
    CLAUDE_GAP_MODEL,
    MIN_CLAIM_LENGTH_FOR_GAP,
    MAX_CLAIMS_PER_GAP,
)
from gap_generation.prompts import GAP_GENERATION_PROMPT, CONTRADICTION_GAP_PROMPT

# System prompt placed in a cache_control block — stable across all gap generation calls.
GAP_SYSTEM_PROMPT = (
    "You are a research gap analyst specializing in identifying underexplored problems "
    "and open questions in scientific literature. You synthesize evidence from multiple "
    "papers to articulate precise, actionable research gaps grounded in the provided claims. "
    "Your output is a single research gap statement beginning with 'Research Gap:' followed "
    "by a clear, specific description. Avoid generic statements like 'more research is needed'. "
    "Reference source papers by name and explain what the claims show versus what is missing."
)


class GapGenerator:
    def __init__(self):
        self.client = anthropic.Anthropic()
        print(f"Using Claude ({CLAUDE_GAP_MODEL}) with adaptive thinking for gap generation")

    def generate_gap(self, cluster_claims: list, contradictions: list = None) -> dict:
        if contradictions:
            return self._generate_contradiction_gap(contradictions)
        return self._generate_cluster_gap(cluster_claims)

    def _generate_cluster_gap(self, cluster_claims: list) -> dict:
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
        gap_text = self._call_claude(prompt)
        source_papers = list({c["paper_title"] for c in clean})

        return {
            "gap": gap_text,
            "supporting_claims": [c["text"] for c in clean],
            "source_papers": source_papers,
            "type": "open_question",
        }

    def _generate_contradiction_gap(self, contradictions: list) -> dict:
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
        gap_text = self._call_claude(prompt)
        source_papers = list({c["paper_title"] for c in all_claims})

        return {
            "gap": gap_text,
            "supporting_claims": [c["text"] for c in all_claims],
            "source_papers": source_papers,
            "type": "contradiction",
        }

    def _call_claude(self, user_prompt: str) -> str:
        """Call Claude Opus 4.7 with adaptive thinking and cached system prompt."""
        try:
            response = self.client.messages.create(
                model=CLAUDE_GAP_MODEL,
                max_tokens=1024,
                thinking={"type": "adaptive"},
                system=[{
                    "type": "text",
                    "text": GAP_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APIError as e:
            print(f"Claude API error during gap generation: {e}")
            return "Research Gap: Unable to synthesize gap due to API error."

        # Extract the text block — response may also contain thinking blocks
        text = next(
            (b.text for b in response.content if b.type == "text"),
            "",
        ).strip()

        if text.lower().startswith("research gap:"):
            return text
        return f"Research Gap: {text}" if text else "Research Gap: Unable to synthesize gap from provided claims."
