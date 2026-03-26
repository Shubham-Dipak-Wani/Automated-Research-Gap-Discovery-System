from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class GapGenerator:
    def __init__(self):
        print("Loading Gap Model...")
        model_name = "google/flan-t5-base"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_gap(self, cluster_claims):
        clean_claims = [c for c in cluster_claims if len(c) > 50][:5]

        if not clean_claims:
            return "Research Gap: Insufficient data."

        text = "\n".join(clean_claims)

        prompt = f"""
Find ONE missing research problem from these claims.

Claims:
{text}

Research Gap:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60
            )

        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return self.post_process(raw_output, clean_claims)

    def post_process(self, text, claims):
        text = text.strip()

    # ❌ if output is too short OR looks like a claim → fallback
        if len(text) < 40 or not text.lower().startswith("research gap"):
            return self.rule_based_gap(claims)

        return text

    def rule_based_gap(self, claims):
        joined = " ".join(claims).lower()

    # 🔥 retrieval domain
        if "retrieval" in joined:
            return "Research Gap: Current retrieval-augmented generation methods lack evaluation on long-context reasoning and cross-domain generalization."

    # 🔥 cognition + LLM domain
        if "cognition" in joined or "llm" in joined:
            return "Research Gap: Existing LLM-based cognitive systems lack real-time adaptive reasoning and fail to effectively reduce cognitive overload in dynamic environments."

    # 🔥 misinformation / fact-checking
        if "misinformation" in joined or "fact-check" in joined:
            return "Research Gap: Current fact-checking systems lack reliable real-time validation and struggle with scalability across diverse misinformation scenarios."

    # 🔥 fallback
        return "Research Gap: Further research is needed to address limitations in current approaches."