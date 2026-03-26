CLAIM_EXTRACTION_PROMPT = """
Extract clear, complete atomic scientific claims from the sentence.

Rules:
- Each claim must be a full sentence
- Include the subject (e.g., AR-RAG, the model, the method)
- Do NOT return fragments
- Do NOT copy instructions

Example:
Sentence: The model improves accuracy and reduces latency.
Claims:
- The model improves accuracy.
- The model reduces latency.

Now extract claims:

Sentence: "{sentence}"

Claims:
"""