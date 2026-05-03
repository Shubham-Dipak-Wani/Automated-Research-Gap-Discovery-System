GAP_GENERATION_PROMPT = """You are a research gap analyst. Given a cluster of related scientific claims from multiple papers, identify ONE specific research gap — something that is missing, underexplored, or contradicted.

Rules:
- The gap must be specific and actionable (not generic like "more research is needed")
- Ground the gap in the provided claims — explain what the claims show and what is missing
- Reference the source papers by name
- Format: Start with "Research Gap:" followed by the gap statement

Claims:
{claims_text}

Research Gap:
"""

CONTRADICTION_GAP_PROMPT = """You are a research gap analyst. The following pairs of claims from different papers contradict each other. Identify the specific research gap this contradiction reveals.

Rules:
- Explain the contradiction clearly
- Identify what experiment, study, or analysis would resolve the disagreement
- Reference the source papers by name
- Format: Start with "Research Gap:" followed by the gap statement

Contradictions:
{contradictions_text}

Research Gap:
"""
