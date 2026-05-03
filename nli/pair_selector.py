"""Choose which claim pairs are worth an NLI call, and collect contradictions.

Running NLI on every pair in a cluster is O(n^2). A cosine-similarity window
keeps only pairs that are about the same thing (>= COSINE_SIM_MIN) but are not
near-duplicates (<= COSINE_SIM_MAX), which is where real disagreements live.
"""
import numpy as np

from config.settings import COSINE_SIM_MAX, COSINE_SIM_MIN, NLI_CONFIDENCE_THRESHOLD


def candidate_pairs(embeddings, lo: float = COSINE_SIM_MIN, hi: float = COSINE_SIM_MAX) -> list:
    """(i, j, similarity) for i < j with lo <= cosine(i, j) <= hi."""
    emb = np.asarray(embeddings, dtype=float)
    if len(emb) < 2:
        return []
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb / norms
    sims = unit @ unit.T
    pairs = []
    for i in range(len(emb)):
        for j in range(i + 1, len(emb)):
            s = float(sims[i, j])
            if lo <= s <= hi:
                pairs.append((i, j, s))
    return pairs


def find_contradictions(groups: dict, claims: list, embeddings, nli,
                        threshold: float = NLI_CONFIDENCE_THRESHOLD) -> list:
    """Run NLI both ways on candidate pairs inside each cluster.

    groups: {cluster_id: [claim indices]}. Returns [(claim_a, claim_b, confidence)].
    NLI is directional, so a pair counts as a contradiction if either direction
    says so with confidence above the threshold (the higher score is kept).
    """
    emb = np.asarray(embeddings)
    found = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        for a, b, _ in candidate_pairs(emb[idxs]):
            ca, cb = claims[idxs[a]], claims[idxs[b]]
            best = 0.0
            for premise, hypothesis in ((ca, cb), (cb, ca)):
                label, conf = nli.predict(premise["text"], hypothesis["text"])
                if label == "contradiction":
                    best = max(best, conf)
            if best > threshold:
                found.append((ca, cb, best))
    return found
