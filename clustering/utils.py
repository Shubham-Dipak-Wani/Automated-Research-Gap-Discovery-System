"""Small, pure helpers for working with cluster labels."""


def group_indices(labels) -> dict:
    """{cluster_id: [claim indices]} in claim order, dropping noise (-1).

    Keeping indices (not just claims) means embeddings and claims can never
    drift out of sync, whatever cluster ids HDBSCAN happens to use.
    """
    groups: dict = {}
    for i, label in enumerate(labels):
        label = int(label)
        if label == -1:
            continue
        groups.setdefault(label, []).append(i)
    return groups


def rank_clusters(groups: dict, claims: list, limit: int | None = None) -> list:
    """Order clusters so the most interesting go to the (expensive) gap model first.

    A cluster whose claims come from several different papers says more about
    the field than one paper repeating itself, so rank by distinct papers,
    then by size. Clusters with fewer than 2 claims are skipped.
    """
    scored = []
    for cid, idxs in groups.items():
        if len(idxs) < 2:
            continue
        papers = {claims[i].get("paper_title", "") for i in idxs}
        scored.append((len(papers), len(idxs), cid))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    ranked = [cid for _, _, cid in scored]
    return ranked[:limit] if limit is not None else ranked
