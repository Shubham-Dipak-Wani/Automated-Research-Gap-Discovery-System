import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config.settings import LIMITATION_SEED_QUERIES, LIMITATION_SIMILARITY_THRESHOLD


class LimitationRetriever:
    """
    Retrieves limitation-adjacent claims using seed query similarity,
    then re-clusters them to identify recurring limitation themes.
    """

    def __init__(self, embedder, clusterer):
        self.embedder = embedder
        self.clusterer = clusterer

    def retrieve(self, all_claims, all_embeddings):
        """
        Find claims similar to limitation seed queries.
        Returns: (limitation_claims, limitation_clusters)
        """
        # Embed seed queries
        seed_embeddings = self.embedder.encode(LIMITATION_SEED_QUERIES)

        # Compute similarity of every claim against every seed query
        similarities = cosine_similarity(all_embeddings, seed_embeddings)

        # A claim is limitation-adjacent if it's similar to ANY seed query
        max_similarities = np.max(similarities, axis=1)

        # Select claims above threshold
        limitation_indices = np.where(max_similarities >= LIMITATION_SIMILARITY_THRESHOLD)[0]

        if len(limitation_indices) == 0:
            return [], {}

        limitation_claims = [all_claims[i] for i in limitation_indices]
        limitation_embeddings = all_embeddings[limitation_indices]

        # Re-cluster limitation claims
        if len(limitation_claims) < 2:
            return limitation_claims, {0: limitation_claims}

        labels = self.clusterer.cluster(limitation_embeddings)
        clusters = self.clusterer.group_clusters(limitation_claims, labels)

        # If all noise, put them in one group
        if not clusters:
            clusters = {0: limitation_claims}

        return limitation_claims, clusters
