from config.settings import HDBSCAN_EPSILON, HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES
from clustering.utils import group_indices


class ClaimClusterer:
    """HDBSCAN over claim embeddings: no need to pick k, and noise is labeled -1."""

    def __init__(self):
        import hdbscan

        print("Initializing HDBSCAN...")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            cluster_selection_epsilon=HDBSCAN_EPSILON,
        )

    def cluster(self, embeddings):
        return self.clusterer.fit_predict(embeddings)

    def group_clusters(self, claims, labels):
        """{cluster_id: [claims]}, noise dropped."""
        return {cid: [claims[i] for i in idxs] for cid, idxs in group_indices(labels).items()}
