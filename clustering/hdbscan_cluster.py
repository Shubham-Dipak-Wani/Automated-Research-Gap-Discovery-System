import hdbscan
from config.settings import HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, HDBSCAN_EPSILON


class ClaimClusterer:
    def __init__(self):
        print("Initializing HDBSCAN...")

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            cluster_selection_epsilon=HDBSCAN_EPSILON,
        )

    def cluster(self, embeddings):
        return self.clusterer.fit_predict(embeddings)

    def group_clusters(self, claims, labels):
        clusters = {}

        for claim, label in zip(claims, labels):
            if label == -1:
                continue

            clusters.setdefault(label, []).append(claim)

        return clusters
