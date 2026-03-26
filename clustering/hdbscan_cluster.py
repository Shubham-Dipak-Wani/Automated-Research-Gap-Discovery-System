import hdbscan


class ClaimClusterer:
    def __init__(self):
        print("Initializing HDBSCAN...")

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            cluster_selection_epsilon=0.8  # relaxed
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