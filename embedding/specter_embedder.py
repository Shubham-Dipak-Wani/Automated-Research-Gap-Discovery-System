from sentence_transformers import SentenceTransformer


class SpecterEmbedder:
    def __init__(self):
        print("Loading SPECTER...")
        self.model = SentenceTransformer("allenai/specter")

    def encode(self, claims):
        return self.model.encode(claims, batch_size=8, show_progress_bar=True)