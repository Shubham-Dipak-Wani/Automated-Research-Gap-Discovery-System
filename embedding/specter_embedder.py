from sentence_transformers import SentenceTransformer
from config.settings import SPECTER_MODEL


class SpecterEmbedder:
    def __init__(self):
        print(f"Loading {SPECTER_MODEL}...")
        self.model = SentenceTransformer(SPECTER_MODEL)

    def encode(self, claims):
        texts = [c["text"] if isinstance(c, dict) else c for c in claims]
        return self.model.encode(texts, batch_size=8, show_progress_bar=True)
