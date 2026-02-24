from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class Embedder:
    """Handles text embedding using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.model.encode(text).tolist()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings."""
        return self.model.encode(texts, show_progress_bar=True)