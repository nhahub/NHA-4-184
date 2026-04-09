from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class Embedder:
    """Handles text embedding using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        start_time = time.time()
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Embedding model loaded successfully: model={model_name}, duration_ms={duration_ms}")
        
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        logger.debug(f"Embedding single text: text_len={len(text)}, model={self.model_name}")
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Failed to embed text: {str(e)}, text_len={len(text)}", exc_info=True)
            raise

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings."""
        logger.info(f"Batch embedding started: batch_size={len(texts)}")
        start_time = time.time()
        try:
            result = self.model.encode(texts, show_progress_bar=True)
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(f"Batch embedding completed: batch_size={len(texts)}, duration_ms={duration_ms}")
            return result
        except Exception as e:
            logger.error(f"Failed to embed texts: {str(e)}, batch_size={len(texts)}", exc_info=True)
            raise