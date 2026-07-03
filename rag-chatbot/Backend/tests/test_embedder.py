import pytest
import numpy as np
from app.nlp.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    """Load the embedder once for the entire module to save time."""
    return Embedder(model_name="all-MiniLM-L6-v2")


def test_embedder_initialization(embedder):
    """Test that the embedder initializes with correct model name."""
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.model is not None


def test_embed_single_text(embedder):
    """Test embedding a single string returns correct dimension list of floats."""
    text = "Hello world"
    embedding = embedder.embed_text(text)
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(val, float) for val in embedding)


def test_embed_texts_batch(embedder):
    """Test batch embedding multiple texts returns a numpy array with correct dimensions."""
    texts = ["How do I login?", "Where is my refund?", "Contact customer support."]
    embeddings = embedder.embed_texts(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 384)


def test_embed_empty_string(embedder):
    """Test embedding an empty string does not crash and returns the correct dimensions."""
    embedding = embedder.embed_text("")
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384
