import pytest
from app.nlp.vector_db import VectorDB


@pytest.fixture
def temp_vector_db(tmp_path):
    """Create a temporary ChromaDB instance in a temp directory."""
    db_dir = tmp_path / "test_chroma_db"
    return VectorDB(db_path=str(db_dir), collection_name="test_collection")


def test_vector_db_initialization(temp_vector_db):
    """Test that ChromaDB initializes and is empty initially."""
    assert temp_vector_db.collection_name == "test_collection"
    assert temp_vector_db.count() == 0


def test_add_chunks(temp_vector_db):
    """Test adding chunks to the vector database."""
    ids = ["doc_1", "doc_2"]
    documents = ["How do I return a product?", "What is the shipping cost?"]
    
    # Generate mock embeddings of size 384
    embeddings = [
        [0.1] * 384,
        [0.2] * 384
    ]
    metadatas = [
        {"source_row": 0, "category": "returns"},
        {"source_row": 1, "category": "shipping"}
    ]
    
    temp_vector_db.add_chunks(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    assert temp_vector_db.count() == 2


def test_search_chunks(temp_vector_db):
    """Test searching for chunks in the database."""
    # First, insert sample chunks
    ids = ["doc_1", "doc_2"]
    documents = ["How do I return a product?", "What is the shipping cost?"]
    
    # Generate mock embeddings of size 384
    embeddings = [
        [0.1] * 384,
        [0.2] * 384
    ]
    metadatas = [
        {"source_row": 0, "category": "returns"},
        {"source_row": 1, "category": "shipping"}
    ]
    temp_vector_db.add_chunks(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    # Run a search query (ChromaDB embeds this string into 384 dimensions now matching database)
    results = temp_vector_db.search("return a product", n_results=1)
    
    assert "ids" in results
    assert len(results["ids"][0]) == 1
    assert results["ids"][0][0] in ["doc_1", "doc_2"]
