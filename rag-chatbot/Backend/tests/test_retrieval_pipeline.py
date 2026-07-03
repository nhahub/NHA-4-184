import pytest
from app.nlp.retrieval import Retriever


@pytest.fixture
def populated_retriever(tmp_path):
    """Create a temporary Retriever instance populated with mock Q&A chunks."""
    db_dir = tmp_path / "test_retrieval_db"
    retriever = Retriever(db_path=str(db_dir), collection_name="test_retrieval_collection")
    
    # Seed data
    ids = ["qa_0_0", "qa_0_1"]
    documents = [
        "Question: How do I return a product?\nAnswer: You can return it within 30 days.",
        "Question: What is the delivery time?\nAnswer: Standard delivery takes 3-5 days."
    ]
    
    # Generate real embeddings using the retriever's embedder
    embeddings = retriever.embedder.embed_texts(documents).tolist()
    metadatas = [
        {"issue_area": "Returns", "source_row": 0},
        {"issue_area": "Delivery", "source_row": 1}
    ]
    
    retriever.vector_db.add_chunks(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    return retriever


@pytest.fixture
def empty_retriever(tmp_path):
    """Create a temporary empty Retriever instance."""
    db_dir = tmp_path / "test_empty_retriever_db"
    return Retriever(db_path=str(db_dir), collection_name="test_empty_collection")


def test_retriever_retrieve(populated_retriever):
    """Test standard retrieve function returns formatted list of results."""
    results = populated_retriever.retrieve("return policy", n_results=1)
    
    assert len(results) == 1
    assert results[0]["chunk_id"] == "qa_0_0"
    assert "return it within 30 days" in results[0]["text"]
    assert "distance" in results[0]
    assert results[0]["metadata"]["issue_area"] == "Returns"


def test_retriever_get_best_answer(populated_retriever):
    """Test get_best_answer returns single best document match dict."""
    result = populated_retriever.get_best_answer("delivery time")
    
    assert result["chunk_id"] == "qa_0_1"
    assert "Standard delivery" in result["text"]
    assert "distance" in result


def test_retriever_get_answer_with_context(populated_retriever):
    """Test get_answer_with_context parses the response keys and formats answer."""
    result = populated_retriever.get_answer_with_context("How do I return my order?", n_results=2)
    
    assert result["best_answer"] == "You can return it within 30 days."
    assert result["matched_question"] == "How do I return a product?"
    assert result["category"] == "Returns"
    assert isinstance(result["confidence"], float)
    assert len(result["context"]) == 1
    assert "Standard delivery takes 3-5 days." in result["context"][0]
    assert len(result["all_results"]) == 2


def test_retriever_get_answer_empty(empty_retriever):
    """Test retrieval pipeline fallback when collection is empty."""
    result = empty_retriever.get_answer_with_context("some query", n_results=3)
    
    assert result["best_answer"] == "Sorry, I couldn't find an answer."
    assert result["confidence"] == 0.0
    assert result["matched_question"] == ""
    assert len(result["context"]) == 0
    assert len(result["all_results"]) == 0
