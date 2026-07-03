import pytest
import io
import base64
from unittest.mock import patch, MagicMock
from app.db.models import Conversation, ChatMessage


@pytest.fixture(autouse=True)
def mock_mlflow_tracker():
    """Mock the MLflow experiment tracker globally for chat endpoint tests to prevent file writing."""
    with patch("app.api.chat.tracker") as mock_track:
        yield mock_track


def test_ask_question_empty_query(client, auth_headers):
    """Test ask endpoint returns 400 when question is empty."""
    payload = {
        "question": "   ",
        "conversation_id": None,
        "n_results": 3
    }
    response = client.post("/chat/ask", json=payload, headers=auth_headers)
    assert response.status_code == 400
    assert "Question cannot be empty" in response.json()["detail"]


def test_ask_question_conversation_not_found(client, auth_headers):
    """Test ask endpoint returns 404 when sending an invalid conversation ID."""
    payload = {
        "question": "Hello",
        "conversation_id": 99999,  # Does not exist
        "n_results": 3
    }
    response = client.post("/chat/ask", json=payload, headers=auth_headers)
    assert response.status_code == 404
    assert "Conversation not found" in response.json()["detail"]


@patch("app.api.chat.llm_router.classify")
@patch("app.api.chat.generator.direct_answer")
def test_ask_question_casual_path(mock_direct_answer, mock_classify, client, auth_headers):
    """Test casual conversation routes to direct answering and skips RAG search."""
    mock_classify.return_value = {"intent": "CASUAL", "duration_ms": 10.0}
    mock_direct_answer.return_value = "Hello! I am BrownBox assistant. How can I help you today?"

    payload = {
        "question": "Hi chatbot",
        "conversation_id": None,
        "n_results": 3
    }
    
    response = client.post("/chat/ask", json=payload, headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "casual"
    assert data["pipeline"] == "casual"
    assert data["answer"] == "Hello! I am BrownBox assistant. How can I help you today?"
    assert data["confidence"] == 1.0
    assert len(data["sources"]) == 0
    assert "conversation_id" in data


@patch("app.api.chat.llm_router.classify")
@patch("app.api.chat.retriever.get_answer_with_context")
@patch("app.api.chat.generator.generate_answer")
def test_ask_question_support_path_high_confidence(
    mock_generate_answer, mock_get_context, mock_classify, client, auth_headers
):
    """Test support path routes to RAG retrieval and returns direct LLM output on high confidence."""
    mock_classify.return_value = {"intent": "SUPPORT", "duration_ms": 10.0}
    
    # Mock retrieval return
    mock_get_context.return_value = {
        "best_answer": "You can return it within 30 days.",
        "confidence": 0.8,  # > 0.5 (HIGH_CONFIDENCE)
        "matched_question": "How do I return a product?",
        "category": "Returns",
        "context": ["Standard delivery takes 3-5 days."],
        "all_results": [
            {
                "chunk_id": "qa_0_0",
                "text": "Question: How do I return a product?\nAnswer: You can return it within 30 days.",
                "distance": 0.2,
                "metadata": {"issue_area": "Returns"}
            }
        ]
    }
    
    mock_generate_answer.return_value = "According to our policy, you can return your item within 30 days."

    payload = {
        "question": "How to return my order?",
        "conversation_id": None,
        "n_results": 3
    }
    
    response = client.post("/chat/ask", json=payload, headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "support"
    assert data["pipeline"] == "high_confidence"
    assert data["answer"] == "According to our policy, you can return your item within 30 days."
    assert data["confidence"] == 0.8
    assert len(data["sources"]) == 1
    assert data["sources"][0]["chunk_id"] == "qa_0_0"


@patch("app.api.chat.llm_router.classify")
@patch("app.api.chat.retriever.get_answer_with_context")
@patch("app.api.chat.generator.generate_answer")
@patch("app.api.chat.generator.verify_answer")
def test_ask_question_support_path_low_confidence_rejected(
    mock_verify_answer, mock_generate_answer, mock_get_context, mock_classify, client, auth_headers
):
    """Test low confidence support question triggers quality check and defaults to fallback on reject."""
    mock_classify.return_value = {"intent": "SUPPORT", "duration_ms": 10.0}
    
    # Mock retrieval return with very low confidence
    mock_get_context.return_value = {
        "best_answer": "Some unrelated answer",
        "confidence": 0.1,  # < 0.2 (LOW_CONFIDENCE)
        "matched_question": "Where is my box?",
        "category": "Shipping",
        "context": [],
        "all_results": [
            {
                "chunk_id": "qa_0_1",
                "text": "Question: Where is my box?\nAnswer: We do not know.",
                "distance": 0.9,
                "metadata": {"issue_area": "Shipping"}
            }
        ]
    }
    
    mock_generate_answer.return_value = "I think it is lost."
    
    # Verification returns invalid
    mock_verify_answer.return_value = {"is_valid": False, "result": "INVALID", "duration_ms": 5.0}

    payload = {
        "question": "Where is my item?",
        "conversation_id": None,
        "n_results": 3
    }
    
    response = client.post("/chat/ask", json=payload, headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["pipeline"] == "low_confidence_rejected"
    assert data["verified"] is False
    # Verifies the fallback message override
    assert "I'm sorry, I don't have enough information to answer that question accurately." in data["answer"]


def test_get_history_conversations(client, auth_headers, test_user, db_session):
    """Test retrieving all conversations for the authenticated user."""
    # Seed a conversation in the DB
    conv = Conversation(user_id=test_user.id, title="Test Topic")
    db_session.add(conv)
    db_session.commit()

    response = client.get("/chat/history", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["title"] == "Test Topic"
    assert data[0]["id"] == conv.id


def test_get_conversation_detail(client, auth_headers, test_user, db_session):
    """Test retrieving conversation details with all associated messages."""
    # Seed conversation and messages
    conv = Conversation(user_id=test_user.id, title="QA Thread")
    db_session.add(conv)
    db_session.commit()
    db_session.refresh(conv)
    
    msg = ChatMessage(
        conversation_id=conv.id,
        user_query="Hello",
        retrieved_context="None",
        llm_response="Hi",
        response_time=0.5
    )
    db_session.add(msg)
    db_session.commit()

    response = client.get(f"/chat/history/{conv.id}", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == conv.id
    assert len(data["messages"]) == 1
    assert data["messages"][0]["user_query"] == "Hello"
    assert data["messages"][0]["llm_response"] == "Hi"


@patch("app.api.chat.transcriber.transcribe")
@patch("app.api.chat.llm_router.classify")
@patch("app.api.chat.generator.direct_answer")
@patch("app.api.chat.tts.synthesize")
def test_voice_query_success(
    mock_synthesize, mock_direct_answer, mock_classify, mock_transcribe, client, auth_headers
):
    """Test uploading an audio file transcribes it, runs the pipeline, and generates base64 speech reply."""
    # Mock transcription output
    mock_transcribe.return_value = {"text": "Hello chatbot", "language": "en"}
    
    # Mock casual routing path
    mock_classify.return_value = {"intent": "CASUAL", "duration_ms": 10.0}
    mock_direct_answer.return_value = "Hello user!"
    
    # Mock TTS audio bytes conversion
    mock_synthesize.return_value = b"fake_mp3_data"
    
    # Force TTS check to think it's active by providing a dummy api_key
    with patch("app.api.chat.tts.api_key", "mock_api_key"):
        # Create a dummy audio file payload
        audio_file = ("test.ogg", io.BytesIO(b"ogg_audio_file_content"), "audio/ogg")
        data = {
            "conversation_id": 0,
            "n_results": 3
        }
        
        response = client.post(
            "/chat/voice",
            files={"audio": audio_file},
            data=data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["transcription"] == "Hello chatbot"
        assert data["answer"] == "Hello user!"
        assert data["intent"] == "casual"
        
        # Audio response should be base64-encoded string of the simulated MP3 output
        expected_audio_base64 = base64.b64encode(b"fake_mp3_data").decode("utf-8")
        assert data["audio_base64"] == expected_audio_base64