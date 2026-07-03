import pytest
from unittest.mock import patch, MagicMock
from app.db.models import Conversation, ChatMessage, Feedback


@pytest.fixture(autouse=True)
def mock_mlflow_tracker():
    """Mock the MLflow experiment tracker globally for feedback tests."""
    with patch("app.api.feedback.tracker") as mock_track:
        yield mock_track


@pytest.fixture
def test_message(db_session, test_user) -> ChatMessage:
    """Helper fixture to seed a conversation and message in the test database."""
    conv = Conversation(user_id=test_user.id, title="Support Conversation")
    db_session.add(conv)
    db_session.commit()
    db_session.refresh(conv)

    msg = ChatMessage(
        conversation_id=conv.id,
        user_query="Can you help me log in?",
        retrieved_context="OTP Instructions",
        llm_response="Enter the code sent to your email.",
        response_time=0.45
    )
    db_session.add(msg)
    db_session.commit()
    db_session.refresh(msg)
    return msg


def test_submit_feedback_new_success(client, auth_headers, test_message, db_session):
    """Test submitting new feedback successfully."""
    payload = {
        "message_id": test_message.id,
        "rating": 1,
        "comment": "Very helpful!"
    }

    response = client.post("/feedback/", json=payload, headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["chat_message_id"] == test_message.id
    assert data["rating"] == 1
    assert data["comment"] == "Very helpful!"

    # Verify it exists in the test DB
    feedback = db_session.query(Feedback).filter(Feedback.chat_message_id == test_message.id).first()
    assert feedback is not None
    assert feedback.rating == 1


def test_submit_feedback_update_success(client, auth_headers, test_message, db_session):
    """Test updating existing feedback successfully."""
    # Seed existing feedback
    feedback = Feedback(
        chat_message_id=test_message.id,
        user_id=test_message.conversation.user_id,
        rating=1,
        comment="Great answer!"
    )
    db_session.add(feedback)
    db_session.commit()

    # Submit updated feedback
    payload = {
        "message_id": test_message.id,
        "rating": -1,
        "comment": "Actually, it did not work."
    }

    response = client.post("/feedback/", json=payload, headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["rating"] == -1
    assert data["comment"] == "Actually, it did not work."

    # Verify updated in DB
    db_session.refresh(feedback)
    assert feedback.rating == -1


def test_submit_feedback_message_not_found(client, auth_headers):
    """Test submitting feedback for a message that does not exist."""
    payload = {
        "message_id": 99999,  # Invalid
        "rating": 1,
        "comment": "Nice"
    }

    response = client.post("/feedback/", json=payload, headers=auth_headers)
    assert response.status_code == 404
    assert "Message not found" in response.json()["detail"]


def test_get_feedback_success(client, auth_headers, test_message, db_session):
    """Test retrieving existing feedback by message ID."""
    # Seed feedback
    feedback = Feedback(
        chat_message_id=test_message.id,
        user_id=test_message.conversation.user_id,
        rating=-1,
        comment="Confusing response"
    )
    db_session.add(feedback)
    db_session.commit()

    response = client.get(f"/feedback/{test_message.id}", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["chat_message_id"] == test_message.id
    assert data["rating"] == -1
    assert data["comment"] == "Confusing response"


def test_get_feedback_not_found(client, auth_headers, test_message):
    """Test retrieving feedback for a message that has no feedback."""
    response = client.get(f"/feedback/{test_message.id}", headers=auth_headers)
    assert response.status_code == 404
    assert "No feedback for this message" in response.json()["detail"]


def test_get_negative_feedback_admin(client, test_message, db_session, auth_headers):
    """Test admin endpoint retrieving only negative feedback."""
    # Seed negative feedback
    feedback_neg = Feedback(
        chat_message_id=test_message.id,
        user_id=test_message.conversation.user_id,
        rating=-1,
        comment="Wrong"
    )
    db_session.add(feedback_neg)
    db_session.commit()

    response = client.get("/feedback/admin/negative", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["message_id"] == test_message.id
    assert data[0]["rating"] == -1
    assert data[0]["comment"] == "Wrong"


@patch("app.nlp.vector_db.VectorDB")
@patch("app.nlp.embedder.Embedder")
def test_retrain_from_feedback_success(mock_embedder, mock_vector_db, client, test_message, db_session, auth_headers):
    """Test admin retraining triggers embedding generation and pushes corrected Q&A into ChromaDB."""
    # Seed negative feedback
    feedback = Feedback(
        chat_message_id=test_message.id,
        user_id=test_message.conversation.user_id,
        rating=-1,
        comment="Wrong info"
    )
    db_session.add(feedback)
    db_session.commit()

    # Mock Embedder & VectorDB
    mock_embed_instance = MagicMock()
    mock_embedder.return_value = mock_embed_instance
    mock_embed_instance.embed_text.return_value = [0.1] * 384

    mock_db_instance = MagicMock()
    mock_vector_db.return_value = mock_db_instance

    params = {
        "message_id": test_message.id,
        "correct_answer": "This is the correct corporate policy answer."
    }

    response = client.post("/feedback/admin/retrain", params=params, headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["correct_answer"] == "This is the correct corporate policy answer."
    
    # Assert vector db add was called
    mock_db_instance.add_chunks.assert_called_once()
    
    # Assert feedback is marked as retrained
    db_session.refresh(feedback)
    assert feedback.is_retrained is True
