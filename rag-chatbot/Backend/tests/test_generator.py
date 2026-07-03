import pytest
from unittest.mock import MagicMock
from app.nlp.generator import Generator


@pytest.fixture
def generator(mock_groq_client):
    """Provide a Generator instance with a mocked Groq client."""
    return Generator()


def test_generate_answer(generator, mock_groq_client):
    """Test generating an answer using retrieved support context chunks."""
    # Set up mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "To log in, please enter the OTP sent to your email."
    mock_groq_client["generator"].chat.completions.create.return_value = mock_response

    # Inputs
    question = "How do I log in?"
    chunks = [
        {"text": "Question: How do I log in?\nAnswer: Enter the code sent to your email."}
    ]
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    # Call method
    answer = generator.generate_answer(question, chunks, history)

    # Asserts
    assert answer == "To log in, please enter the OTP sent to your email."
    mock_groq_client["generator"].chat.completions.create.assert_called_once()
    
    # Verify call arguments structure
    called_args = mock_groq_client["generator"].chat.completions.create.call_args[1]
    assert called_args["model"] == "llama-3.3-70b-versatile"
    assert called_args["temperature"] == 0.3
    assert len(called_args["messages"]) == 4  # system + 2 history + current question


def test_rewrite_query(generator, mock_groq_client):
    """Test query rewriting on a vague follow-up question."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "How do I return my package?"
    mock_groq_client["generator"].chat.completions.create.return_value = mock_response

    question = "How do I return it?"
    history = [{"role": "user", "content": "I bought a box"}, {"role": "assistant", "content": "Awesome"}]

    rewritten = generator.rewrite_query(question, history)

    assert rewritten == "How do I return my package?"


def test_direct_answer(generator, mock_groq_client):
    """Test direct answering for casual conversation without context lookup."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello! How can I help you today?"
    mock_groq_client["generator"].chat.completions.create.return_value = mock_response

    answer = generator.direct_answer("Hello")
    assert answer == "Hello! How can I help you today?"


def test_verify_answer_valid(generator, mock_groq_client):
    """Test verify_answer returns VALID if LLM output matches context."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "VALID"
    mock_groq_client["generator"].chat.completions.create.return_value = mock_response

    verification = generator.verify_answer(
        question="How to log in?",
        answer="Enter your OTP.",
        context="To log in enter the OTP."
    )

    assert verification["is_valid"] is True
    assert verification["result"] == "VALID"


def test_verify_answer_invalid(generator, mock_groq_client):
    """Test verify_answer returns INVALID if LLM output does not match context."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "INVALID"
    mock_groq_client["generator"].chat.completions.create.return_value = mock_response

    verification = generator.verify_answer(
        question="How to log in?",
        answer="Send a check.",
        context="To log in enter the OTP."
    )

    assert verification["is_valid"] is False
    assert verification["result"] == "INVALID"
