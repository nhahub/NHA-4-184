import pytest
from unittest.mock import MagicMock
from app.nlp.router import LLMRouter


@pytest.fixture
def router(mock_groq_client):
    """Provide an LLMRouter instance with a mocked Groq client."""
    return LLMRouter()


def test_router_casual_classification(router, mock_groq_client):
    """Test router classifies greeting query as CASUAL."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "CASUAL"
    mock_groq_client["router"].chat.completions.create.return_value = mock_response

    result = router.classify("hello there")
    
    assert result["intent"] == "CASUAL"
    mock_groq_client["router"].chat.completions.create.assert_called_once()


def test_router_support_classification(router, mock_groq_client):
    """Test router classifies product query as SUPPORT."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "SUPPORT"
    mock_groq_client["router"].chat.completions.create.return_value = mock_response

    result = router.classify("how to refund my item?")
    
    assert result["intent"] == "SUPPORT"


def test_router_invalid_intent_fallback(router, mock_groq_client):
    """Test that the router defaults to SUPPORT if the LLM returns an invalid category."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "SOMETHING_ELSE"
    mock_groq_client["router"].chat.completions.create.return_value = mock_response

    result = router.classify("hello")
    
    assert result["intent"] == "SUPPORT"  # Defaults to SUPPORT for safety


def test_router_api_failure_fallback(router, mock_groq_client):
    """Test that the router defaults to SUPPORT if the Groq API call fails."""
    # Simulate API exception
    mock_groq_client["router"].chat.completions.create.side_effect = Exception("API Outage")

    result = router.classify("hello")
    
    assert result["intent"] == "SUPPORT"  # Defaults to SUPPORT on error
    assert result["duration_ms"] == 0
