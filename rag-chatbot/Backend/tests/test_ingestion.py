import pytest
import os
import json
import pandas as pd
from app.nlp.ingestion import create_qa_chunks


@pytest.fixture
def sample_csv_path(tmp_path):
    """Create a temporary CSV file with sample data matching the exact schema."""
    # The ingestion script expects a column 'qa' with a JSON structure containing 'knowledge'
    qa_row_1 = {
        "knowledge": [
            {
                "customer_summary_question": "How to log in?",
                "agent_summary_solution": "Enter the code sent to your email."
            },
            {
                "customer_summary_question": "Why request code?",
                "agent_summary_solution": "For verification purposes."
            }
        ]
    }
    
    qa_row_2 = {
        "knowledge": [
            {
                "customer_summary_question": "Can I get a refund?",
                "agent_summary_solution": "Yes, within 24 hours."
            }
        ]
    }

    data = {
        "issue_area": ["Login and Account", "Cancellations and returns"],
        "issue_category": ["Mobile Verification", "Refund Process"],
        "product_category": ["App", "Billing"],
        "customer_sentiment": ["neutral", "frustrated"],
        "issue_complexity": ["low", "medium"],
        "qa": [json.dumps(qa_row_1), json.dumps(qa_row_2)]
    }
    
    df = pd.DataFrame(data)
    csv_file = tmp_path / "test_data.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)


def test_create_qa_chunks(sample_csv_path):
    """Test that create_qa_chunks correctly parses the CSV and formats QA pair chunks."""
    chunks = create_qa_chunks(sample_csv_path)
    
    # Row 1 has 2 QA pairs, Row 2 has 1 QA pair.
    # Total expected chunks = 3.
    assert len(chunks) == 3
    
    # Verify the structure of the first chunk
    first_chunk = chunks[0]
    assert first_chunk["chunk_id"] == "qa_0_0"
    assert "text" in first_chunk
    assert first_chunk["question"] == "How to log in?"
    assert first_chunk["answer"] == "Enter the code sent to your email."
    
    # Verify metadata fields are preserved correctly
    assert first_chunk["metadata"]["source_row"] == 0
    assert first_chunk["metadata"]["issue_area"] == "Login and Account"
    assert first_chunk["metadata"]["issue_category"] == "Mobile Verification"
    assert first_chunk["metadata"]["product_category"] == "App"
    assert first_chunk["metadata"]["customer_sentiment"] == "neutral"
    assert first_chunk["metadata"]["issue_complexity"] == "low"
    
    # Verify the third chunk (from the second row)
    third_chunk = chunks[2]
    assert third_chunk["chunk_id"] == "qa_1_0"
    assert third_chunk["question"] == "Can I get a refund?"
    assert third_chunk["metadata"]["source_row"] == 1
    assert third_chunk["metadata"]["issue_area"] == "Cancellations and returns"
