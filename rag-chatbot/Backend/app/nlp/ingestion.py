# Backend/app/nlp/ingestion.py

import pandas as pd
import json
from typing import List, Dict


def create_qa_chunks(csv_path: str) -> List[Dict]:
    """
    Read cleaned CSV and create QA pair chunks for vector database.
    
    Args:
        csv_path: Path to cleaned_data.csv
    
    Returns:
        List of chunk dictionaries with text and metadata
    """
    df = pd.read_csv(csv_path)
    chunks = []

    for idx, row in df.iterrows():
        try:
            qa = json.loads(row['qa'])
            for i, pair in enumerate(qa['knowledge']):
                question = pair.get('customer_summary_question', '').strip()
                answer = pair.get('agent_summary_solution', '').strip()

                if not question or not answer:
                    continue

                chunk = {
                    "chunk_id": f"qa_{idx}_{i}",
                    "text": f"Question: {question}\nAnswer: {answer}",
                    "question": question,
                    "answer": answer,
                    "metadata": {
                        "source_row": idx,
                        "chunk_type": "qa_pair",
                        "issue_area": row['issue_area'],
                        "issue_category": row['issue_category'],
                        "product_category": row['product_category'],
                        "customer_sentiment": row['customer_sentiment'],
                        "issue_complexity": row['issue_complexity']
                    }
                }
                chunks.append(chunk)
        except Exception:
            continue

    return chunks


if __name__ == "__main__":
    chunks = create_qa_chunks("../../data/processed/cleaned_data.csv")
    print(f"Created {len(chunks)} QA chunks")