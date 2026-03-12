from typing import List, Dict, Optional
from app.nlp.embedder import Embedder
from app.nlp.vector_db import VectorDB


class Retriever:
    """Retrieves the most relevant QA chunks for a given user question."""

    def __init__(self, db_path: str = "../data/vector_db", collection_name: str = "qa_chunks"):
        self.embedder = Embedder()
        self.vector_db = VectorDB(db_path=db_path, collection_name=collection_name)

    def retrieve(self, question: str, n_results: int = 3,
                 filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search for the most relevant chunks matching the user's question.

        Args:
            question: The user's question
            n_results: Number of results to return
            filter_metadata: Optional metadata filter (e.g. {"issue_area": "Order"})

        Returns:
            List of results with text, metadata, and similarity score
        """
        # Build query kwargs
        query_kwargs = {
            "query_texts": [question],
            "n_results": n_results
        }

        # Add metadata filter if provided
        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        results = self.vector_db.collection.query(**query_kwargs)

        # Format results into clean list
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            })

        return formatted

    def get_best_answer(self, question: str) -> Dict:
        """Get the single best matching answer for a question."""
        results = self.retrieve(question, n_results=1)
        if results:
            return results[0]
        return {"text": "Sorry, I couldn't find an answer to your question.", "distance": 999}

    def get_answer_with_context(self, question: str, n_results: int = 3) -> Dict:
        """
        Get the best answer along with supporting context from other matches.

        Returns:
            Dict with 'best_answer', 'context', and 'all_results'
        """
        results = self.retrieve(question, n_results=n_results)

        if not results:
            return {
                "best_answer": "Sorry, I couldn't find an answer.",
                "confidence": 0.0,
                "matched_question": "",
                "category": "",
                "context": [],
                "all_results": []
            }


        # Best match
        best = results[0]

        # Extract answer part from the text (format is "Question: ...\nAnswer: ...")
        text = best["text"]
        if "Answer:" in text:
            answer = text.split("Answer:")[-1].strip()
        else:
            answer = text

        return {
            "best_answer": answer,
            "confidence": round(1 - best["distance"], 4),  # Convert distance to confidence
            "matched_question": text.split("\nAnswer:")[0].replace("Question: ", "") if "Answer:" in text else "",
            "category": best["metadata"].get("issue_area", ""),
            "context": [r["text"] for r in results[1:]],  # Other supporting chunks
            "all_results": results
        }
