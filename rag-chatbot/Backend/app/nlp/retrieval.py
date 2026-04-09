from typing import List, Dict, Optional
from app.nlp.embedder import Embedder
from app.nlp.vector_db import VectorDB
import logging
import time

logger = logging.getLogger(__name__)


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
        logger.debug(f"Vector search started: question_len={len(question)}, n_results={n_results}, filter_metadata={filter_metadata}")
        start_time = time.time()

        # Build query kwargs
        query_kwargs = {
            "query_texts": [question],
            "n_results": n_results
        }

        # Add metadata filter if provided
        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        try:
            results = self.vector_db.collection.query(**query_kwargs)
        except Exception as e:
            logger.error(f"ChromaDB query failed: {str(e)}, question_len={len(question)}, n_results={n_results}", exc_info=True)
            raise

        # Format results into clean list
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            })

        duration_ms = round((time.time() - start_time) * 1000, 2)
        top_distance = formatted[0]["distance"] if formatted else "N/A"
        logger.info(f"Vector search completed: results_found={len(formatted)}, top_distance={top_distance}, duration_ms={duration_ms}")

        return formatted

    def get_best_answer(self, question: str) -> Dict:
        """Get the single best matching answer for a question."""
        try:
            results = self.retrieve(question, n_results=1)
            if results:
                logger.debug(f"Best answer found: distance={results[0]['distance']}")
                return results[0]
            logger.warning(f"No best answer found for question_len={len(question)}")
            return {"text": "Sorry, I couldn't find an answer to your question.", "distance": 999}
        except Exception as e:
            logger.error(f"get_best_answer failed: {str(e)}, question_len={len(question)}", exc_info=True)
            raise

    def get_answer_with_context(self, question: str, n_results: int = 3) -> Dict:   
        logger.debug(f"Answer extraction started: question_len={len(question)}, n_results={n_results}")
        start_time = time.time()
    
        try:
            results = self.retrieve(question, n_results=n_results)

            if not results:
                logger.info(f"Answer extraction completed: no results found")
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

            # Extract answer part from the text
            text = best["text"]
            try:
                if "Answer:" in text:
                    answer = text.split("Answer:")[-1].strip()
                else:
                    logger.warning(f"Answer parsing: malformed response format, chunk_id={best['chunk_id']}")
                    answer = text

                matched_question = text.split("\nAnswer:")[0].replace("Question: ", "") if "Answer:" in text else ""
                confidence = round(1 - best["distance"], 4)

                duration_ms = round((time.time() - start_time) * 1000, 2)
                logger.info(f"Answer extraction completed: matched_question_len={len(matched_question)}, confidence={confidence}, duration_ms={duration_ms}")

                return {
                    "best_answer": answer,
                    "confidence": confidence,
                    "matched_question": matched_question,
                    "category": best["metadata"].get("issue_area", ""),
                    "context": [r["text"] for r in results[1:]],
                    "all_results": results
                }
            except Exception as e:
                logger.error(f"Answer extraction parsing failed: {str(e)}, chunk_id={best.get('chunk_id', 'unknown')}", exc_info=True)
                raise
            
        except Exception as e:
            logger.error(f"get_answer_with_context failed: {str(e)}, question_len={len(question)}", exc_info=True)
            raise