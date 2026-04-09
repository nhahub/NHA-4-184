import os
from groq import Groq
from typing import List, Dict
import logging
import time

logger = logging.getLogger(__name__)


class Generator:
    """Generates natural language answers using Groq LLM."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY not found in environment variables")
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.model = "llama-3.3-70b-versatile"
        self.client = Groq(api_key=api_key)
        logger.info(f"Groq LLM client initialized successfully: model={self.model}")

    def generate_answer(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """
        Generate a natural answer using the retrieved chunks as context.
        """
        # Build context from retrieved chunks
        context = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            context += f"\n--- Source {i} ---\n{chunk['text']}\n"

        context_len = len(context)
        logger.info(f"LLM generation started: question_len={len(question)}, chunks_count={len(retrieved_chunks)}, context_len={context_len}, temperature=0.3, max_tokens=500")
        start_time = time.time()

        # Create the prompt
        prompt = f"""You are a helpful customer support assistant for BrownBox (an e-commerce company).
Answer the customer's question based ONLY on the provided context below.
If the context doesn't contain enough information to answer, say "I'm sorry, I don't have enough information to answer that question."

Keep your answer:
- Clear and concise
- Friendly and professional
- Based only on the provided context

Context:
{context}

Customer Question: {question}

Answer:"""

        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )

            answer = response.choices[0].message.content
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(f"LLM generation completed: answer_len={len(answer)}, duration_ms={duration_ms}")
            return answer

        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.error(f"Groq API call failed: {str(e)}, duration_ms={duration_ms}", exc_info=True)
            raise
