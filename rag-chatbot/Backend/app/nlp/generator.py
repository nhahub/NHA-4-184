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

    def generate_answer(self, question: str, retrieved_chunks: List[Dict],
                        conversation_history: List[Dict] = None) -> str:
        """
        Generate a natural answer using the retrieved chunks as context.
        conversation_history: list of {"role": "user"/"assistant", "content": "..."}
        """
        # Build context from retrieved chunks
        context = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            context += f"\n--- Source {i} ---\n{chunk['text']}\n"

        context_len = len(context)
        history_count = len(conversation_history) if conversation_history else 0
        logger.info(f"LLM generation started: question_len={len(question)}, chunks_count={len(retrieved_chunks)}, context_len={context_len}, history_messages={history_count}, temperature=0.3, max_tokens=500")
        start_time = time.time()

        # System prompt with context
        system_prompt = f"""You are a helpful customer support assistant for BrownBox (an e-commerce company).
Answer the customer's question based on the provided context below.
If the context doesn't contain enough information, say "I'm sorry, I don't have enough information to answer that question."
Use the conversation history to understand follow-up questions and pronouns (like "it", "that", "this").

Keep your answer:
- Clear and concise
- Friendly and professional

Context:
{context}"""

        # Build messages array (the proper way for chat LLMs)
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current question
        messages.append({"role": "user", "content": question})

        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
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

    def rewrite_query(self, question: str, conversation_history: List[Dict] = None) -> str:
        """Rewrite a vague follow-up question into a clear standalone question."""
        if not conversation_history:
            return question

        messages = [
            {"role": "system", "content": (
                "You are a query rewriter. Rewrite the user's latest question "
                "into a clear, standalone question using the conversation history. "
                "Output ONLY the rewritten question, nothing else. "
                "If the question is already clear, return it as-is."
            )}
        ]

        for msg in conversation_history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": question})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=100
            )
            rewritten = response.choices[0].message.content.strip()
            logger.info(f"Query rewritten: original='{question}', rewritten='{rewritten}'")
            return rewritten
        except Exception:
            logger.warning(f"Query rewrite failed, using original question")
            return question
    
    def direct_answer(self, question: str, conversation_history: List[Dict] = None) -> str:
        """Answer casual questions directly without any RAG context."""
        logger.info(f"Direct answer started: question_len={len(question)}")
        start_time = time.time()

        messages = [
            {"role": "system", "content": (
                "You are a friendly customer support assistant for BrownBox (an e-commerce company). "
                "The user is making casual conversation (greeting, thanks, goodbye, etc). "
                "Respond naturally and briefly. Keep it warm and professional. "
                "If they greet you, greet them back and ask how you can help. "
                "Do NOT make up any product or order information."
            )}
        ]

        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": question})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=200
            )
            answer = response.choices[0].message.content
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(f"Direct answer completed: answer_len={len(answer)}, duration_ms={duration_ms}")
            return answer
        except Exception as e:
            logger.error(f"Direct answer failed: {str(e)}", exc_info=True)
            raise

    def verify_answer(self, question: str, answer: str, context: str) -> Dict:
        """Verify if the generated answer actually addresses the question."""
        logger.info(f"Verification started: question_len={len(question)}, answer_len={len(answer)}")
        start_time = time.time()

        messages = [
            {"role": "system", "content": (
                "You are an answer quality checker. Given a customer question, "
                "a generated answer, and the source context, determine if the answer "
                "correctly addresses the question.\n\n"
                "Output ONLY one word:\n"
                "VALID - The answer is relevant and addresses the question\n"
                "INVALID - The answer does not address the question or is made up"
            )},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Answer: {answer}\n\n"
                f"Source Context: {context[:500]}"
            )}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip().upper()
            is_valid = "VALID" in result and "INVALID" not in result
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(f"Verification completed: result={result}, is_valid={is_valid}, duration_ms={duration_ms}")

            return {
                "is_valid": is_valid,
                "result": result,
                "duration_ms": duration_ms
            }
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}", exc_info=True)
            # Default to valid on failure (don't block the answer)
            return {"is_valid": True, "result": "ERROR", "duration_ms": 0}

