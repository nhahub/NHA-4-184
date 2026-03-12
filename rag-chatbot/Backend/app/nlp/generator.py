import os
from groq import Groq
from typing import List, Dict


class Generator:
    """Generates natural language answers using Groq LLM."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def generate_answer(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """
        Generate a natural answer using the retrieved chunks as context.
        """
        # Build context from retrieved chunks
        context = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            context += f"\n--- Source {i} ---\n{chunk['text']}\n"

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

        # Call Groq API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        return response.choices[0].message.content
