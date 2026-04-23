import os
import logging
import time
from groq import Groq
from typing import Dict

logger = logging.getLogger(__name__)


class LLMRouter:
    """Routes user queries to the appropriate pipeline using LLM classification."""

    VALID_INTENTS = ["CASUAL", "SUPPORT"]

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        logger.info(f"LLM Router initialized: model={self.model}")

    def classify(self, question: str) -> Dict:
        """
        Classify a question as CASUAL or SUPPORT.
        
        CASUAL = greetings, thanks, chitchat, jokes, goodbyes
        SUPPORT = product questions, orders, returns, shipping, account, complaints
        """
        start_time = time.time()

        messages = [
            {"role": "system", "content": (
                "You are a query classifier for a customer support chatbot. "
                "Classify the user's message into exactly one category.\n\n"
                "CASUAL - Greetings, thank you, goodbye, chitchat, jokes, "
                "compliments, or any message that does NOT need product/order information.\n"
                "Examples: 'hello', 'thanks', 'how are you', 'bye', 'you're great'\n\n"
                "SUPPORT - Any question about products, orders, returns, shipping, "
                "payments, accounts, refunds, complaints, or anything that needs "
                "information from the knowledge base.\n"
                "Examples: 'how to return?', 'where is my order?', 'refund policy'\n\n"
                "Output ONLY one word: CASUAL or SUPPORT"
            )},
            {"role": "user", "content": question}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=10
            )

            intent = response.choices[0].message.content.strip().upper()

            # Validate — default to SUPPORT if unclear
            if intent not in self.VALID_INTENTS:
                logger.warning(f"Router returned invalid intent: '{intent}', defaulting to SUPPORT")
                intent = "SUPPORT"

            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(f"Router classified: question='{question[:80]}', intent={intent}, duration_ms={duration_ms}")

            return {
                "intent": intent,
                "duration_ms": duration_ms
            }

        except Exception as e:
            logger.error(f"Router classification failed: {str(e)}", exc_info=True)
            # Default to SUPPORT on failure (safer — always try to help)
            return {
                "intent": "SUPPORT",
                "duration_ms": 0
            }
