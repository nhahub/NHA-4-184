import os
import sys
import time
import mlflow
from dotenv import load_dotenv

load_dotenv()

from app.nlp.retrieval import Retriever
from app.nlp.generator import Generator

# --- Test Questions ---
TEST_QUESTIONS = [
    "How can I track my order?",
    "What is the return policy?",
    "How do I cancel my order?",
    "I received a damaged product, what should I do?",
    "How long does shipping take?",
    "Can I change my delivery address?",
    "How do I get a refund?",
    "What payment methods do you accept?",
    "How do I contact customer support?",
    "Where is my order?",
]

# --- Configurations to Compare ---
CONFIGS = [
    {"n_results": 3, "temperature": 0.3, "max_tokens": 500},
    {"n_results": 5, "temperature": 0.3, "max_tokens": 500},
    {"n_results": 3, "temperature": 0.7, "max_tokens": 500},
    {"n_results": 5, "temperature": 0.7, "max_tokens": 500},
]


def run_experiment(config: dict):
    """Run all test questions with a specific config and log to MLflow."""

    retriever = Retriever()
    generator = Generator()

    # Override generator settings
    original_model = generator.model

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("rag-config-comparison")

    run_name = f"n{config['n_results']}_t{config['temperature']}_m{config['max_tokens']}"

    with mlflow.start_run(run_name=run_name):
        # Log config parameters
        mlflow.log_param("n_results", config["n_results"])
        mlflow.log_param("temperature", config["temperature"])
        mlflow.log_param("max_tokens", config["max_tokens"])
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("llm_model", "llama-3.3-70b-versatile")
        mlflow.log_param("num_test_questions", len(TEST_QUESTIONS))

        total_time = 0
        total_confidence = 0
        total_answer_length = 0

        for i, question in enumerate(TEST_QUESTIONS):
            print(f"  [{i+1}/{len(TEST_QUESTIONS)}] {question}")

            start = time.time()

            # Retrieve
            result = retriever.get_answer_with_context(
                question=question,
                n_results=config["n_results"]
            )

            # Generate (with custom temperature)
            response = generator.client.chat.completions.create(
                model=generator.model,
                messages=[{
                    "role": "user",
                    "content": f"""You are a helpful customer support assistant.
Answer based ONLY on the context below.

Context:
{chr(10).join([r['text'] for r in result['all_results']])}

Question: {question}

Answer:"""
                }],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"]
            )
            answer = response.choices[0].message.content

            elapsed = time.time() - start
            total_time += elapsed
            total_confidence += result["confidence"]
            total_answer_length += len(answer)

            # Log per-question metrics
            mlflow.log_metric(f"q{i}_confidence", result["confidence"])
            mlflow.log_metric(f"q{i}_response_time", round(elapsed, 3))

        # Log aggregate metrics
        n = len(TEST_QUESTIONS)
        mlflow.log_metric("avg_confidence", round(total_confidence / n, 4))
        mlflow.log_metric("avg_response_time", round(total_time / n, 3))
        mlflow.log_metric("total_time", round(total_time, 3))
        mlflow.log_metric("avg_answer_length", round(total_answer_length / n, 1))

    print(f"  ✅ Config {run_name} done! Avg confidence: {total_confidence/n:.4f}, Avg time: {total_time/n:.2f}s")


def main():
    print("🧪 Starting RAG Configuration Experiments")
    print(f"📋 {len(TEST_QUESTIONS)} test questions × {len(CONFIGS)} configs = {len(TEST_QUESTIONS) * len(CONFIGS)} total queries")
    print("-" * 60)

    for i, config in enumerate(CONFIGS):
        print(f"\n🔬 Experiment {i+1}/{len(CONFIGS)}: {config}")
        run_experiment(config)

    print("\n" + "=" * 60)
    print("🎉 All experiments done!")
    print("📊 View results: mlflow ui --port 5001")
    print("   Then open: http://localhost:5001")


if __name__ == "__main__":
    main()
