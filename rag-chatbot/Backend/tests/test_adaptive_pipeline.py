"""
Adaptive RAG Pipeline Benchmark
Tests all pipeline paths and logs results to MLflow.

Usage:
    1. Make sure the FastAPI server is running (uvicorn)
    2. Run: python -m tests.test_adaptive_pipeline
"""

import os
import requests
import time
import mlflow

# ============ Config ============
BASE_URL = "http://localhost:8000"
MLFLOW_EXPERIMENT = "Adaptive Pipeline Benchmark"

# Point MLflow to Backend/mlruns (same as the main app)
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"file://{BACKEND_DIR}/mlruns")

# ============ Test Cases ============
TEST_CASES = [
    {
        "name": "Casual - Greeting",
        "question": "Hello how are you today?",
        "expected_intent": "casual",
        "expected_pipeline": "casual"
    },
    {
        "name": "Casual - Thank You",
        "question": "thank you so much for your help!",
        "expected_intent": "casual",
        "expected_pipeline": "casual"
    },
    {
        "name": "Support - High Confidence",
        "question": "Can you resend the verification link?",
        "expected_intent": "support",
        "expected_pipeline": "high_confidence"
    },
    {
        "name": "Support - Medium Confidence",
        "question": "how can I return my product?",
        "expected_intent": "support",
        "expected_pipeline": "medium_confidence"
    },
    {
        "name": "Support - Low Confidence (Rejected)",
        "question": "do you accept cryptocurrency payments like Bitcoin?",
        "expected_intent": "support",
        "expected_pipeline": "low_confidence_rejected"
    },
]


def get_auth_token():
    """Login via Google OAuth and get token. Uses existing token approach."""
    print("Getting auth token...")
    response = requests.get(f"{BASE_URL}/auth/google/login", allow_redirects=False)
    print("Please login via browser and paste your JWT token below:")
    token = input("Token: ").strip()
    return token


def run_test(question: str, token: str):
    """Send a question to /chat/ask and measure response time."""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"question": question, "conversation_id": 0, "n_results": 3}

    start = time.time()
    response = requests.post(
        f"{BASE_URL}/chat/ask",
        json=payload,
        headers=headers
    )
    elapsed = round(time.time() - start, 3)

    if response.status_code != 200:
        print(f"  ERROR: {response.status_code} - {response.text}")
        return None

    data = response.json()
    data["client_response_time"] = elapsed
    return data


def main():
    print("=" * 60)
    print("  Adaptive RAG Pipeline Benchmark")
    print("=" * 60)

    # Get auth token
    token = input("Paste your JWT token: ").strip()

    # Setup MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results = []

    with mlflow.start_run(run_name="pipeline_benchmark"):

        for i, test in enumerate(TEST_CASES, 1):
            print(f"\n[{i}/{len(TEST_CASES)}] Testing: {test['name']}")
            print(f"  Question: \"{test['question']}\"")

            data = run_test(test["question"], token)

            if data is None:
                continue

            # Extract results
            result = {
                "name": test["name"],
                "question": test["question"],
                "intent": data.get("intent", ""),
                "pipeline": data.get("pipeline", ""),
                "confidence": data.get("confidence", 0),
                "verified": data.get("verified"),
                "response_time": data["client_response_time"],
                "expected_intent": test["expected_intent"],
                "expected_pipeline": test["expected_pipeline"],
                "intent_match": data.get("intent") == test["expected_intent"],
                "pipeline_match": data.get("pipeline") == test["expected_pipeline"],
            }
            results.append(result)

            # Print result
            intent_icon = "✅" if result["intent_match"] else "❌"
            pipeline_icon = "✅" if result["pipeline_match"] else "❌"
            print(f"  Intent:     {result['intent']} {intent_icon}")
            print(f"  Pipeline:   {result['pipeline']} {pipeline_icon}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Verified:   {result['verified']}")
            print(f"  Time:       {result['response_time']}s")

            # Log each test to MLflow
            safe_name = test['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
            mlflow.log_metric(f"time_{safe_name}", result["response_time"])
            mlflow.log_metric(f"confidence_{safe_name}", result["confidence"])

        # Log summary metrics
        total_tests = len(results)
        intent_correct = sum(1 for r in results if r["intent_match"])
        pipeline_correct = sum(1 for r in results if r["pipeline_match"])
        avg_time = round(sum(r["response_time"] for r in results) / total_tests, 3) if total_tests > 0 else 0

        mlflow.log_metric("total_tests", total_tests)
        mlflow.log_metric("intent_accuracy", intent_correct / total_tests if total_tests > 0 else 0)
        mlflow.log_metric("pipeline_accuracy", pipeline_correct / total_tests if total_tests > 0 else 0)
        mlflow.log_metric("avg_response_time", avg_time)

        # Print summary table
        print("\n" + "=" * 80)
        print("  BENCHMARK RESULTS")
        print("=" * 80)
        print(f"{'Test':<35} {'Intent':<10} {'Pipeline':<28} {'Time':<8} {'Conf':<8}")
        print("-" * 80)
        for r in results:
            i_icon = "✅" if r["intent_match"] else "❌"
            p_icon = "✅" if r["pipeline_match"] else "❌"
            print(f"{r['name']:<35} {r['intent']:<8}{i_icon} {r['pipeline']:<26}{p_icon} {r['response_time']:<8} {r['confidence']:<8}")

        print("-" * 80)
        print(f"Intent Accuracy:   {intent_correct}/{total_tests}")
        print(f"Pipeline Accuracy: {pipeline_correct}/{total_tests}")
        print(f"Avg Response Time: {avg_time}s")
        print("=" * 80)
        print(f"\nResults logged to MLflow experiment: '{MLFLOW_EXPERIMENT}'")


if __name__ == "__main__":
    main()
