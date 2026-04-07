import mlflow
import time
import logging

logger = logging.getLogger(__name__)


class RAGExperimentTracker:
    """Tracks RAG pipeline experiments using MLflow."""

    def __init__(self, experiment_name: str = "rag-chatbot"):
        # Store MLflow data locally in Backend/mlruns/ folder
        mlflow.set_tracking_uri("file:./mlruns")

        # Create or get the experiment
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        logger.info(f"MLflow tracker initialized with experiment: {experiment_name}")

    def log_chat_query(
        self,
        question: str,
        answer: str,
        confidence: float,
        response_time: float,
        n_results: int,
        sources_count: int,
        category: str,
        user_id: int,
        conversation_id: int
    ):
        """
        Log a single chat query as an MLflow run.
        
        This records:
        - Parameters: what settings were used (model, temperature, etc.)
        - Metrics: how well it performed (confidence, speed, etc.)
        """
        try:
            with mlflow.start_run(run_name=f"query_{conversation_id}_{int(time.time())}"):
                # --- Parameters (the CONFIG used) ---
                mlflow.log_param("question", question[:250])
                mlflow.log_param("n_results", n_results)
                mlflow.log_param("category", category)
                mlflow.log_param("user_id", user_id)
                mlflow.log_param("conversation_id", conversation_id)
                mlflow.log_param("llm_model", "llama-3.3-70b-versatile")
                mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
                mlflow.log_param("temperature", 0.3)
                mlflow.log_param("max_tokens", 500)

                # --- Metrics (the RESULTS) ---
                mlflow.log_metric("confidence", confidence)
                mlflow.log_metric("response_time_seconds", response_time)
                mlflow.log_metric("sources_count", sources_count)
                mlflow.log_metric("answer_length", len(answer))
                mlflow.log_metric("question_length", len(question))

        except Exception as e:
            # Don't crash the app if MLflow fails
            logger.error(f"MLflow logging failed: {e}")

    def log_feedback(self, message_id: int, rating: int):
        """
        Log user feedback as an MLflow run.
        
        This helps track user satisfaction over time.
        rating: 1 = thumbs up, -1 = thumbs down
        """
        try:
            with mlflow.start_run(run_name=f"feedback_{message_id}_{int(time.time())}"):
                mlflow.log_param("message_id", message_id)
                mlflow.log_param("feedback_type", "positive" if rating == 1 else "negative")
                mlflow.log_metric("rating", rating)

        except Exception as e:
            logger.error(f"MLflow feedback logging failed: {e}")


# --- Create a single global instance ---
# This way all parts of the app use the same tracker
tracker = RAGExperimentTracker()
