from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# ==================== CHAT METRICS ====================

# Counts how many chat questions have been asked (total, ever)
chat_requests_total = Counter(
    "chat_requests_total",
    "Total number of chat requests",
    ["status"]  # label: "success" or "error"
)

# Tracks the distribution of response times (how many were fast vs slow)
chat_response_seconds = Histogram(
    "chat_response_seconds",
    "Chat response time in seconds",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0]  # time brackets
)

# Tracks the distribution of retrieval confidence scores
retrieval_confidence = Histogram(
    "retrieval_confidence",
    "Retrieval confidence score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# ==================== FEEDBACK METRICS ====================

# Counts thumbs up
feedback_positive_total = Counter(
    "feedback_positive_total",
    "Total positive feedback (thumbs up)"
)

# Counts thumbs down
feedback_negative_total = Counter(
    "feedback_negative_total",
    "Total negative feedback (thumbs down)"
)

# ==================== AUTH METRICS ====================

# Counts login attempts
auth_login_total = Counter(
    "auth_login_total",
    "Total login attempts",
    ["status"]  # "success" or "failed"
)

# Counts registrations
auth_register_total = Counter(
    "auth_register_total",
    "Total registration attempts",
    ["status"]  # "success" or "failed"
)

# ==================== SYSTEM METRICS ====================

# Current number of active conversations (can go up AND down)
active_conversations = Gauge(
    "active_conversations",
    "Number of active conversations in the last 24h"
)


def get_metrics():
    """Generate the Prometheus metrics output text."""
    return generate_latest()


def get_metrics_content_type():
    """Return the correct content type for Prometheus scraping."""
    return CONTENT_TYPE_LATEST
