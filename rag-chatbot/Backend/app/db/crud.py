"""
CRUD operations for database.
"""
from sqlalchemy.orm import Session
from . import models


def create_chat_log(
    db: Session,
    user_query: str,
    llm_response: str,
    retrieved_context: str = None,
    response_time: float = None
) -> models.ChatLog:
    """Create a new chat log entry."""
    chat_log = models.ChatLog(
        user_query=user_query,
        llm_response=llm_response,
        retrieved_context=retrieved_context,
        response_time=response_time
    )
    db.add(chat_log)
    db.commit()
    db.refresh(chat_log)
    return chat_log


def create_feedback(
    db: Session,
    chat_log_id: int,
    rating: int,
    comment: str = None
) -> models.Feedback:
    """Create a new feedback entry."""
    feedback = models.Feedback(
        chat_log_id=chat_log_id,
        rating=rating,
        comment=comment
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback
