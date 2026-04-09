"""
CRUD operations for database.
"""
from sqlalchemy.orm import Session
import logging
import time
from . import models

logger = logging.getLogger(__name__)


def create_chat_log(
    db: Session,
    user_query: str,
    llm_response: str,
    retrieved_context: str = None,
    response_time: float = None
) -> models.ChatLog:
    """Create a new chat log entry."""
    logger.info(f"Creating chat log: query_len={len(user_query)}, response_len={len(llm_response)}, response_time={response_time}")
    start_time = time.time()
    try:
        chat_log = models.ChatLog(
            user_query=user_query,
            llm_response=llm_response,
            retrieved_context=retrieved_context,
            response_time=response_time
        )
        db.add(chat_log)
        db.commit()
        db.refresh(chat_log)
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Chat log created successfully: chat_log_id={chat_log.id}, duration_ms={duration_ms}")
        return chat_log
    except Exception as e:
        logger.error(f"Failed to create chat log: {str(e)}", exc_info=True)
        db.rollback()
        raise


def create_feedback(
    db: Session,
    chat_log_id: int,
    rating: int,
    comment: str = None
) -> models.Feedback:
    """Create a new feedback entry."""
    logger.info(f"Creating feedback: chat_log_id={chat_log_id}, rating={rating}, comment_len={len(comment or '')}")
    start_time = time.time()
    try:
        feedback = models.Feedback(
            chat_log_id=chat_log_id,
            rating=rating,
            comment=comment
        )
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Feedback created successfully: feedback_id={feedback.id}, chat_log_id={chat_log_id}, duration_ms={duration_ms}")
        return feedback
    except Exception as e:
        logger.error(f"Failed to create feedback: {str(e)}, chat_log_id={chat_log_id}", exc_info=True)
        db.rollback()
        raise
