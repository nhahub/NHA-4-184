from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import logging

from app.db.session import get_db
from app.db.models import User, ChatMessage, Feedback, Conversation
from app.models.request import FeedbackRequest
from app.models.response import FeedbackResponse
from app.core.security import get_current_user
from app.mlops.tracker import tracker
from app.mlops.metrics import feedback_positive_total, feedback_negative_total

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post("/", response_model=FeedbackResponse)
def submit_feedback(
    request: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    logger.info(f"Feedback submission started: user_id={current_user.id}, message_id={request.message_id}, rating={request.rating}")

    # Verify the message exists and belongs to the user
    message = db.query(ChatMessage).join(Conversation).filter(
        ChatMessage.id == request.message_id,
        Conversation.user_id == current_user.id
    ).first()

    if not message:
        logger.warning(f"Feedback failed: message not found, message_id={request.message_id}, user_id={current_user.id}")
        raise HTTPException(status_code=404, detail="Message not found")

    # Check if feedback already exists for this message
    existing = db.query(Feedback).filter(
        Feedback.chat_message_id == request.message_id
    ).first()

    if existing:
        # Update existing feedback
        existing.rating = request.rating
        existing.comment = request.comment
        try:
            db.commit()
            db.refresh(existing)
            logger.info(f"Feedback updated: message_id={request.message_id}, user_id={current_user.id}, rating={request.rating}")
            tracker.log_feedback(message_id=request.message_id, rating=request.rating)
            # --- Record Prometheus metric ---
            if request.rating == 1:
                feedback_positive_total.inc()
            else:
                feedback_negative_total.inc()
        except Exception as e:
            logger.error(f"Failed to update feedback: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to save feedback")
        return existing
    else:
        # Create new feedback
        feedback = Feedback(
            chat_message_id=request.message_id,
            user_id=current_user.id,
            rating=request.rating,
            comment=request.comment
        )
        db.add(feedback)
        try:
            db.commit()
            db.refresh(feedback)
            logger.info(f"Feedback submitted: message_id={request.message_id}, user_id={current_user.id}, rating={request.rating}")
            tracker.log_feedback(message_id=request.message_id, rating=request.rating)
            # --- Record Prometheus metric ---
            if request.rating == 1:
                feedback_positive_total.inc()
            else:
                feedback_negative_total.inc()
        except Exception as e:
            logger.error(f"Failed to create feedback: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to save feedback")
        return feedback


@router.get("/{message_id}", response_model=FeedbackResponse)
def get_feedback(
    message_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    logger.info(f"Feedback retrieval started: user_id={current_user.id}, message_id={message_id}")

    # Verify the message belongs to the user
    message = db.query(ChatMessage).join(Conversation).filter(
        ChatMessage.id == message_id,
        Conversation.user_id == current_user.id
    ).first()

    if not message:
        logger.warning(f"Get feedback failed: message not found, message_id={message_id}, user_id={current_user.id}")
        raise HTTPException(status_code=404, detail="Message not found")

    feedback = db.query(Feedback).filter(
        Feedback.chat_message_id == message_id
    ).first()

    if not feedback:
        logger.warning(f"Get feedback failed: no feedback exists, message_id={message_id}, user_id={current_user.id}")
        raise HTTPException(status_code=404, detail="No feedback for this message")

    logger.info(f"Feedback retrieved: message_id={message_id}, user_id={current_user.id}, rating={feedback.rating}")
    return feedback


# ==================== ADMIN: NEGATIVE FEEDBACK ====================

@router.get("/admin/negative", tags=["Admin"])
def get_negative_feedback(db: Session = Depends(get_db)):
    """Get all messages with negative feedback (for retraining)."""
    logger.info("Admin: fetching negative feedback")

    results = db.query(
        ChatMessage.id,
        ChatMessage.user_query,
        ChatMessage.llm_response,
        ChatMessage.created_at,
        Feedback.rating,
        Feedback.comment
    ).join(Feedback, Feedback.chat_message_id == ChatMessage.id).filter(
        Feedback.rating == -1
    ).order_by(ChatMessage.created_at.desc()).all()

    return [
        {
            "message_id": r.id,
            "question": r.user_query,
            "bad_answer": r.llm_response,
            "rating": r.rating,
            "comment": r.comment,
            "created_at": str(r.created_at)
        }
        for r in results
    ]


@router.post("/admin/retrain", tags=["Admin"])
def retrain_from_feedback(
    message_id: int,
    correct_answer: str,
    db: Session = Depends(get_db)
):
    """Add a corrected Q&A pair to ChromaDB knowledge base."""
    from app.nlp.embedder import Embedder
    from app.nlp.vector_db import VectorDB

    logger.info(f"Admin retrain: message_id={message_id}")

    # Get the original question
    message = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    # Create the document to add to ChromaDB
    document = f"Q: {message.user_query}\nA: {correct_answer}"

    # Embed and add to ChromaDB
    embedder = Embedder()
    vector_db = VectorDB()

    embedding = embedder.embed_text(document)
    doc_id = f"retrain_{message_id}"

    vector_db.add_chunks(
        ids=[doc_id],
        documents=[document],
        embeddings=[embedding],
        metadatas=[{
            "source": "admin_retrain",
            "original_message_id": str(message_id),
            "question": message.user_query,
            "correct_answer": correct_answer
        }]
    )

    logger.info(f"Admin retrain successful: message_id={message_id}, doc_id={doc_id}")

    return {
        "status": "success",
        "message": f"Added corrected answer to knowledge base",
        "doc_id": doc_id,
        "question": message.user_query,
        "correct_answer": correct_answer
    }

