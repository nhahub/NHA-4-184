from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import logging

from app.db.session import get_db
from app.db.models import User, ChatMessage, Feedback, Conversation
from app.models.request import FeedbackRequest
from app.models.response import FeedbackResponse
from app.core.security import get_current_user
from app.mlops.tracker import tracker

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
