from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import User, ChatMessage, Feedback, Conversation
from app.models.request import FeedbackRequest
from app.models.response import FeedbackResponse
from app.core.security import get_current_user

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post("/", response_model=FeedbackResponse)
def submit_feedback(
    request: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Verify the message exists and belongs to the user
    message = db.query(ChatMessage).join(Conversation).filter(
        ChatMessage.id == request.message_id,
        Conversation.user_id == current_user.id
    ).first()

    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    # Check if feedback already exists for this message
    existing = db.query(Feedback).filter(
        Feedback.chat_message_id == request.message_id
    ).first()

    if existing:
        # Update existing feedback
        existing.rating = request.rating
        existing.comment = request.comment
        db.commit()
        db.refresh(existing)
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
        db.commit()
        db.refresh(feedback)
        return feedback


@router.get("/{message_id}", response_model=FeedbackResponse)
def get_feedback(
    message_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Verify the message belongs to the user
    message = db.query(ChatMessage).join(Conversation).filter(
        ChatMessage.id == message_id,
        Conversation.user_id == current_user.id
    ).first()

    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    feedback = db.query(Feedback).filter(
        Feedback.chat_message_id == message_id
    ).first()

    if not feedback:
        raise HTTPException(status_code=404, detail="No feedback for this message")

    return feedback
