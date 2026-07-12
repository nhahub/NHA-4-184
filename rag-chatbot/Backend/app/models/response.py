from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    is_admin: bool = False

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    distance: float
    category: str


class ChatResponse(BaseModel):
    message_id: int
    answer: str
    confidence: float
    matched_question: str
    category: str
    sources: List[SourceChunk]
    conversation_id: int
    intent: Optional[str] = None
    pipeline: Optional[str] = None
    verified: Optional[bool] = None


# --- Feedback Models ---

class FeedbackInfo(BaseModel):
    rating: int
    comment: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class FeedbackResponse(BaseModel):
    id: int
    chat_message_id: int
    rating: int
    comment: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class OTPResponse(BaseModel):
    message: str
    expires_in_minutes: int = 5


class ResetTokenResponse(BaseModel):
    reset_token: str
    user: UserResponse


# --- Chat History Models ---

class MessageItem(BaseModel):
    id: int
    user_query: str
    llm_response: str
    response_time: Optional[float]
    created_at: datetime
    feedback: Optional[FeedbackInfo] = None

    class Config:
        from_attributes = True


class ConversationListItem(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConversationDetail(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[MessageItem]

    class Config:
        from_attributes = True


# --- Ticket Models ---

class TicketResponseInfo(BaseModel):
    id: int
    admin_id: int
    answer: str
    created_at: datetime

    class Config:
        from_attributes = True


class TicketOut(BaseModel):
    """Used by admin to see full ticket details."""
    id: int
    question: str
    status: str
    priority: str
    created_at: datetime
    user_id: int
    responses: List[TicketResponseInfo] = []

    class Config:
        from_attributes = True


class UserTicketOut(BaseModel):
    """Used by user to see their own tickets and the admin answer."""
    id: int
    conversation_id: Optional[int] = None
    question: str
    status: str
    created_at: datetime
    answer: Optional[str] = None

    class Config:
        from_attributes = True
