from pydantic import BaseModel
from typing import List


class UserResponse(BaseModel):
    id: int
    username: str
    is_active: bool

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
    answer: str
    confidence: float
    matched_question: str
    category: str
    sources: List[SourceChunk]
