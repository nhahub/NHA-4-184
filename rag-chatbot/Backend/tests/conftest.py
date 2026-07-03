import os
import sys
from typing import Generator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from unittest.mock import MagicMock, patch

# Add the Backend path to sys.path so we can import app modules directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 1. Global mock for Redis to prevent connection errors during test imports
mock_redis = MagicMock()
sys.modules['redis'] = MagicMock()
import redis
redis.Redis = MagicMock(return_value=mock_redis)

from app.main import app
from app.db.session import get_db, Base
from app.core.security import create_access_token
from app.db.models import User

# 2. SQLite Database Setup for Testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    """Create all tables in the test database on startup and drop them on teardown."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Provide a clean transactional database session for a single test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """Provide a TestClient with overridden get_db dependency pointing to the test database."""
    def _get_test_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _get_test_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# 3. Mocks for External Services (LLMs, TTS, Speech)
@pytest.fixture
def mock_groq_client():
    """Mock Groq client completions interface."""
    with patch("app.nlp.generator.Groq") as mock_groq, \
         patch("app.nlp.router.Groq") as mock_router:
        
        # Generator Mock Setup
        mock_gen_instance = MagicMock()
        mock_groq.return_value = mock_gen_instance
        
        # Router Mock Setup
        mock_route_instance = MagicMock()
        mock_router.return_value = mock_route_instance
        
        yield {
            "generator": mock_gen_instance,
            "router": mock_route_instance
        }


@pytest.fixture
def mock_whisper():
    """Mock OpenAI Whisper model loading and transcribing."""
    with patch("app.nlp.transcriber.whisper") as mock_whisp:
        mock_model = MagicMock()
        mock_whisp.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {
            "text": "This is a mock transcription from voice.",
            "language": "en"
        }
        yield mock_model


@pytest.fixture
def mock_tts():
    """Mock ElevenLabs TTS synthesize function."""
    with patch("app.nlp.tts.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake_mp3_audio_bytes"
        mock_post.return_value = mock_response
        yield mock_post


# 4. Helpers and User Fixtures
@pytest.fixture
def test_user(db_session: Session) -> User:
    """Create a default active user in the test database."""
    from app.core.security import hash_password
    user = User(
        username="testuser",
        email="testuser@example.com",
        hashed_password=hash_password("Password123"),
        is_active=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(test_user: User) -> dict:
    """Generate JWT authorization headers for the test user."""
    access_token = create_access_token(data={"sub": str(test_user.id)})
    return {"Authorization": f"Bearer {access_token}"}

