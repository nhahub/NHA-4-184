import pytest
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    SECRET_KEY,
    ALGORITHM
)
from app.db.models import User
from unittest.mock import MagicMock


def test_password_hashing():
    """Test that passwords are hashed and can be verified correctly."""
    password = "MySecurePassword123"
    hashed = hash_password(password)
    
    assert hashed != password
    assert len(hashed) > 0
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword123", hashed) is False


def test_password_hashing_empty():
    """Test hashing an empty password string."""
    hashed = hash_password("")
    assert verify_password("", hashed) is True


def test_create_access_token():
    """Test that JWT access tokens are created with correct payloads and expiry."""
    data = {"sub": "12345", "role": "admin"}
    token = create_access_token(data)
    
    # Decode the token and verify contents
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert payload["sub"] == "12345"
    assert payload["role"] == "admin"
    assert "exp" in payload


def test_get_current_user_valid(db_session, test_user):
    """Test get_current_user successfully returns the user when given a valid token."""
    # Generate token using our fixture user's ID
    token = create_access_token(data={"sub": str(test_user.id)})
    
    # Mock FastAPI authorization credentials
    credentials = MagicMock()
    credentials.credentials = token
    
    # Authenticate
    authenticated_user = get_current_user(credentials=credentials, db=db_session)
    
    assert authenticated_user.id == test_user.id
    assert authenticated_user.username == test_user.username


def test_get_current_user_expired(db_session, test_user):
    """Test get_current_user raises 401 when the token has expired."""
    # Create an expired payload
    data = {
        "sub": str(test_user.id),
        "exp": datetime.utcnow() - timedelta(minutes=10)  # expired 10 mins ago
    }
    expired_token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
    
    credentials = MagicMock()
    credentials.credentials = expired_token
    
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(credentials=credentials, db=db_session)
        
    assert exc_info.value.status_code == 401
    assert "Invalid or expired token" in exc_info.value.detail


def test_get_current_user_invalid_token(db_session):
    """Test get_current_user raises 401 when token is malformed."""
    credentials = MagicMock()
    credentials.credentials = "this-is-not-a-valid-jwt-token"
    
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(credentials=credentials, db=db_session)
        
    assert exc_info.value.status_code == 401


def test_get_current_user_not_found(db_session):
    """Test get_current_user raises 401 when token is valid but user does not exist in DB."""
    # Use user ID 99999 which does not exist in the in-memory database
    token = create_access_token(data={"sub": "99999"})
    
    credentials = MagicMock()
    credentials.credentials = token
    
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(credentials=credentials, db=db_session)
        
    assert exc_info.value.status_code == 401
    assert "User not found" in exc_info.value.detail
