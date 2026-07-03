import pytest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
from app.db.models import User, OTPCode
from app.core.security import create_access_token


def test_register_user_success(client):
    """Test successful user registration."""
    payload = {
        "username": "newuser",
        "email": "newuser@example.com",
        "password": "SecretPassword123",
        "confirm_password": "SecretPassword123"
    }
    
    response = client.post("/auth/register", json=payload)
    
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "newuser"
    assert data["email"] == "newuser@example.com"
    assert "id" in data


def test_register_user_duplicate_username(client, test_user):
    """Test registration fails with duplicate username."""
    payload = {
        "username": test_user.username,  # Duplicate
        "email": "anotheremail@example.com",
        "password": "Password123",
        "confirm_password": "Password123"
    }
    
    response = client.post("/auth/register", json=payload)
    
    assert response.status_code == 400
    assert "Username already taken" in response.json()["detail"]


def test_register_user_duplicate_email(client, test_user):
    """Test registration fails with duplicate email."""
    payload = {
        "username": "anotheruser",
        "email": test_user.email,  # Duplicate
        "password": "Password123",
        "confirm_password": "Password123"
    }
    
    response = client.post("/auth/register", json=payload)
    
    assert response.status_code == 400
    assert "Email already registered" in response.json()["detail"]


def test_login_user_success(client, test_user):
    """Test successful user login."""
    payload = {
        "username": test_user.username,
        "password": "Password123"  # Matches conftest.py test_user password
    }
    
    response = client.post("/auth/login", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["username"] == test_user.username


def test_login_user_invalid_credentials(client, test_user):
    """Test login fails with incorrect password."""
    payload = {
        "username": test_user.username,
        "password": "WrongPassword"
    }
    
    response = client.post("/auth/login", json=payload)
    
    assert response.status_code == 401
    assert "Invalid username or password" in response.json()["detail"]


@patch("app.api.auth.send_otp_email")
def test_forgot_password_success(mock_send_email, client, test_user, db_session):
    """Test requesting a password reset email OTP."""
    payload = {
        "email": test_user.email
    }
    
    response = client.post("/auth/forgot-password", json=payload)
    
    assert response.status_code == 200
    assert response.json()["message"] == "OTP sent to your email"
    mock_send_email.assert_called_once()
    
    # Check that the OTP is created in the test database
    otp = db_session.query(OTPCode).filter(OTPCode.user_id == test_user.id).first()
    assert otp is not None
    assert len(otp.code) == 6


def test_verify_otp_success(client, test_user, db_session):
    """Test verifying a correct OTP returns a reset token."""
    # Seed OTP in database
    otp = OTPCode(
        user_id=test_user.id,
        code="123456",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        is_used=False
    )
    db_session.add(otp)
    db_session.commit()
    
    payload = {
        "email": test_user.email,
        "otp": "123456"
    }
    
    response = client.post("/auth/verify-otp", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "reset_token" in data
    assert data["user"]["email"] == test_user.email


def test_reset_password_success(client, test_user):
    """Test resetting the password using a valid reset token."""
    # Create valid reset token
    reset_token = create_access_token(data={"sub": str(test_user.id), "purpose": "reset"})
    
    payload = {
        "reset_token": reset_token,
        "new_password": "NewSecretPassword123",
        "confirm_password": "NewSecretPassword123"
    }
    
    response = client.post("/auth/reset-password", json=payload)
    
    assert response.status_code == 200
    assert response.json()["message"] == "Password reset successfully"


def test_get_current_user_info_success(client, auth_headers, test_user):
    """Test retrieving current logged in user details (/me endpoint)."""
    response = client.get("/auth/me", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == test_user.id
    assert data["username"] == test_user.username
    assert data["email"] == test_user.email
