import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import logging

from app.db.session import get_db
from app.db.models import User

logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv("SECRET_KEY", "rsDwEzGRFWNq-ZKpmsIRvHEIYPyk8n14HwoHKy37QV8")
if SECRET_KEY == "rsDwEzGRFWNq-ZKpmsIRvHEIYPyk8n14HwoHKy37QV8":
    logger.warning("SECURITY WARNING: Using default SECRET_KEY! Set SECRET_KEY environment variable!")
else:
    logger.info("SECRET_KEY loaded from environment variables")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

security_scheme = HTTPBearer()


def hash_password(password: str) -> str:
    try:
        logger.debug(f"Hashing password: length={len(password)}")
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        logger.debug(f"Password hashed successfully")
        return hashed
    except Exception as e:
        logger.error(f"Failed to hash password: {str(e)}", exc_info=True)
        raise


def verify_password(plain: str, hashed: str) -> bool:
    try:
        result = bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
        if not result:
            logger.debug(f"Password verification failed: incorrect password")
        return result
    except Exception as e:
        logger.error(f"Password verification error: {str(e)}", exc_info=True)
        return False


def create_access_token(data: dict) -> str:
    try:
        payload = data.copy()
        payload["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        logger.debug(f"Creating access token: user_id={payload.get('sub')}, expire_minutes={ACCESS_TOKEN_EXPIRE_MINUTES}")
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        logger.debug(f"Access token created successfully: user_id={payload.get('sub')}")
        return token
    except Exception as e:
        logger.error(f"Failed to create access token: {str(e)}", exc_info=True)
        raise


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    db: Session = Depends(get_db)
) -> User:
    logger.debug(f"User authentication started")
    token = credentials.credentials
    user_id = None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
    except jwt.ExpiredSignatureError:
        logger.warning(f"Authentication failed: token_expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    except jwt.DecodeError:
        logger.warning(f"Authentication failed: invalid_token_format")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    except ValueError:
        logger.warning(f"Authentication failed: invalid_user_id_format")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        logger.warning(f"Authentication failed: user not found, user_id={user_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    logger.debug(f"User authenticated: user_id={user.id}, username={user.username}")
    return user
