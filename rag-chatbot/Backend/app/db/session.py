"""
Database session management.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from typing import Generator
import os
import logging
import time
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag_project")

# Mask password in logs for security
db_url_masked = DATABASE_URL.split("@")[1] if "@" in DATABASE_URL else "unknown"
logger.info(f"Initializing database engine: url={db_url_masked}")

# Create SQLAlchemy engine
try:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Test connection before using
        echo=False  # Set to True to see SQL queries
    )
    logger.info(f"Database engine initialized successfully: url={db_url_masked}")
except Exception as e:
    logger.error(f"Failed to initialize database engine: {str(e)}, url={db_url_masked}", exc_info=True)
    raise

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.
    Yields a database session and closes it after use.
    """
    db = SessionLocal()
    session_id = str(id(db))[:8]
    session_start = time.time()
    logger.debug(f"Database session created: session_id={session_id}")
    
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}, session_id={session_id}, error_type={type(e).__name__}", exc_info=True)
        try:
            db.rollback()
            logger.debug(f"Database session rolled back: session_id={session_id}")
        except Exception as rollback_error:
            logger.warning(f"Error during rollback: {str(rollback_error)}, session_id={session_id}")
        raise
    finally:
        session_duration_ms = round((time.time() - session_start) * 1000, 2)
        try:
            db.close()
            logger.debug(f"Database session closed: session_id={session_id}, duration_ms={session_duration_ms}")
        except Exception as close_error:
            logger.warning(f"Error closing database session: {str(close_error)}, session_id={session_id}")
