from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# UNIFIED SQLITE DATABASE CONFIGURATION
# ============================================================================
# Single SQLite database for both AI assistant and operational PORTNET data
# Simplified setup - no MySQL required!
# ============================================================================

# Unified Database (SQLite) - All data in one place
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./duty_officer_assistant.db")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all tables in database"""
    from app.models.database import Base
    Base.metadata.create_all(bind=engine)