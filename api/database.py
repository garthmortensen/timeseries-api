#!/usr/bin/env python3
# timeseries-api/api/database.py
"""Database connection and session management."""

import os
import time
import logging as l
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, ForeignKey, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.exc import OperationalError
import datetime
from dotenv import load_dotenv
from typing import Optional, Generator

# Database connection details from environment variables
load_dotenv()
DB_USER = os.getenv("DB_USER", "timeseriesuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "timeseriespass")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "timeseriesdb")

# Construct database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLAlchemy setup
Base = declarative_base()

# Hardcode database as disabled
DISABLE_DATABASE = True

# Type definitions for proper typing
SessionLocal: Optional[sessionmaker[Session]] = None
engine: Optional[object] = None

# Only define engine and session if database is enabled
if not DISABLE_DATABASE:
    def get_engine(max_retries=5, retry_delay=5):
        """Create database engine with retry logic for container orchestration."""
        retries = 0
        last_exception = None
        
        while retries < max_retries:
            try:
                l.info(f"Attempting to connect to database (attempt {retries+1}/{max_retries})...")
                
                engine = create_engine(
                    DATABASE_URL,
                    pool_pre_ping=True,  # Verify connections before using from pool
                    pool_size=10,
                    max_overflow=20,
                    pool_recycle=3600    # Recycle connections after 1 hour
                )
                
                # Test connection
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    
                l.info("Successfully connected to PostgreSQL database")
                return engine
            
            except OperationalError as e:
                last_exception = e
                retries += 1
                l.warning(f"Database connection failed: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        l.error(f"Failed to connect to database after {max_retries} attempts")
        raise last_exception or Exception("Failed to connect to the database after multiple attempts")
    # Create engine instance
    engine = get_engine()
    # Session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Optional[Session], None, None]:
    """Dependency for getting DB session. Returns None if DB is disabled."""
    if DISABLE_DATABASE or SessionLocal is None:
        yield None
    else:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

def init_db() -> None:
    """Initialize the database tables."""
    if not DISABLE_DATABASE and engine is not None:
        Base.metadata.create_all(bind=engine)

# Define SQLAlchemy models - always define them for type consistency
class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    status = Column(String, nullable=False)
    source_type = Column(String)
    start_date = Column(String)
    end_date = Column(String) 
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    
    # Only define relationship if database is enabled
    if not DISABLE_DATABASE:
        results = relationship("PipelineResult", back_populates="pipeline_run")

class PipelineResult(Base):
    __tablename__ = "pipeline_results"
    
    id = Column(Integer, primary_key=True, index=True)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id"))
    symbol = Column(String, nullable=False)
    result_type = Column(String, nullable=False)  # 'arima', 'garch', 'stationarity'
    is_stationary = Column(Boolean, nullable=True)
    adf_statistic = Column(Float, nullable=True)
    p_value = Column(Float, nullable=True)
    model_summary = Column(Text, nullable=True)
    forecast = Column(Text, nullable=True)  # JSON string of forecast values
    interpretation = Column(Text, nullable=True)
    
    # Only define relationship if database is enabled
    if not DISABLE_DATABASE:
        pipeline_run = relationship("PipelineRun", back_populates="results")
