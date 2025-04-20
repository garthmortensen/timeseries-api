import os
import time
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Connection retry settings
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

# SQLAlchemy setup
Base = declarative_base()

def get_engine(max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    """Create database engine with retry logic for container orchestration."""
    retries = 0
    last_exception = None
    
    while retries < max_retries:
        try:
            engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True,  # Verify connections before using from pool
                pool_size=5,         # Adjust based on your workload
                max_overflow=10,
                pool_recycle=3600    # Recycle connections after 1 hour
            )
            
            # Test connection
            with engine.connect() as conn:
                conn.execute("SELECT 1")
                
            logger.info("Successfully connected to database")
            return engine
            
        except OperationalError as e:
            last_exception = e
            retries += 1
            logger.warning(f"Database connection attempt {retries} failed. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    logger.error(f"Failed to connect to database after {max_retries} attempts")
    raise last_exception or Exception("Failed to connect to the database after multiple attempts")

# Create engine instance
engine = get_engine()

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for getting DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
