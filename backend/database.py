from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

# Database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

# For local development, use SQLite if no DATABASE_URL is provided
if not DATABASE_URL:
    # Use the provided Supabase connection string
    password = "Shanjan-27"
    encoded_password = quote_plus(password)  # URL encode the password
    DATABASE_URL = f"postgresql://postgres:{encoded_password}@db.lvhhaaqflmxxrrrwpvjw.supabase.co:5432/postgres"
    print("📝 Using Supabase PostgreSQL database")

# Create engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    print("🔧 SQLite engine created")
else:
    # For PostgreSQL (Supabase/Railway)
    engine = create_engine(DATABASE_URL)
    print("🔧 PostgreSQL engine created")

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables
def create_tables():
    """Create all tables in the database"""
    try:
        # Import models to ensure they're registered
        from models import User, Favorite, RecommenderItem, ChatHistory, UploadedImage
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully!")
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False 