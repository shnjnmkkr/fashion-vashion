#!/usr/bin/env python3
"""
Initialize database tables for Fashion-Vashion
"""

import os
from dotenv import load_dotenv
from database import create_tables, engine
from models import Base

load_dotenv()

def init_database():
    """Initialize database tables"""
    print("🗄️  Initializing Fashion-Vashion Database...")
    
    try:
        # Drop all tables first (for clean start)
        print("🧹 Dropping existing tables...")
        Base.metadata.drop_all(bind=engine)
        print("✅ Existing tables dropped")
        
        # Create all tables
        print("🏗️  Creating new tables...")
        create_tables()
        print("✅ Database tables created successfully!")
        
        print("🎉 Database initialization completed!")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    init_database() 