#!/usr/bin/env python3
"""
Test script to verify database connection and table creation
"""

import os
from dotenv import load_dotenv
from database import create_tables, get_db, engine
from models import User, Favorite, RecommenderItem, ChatHistory, UploadedImage

load_dotenv()

def test_database_connection():
    """Test database connection and table creation"""
    print("🔍 Testing database connection...")
    
    try:
        # Create tables
        create_tables()
        print("✅ Database tables created successfully!")
        
        # Test database session
        db = next(get_db())
        print("✅ Database session created successfully!")
        
        # Test creating a user
        test_user = User(email="test@example.com")
        db.add(test_user)
        db.commit()
        print("✅ Test user created successfully!")
        
        # Clean up
        db.delete(test_user)
        db.commit()
        print("✅ Test user cleaned up successfully!")
        
        db.close()
        print("✅ Database connection test completed successfully!")
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_database_connection() 