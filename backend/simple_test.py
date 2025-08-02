#!/usr/bin/env python3
"""
Simple database test
"""

from database import engine, create_tables
from models import Base
import sqlalchemy as sa

def simple_test():
    """Simple database test"""
    print("🔍 Simple database test...")
    
    try:
        # Create tables
        create_tables()
        print("✅ Tables creation called")
        
        # Check if tables exist
        inspector = sa.inspect(engine)
        tables = inspector.get_table_names()
        print(f"📋 Tables in database: {tables}")
        
        # Try to create a simple table
        with engine.connect() as conn:
            result = conn.execute(sa.text("SELECT current_database()"))
            db_name = result.fetchone()[0]
            print(f"📊 Connected to database: {db_name}")
            
            # Check if users table exists
            result = conn.execute(sa.text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'users'
            """))
            if result.fetchone():
                print("✅ Users table exists!")
            else:
                print("❌ Users table does not exist")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    simple_test() 