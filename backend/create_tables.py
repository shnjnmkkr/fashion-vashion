#!/usr/bin/env python3
"""
Create database tables using raw SQL
"""

from database import engine
import sqlalchemy as sa

def create_tables_sql():
    """Create tables using raw SQL"""
    print("🗄️  Creating database tables with SQL...")
    
    # SQL to create tables
    create_users_table = """
    CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        email VARCHAR UNIQUE,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    
    create_favorites_table = """
    CREATE TABLE IF NOT EXISTS favorites (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(id),
        item_id VARCHAR NOT NULL,
        item_name VARCHAR NOT NULL,
        cloth_image_url VARCHAR NOT NULL,
        person_image_url VARCHAR NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    
    create_recommender_items_table = """
    CREATE TABLE IF NOT EXISTS recommender_items (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(id),
        item_id VARCHAR NOT NULL,
        item_name VARCHAR NOT NULL,
        cloth_image_url VARCHAR NOT NULL,
        person_image_url VARCHAR NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    
    create_chat_history_table = """
    CREATE TABLE IF NOT EXISTS chat_history (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(id),
        message_type VARCHAR NOT NULL,
        message_content TEXT NOT NULL,
        recommendations JSONB,
        llm_analysis JSONB,
        gemini_analysis JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    
    create_uploaded_images_table = """
    CREATE TABLE IF NOT EXISTS uploaded_images (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(id),
        image_url VARCHAR NOT NULL,
        filename VARCHAR NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    
    try:
        with engine.connect() as conn:
            # Create tables
            conn.execute(sa.text(create_users_table))
            conn.execute(sa.text(create_favorites_table))
            conn.execute(sa.text(create_recommender_items_table))
            conn.execute(sa.text(create_chat_history_table))
            conn.execute(sa.text(create_uploaded_images_table))
            conn.commit()
            
            print("✅ All tables created successfully!")
            
            # Verify tables exist
            result = conn.execute(sa.text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result.fetchall()]
            print(f"📋 Tables in database: {tables}")
            
    except Exception as e:
        print(f"❌ Error creating tables: {e}")

if __name__ == "__main__":
    create_tables_sql() 