#!/usr/bin/env python3
"""
Database setup script for Fashion-Vashion
"""

import os
from dotenv import load_dotenv

load_dotenv()

def setup_database():
    """Guide user through database setup"""
    print("🗄️  Fashion-Vashion Database Setup")
    print("=" * 50)
    
    # Check if DATABASE_URL is set
    database_url = os.getenv("DATABASE_URL")
    
    if database_url:
        print(f"✅ DATABASE_URL found: {database_url[:50]}...")
        print("📝 Using existing database configuration")
    else:
        print("❌ DATABASE_URL not found in .env file")
        print("\n🔧 Please add your database URL to the .env file:")
        print("\nFor Supabase:")
        print("DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.lvhhaaqflmxxrrrwpvjw.supabase.co:5432/postgres")
        print("\nFor local development (SQLite):")
        print("DATABASE_URL=sqlite:///./fashion_vashion.db")
        print("\nFor Railway:")
        print("DATABASE_URL=postgresql://postgres:password@containers-us-west-1.railway.app:5432/railway")
        
        # Ask user for database URL
        print("\n🤔 Which database would you like to use?")
        print("1. Supabase (Recommended for deployment)")
        print("2. SQLite (Local development)")
        print("3. Railway")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n📋 For Supabase, you need:")
            print("1. Go to your Supabase project dashboard")
            print("2. Navigate to Settings → Database")
            print("3. Copy the connection string")
            print("4. Add it to your .env file as DATABASE_URL")
            
            password = input("\nEnter your Supabase database password: ").strip()
            if password:
                database_url = f"postgresql://postgres:{password}@db.lvhhaaqflmxxrrrwpvjw.supabase.co:5432/postgres"
                print(f"\n📝 Add this to your .env file:")
                print(f"DATABASE_URL={database_url}")
            else:
                print("❌ Password required for Supabase connection")
                return False
                
        elif choice == "2":
            database_url = "sqlite:///./fashion_vashion.db"
            print(f"\n📝 Add this to your .env file:")
            print(f"DATABASE_URL={database_url}")
            
        elif choice == "3":
            print("\n📋 For Railway:")
            print("1. Create a Railway account")
            print("2. Add a PostgreSQL service")
            print("3. Copy the connection string")
            print("4. Add it to your .env file as DATABASE_URL")
            return False
        else:
            print("❌ Invalid choice")
            return False
    
    print("\n✅ Database configuration complete!")
    print("🚀 You can now run the backend with: python start_backend.py")
    return True

if __name__ == "__main__":
    setup_database() 