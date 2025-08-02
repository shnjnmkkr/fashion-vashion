#!/usr/bin/env python3
"""
Fashion-Vashion AI Backend Startup Script
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_requirements():
    """Check if all required files and directories exist"""
    print("🔍 Checking requirements...")
    
    # Check if dataset directory exists
    dataset_path = "../clothes_tryon_dataset"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset directory not found: {dataset_path}")
        print("Please ensure the clothes_tryon_dataset directory exists with the following structure:")
        print("clothes_tryon_dataset/")
        print("├── train/")
        print("│   ├── cloth/     # Clothing images")
        print("│   └── image/     # Person images")
        return False
    
    # Check if catalog directory exists
    catalog_path = os.path.join(dataset_path, "train", "cloth")
    if not os.path.exists(catalog_path):
        print(f"❌ Catalog directory not found: {catalog_path}")
        return False
    
    # Check if there are catalog images
    catalog_files = [f for f in os.listdir(catalog_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(catalog_files) == 0:
        print(f"❌ No catalog images found in {catalog_path}")
        return False
    
    print(f"✅ Found {len(catalog_files)} catalog images")
    
    # Check Gemini API key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        print("⚠️  Warning: GEMINI_API_KEY not set or using default value")
        print("Please set your Gemini API key in .env file or environment variable")
        print("You can still run the backend, but AI features will be limited")
    
    return True

def main():
    """Main startup function"""
    print("🚀 Fashion-Vashion AI Backend Startup")
    print("="*50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix the issues above.")
        sys.exit(1)
    
    print("\n✅ All requirements met!")
    print("\n🎯 Starting backend server...")
    print("📡 API will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    print("="*50)
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 