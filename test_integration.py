#!/usr/bin/env python3
"""
Integration test script for Fashion-Vashion
"""

import requests
import json
import time

def test_backend_integration():
    """Test all backend endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Fashion-Vashion Backend Integration")
    print("=" * 50)
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Health check: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: Create user
    try:
        user_data = {"email": "test@example.com"}
        response = requests.post(f"{base_url}/api/users", data=user_data)
        if response.status_code == 200:
            user = response.json()
            user_id = user["id"]
            print(f"✅ User created: {user_id}")
        else:
            print(f"❌ User creation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ User creation error: {e}")
        return False
    
    # Test 3: Add to favorites
    try:
        favorite_data = {
            "item_id": "test-item-1",
            "item_name": "Test Fashion Item",
            "cloth_image_url": "/api/clothes/test.jpg",
            "person_image_url": "/api/images/test.jpg"
        }
        response = requests.post(f"{base_url}/api/users/{user_id}/favorites", data=favorite_data)
        print(f"✅ Add to favorites: {response.status_code}")
    except Exception as e:
        print(f"❌ Add to favorites error: {e}")
    
    # Test 4: Get favorites
    try:
        response = requests.get(f"{base_url}/api/users/{user_id}/favorites")
        if response.status_code == 200:
            favorites = response.json()
            print(f"✅ Get favorites: {len(favorites)} items")
        else:
            print(f"❌ Get favorites failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Get favorites error: {e}")
    
    # Test 5: Add to recommender
    try:
        recommender_data = {
            "item_id": "test-item-2",
            "item_name": "Test Recommender Item",
            "cloth_image_url": "/api/clothes/test2.jpg",
            "person_image_url": "/api/images/test2.jpg"
        }
        response = requests.post(f"{base_url}/api/users/{user_id}/recommender-items", data=recommender_data)
        print(f"✅ Add to recommender: {response.status_code}")
    except Exception as e:
        print(f"❌ Add to recommender error: {e}")
    
    # Test 6: Get recommender items
    try:
        response = requests.get(f"{base_url}/api/users/{user_id}/recommender-items")
        if response.status_code == 200:
            items = response.json()
            print(f"✅ Get recommender items: {len(items)} items")
        else:
            print(f"❌ Get recommender items failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Get recommender items error: {e}")
    
    # Test 7: Save chat message
    try:
        chat_data = {
            "message_type": "user",
            "message_content": "Test message",
            "recommendations": json.dumps({"test": "data"}),
            "llm_analysis": json.dumps({"strategy": "test"}),
            "gemini_analysis": json.dumps({"analysis": "test"})
        }
        response = requests.post(f"{base_url}/api/users/{user_id}/chat-history", data=chat_data)
        print(f"✅ Save chat message: {response.status_code}")
    except Exception as e:
        print(f"❌ Save chat message error: {e}")
    
    # Test 8: Get chat history
    try:
        response = requests.get(f"{base_url}/api/users/{user_id}/chat-history")
        if response.status_code == 200:
            history = response.json()
            print(f"✅ Get chat history: {len(history)} messages")
        else:
            print(f"❌ Get chat history failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Get chat history error: {e}")
    
    # Test 9: Test recommendations endpoint
    try:
        form_data = {
            "user_prompt": "Show me casual clothing",
            "catalog_items": json.dumps(["test-item-1", "test-item-2"]),
            "top_k": "5"
        }
        response = requests.post(f"{base_url}/api/recommendations", data=form_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Recommendations endpoint: {response.status_code}")
            print(f"   Recommendations: {len(data.get('recommendations', []))}")
        else:
            print(f"❌ Recommendations endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Recommendations endpoint error: {e}")
    
    print("\n🎉 Integration testing completed!")
    return True

if __name__ == "__main__":
    # Wait a bit for backend to start
    print("⏳ Waiting for backend to start...")
    time.sleep(3)
    test_backend_integration() 