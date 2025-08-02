from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import json
import numpy as np
from typing import List, Dict, Optional
import uvicorn
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import faiss
import google.generativeai as genai
import requests
from io import BytesIO
import re
import warnings
from transformers import CLIPProcessor, CLIPModel
import tempfile
import shutil

# Database imports
from database import get_db, create_tables
from models import User, Favorite, RecommenderItem, ChatHistory, UploadedImage
from sqlalchemy.orm import Session

warnings.filterwarnings('ignore')

app = FastAPI(title="Fashion-Vashion AI Backend", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://fashion-vashion.vercel.app",
        "https://fashion-vashion-frontend.vercel.app",
        "https://*.vercel.app",
        "https://*.railway.app",
        "https://*.render.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dataset path - handle missing dataset gracefully
DATASET_PATH = "../clothes_tryon_dataset"
if not os.path.exists(DATASET_PATH):
    print(f"⚠️ Dataset not found at {DATASET_PATH}")
    print("📝 Running in deployment mode - dataset will be served from external storage")
    DATASET_PATH = None
else:
    print(f"✅ Dataset found at {DATASET_PATH}")

# Configuration
CATALOG_PATH = os.path.join(DATASET_PATH, "train/cloth")
USER_IMAGES_PATH = os.path.join(DATASET_PATH, "train/image")

# Model Configuration
EMBEDDING_DIM = 512
TOP_K = 5
BATCH_SIZE = 32

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_name)

# Initialize FAISS index and catalog files
catalog_index = None
catalog_files = []

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
genai.configure(api_key=GEMINI_API_KEY)

# Pydantic models
class RecommendationRequest(BaseModel):
    user_prompt: str
    catalog_items: Optional[List[str]] = []
    top_k: int = 5

class RecommendationResponse(BaseModel):
    recommendations: List[Dict]
    llm_analysis: Dict
    gemini_analysis: Dict
    search_strategy: str

# Utility functions
def load_and_preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> torch.Tensor:
    """Load and preprocess image for CLIP model"""
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        
        # Use CLIP processor for consistent preprocessing
        inputs = clip_processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(device)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def extract_embedding(image_tensor: torch.Tensor) -> np.ndarray:
    """Extract embedding from image tensor using CLIP"""
    with torch.no_grad():
        image_features = clip_model.get_image_features(image_tensor)
        # Normalize embeddings
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

def build_catalog_index():
    """Build FAISS index from catalog images"""
    global catalog_index, catalog_files
    
    print("Building catalog index...")
    
    # Get all catalog images
    catalog_files = [f for f in os.listdir(CATALOG_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(catalog_files)} catalog images")
    
    if len(catalog_files) == 0:
        print(f"❌ No catalog images found in {CATALOG_PATH}")
        return False
    
    # Initialize FAISS index
    catalog_index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product for cosine similarity
    
    catalog_embeddings = []
    valid_files = []
    
    for i, filename in enumerate(catalog_files):
        if i % 1000 == 0:
            print(f"Processing {i}/{len(catalog_files)} images...")
        
        image_path = os.path.join(CATALOG_PATH, filename)
        
        if not os.path.exists(image_path):
            print(f"⚠️ File not found: {image_path}")
            continue
            
        image_tensor = load_and_preprocess_image(image_path)
        
        if image_tensor is not None:
            embedding = extract_embedding(image_tensor)
            catalog_embeddings.append(embedding)
            valid_files.append(filename)
        else:
            print(f"⚠️ Failed to process: {filename}")
    
    if catalog_embeddings:
        # Stack embeddings and add to index
        catalog_embeddings = np.vstack(catalog_embeddings)
        catalog_index.add(catalog_embeddings)
        catalog_files = valid_files
        
        print(f"✅ Built index with {len(valid_files)} catalog items")
        print(f"Index size: {catalog_index.ntotal}")
        return True
    else:
        print("❌ No valid catalog embeddings found!")
        return False

def analyze_user_intent_and_guide_search(user_images: List[str], user_prompt: str) -> dict:
    """
    LLM analyzes user intent, extracts attributes, and generates CLIP search terms.
    Now supports multiple images.
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Load all user images
        user_image_objects = []
        for image_path in user_images:
            try:
                user_image_objects.append(Image.open(image_path))
                print(f"   ✅ Loaded image: {image_path}")
            except Exception as e:
                print(f"   ❌ Error loading image {image_path}: {e}")
        
        if not user_image_objects:
            print("   ⚠️ No valid images loaded, using fallback")
            return {
                "search_strategy": "similar",
                "clip_search_terms": ["clothing"],
                "user_intent": "casual",
                "style_analysis": "No valid images provided",
                "confidence": "low"
            }

        prompt = f"""
        You are an AI fashion stylist. Analyze the user's images and request to determine the best search strategy.

        USER REQUEST: "{user_prompt}"
        NUMBER OF IMAGES: {len(user_image_objects)}

        In addition to previous fields, extract:
        - "detected_gender": "male|female|unisex|other|unknown"
        - "skin_tone": "fair|medium|dark|olive|other|unknown"
        - "body_type": "slim|average|plus-size|athletic|petite|tall|other|unknown"
        - "age_group": "child|teen|adult|senior|unknown"
        - "season": "summer|winter|spring|autumn|all-season|unknown"
        - "occasion": "casual|formal|business|party|wedding|sports|vacation|other|unknown"
        - "color_palette": ["red", "blue", ...]  # main colors detected
        - "fashion_era": "modern|retro|vintage|y2k|other|unknown"
        - "detected_patterns": ["stripes", "floral", ...]  # if visible
        - "detected_materials": ["denim", "cotton", ...]  # if visible
        - "detected_brands": ["Nike", "Adidas", ...]  # if visible

        Also, generate 3-5 CLIP search terms that combine these attributes, e.g.:
        ["red floral dress for plus-size women", "Nike logo t-shirt for teens", "outfit for dark skin tone in summer"]

        Respond in this JSON:
        {{
            "search_strategy": "similar|complementary|evaluation_only|trendy|formal|casual",
            "clip_search_terms": ["term1", "term2", "term3"],
            "user_intent": "casual|formal|trendy|evaluation|other",
            "style_analysis": "Detailed analysis of the user's style",
            "confidence": "high|medium|low",
            "detected_gender": "male|female|unisex|other|unknown",
            "skin_tone": "fair|medium|dark|olive|other|unknown",
            "body_type": "slim|average|plus-size|athletic|petite|tall|other|unknown",
            "age_group": "child|teen|adult|senior|unknown",
            "season": "summer|winter|spring|autumn|all-season|unknown",
            "occasion": "casual|formal|business|party|wedding|sports|vacation|other|unknown",
            "color_palette": ["red", "blue"],
            "fashion_era": "modern|retro|vintage|y2k|other|unknown",
            "detected_patterns": ["stripes"],
            "detected_materials": ["cotton"],
            "detected_brands": ["Nike"]
        }}
        """

        # Generate content with images
        response = model.generate_content([prompt] + user_image_objects)
        
        # Parse response
        try:
            result = json.loads(response.text)
            print(f"   ✅ LLM analysis successful: {result.get('search_strategy', 'unknown')}")
            return result
        except json.JSONDecodeError as e:
            print(f"   ❌ Failed to parse LLM response: {e}")
            print(f"   Raw response: {response.text}")
            # Return fallback response
            return {
                "search_strategy": "similar",
                "clip_search_terms": ["clothing", "fashion"],
                "user_intent": "casual",
                "style_analysis": "Analysis failed, using fallback",
                "confidence": "low"
            }
            
    except Exception as e:
        print(f"   ❌ Error in analyze_user_intent_and_guide_search: {e}")
        import traceback
        traceback.print_exc()
        # Return fallback response
        return {
            "search_strategy": "similar",
            "clip_search_terms": ["clothing"],
            "user_intent": "casual",
            "style_analysis": f"Error occurred: {str(e)}",
            "confidence": "low"
        }

def search_by_text_query(search_term: str, top_k: int) -> List[Dict]:
    """Search catalog using text query"""
    text_inputs = clip_processor(text=search_term, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    similarities, indices = catalog_index.search(text_features.cpu().numpy(), top_k)
    results = []
    
    for i, (score, idx) in enumerate(zip(similarities[0], indices[0])):
        if idx < len(catalog_files):
            filename = catalog_files[idx]
            image_path = os.path.join(CATALOG_PATH, filename)
            results.append({
                'image_path': image_path,
                'filename': filename,
                'similarity_score': float(score),
                'search_type': 'text',
                'search_term': search_term,
                'rank': i + 1
            })
    return results

def search_with_llm_guidance(user_images: List[str], user_prompt: str, top_k: int = TOP_K):
    """
    Use LLM to guide CLIP search strategy, combining image and text-based retrieval.
    Now supports multiple images.
    """
    try:
        print("🎯 LLM-Guided CLIP Search...")

        # Step 1: LLM analyzes intent and provides search strategy and attributes
        llm_analysis = analyze_user_intent_and_guide_search(user_images, user_prompt)
        print(f"✅ LLM analysis completed: {llm_analysis.get('search_strategy', 'unknown')}")

        # Step 2: Handle different search strategies
        if llm_analysis["search_strategy"] == "evaluation_only":
            print("📝 User wants evaluation only - no catalog search needed")
            return [], llm_analysis

        # Step 3: Retrieve using both image and LLM-generated text queries
        recommendations = []

        # Image-based retrieval (average embeddings from multiple images)
        all_embeddings = []
        for image_path in user_images:
            print(f"   Processing image: {image_path}")
            user_image_tensor = load_and_preprocess_image(image_path)
            if user_image_tensor is not None:
                user_embedding = extract_embedding(user_image_tensor)
                all_embeddings.append(user_embedding)
                print(f"   ✅ Image processed successfully")
            else:
                print(f"   ❌ Failed to process image: {image_path}")
        
        if all_embeddings:
            print(f"   Processing {len(all_embeddings)} image embeddings")
            # Average all embeddings
            avg_embedding = np.mean(all_embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)  # Normalize
            
            similarities, indices = catalog_index.search(avg_embedding.reshape(1, -1), top_k)
            print(f"   Found {len(similarities[0])} similar items")
            
            for i, (score, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(catalog_files):
                    filename = catalog_files[idx]
                    image_path = os.path.join(CATALOG_PATH, filename)
                    recommendations.append({
                        'image_path': image_path,
                        'filename': filename,
                        'similarity_score': float(score),
                        'search_type': 'image',
                        'rank': i + 1
                    })
                    print(f"   ✅ Added recommendation: {filename} (score: {score:.3f})")

        # Text-based retrieval (LLM-guided)
        search_terms = llm_analysis.get("clip_search_terms", [])
        print(f"   Processing {len(search_terms)} text search terms")
        for term in search_terms:
            print(f"   Searching for: {term}")
            text_results = search_by_text_query(term, top_k)
            recommendations.extend(text_results)
            print(f"   ✅ Found {len(text_results)} text-based results")

        # Deduplicate by filename, keep highest score
        unique = {}
        for rec in recommendations:
            fname = rec['filename']
            if fname not in unique or rec['similarity_score'] > unique[fname]['similarity_score']:
                unique[fname] = rec
        deduped = list(unique.values())

        # Sort by similarity score
        deduped.sort(key=lambda x: x['similarity_score'], reverse=True)
        # Re-rank
        for i, rec in enumerate(deduped):
            rec['rank'] = i + 1

        print(f"✅ Final recommendations: {len(deduped[:top_k])} items")
        return deduped[:top_k], llm_analysis
        
    except Exception as e:
        print(f"❌ Error in search_with_llm_guidance: {e}")
        import traceback
        traceback.print_exc()
        # Return empty results with error info
        return [], {
            "search_strategy": "error",
            "error": str(e),
            "clip_search_terms": [],
            "user_intent": "error",
            "style_analysis": f"Error occurred: {str(e)}",
            "confidence": "error"
        }

def enhance_with_gemini_analysis(user_images: List[str], user_prompt: str, catalog_recommendations: List[Dict], llm_analysis: Dict):
    """Enhanced Gemini analysis based on search strategy"""
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Load all user images
    user_image_objects = []
    for image_path in user_images:
        try:
            user_image_objects.append(Image.open(image_path))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    
    # Prepare catalog info
    catalog_info = ""
    for i, rec in enumerate(catalog_recommendations[:3]):
        catalog_info += f"Item {i+1}: {rec['filename']} (Score: {rec['similarity_score']:.3f})\n"
    
    # Different prompts based on search strategy
    if llm_analysis["search_strategy"] == "evaluation_only":
        prompt = f"""
        You are an AI fashion stylist. The user wants feedback on their current outfit.
        
        USER REQUEST: "{user_prompt}"
        
        Provide a detailed outfit evaluation in this JSON format:
        {{
            "outfit_rating": "1-10",
            "style_analysis": "Detailed analysis of the outfit",
            "strengths": ["What looks good"],
            "improvements": ["What could be better"],
            "occasion_suitability": "casual|formal|business|party|wedding|other",
            "styling_tips": ["How to improve the look"],
            "accessory_suggestions": ["Accessories that would enhance the outfit"]
        }}
        """
    
    elif llm_analysis["search_strategy"] == "similar":
        prompt = f"""
        You are an AI fashion stylist. The user wants items similar to what they're wearing.
        
        USER REQUEST: "{user_prompt}"
        
        CATALOG RESULTS:
        {catalog_info}
        
        Provide enhanced recommendations in this JSON format:
        {{
            "analysis": {{
                "style_match": "How well the items match the user's style",
                "catalog_feedback": "Quality of similar items found"
            }},
            "styling_tips": ["How to style the recommended items"],
            "accessory_suggestions": ["Accessories that would work well"],
            "search_suggestions": [
                {{
                    "item_type": "Additional items to search for",
                    "search_terms": "Exact search terms",
                    "websites": ["myntra", "amazon", "flipkart"],
                    "reason": "Why this would be good"
                }}
            ]
        }}
        """
    
    else:  # complementary
        prompt = f"""
        You are an AI fashion stylist. The user wants items that complement what they're wearing.
        
        USER REQUEST: "{user_prompt}"
        
        CATALOG RESULTS:
        {catalog_info}
        
        Provide enhanced recommendations in this JSON format:
        {{
            "analysis": {{
                "compatibility": "How well the items complement the user's outfit",
                "catalog_feedback": "Quality of complementary items found"
            }},
            "outfit_combinations": ["How to combine the items"],
            "styling_tips": ["How to style the complete outfit"],
            "accessory_suggestions": ["Accessories for the complete look"],
            "search_suggestions": [
                {{
                    "item_type": "Additional complementary items",
                    "search_terms": "Exact search terms",
                    "websites": ["myntra", "amazon", "flipkart"],
                    "reason": "Why this would complement well"
                }}
            ]
        }}
        """
    
    try:
        # Pass all images to Gemini
        content = [prompt] + user_image_objects
        response = model.generate_content(content)
        
        try:
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                return {"error": "Unable to parse response"}
        except json.JSONDecodeError:
            return {"error": "JSON decode error"}
            
    except Exception as e:
        return {"error": f"API error: {e}"}

def get_intelligent_recommendations(user_images: List[str], user_prompt: str, top_k: int = TOP_K):
    """Main function that orchestrates the entire process"""
    
    print("🧠 INTELLIGENT FASHION RECOMMENDATION SYSTEM")
    print("="*60)
    
    # Step 1: LLM analyzes intent and guides search
    catalog_recommendations, llm_analysis = search_with_llm_guidance(user_images, user_prompt, top_k)
    
    # Step 2: Enhanced analysis based on strategy
    if llm_analysis["search_strategy"] == "evaluation_only":
        print("📝 Providing outfit evaluation...")
        gemini_analysis = enhance_with_gemini_analysis(user_images, user_prompt, [], llm_analysis)
        return [], llm_analysis, gemini_analysis
    else:
        print("🎯 Enhancing recommendations...")
        gemini_analysis = enhance_with_gemini_analysis(user_images, user_prompt, catalog_recommendations, llm_analysis)
        return catalog_recommendations, llm_analysis, gemini_analysis

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    print("🚀 Starting Fashion-Vashion Backend...")
    
    # Check if dataset exists
    if DATASET_PATH is None:
        print("⚠️ Dataset not found, skipping dataset checks.")
    else:
        if not os.path.exists(DATASET_PATH):
            print(f"❌ Dataset directory not found: {DATASET_PATH}")
            print("Please ensure the dataset is in the correct location.")
            return
        
        print("✅ Dataset found!")
        print(f"📁 Dataset path: {DATASET_PATH}")
    
    # Create database tables
    try:
        create_tables()
        print("✅ Database tables created successfully!")
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
    
    # Build FAISS index
    if DATASET_PATH is not None:
        try:
            build_catalog_index()
            print("✅ FAISS index built successfully!")
        except Exception as e:
            print(f"❌ Error building FAISS index: {e}")
    else:
        print("⚠️ Skipping FAISS index build due to missing dataset.")
    
    print("✅ Backend initialized successfully!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Fashion-Vashion AI Backend",
        "status": "running",
        "catalog_size": len(catalog_files) if catalog_files else 0
    }

@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    user_prompt: str = Form(...),
    catalog_items: Optional[str] = Form("[]"),  # JSON string of catalog item IDs
    top_k: int = Form(5),
    files: List[UploadFile] = File([])
):
    """
    Get fashion recommendations based on user images and prompt.
    Supports multiple uploaded images and catalog items.
    """
    try:
        print(f"🔍 Recommendations request received:")
        print(f"   User prompt: {user_prompt}")
        print(f"   Catalog items: {catalog_items}")
        print(f"   Top K: {top_k}")
        print(f"   Files: {len(files)}")
        
        # Check if FAISS index is built
        if catalog_index is None:
            print("❌ FAISS index not built!")
            raise HTTPException(status_code=500, detail="Catalog index not initialized. Please restart the backend.")
        
        if DATASET_PATH is None:
            print("⚠️ Dataset not found, serving recommendations from external storage.")
            # In deployment mode, we can't rely on catalog_files directly.
            # We need to provide a placeholder or a way to fetch images.
            # For now, we'll return an error or a placeholder.
            raise HTTPException(status_code=503, detail="Dataset not available for catalog search.")

        if not catalog_files:
            print("❌ No catalog files available!")
            raise HTTPException(status_code=500, detail="No catalog files found. Please check the dataset path.")
        
        print(f"✅ FAISS index ready: {catalog_index.ntotal} items")
        
        # Parse catalog items
        catalog_item_list = json.loads(catalog_items) if catalog_items else []
        print(f"   Parsed catalog items: {catalog_item_list}")
        
        # Save uploaded files temporarily
        user_image_paths = []
        for file in files:
            if file.content_type.startswith('image/'):
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                shutil.copyfileobj(file.file, temp_file)
                temp_file.close()
                user_image_paths.append(temp_file.name)
                print(f"   Saved uploaded file: {temp_file.name}")
        
        # Add catalog items as user images if provided
        for item_id in catalog_item_list:
            if item_id in catalog_files:
                catalog_image_path = os.path.join(CATALOG_PATH, item_id)
                if os.path.exists(catalog_image_path):
                    user_image_paths.append(catalog_image_path)
                    print(f"   Added catalog item: {catalog_image_path}")
                else:
                    print(f"   ⚠️ Catalog item not found: {catalog_image_path}")
            else:
                print(f"   ⚠️ Catalog item ID not in files: {item_id}")
        
        if not user_image_paths:
            print("⚠️ No images provided, using text-based search only")
            # Use text-based search when no images are provided
            llm_analysis = {
                "search_strategy": "text_only",
                "clip_search_terms": [user_prompt],
                "user_intent": "general",
                "style_analysis": "Text-based search",
                "confidence": "medium"
            }
            
            # Get text-based recommendations
            catalog_recommendations = search_by_text_query(user_prompt, top_k)
            
            # Get Gemini analysis
            gemini_analysis = enhance_with_gemini_analysis([], user_prompt, catalog_recommendations, llm_analysis)
            
            return RecommendationResponse(
                recommendations=catalog_recommendations,
                llm_analysis=llm_analysis,
                gemini_analysis=gemini_analysis,
                search_strategy=llm_analysis.get("search_strategy", "text_only")
            )
        
        print(f"✅ Processing {len(user_image_paths)} images")
        
        # Get recommendations
        catalog_recommendations, llm_analysis, gemini_analysis = get_intelligent_recommendations(
            user_image_paths, user_prompt, top_k
        )
        
        print(f"✅ Got {len(catalog_recommendations)} recommendations")
        
        # Clean up temporary files
        for temp_path in user_image_paths:
            if temp_path.startswith('/tmp') or temp_path.startswith(tempfile.gettempdir()):
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        return RecommendationResponse(
            recommendations=catalog_recommendations,
            llm_analysis=llm_analysis,
            gemini_analysis=gemini_analysis,
            search_strategy=llm_analysis.get("search_strategy", "unknown")
        )
        
    except Exception as e:
        print(f"❌ Error in recommendations endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/catalog")
async def get_catalog_info():
    """Get information about the catalog"""
    return {
        "total_items": len(catalog_files) if catalog_files else 0,
        "index_size": catalog_index.ntotal if catalog_index else 0,
        "sample_items": catalog_files[:10] if catalog_files else []
    }

@app.get("/api/catalog/{filename}")
async def get_catalog_image(filename: str):
    """Serve catalog images"""
    try:
        image_path = os.path.join(CATALOG_PATH, filename)
        if os.path.exists(image_path):
            return FileResponse(image_path)
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving catalog image: {str(e)}")

@app.get("/api/clothes/{filename}")
async def serve_cloth_image(filename: str):
    """Serve cloth images"""
    try:
        if DATASET_PATH is None:
            # In deployment, serve from external storage or return placeholder
            raise HTTPException(status_code=404, detail="Dataset not available in deployment")
        
        image_path = os.path.join(CATALOG_PATH, filename)
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(image_path, media_type="image/jpeg")
    except Exception as e:
        print(f"Error serving cloth image: {e}")
        raise HTTPException(status_code=500, detail="Error serving image")

@app.get("/api/images/{filename}")
async def serve_person_image(filename: str):
    """Serve person images"""
    try:
        if DATASET_PATH is None:
            # In deployment, serve from external storage or return placeholder
            raise HTTPException(status_code=404, detail="Dataset not available in deployment")
        
        image_path = os.path.join(USER_IMAGES_PATH, filename)
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(image_path, media_type="image/jpeg")
    except Exception as e:
        print(f"Error serving person image: {e}")
        raise HTTPException(status_code=500, detail="Error serving image")

# Database endpoints
@app.post("/api/users")
async def create_user(email: str = Form(...), db: Session = Depends(get_db)):
    """Create a new user"""
    try:
        user = User(email=email)
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"id": str(user.id), "email": user.email}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/favorites")
async def get_user_favorites(user_id: str, db: Session = Depends(get_db)):
    """Get user's favorites"""
    try:
        favorites = db.query(Favorite).filter(Favorite.user_id == user_id).all()
        return [{"id": str(f.id), "item_id": f.item_id, "item_name": f.item_name, 
                "cloth_image_url": f.cloth_image_url, "person_image_url": f.person_image_url} 
                for f in favorites]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/users/{user_id}/favorites")
async def add_favorite(user_id: str, item_id: str, item_name: str, 
                      cloth_image_url: str, person_image_url: str, db: Session = Depends(get_db)):
    """Add item to user's favorites"""
    try:
        favorite = Favorite(
            user_id=user_id,
            item_id=item_id,
            item_name=item_name,
            cloth_image_url=cloth_image_url,
            person_image_url=person_image_url
        )
        db.add(favorite)
        db.commit()
        return {"message": "Added to favorites"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/users/{user_id}/favorites/{item_id}")
async def remove_favorite(user_id: str, item_id: str, db: Session = Depends(get_db)):
    """Remove item from user's favorites"""
    try:
        favorite = db.query(Favorite).filter(
            Favorite.user_id == user_id, 
            Favorite.item_id == item_id
        ).first()
        if favorite:
            db.delete(favorite)
            db.commit()
            return {"message": "Removed from favorites"}
        else:
            raise HTTPException(status_code=404, detail="Favorite not found")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/recommender-items")
async def get_user_recommender_items(user_id: str, db: Session = Depends(get_db)):
    """Get user's recommender items"""
    try:
        items = db.query(RecommenderItem).filter(RecommenderItem.user_id == user_id).all()
        return [{"id": str(i.id), "item_id": i.item_id, "item_name": i.item_name,
                "cloth_image_url": i.cloth_image_url, "person_image_url": i.person_image_url}
                for i in items]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/users/{user_id}/recommender-items")
async def add_recommender_item(user_id: str, item_id: str, item_name: str,
                              cloth_image_url: str, person_image_url: str, db: Session = Depends(get_db)):
    """Add item to user's recommender"""
    try:
        item = RecommenderItem(
            user_id=user_id,
            item_id=item_id,
            item_name=item_name,
            cloth_image_url=cloth_image_url,
            person_image_url=person_image_url
        )
        db.add(item)
        db.commit()
        return {"message": "Added to recommender"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/users/{user_id}/recommender-items/{item_id}")
async def remove_recommender_item(user_id: str, item_id: str, db: Session = Depends(get_db)):
    """Remove item from user's recommender"""
    try:
        item = db.query(RecommenderItem).filter(
            RecommenderItem.user_id == user_id,
            RecommenderItem.item_id == item_id
        ).first()
        if item:
            db.delete(item)
            db.commit()
            return {"message": "Removed from recommender"}
        else:
            raise HTTPException(status_code=404, detail="Recommender item not found")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/chat-history")
async def get_chat_history(user_id: str, db: Session = Depends(get_db)):
    """Get user's chat history"""
    try:
        history = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).order_by(ChatHistory.created_at.desc()).limit(50).all()
        return [{"id": str(h.id), "message_type": h.message_type, "message_content": h.message_content,
                "recommendations": h.recommendations, "llm_analysis": h.llm_analysis, 
                "gemini_analysis": h.gemini_analysis, "created_at": h.created_at.isoformat()}
                for h in history]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/users/{user_id}/chat-history")
async def save_chat_message(user_id: str, message_type: str, message_content: str,
                           recommendations: Optional[dict] = None, llm_analysis: Optional[dict] = None,
                           gemini_analysis: Optional[dict] = None, db: Session = Depends(get_db)):
    """Save a chat message"""
    try:
        chat_message = ChatHistory(
            user_id=user_id,
            message_type=message_type,
            message_content=message_content,
            recommendations=recommendations,
            llm_analysis=llm_analysis,
            gemini_analysis=gemini_analysis
        )
        db.add(chat_message)
        db.commit()
        return {"message": "Chat message saved"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 