# Fashion-Vashion AI Backend

A sophisticated FastAPI backend that provides AI-powered fashion recommendations using CLIP embeddings, FAISS similarity search, and Gemini LLM analysis.

## 🚀 Features

- **Multi-Image Support**: Process multiple user images simultaneously
- **Smart Attribute Extraction**: Extract skin tone, body type, age, gender, etc.
- **LLM-Guided Search**: Gemini generates intelligent search queries
- **FAISS Similarity Search**: Fast vector similarity search across 11,647+ catalog items
- **Flexible Input Sources**: Upload images or select from catalog
- **Real-time Analysis**: Instant fashion recommendations and styling tips

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Images   │    │   Catalog Items │    │   User Prompt   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   LLM Analysis (Gemini)   │
                    │  - Extract attributes     │
                    │  - Generate search terms  │
                    │  - Determine strategy     │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   CLIP Embeddings        │
                    │  - Image embeddings       │
                    │  - Text embeddings        │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   FAISS Search           │
                    │  - Similarity search     │
                    │  - Top-K retrieval       │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   Enhanced Analysis      │
                    │  - Styling tips          │
                    │  - Accessory suggestions │
                    │  - Search suggestions    │
                    └─────────────────────────┘
```

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- 8GB+ RAM (for FAISS index)
- Gemini API key

## 🛠️ Installation

### 1. Clone and Setup

```bash
cd backend
pip install -r requirements.txt
```

### 2. Dataset Setup

Ensure your dataset structure:
```
clothes_tryon_dataset/
├── train/
│   ├── cloth/     # 11,647+ clothing images
│   └── image/     # Person images (optional)
```

### 3. Environment Variables

Create a `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Start the Backend

```bash
# Option 1: Using startup script (recommended)
python start_backend.py

# Option 2: Direct uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📡 API Endpoints

### 1. Health Check
```http
GET /
```
**Response:**
```json
{
  "message": "Fashion-Vashion AI Backend",
  "status": "running",
  "catalog_size": 11647
}
```

### 2. Get Recommendations
```http
POST /api/recommendations
Content-Type: multipart/form-data

Form Data:
- user_prompt: "Show me clothes similar to this"
- catalog_items: ["14602_00.jpg", "14603_00.jpg"] (optional)
- top_k: 5 (optional)
- files: [image1.jpg, image2.jpg] (optional)
```

**Response:**
```json
{
  "recommendations": [
    {
      "image_path": "/path/to/image.jpg",
      "filename": "14602_00.jpg",
      "similarity_score": 0.95,
      "search_type": "image",
      "rank": 1
    }
  ],
  "llm_analysis": {
    "search_strategy": "similar",
    "user_intent": "Find similar clothing",
    "clip_search_terms": ["red floral dress", "casual t-shirt"],
    "detected_gender": "female",
    "skin_tone": "medium",
    "body_type": "average",
    "age_group": "adult",
    "season": "summer",
    "occasion": "casual",
    "color_palette": ["red", "blue"],
    "fashion_era": "modern",
    "detected_patterns": ["floral"],
    "detected_materials": ["cotton"],
    "detected_brands": [],
    "confidence": "high"
  },
  "gemini_analysis": {
    "analysis": {
      "style_match": "Excellent match found",
      "catalog_feedback": "High quality recommendations"
    },
    "styling_tips": ["Pair with jeans", "Add accessories"],
    "accessory_suggestions": ["Silver necklace", "Leather bag"],
    "search_suggestions": [
      {
        "item_type": "Matching shoes",
        "search_terms": "white sneakers",
        "websites": ["myntra", "amazon"],
        "reason": "Would complete the casual look"
      }
    ]
  },
  "search_strategy": "similar"
}
```

### 3. Get Catalog Info
```http
GET /api/catalog
```

### 4. Get Catalog Image
```http
GET /api/catalog/{filename}
```

## 🔧 Configuration

### Model Settings
```python
# In main.py
EMBEDDING_DIM = 512      # CLIP embedding dimension
TOP_K = 5               # Default number of recommendations
BATCH_SIZE = 32         # Processing batch size
```

### Search Strategies
The backend supports three search strategies:

1. **Similar**: Find items similar to user's images
2. **Complementary**: Find items that go well with user's images
3. **Evaluation**: Provide outfit analysis without catalog search

## 🧠 AI Components

### 1. CLIP Model
- **Model**: `openai/clip-vit-base-patch32`
- **Purpose**: Extract image and text embeddings
- **Features**: 512-dimensional embeddings

### 2. FAISS Index
- **Type**: `IndexFlatIP` (Inner Product for cosine similarity)
- **Size**: 11,647+ catalog items
- **Performance**: Sub-second similarity search

### 3. Gemini LLM
- **Model**: `gemini-2.0-flash-exp`
- **Purpose**: 
  - Analyze user intent
  - Extract fashion attributes
  - Generate search queries
  - Provide styling advice

## 📊 Performance

- **Index Build Time**: ~5-10 minutes (first run)
- **Recommendation Time**: 2-5 seconds
- **Memory Usage**: ~2GB (with FAISS index)
- **GPU Acceleration**: Available with CUDA

## 🚀 Deployment

### Local Development
```bash
python start_backend.py
```

### Production (Railway/Render)
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Set environment variable
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **FAISS Index Build Fails**
   ```bash
   # Check available memory
   free -h
   # Reduce batch size in main.py
   ```

3. **Gemini API Errors**
   ```bash
   # Check API key
   echo $GEMINI_API_KEY
   # Verify quota
   ```

### Logs
```bash
# View detailed logs
uvicorn main:app --log-level debug
```

## 📈 Monitoring

### Health Checks
- **Endpoint**: `GET /`
- **Frequency**: Every 30 seconds
- **Metrics**: Catalog size, model status

### Performance Metrics
- Response time: < 5 seconds
- Memory usage: < 4GB
- GPU utilization: < 80%

## 🔐 Security

- CORS enabled for frontend integration
- File upload validation
- Temporary file cleanup
- API key environment variable

## 🤝 Integration

### Frontend Integration
```javascript
// Example API call
const formData = new FormData();
formData.append('user_prompt', 'Show me similar clothes');
formData.append('catalog_items', JSON.stringify(['14602_00.jpg']));

// Add uploaded files
for (const file of uploadedFiles) {
  formData.append('files', file);
}

const response = await fetch('/api/recommendations', {
  method: 'POST',
  body: formData
});
```

## 📝 License

MIT License - see LICENSE file for details.

---

**Happy coding! 🚀** 