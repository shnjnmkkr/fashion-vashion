# 🚀 Fashion-Vashion Deployment Guide

## 📋 **Deployment Overview**

This guide will help you deploy Fashion-Vashion to production using free services:

- **Frontend**: Vercel (Next.js)
- **Backend**: Railway/Render (FastAPI)
- **Database**: Supabase (PostgreSQL)

## 🗄️ **Step 1: Database Setup (Supabase)**

### 1.1 Create Supabase Project
1. Go to [supabase.com](https://supabase.com)
2. Click "Start your project"
3. Create a new project
4. Note your project URL and anon key

### 1.2 Get Database Connection String
1. Go to Settings → Database
2. Copy the connection string
3. Format: `postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres`

### 1.3 Update Environment Variables
Add to your `.env` file:
```env
DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
GEMINI_API_KEY=your_gemini_api_key_here
```

## ⚙️ **Step 2: Backend Deployment (Railway)**

### 2.1 Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Set the root directory to `backend/`

### 2.2 Configure Environment Variables
In Railway dashboard, add these environment variables:
```env
DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2.3 Get Backend URL
1. Railway will provide a URL like: `https://your-app.railway.app`
2. Note this URL for frontend configuration

## 🌐 **Step 3: Frontend Deployment (Vercel)**

### 3.1 Deploy to Vercel
1. Go to [vercel.com](https://vercel.com)
2. Sign up with GitHub
3. Click "New Project"
4. Import your GitHub repository
5. Set the root directory to the project root (not backend/)

### 3.2 Configure Environment Variables
In Vercel dashboard, add:
```env
NEXT_PUBLIC_BACKEND_URL=https://your-app.railway.app
```

### 3.3 Update Frontend Configuration
Update `app/page.tsx` to use the environment variable:
```typescript
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
```

## 🔧 **Step 4: Update Frontend for Production**

### 4.1 Update API Calls
Replace all `http://localhost:8000` with the environment variable:

```typescript
// In app/page.tsx
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

// Update all fetch calls
const response = await fetch(`${BACKEND_URL}/api/users`, {
  // ... rest of the code
})
```

### 4.2 Update Image Serving
Update the image API routes to use the backend URL:

```typescript
// In app/api/clothes/[filename]/route.ts
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
```

## 🧪 **Step 5: Testing Deployment**

### 5.1 Test Backend
1. Visit your Railway URL: `https://your-app.railway.app/`
2. Should see: `{"message": "Fashion-Vashion AI Backend", "status": "running"}`

### 5.2 Test Frontend
1. Visit your Vercel URL: `https://your-app.vercel.app`
2. Test all features:
   - Browse fashion items
   - Add to favorites
   - Use AI recommender
   - Upload images

### 5.3 Test Database
1. Check if favorites are saved
2. Check if chat history is saved
3. Verify recommender items persist

## 🔍 **Troubleshooting**

### Common Issues:

**1. CORS Errors**
- Check if backend URL is correct in frontend
- Verify CORS origins in backend

**2. Database Connection**
- Check DATABASE_URL format
- Verify Supabase credentials

**3. Image Loading**
- Check if dataset is accessible
- Verify image paths

**4. AI Recommendations**
- Check GEMINI_API_KEY
- Verify API quotas

## 📊 **Monitoring**

### Railway Backend
- Check logs in Railway dashboard
- Monitor resource usage
- Set up alerts

### Vercel Frontend
- Check build logs
- Monitor performance
- Set up analytics

## 🎉 **Success Indicators**

✅ Backend responds to health check
✅ Frontend loads without errors
✅ Database operations work
✅ AI recommendations function
✅ Image upload works
✅ Favorites persist across sessions

## 🔗 **Final URLs**

- **Frontend**: `https://your-app.vercel.app`
- **Backend**: `https://your-app.railway.app`
- **Database**: Supabase dashboard

## 📝 **Next Steps**

1. Set up custom domain (optional)
2. Configure monitoring and alerts
3. Set up CI/CD pipelines
4. Add analytics
5. Optimize performance

---

**Need help?** Check the logs in Railway and Vercel dashboards for specific error messages. 