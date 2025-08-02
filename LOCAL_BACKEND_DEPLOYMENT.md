# 🏠 Local Backend + Deployed Frontend Setup

This guide shows how to deploy only the frontend while running the backend locally on your machine.

## 🎯 **Benefits of This Approach**

- ✅ **No Railway size limits** - Backend runs locally with full dataset
- ✅ **Full AI functionality** - All CLIP, FAISS, and Gemini features work
- ✅ **Cost effective** - Only frontend hosting costs
- ✅ **Easy development** - Modify backend code instantly
- ✅ **Privacy** - Your AI models stay on your machine

## 🚀 **Step 1: Deploy Frontend to Vercel**

### 1.1 Create Vercel Project

1. Go to [vercel.com](https://vercel.com)
2. Sign up/Login with GitHub
3. Click "New Project"
4. Import repository: `shnjnmkkr/fashion-vashion`
5. Configure:
   - **Framework**: Next.js (auto-detected)
   - **Root Directory**: `/` (project root)
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`

### 1.2 Set Environment Variables

In Vercel dashboard → Settings → Environment Variables:

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

**Note**: This will only work when you're running the backend locally.

## 🔧 **Step 2: Run Backend Locally**

### 2.1 Start Backend

**Option A: Use the batch file (Windows)**
```bash
start_backend_local.bat
```

**Option B: Manual start**
```bash
cd backend
pip install -r requirements.txt
python start_backend.py
```

### 2.2 Verify Backend is Running

Visit: `http://localhost:8000/`

**Expected Response:**
```json
{
  "message": "Fashion-Vashion AI Backend",
  "status": "running"
}
```

## 🌐 **Step 3: Configure for Public Access**

### 3.1 For Local Network Access

If you want others on your network to access it:

1. **Find your local IP**:
   ```bash
   ipconfig  # Windows
   ifconfig  # Mac/Linux
   ```

2. **Update Vercel environment variable**:
   ```env
   NEXT_PUBLIC_BACKEND_URL=http://YOUR_LOCAL_IP:8000
   ```

3. **Configure firewall** to allow port 8000

### 3.2 For Internet Access (Advanced)

**Option A: ngrok (Temporary)**
```bash
# Install ngrok
npm install -g ngrok

# Create tunnel
ngrok http 8000

# Use the ngrok URL in Vercel environment variable
NEXT_PUBLIC_BACKEND_URL=https://your-ngrok-url.ngrok.io
```

**Option B: Cloudflare Tunnel (Permanent)**
1. Install cloudflared
2. Create tunnel to localhost:8000
3. Use tunnel URL in environment variable

## 🧪 **Step 4: Testing**

### 4.1 Test Local Setup

1. **Start backend**: `start_backend_local.bat`
2. **Visit frontend**: Your Vercel URL
3. **Test features**:
   - ✅ Browse fashion catalog
   - ✅ Add to favorites
   - ✅ Use AI recommender
   - ✅ Upload images

### 4.2 Test Public Access

1. **Update environment variable** with your public URL
2. **Restart backend** if needed
3. **Test from different devices/networks**

## 🔄 **Step 5: Development Workflow**

### 5.1 Making Changes

**Frontend Changes:**
1. Edit code locally
2. Push to GitHub
3. Vercel auto-deploys

**Backend Changes:**
1. Edit code locally
2. Restart backend: `python start_backend.py`
3. Test immediately

### 5.2 Environment Variables

**For Development:**
```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

**For Production (when you're online):**
```env
NEXT_PUBLIC_BACKEND_URL=http://YOUR_PUBLIC_IP:8000
```

## 🛠️ **Troubleshooting**

### Common Issues

**1. Frontend can't connect to backend**
- Check if backend is running on port 8000
- Verify environment variable is correct
- Check browser console for CORS errors

**2. Backend won't start**
- Check if port 8000 is available
- Verify Python dependencies are installed
- Check if dataset path is correct

**3. Public access not working**
- Verify firewall allows port 8000
- Check if IP address is correct
- Test with ngrok first

**4. Images not loading**
- Ensure dataset is in correct location
- Check file permissions
- Verify image paths in code

### Debug Commands

```bash
# Check if backend is running
curl http://localhost:8000/

# Check if port is open
netstat -an | findstr :8000

# Test backend API
curl -X POST http://localhost:8000/api/recommendations \
  -F "user_prompt=test" \
  -F "top_k=5"
```

## 📊 **Monitoring**

### Backend Monitoring
- Check console output for errors
- Monitor CPU/Memory usage
- Watch for API request logs

### Frontend Monitoring
- Use Vercel analytics
- Check browser console
- Monitor user interactions

## 🔒 **Security Considerations**

1. **Firewall**: Only open port 8000 when needed
2. **Authentication**: Consider adding API keys
3. **Rate Limiting**: Implement to prevent abuse
4. **HTTPS**: Use ngrok or similar for secure access

## 🎉 **Success Indicators**

✅ Backend responds to health check  
✅ Frontend loads without errors  
✅ AI recommendations work  
✅ Images load properly  
✅ Database operations work  
✅ Public access functions  

---

**This setup gives you the best of both worlds: deployed frontend with local backend control!** 🚀 