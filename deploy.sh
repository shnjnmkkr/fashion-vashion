#!/bin/bash

echo "🚀 Fashion-Vashion Deployment Script"
echo "=================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit for deployment"
fi

# Check if remote is set
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "🔗 Please set up your GitHub repository:"
    echo "   git remote add origin https://github.com/yourusername/fashion-vashion.git"
    echo "   git push -u origin main"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ Git repository ready"
echo ""

echo "📋 Deployment Checklist:"
echo "1. ✅ Git repository initialized"
echo "2. ⏳ Create Supabase project (see DEPLOYMENT.md)"
echo "3. ⏳ Deploy backend to Railway (see DEPLOYMENT.md)"
echo "4. ⏳ Deploy frontend to Vercel (see DEPLOYMENT.md)"
echo ""

echo "🔧 Environment Variables needed:"
echo "Backend (Railway):"
echo "  - DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres"
echo "  - GEMINI_API_KEY=your_gemini_api_key"
echo ""
echo "Frontend (Vercel):"
echo "  - NEXT_PUBLIC_BACKEND_URL=https://your-app.railway.app"
echo ""

echo "📖 See DEPLOYMENT.md for detailed instructions"
echo ""

echo "🎯 Quick Commands:"
echo "  - Push to GitHub: git push origin main"
echo "  - Test locally: npm run dev (frontend) && cd backend && python start_backend.py"
echo ""

echo "✅ Ready for deployment!" 