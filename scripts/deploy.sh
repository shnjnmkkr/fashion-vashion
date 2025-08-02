#!/bin/bash

# Fashion Marketplace Deployment Script
echo "🚀 Starting deployment process..."

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Please run this script from the project root."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build the project
echo "🔨 Building the project..."
npm run build

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed!"
    exit 1
fi

# Check for environment variables
if [ ! -f ".env.local" ]; then
    echo "⚠️  Warning: .env.local not found. Please create it with your environment variables."
    echo "   Copy env.example to .env.local and fill in your values."
fi

echo "🎉 Deployment script completed!"
echo ""
echo "Next steps:"
echo "1. Push to GitHub: git push origin main"
echo "2. Deploy on Vercel: https://vercel.com"
echo "3. Set up environment variables in your deployment platform"
echo ""
echo "For detailed instructions, see README.md" 