# Fashion-Vashion

A beautiful, minimalistic fashion marketplace built with Next.js, shadcn/ui, and integrated with a sophisticated AI-powered recommendation system.

## Features

- 🛍️ **Elegant Catalog**: Browse through a curated collection of fashion items
- 🖱️ **Hover Preview**: See how clothes look on people when you hover over items
- ❤️ **Favorites**: Save your favorite items for later
- 🤖 **AI Recommender**: Add items to your AI-powered recommendation system
- 📱 **Responsive Design**: Works perfectly on desktop and mobile
- ⚡ **Fast Performance**: Optimized for speed and user experience

## Tech Stack

- **Frontend**: Next.js 14, React 18, TypeScript
- **UI Components**: shadcn/ui, Tailwind CSS
- **Icons**: Lucide React
- **Backend**: Python with CLIP, FAISS, Gemini LLM
- **Database**: Supabase (for deployment)

## Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Python 3.8+ (for backend)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd fashion-marketplace
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env.local
   ```
   
   Add your environment variables:
   ```env
   NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
   NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
   ```

4. **Start the development server**
   ```bash
   npm run dev
   ```

5. **Open your browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

## Project Structure

```
fashion-marketplace/
├── app/                    # Next.js app directory
│   ├── api/               # API routes
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Main page
├── components/            # React components
│   └── ui/               # shadcn/ui components
├── lib/                  # Utility functions
├── clothes_tryon_dataset/ # Fashion dataset
└── recommendation_engine_backend.ipynb # AI backend
```

## Deployment Guide

### Option 1: Vercel (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Connect your GitHub repository
   - Vercel will automatically detect Next.js and deploy

3. **Environment Variables**
   - Add your environment variables in Vercel dashboard
   - Go to Project Settings → Environment Variables

### Option 2: Supabase + Vercel

1. **Set up Supabase Database**
   ```sql
   -- Create tables for favorites and recommender items
   CREATE TABLE favorites (
     id SERIAL PRIMARY KEY,
     user_id TEXT NOT NULL,
     item_id TEXT NOT NULL,
     created_at TIMESTAMP DEFAULT NOW()
   );

   CREATE TABLE recommender_items (
     id SERIAL PRIMARY KEY,
     user_id TEXT NOT NULL,
     item_id TEXT NOT NULL,
     created_at TIMESTAMP DEFAULT NOW()
   );
   ```

2. **Deploy Backend API**
   - Convert your Jupyter notebook to a FastAPI service
   - Deploy on Railway, Render, or similar platforms
   - Update frontend API calls to use the deployed backend

### Option 3: Full Stack with Railway

1. **Create Railway account**
   - Go to [railway.app](https://railway.app)
   - Connect your GitHub repository

2. **Deploy both frontend and backend**
   - Railway can handle both Next.js and Python services
   - Set up environment variables in Railway dashboard

## AI Recommendation System

The backend uses:
- **CLIP**: For image and text embeddings
- **FAISS**: For fast similarity search
- **Gemini LLM**: For intelligent analysis and recommendations

### FAISS Index Deployment

For production deployment, you have several options:

1. **Pre-computed Index** (Recommended)
   ```python
   # Save index during development
   faiss.write_index(catalog_index, "catalog_index.faiss")
   
   # Load in production
   catalog_index = faiss.read_index("catalog_index.faiss")
   ```

2. **Cloud Storage**
   - Upload the FAISS index to AWS S3, Google Cloud Storage, or similar
   - Download and load on server startup

3. **Database Storage**
   - Store embeddings in PostgreSQL with pgvector extension
   - Use Supabase's vector capabilities

## Environment Variables

Create a `.env.local` file:

```env
# Supabase Configuration
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# AI Backend URL (if deployed separately)
NEXT_PUBLIC_AI_API_URL=https://your-backend-url.com

# Optional: Analytics
NEXT_PUBLIC_GA_ID=your_google_analytics_id
```

## Customization

### Adding New Features

1. **New UI Components**: Add to `components/ui/`
2. **API Routes**: Create in `app/api/`
3. **Pages**: Add to `app/` directory

### Styling

- **Global Styles**: Edit `app/globals.css`
- **Component Styles**: Use Tailwind classes
- **Theme**: Modify `tailwind.config.js`

## Performance Optimization

1. **Image Optimization**
   - Use Next.js Image component for automatic optimization
   - Implement lazy loading for catalog items

2. **Caching**
   - Implement Redis for session storage
   - Use CDN for static assets

3. **Database**
   - Add indexes for frequently queried columns
   - Implement connection pooling

## Troubleshooting

### Common Issues

1. **Images not loading**
   - Check file paths in `clothes_tryon_dataset/`
   - Verify API routes are working

2. **Build errors**
   - Clear `.next` folder: `rm -rf .next`
   - Reinstall dependencies: `npm install`

3. **Deployment issues**
   - Check environment variables
   - Verify build logs in deployment platform

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For questions or issues:
- Create an issue on GitHub
- Check the documentation
- Contact the development team

---

**Happy coding! 🚀** 