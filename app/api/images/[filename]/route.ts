import { NextRequest, NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'

export async function GET(
  request: NextRequest,
  { params }: { params: { filename: string } }
) {
  try {
    const filename = params.filename
    const imagePath = path.join(process.cwd(), 'clothes_tryon_dataset', 'train', 'image', filename)
    
    const imageBuffer = await fs.readFile(imagePath)
    
    return new NextResponse(imageBuffer, {
      headers: {
        'Content-Type': 'image/jpeg',
        'Cache-Control': 'public, max-age=31536000, immutable',
      },
    })
  } catch (error) {
    console.error('Error serving person image:', error)
    return new NextResponse('Image not found', { status: 404 })
  }
} 