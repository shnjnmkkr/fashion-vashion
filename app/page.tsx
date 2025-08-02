'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'
import { Heart, Star, ShoppingCart, Eye, Plus, MessageCircle, Send, X, Brain, Search, Sparkles, Loader2 } from 'lucide-react'

interface ClothingItem {
  id: string
  name: string
  clothImage: string
  personImage: string
  isFavorite: boolean
  isInRecommender: boolean
}

interface ThinkingStep {
  type: 'thinking' | 'clip' | 'gemini' | 'faiss' | 'analysis' | 'recommendation'
  title: string
  content: string
  status: 'pending' | 'processing' | 'completed' | 'error'
  details?: any
}

interface ChatMessage {
  type: 'user' | 'bot' | 'thinking'
  message: string
  thinkingSteps?: ThinkingStep[]
  recommendations?: any[]
  llmAnalysis?: any
  geminiAnalysis?: any
}

export default function FashionVashion() {
  const [clothingItems, setClothingItems] = useState<ClothingItem[]>([])
  const [favorites, setFavorites] = useState<string[]>([])
  const [recommenderItems, setRecommenderItems] = useState<string[]>([])
  const [selectedItem, setSelectedItem] = useState<ClothingItem | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [chatbotOpen, setChatbotOpen] = useState(false)
  const [favoritesOpen, setFavoritesOpen] = useState(false)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [userInput, setUserInput] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentImageIndex, setCurrentImageIndex] = useState(0)
  const [autoScroll, setAutoScroll] = useState(true)
  const [uploadedImages, setUploadedImages] = useState<File[]>([])
  const [currentUser, setCurrentUser] = useState<string | null>(null)
  const [isUserLoading, setIsUserLoading] = useState(false)

  // Auto-scroll effect
  useEffect(() => {
    if (selectedItem && autoScroll) {
      const interval = setInterval(() => {
        setCurrentImageIndex(prev => prev === 0 ? 1 : 0)
      }, 3000) // Switch every 3 seconds
      
      return () => clearInterval(interval)
    }
  }, [selectedItem, autoScroll])

  // Actual image filenames from your dataset
  const imageFiles = [
    '14602_00.jpg', '14603_00.jpg', '14604_00.jpg', '14605_00.jpg', '14606_00.jpg',
    '14608_00.jpg', '14609_00.jpg', '14610_00.jpg', '14611_00.jpg', '14613_00.jpg',
    '14614_00.jpg', '14617_00.jpg', '14618_00.jpg', '14619_00.jpg', '14620_00.jpg',
    '14622_00.jpg', '14623_00.jpg', '14624_00.jpg', '14625_00.jpg', '14626_00.jpg',
    '14628_00.jpg', '14630_00.jpg', '14631_00.jpg', '14632_00.jpg', '14633_00.jpg',
    '14634_00.jpg', '14635_00.jpg', '14636_00.jpg', '14637_00.jpg', '14638_00.jpg',
    '14640_00.jpg', '14641_00.jpg', '14642_00.jpg', '14643_00.jpg', '14644_00.jpg',
    '14646_00.jpg', '14647_00.jpg', '14648_00.jpg', '14649_00.jpg', '14650_00.jpg',
    '14652_00.jpg', '14653_00.jpg', '14654_00.jpg', '14656_00.jpg', '14657_00.jpg',
    '14658_00.jpg', '14659_00.jpg', '14661_00.jpg', '14662_00.jpg', '14663_00.jpg',
    '14664_00.jpg', '14665_00.jpg', '14666_00.jpg', '14667_00.jpg', '14668_00.jpg',
    '14670_00.jpg', '14672_00.jpg', '14677_00.jpg', '14678_00.jpg', '14680_00.jpg',
    '14681_00.jpg', '14682_00.jpg', '14683_00.jpg', '14684_00.jpg', '14373_00.jpg',
    '14374_00.jpg', '14376_00.jpg', '14377_00.jpg', '14378_00.jpg', '14379_00.jpg',
    '14380_00.jpg', '14381_00.jpg', '14382_00.jpg', '14385_00.jpg', '14386_00.jpg',
    '14387_00.jpg', '14388_00.jpg', '14389_00.jpg', '14390_00.jpg', '14391_00.jpg',
    '14392_00.jpg', '14394_00.jpg', '14395_00.jpg', '14396_00.jpg', '14398_00.jpg',
    '14399_00.jpg', '14400_00.jpg', '14401_00.jpg', '14402_00.jpg', '14404_00.jpg',
    '14405_00.jpg', '14406_00.jpg', '14408_00.jpg', '14409_00.jpg', '14410_00.jpg',
    '14411_00.jpg', '14412_00.jpg', '14413_00.jpg', '14414_00.jpg', '14416_00.jpg'
  ]

  // Simulate loading clothing data
  useEffect(() => {
    const loadClothingData = async () => {
      try {
        // Create clothing items using actual filenames
        const mockData: ClothingItem[] = imageFiles.map((filename, i) => ({
          id: `item-${i + 1}`,
          name: `Fashion Item ${i + 1}`,
          clothImage: `/api/clothes/${filename}`,
          personImage: `/api/images/${filename}`,
          isFavorite: false,
          isInRecommender: false
        }))
        
        setClothingItems(mockData)
        setIsLoading(false)
      } catch (error) {
        console.error('Error loading clothing data:', error)
        setIsLoading(false)
      }
    }

    loadClothingData()
  }, [])

  // Load user on component mount
  useEffect(() => {
    const initializeUser = async () => {
      setIsUserLoading(true)
      const userId = await loadUser()
      if (!userId) {
        // Create a new user if none exists
        await createUser('user@example.com')
      }
      setIsUserLoading(false)
    }

    initializeUser()
  }, [])

  const toggleFavorite = async (itemId: string) => {
    const item = clothingItems.find(item => item.id === itemId)
    if (!item) return

    if (favorites.includes(itemId)) {
      await removeFromFavorites(itemId)
    } else {
      await addToFavorites(itemId, item.name, item.clothImage, item.personImage)
    }
  }

  const toggleRecommender = async (itemId: string) => {
    const item = clothingItems.find(item => item.id === itemId)
    if (!item) return

    if (recommenderItems.includes(itemId)) {
      await removeFromRecommender(itemId)
    } else {
      await addToRecommender(itemId, item.name, item.clothImage, item.personImage)
    }
  }

  const handleItemClick = (item: ClothingItem) => {
    setSelectedItem(item)
    setCurrentImageIndex(0) // Reset to first image when opening
  }

  const nextImage = () => {
    setCurrentImageIndex(prev => prev === 0 ? 1 : 0)
  }

  const prevImage = () => {
    setCurrentImageIndex(prev => prev === 0 ? 1 : 0)
  }

  const simulateThinkingSteps = async (userMessage: string) => {
    // Add thinking message to chat
    setChatMessages(prev => [...prev, {
      type: 'thinking',
      message: 'Processing your request...',
      thinkingSteps: []
    }])

    // Check if backend is available
    if (!BACKEND_URL) {
      // Replace thinking message with offline response
      setChatMessages(prev => {
        const newMessages = [...prev]
        const lastMessage = newMessages[newMessages.length - 1]
        if (lastMessage.type === 'thinking') {
          newMessages[newMessages.length - 1] = {
            type: 'bot',
            message: `I'm currently offline. Please make sure the backend server is running on your local machine (localhost:8000) and you're connected to the internet.`,
            recommendations: []
          }
        }
        return newMessages
      })
      return
    }

    // Call backend API immediately without fake steps
    try {
      const formData = new FormData()
      formData.append('user_prompt', userMessage)
      formData.append('catalog_items', JSON.stringify(recommenderItems))
      formData.append('top_k', '5')

      // Add uploaded images
      uploadedImages.forEach((file, index) => {
        formData.append('files', file)
      })

      const response = await fetch(`${BACKEND_URL}/api/recommendations`, {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        const data = await response.json()
        
        // Replace thinking message with actual response
        setChatMessages(prev => {
          const newMessages = [...prev]
          const lastMessage = newMessages[newMessages.length - 1]
          if (lastMessage.type === 'thinking') {
            const botMessage: ChatMessage = {
              type: 'bot',
              message: `Based on your request "${userMessage}", here are my recommendations:`,
              recommendations: data.recommendations || [],
              llmAnalysis: data.llm_analysis,
              geminiAnalysis: data.gemini_analysis
            }
            newMessages[newMessages.length - 1] = botMessage
            
            // Save bot message to database
            saveChatMessage('bot', botMessage.message, botMessage.recommendations, botMessage.llmAnalysis, botMessage.geminiAnalysis)
          }
          return newMessages
        })
      } else {
        throw new Error('Backend request failed')
      }
    } catch (error) {
      console.error('Backend error:', error)
      
      // Replace thinking message with error response
      setChatMessages(prev => {
        const newMessages = [...prev]
        const lastMessage = newMessages[newMessages.length - 1]
        if (lastMessage.type === 'thinking') {
          newMessages[newMessages.length - 1] = {
            type: 'bot',
            message: `I'm having trouble connecting to the AI backend. Please make sure the backend server is running on port 8000 and you're connected to the internet.`,
            recommendations: []
          }
        }
        return newMessages
      })
    }
  }

  const sendMessage = async () => {
    if (!userInput.trim() || isProcessing) return

    const userMessage = userInput.trim()
    setUserInput('')
    setIsProcessing(true)

    // Add user message
    setChatMessages(prev => [...prev, { type: 'user', message: userMessage }])
    
    // Save user message to database
    await saveChatMessage('user', userMessage)

    // Simulate AI thinking process
    await simulateThinkingSteps(userMessage)
    
    setIsProcessing(false)
  }

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    setUploadedImages(prev => [...prev, ...files])
  }

  const removeUploadedImage = (index: number) => {
    setUploadedImages(prev => prev.filter((_, i) => i !== index))
  }

  // Database interaction functions
  const getBackendUrl = () => {
    // Check if we're in development
    if (process.env.NODE_ENV === 'development') {
      return 'http://localhost:8000'
    }
    
    // In production, try to use environment variable or fallback
    const envBackendUrl = process.env.NEXT_PUBLIC_BACKEND_URL
    if (envBackendUrl) {
      return envBackendUrl
    }
    
    // Fallback for when backend is not available
    return null
  }
  
  const BACKEND_URL = getBackendUrl()
  
  const createUser = async (email: string) => {
    if (!BACKEND_URL) {
      console.error('Backend URL not configured. Cannot create user.')
      return null
    }

    try {
      const formData = new FormData()
      formData.append('email', email)
      
      const response = await fetch(`${BACKEND_URL}/api/users`, {
        method: 'POST',
        body: formData
      })
      
      if (response.ok) {
        const user = await response.json()
        setCurrentUser(user.id)
        localStorage.setItem('fashion_vashion_user_id', user.id)
        return user.id
      } else {
        throw new Error('Failed to create user')
      }
    } catch (error) {
      console.error('Error creating user:', error)
      return null
    }
  }

  const loadUser = async () => {
    if (!BACKEND_URL) {
      console.error('Backend URL not configured. Cannot load user.')
      return null
    }

    const savedUserId = localStorage.getItem('fashion_vashion_user_id')
    if (savedUserId) {
      setCurrentUser(savedUserId)
      return savedUserId
    }
    return null
  }

  const addToFavorites = async (itemId: string, itemName: string, clothImageUrl: string, personImageUrl: string) => {
    if (!currentUser) {
      const userId = await createUser('user@example.com')
      if (!userId) return
    }

    if (!BACKEND_URL) {
      console.error('Backend URL not configured. Cannot add to favorites.')
      return
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/users/${currentUser}/favorites`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `item_id=${encodeURIComponent(itemId)}&item_name=${encodeURIComponent(itemName)}&cloth_image_url=${encodeURIComponent(clothImageUrl)}&person_image_url=${encodeURIComponent(personImageUrl)}`
      })
      
      if (response.ok) {
        setFavorites(prev => [...prev, itemId])
      }
    } catch (error) {
      console.error('Error adding to favorites:', error)
    }
  }

  const removeFromFavorites = async (itemId: string) => {
    if (!currentUser) return

    if (!BACKEND_URL) {
      console.error('Backend URL not configured. Cannot remove from favorites.')
      return
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/users/${currentUser}/favorites/${itemId}`, {
        method: 'DELETE'
      })
      
      if (response.ok) {
        setFavorites(prev => prev.filter(id => id !== itemId))
      }
    } catch (error) {
      console.error('Error removing from favorites:', error)
    }
  }

  const addToRecommender = async (itemId: string, itemName: string, clothImageUrl: string, personImageUrl: string) => {
    if (!currentUser) {
      const userId = await createUser('user@example.com')
      if (!userId) return
    }

    if (!BACKEND_URL) {
      console.error('Backend URL not configured. Cannot add to recommender.')
      return
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/users/${currentUser}/recommender-items`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `item_id=${encodeURIComponent(itemId)}&item_name=${encodeURIComponent(itemName)}&cloth_image_url=${encodeURIComponent(clothImageUrl)}&person_image_url=${encodeURIComponent(personImageUrl)}`
      })
      
      if (response.ok) {
        setRecommenderItems(prev => [...prev, itemId])
      }
    } catch (error) {
      console.error('Error adding to recommender:', error)
    }
  }

  const removeFromRecommender = async (itemId: string) => {
    if (!currentUser) return

    if (!BACKEND_URL) {
      console.error('Backend URL not configured. Cannot remove from recommender.')
      return
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/users/${currentUser}/recommender-items/${itemId}`, {
        method: 'DELETE'
      })
      
      if (response.ok) {
        setRecommenderItems(prev => prev.filter(id => id !== itemId))
      }
    } catch (error) {
      console.error('Error removing from recommender:', error)
    }
  }

  const saveChatMessage = async (messageType: string, messageContent: string, recommendations?: any, llmAnalysis?: any, geminiAnalysis?: any) => {
    if (!currentUser) return

    if (!BACKEND_URL) {
      console.error('Backend URL not configured. Cannot save chat message.')
      return
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/users/${currentUser}/chat-history`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `message_type=${encodeURIComponent(messageType)}&message_content=${encodeURIComponent(messageContent)}&recommendations=${encodeURIComponent(JSON.stringify(recommendations || {}))}&llm_analysis=${encodeURIComponent(JSON.stringify(llmAnalysis || {}))}&gemini_analysis=${encodeURIComponent(JSON.stringify(geminiAnalysis || {}))}`
      })
      
      if (!response.ok) {
        console.error('Failed to save chat message')
      }
    } catch (error) {
      console.error('Error saving chat message:', error)
    }
  }

  const getStepIcon = (type: string) => {
    switch (type) {
      case 'thinking': return <Brain className="w-4 h-4" />
      case 'gemini': return <Sparkles className="w-4 h-4" />
      case 'clip': return <Search className="w-4 h-4" />
      case 'faiss': return <Search className="w-4 h-4" />
      case 'analysis': return <Brain className="w-4 h-4" />
      case 'recommendation': return <Sparkles className="w-4 h-4" />
      default: return <Brain className="w-4 h-4" />
    }
  }

  const getStepColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-400'
      case 'processing': return 'text-yellow-400'
      case 'error': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-purple-400 mx-auto"></div>
          <p className="mt-4 text-lg text-white">Loading fashion catalog...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black">
      {/* Header */}
      <header className="bg-black/40 backdrop-blur-sm border-b border-purple-500/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <h1 className="text-3xl font-bold text-white">Fashion-Vashion</h1>
            </div>
            <div className="flex items-center space-x-4">
              <Button 
                variant="outline" 
                size="sm" 
                className="border-purple-500 text-black bg-white hover:bg-purple-500 hover:text-white"
                onClick={() => setFavoritesOpen(true)}
              >
                <Heart className="w-4 h-4 mr-2" />
                Favorites ({favorites.length})
              </Button>
              <Button 
                variant="outline" 
                size="sm" 
                className="border-purple-500 text-black bg-white hover:bg-purple-500 hover:text-white"
                onClick={() => setChatbotOpen(true)}
              >
                <MessageCircle className="w-4 h-4 mr-2" />
                AI Recommender
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-4 gap-8">
          {clothingItems.map((item) => (
            <HoverCard key={item.id}>
              <HoverCardTrigger asChild>
                <Card 
                  className="clothing-card group cursor-pointer overflow-hidden bg-white/10 backdrop-blur-sm border-purple-500/20 hover:bg-white/20 transition-all duration-300 w-full"
                  onClick={() => handleItemClick(item)}
                >
                  <div className="relative">
                    <div className="w-full h-56 flex items-center justify-center pt-8">
                      <div className="w-full h-full flex items-center justify-center">
                        <img
                          src={item.clothImage}
                          alt={item.name}
                          className="max-h-48 max-w-full object-contain transition-transform duration-300 group-hover:scale-105"
                        />
                      </div>
                    </div>
                    <div className="absolute inset-0 w-full h-full flex items-center justify-center opacity-0 hover-image pt-8">
                      <div className="w-full h-full flex items-center justify-center">
                        <img
                          src={item.personImage}
                          alt={`${item.name} on person`}
                          className="max-h-48 max-w-full object-contain"
                        />
                      </div>
                    </div>
                  </div>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-xl text-white">{item.name}</CardTitle>
                  </CardHeader>
                  <CardFooter className="pt-0 pb-4">
                    <div className="flex gap-3 w-full items-center justify-center">
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="flex-1 border-purple-500 text-black bg-white hover:bg-purple-500 hover:text-white text-sm h-10 px-3"
                        onClick={(e) => {
                          e.stopPropagation()
                          toggleRecommender(item.id)
                        }}
                      >
                        <Plus className="w-4 h-4 mr-2" />
                        {recommenderItems.includes(item.id) ? 'Remove' : 'Add to Recommender'}
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        className="w-10 h-10 border-purple-500 text-black bg-white hover:bg-purple-500 hover:text-white px-0"
                        onClick={(e) => {
                          e.stopPropagation()
                          toggleFavorite(item.id)
                        }}
                      >
                        <Heart 
                          className={`w-4 h-4 ${favorites.includes(item.id) ? 'fill-red-500 text-red-500' : ''}`} 
                        />
                      </Button>
                    </div>
                  </CardFooter>
                </Card>
              </HoverCardTrigger>
              <HoverCardContent className="w-96 bg-black/90 backdrop-blur-sm border-purple-500/20">
                <div className="space-y-3">
                  <h4 className="font-semibold text-white text-lg">{item.name}</h4>
                  <div className="w-full h-56 flex items-center justify-center pt-8">
                    <div className="w-full h-full flex items-center justify-center">
                      <img
                        src={item.personImage}
                        alt={`${item.name} on person`}
                        className="max-h-48 max-w-full object-contain"
                      />
                    </div>
                  </div>
                  <p className="text-sm text-gray-300">
                    Hover to see how this item looks when worn.
                  </p>
                </div>
              </HoverCardContent>
            </HoverCard>
          ))}
        </div>
      </main>

      {/* Item Detail Dialog */}
      <Dialog open={!!selectedItem} onOpenChange={() => setSelectedItem(null)}>
        <DialogContent className="max-w-4xl bg-black/90 backdrop-blur-sm border-purple-500/20">
          <DialogHeader>
            <DialogTitle className="text-white">{selectedItem?.name}</DialogTitle>
          </DialogHeader>
          {selectedItem && (
            <>
              <div className="relative">
                {/* Image Container */}
                <div className="relative w-full h-96 overflow-hidden rounded-lg">
                  <div className="flex transition-transform duration-300 ease-in-out" style={{ transform: `translateX(-${currentImageIndex * 100}%)` }}>
                    <div className="w-full h-96 flex items-center justify-center flex-shrink-0 pt-12">
                      <div className="w-full h-full flex items-center justify-center">
                        <img
                          src={selectedItem.clothImage}
                          alt={selectedItem.name}
                          className="max-h-80 max-w-full object-contain"
                        />
                      </div>
                    </div>
                    <div className="w-full h-96 flex items-center justify-center flex-shrink-0 pt-12">
                      <div className="w-full h-full flex items-center justify-center">
                        <img
                          src={selectedItem.personImage}
                          alt={`${selectedItem.name} on person`}
                          className="max-h-80 max-w-full object-contain"
                        />
                      </div>
                    </div>
                  </div>
                  
                  {/* Navigation Arrows */}
                  <button 
                    className="absolute left-2 top-1/2 transform -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white p-2 rounded-full"
                    onClick={() => {
                      setAutoScroll(false)
                      prevImage()
                    }}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                  </button>
                  <button 
                    className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white p-2 rounded-full"
                    onClick={() => {
                      setAutoScroll(false)
                      nextImage()
                    }}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </button>
                  
                  {/* Dots Indicator */}
                  <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-2">
                    <div 
                      className={`w-2 h-2 rounded-full cursor-pointer ${currentImageIndex === 0 ? 'bg-white' : 'bg-white/50'}`} 
                      onClick={() => {
                        setAutoScroll(false)
                        setCurrentImageIndex(0)
                      }}
                    ></div>
                    <div 
                      className={`w-2 h-2 rounded-full cursor-pointer ${currentImageIndex === 1 ? 'bg-white' : 'bg-white/50'}`} 
                      onClick={() => {
                        setAutoScroll(false)
                        setCurrentImageIndex(1)
                      }}
                    ></div>
                  </div>
                  
                  {/* Auto-scroll indicator */}
                  <div className="absolute top-4 right-4">
                    <button
                      className={`px-2 py-1 text-xs rounded ${autoScroll ? 'bg-green-500 text-white' : 'bg-gray-500 text-white'}`}
                      onClick={() => setAutoScroll(!autoScroll)}
                    >
                      {autoScroll ? 'Auto ON' : 'Auto OFF'}
                    </button>
                  </div>
                </div>
              </div>
              
              <div className="flex justify-between items-center pt-4">
                <div>
                  {/* Removed category display */}
                </div>
                <div className="flex space-x-2">
                  <Button
                    variant="outline"
                    className="border-purple-500 text-purple-300 hover:bg-purple-500 hover:text-white"
                    onClick={() => toggleFavorite(selectedItem.id)}
                  >
                    <Heart className={`w-4 h-4 mr-2 ${favorites.includes(selectedItem.id) ? 'fill-red-500 text-red-500' : ''}`} />
                    {favorites.includes(selectedItem.id) ? 'Remove from Favorites' : 'Add to Favorites'}
                  </Button>
                  <Button
                    variant="outline"
                    className="border-purple-500 text-black bg-white hover:bg-purple-500 hover:text-white"
                    onClick={() => toggleRecommender(selectedItem.id)}
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    {recommenderItems.includes(selectedItem.id) ? 'Remove from Recommender' : 'Add to Recommender'}
                  </Button>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* Enhanced Chatbot Dialog */}
      <Dialog open={chatbotOpen} onOpenChange={setChatbotOpen}>
        <DialogContent className="max-w-5xl h-[800px] bg-black/90 backdrop-blur-sm border-purple-500/20">
          <DialogHeader>
            <DialogTitle className="text-white flex items-center justify-between">
              <span className="flex items-center">
                <Brain className="w-5 h-5 mr-2" />
                AI Fashion Recommender
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setChatbotOpen(false)}
                className="text-purple-300 hover:text-white"
              >
                <X className="w-4 h-4" />
              </Button>
            </DialogTitle>
          </DialogHeader>
          
          <div className="flex flex-col h-full">
            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto space-y-4 mb-4 p-4 bg-black/20 rounded-lg">
              {chatMessages.length === 0 ? (
                <div className="text-center text-purple-300">
                  <Brain className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <p>Ask me about fashion recommendations!</p>
                  <p className="text-sm opacity-70 mt-1">I'll show you my thinking process step by step</p>
                  
                  {/* Show recommender items if any */}
                  {recommenderItems.length > 0 && (
                    <div className="mt-4 p-3 bg-purple-900/20 rounded-lg border border-purple-500/20">
                      <p className="text-sm font-semibold text-purple-300 mb-2">Items in Recommender:</p>
                      <div className="grid grid-cols-2 gap-2">
                        {clothingItems
                          .filter(item => recommenderItems.includes(item.id))
                          .map((item) => (
                            <div key={item.id} className="flex items-center space-x-2 p-2 bg-black/30 rounded">
                              <img 
                                src={item.clothImage} 
                                alt={item.name} 
                                className="w-8 h-8 object-contain"
                              />
                              <span className="text-xs truncate">{item.name}</span>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Show uploaded images if any */}
                  {uploadedImages.length > 0 && (
                    <div className="mt-4 p-3 bg-blue-900/20 rounded-lg border border-blue-500/20">
                      <p className="text-sm font-semibold text-blue-300 mb-2">Uploaded Images:</p>
                      <div className="grid grid-cols-2 gap-2">
                        {uploadedImages.map((file, index) => (
                          <div key={index} className="relative">
                            <img 
                              src={URL.createObjectURL(file)} 
                              alt={`Uploaded ${index + 1}`} 
                              className="w-full h-16 object-cover rounded"
                            />
                            <button
                              onClick={() => removeUploadedImage(index)}
                              className="absolute -top-1 -right-1 bg-red-500 text-white rounded-full w-5 h-5 text-xs"
                            >
                              ×
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                chatMessages.map((msg, index) => (
                  <div key={index} className="space-y-3">
                    {/* User Message */}
                    {msg.type === 'user' && (
                      <div className="flex justify-end">
                        <div className="max-w-xs px-4 py-2 rounded-lg bg-purple-600 text-white">
                          {msg.message}
                        </div>
                      </div>
                    )}
                    
                    {/* Bot Message */}
                    {msg.type === 'bot' && (
                      <div className="flex justify-start">
                        <div className="max-w-2xl px-4 py-3 rounded-lg bg-white/10 text-white border border-purple-500/20">
                          <p className="mb-3">{msg.message}</p>
                          
                          {msg.recommendations && msg.recommendations.length > 0 && (
                            <div className="space-y-3">
                              <p className="text-sm font-semibold text-purple-300">Top Recommendations:</p>
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-60 overflow-y-auto">
                                {msg.recommendations.map((rec, i) => (
                                  <div key={rec.filename || i} className="flex items-center space-x-3 p-3 bg-black/30 rounded-lg border border-purple-500/20">
                                    <div className="w-16 h-16 flex items-center justify-center">
                                      <img 
                                        src={`http://localhost:8000/api/catalog/${rec.filename}`} 
                                        alt={rec.filename} 
                                        className="w-auto h-full object-contain"
                                      />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                      <p className="text-sm font-medium truncate">{rec.filename}</p>
                                      <p className="text-xs text-purple-300">
                                        Match: {((rec.similarity_score || 0) * 100).toFixed(0)}%
                                      </p>
                                      <p className="text-xs text-gray-400">
                                        Rank: #{rec.rank || i + 1}
                                      </p>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          {msg.llmAnalysis && (
                            <div className="mt-3 p-3 bg-purple-900/20 rounded-lg border border-purple-500/20">
                              <p className="text-sm font-semibold text-purple-300 mb-2">LLM Analysis:</p>
                              <div className="text-xs text-gray-300 space-y-1">
                                {typeof msg.llmAnalysis === 'object' ? (
                                  <div>
                                    <p><strong>Strategy:</strong> {msg.llmAnalysis.search_strategy}</p>
                                    <p><strong>Intent:</strong> {msg.llmAnalysis.user_intent}</p>
                                    <p><strong>Confidence:</strong> {msg.llmAnalysis.confidence}</p>
                                  </div>
                                ) : (
                                  <p>{msg.llmAnalysis}</p>
                                )}
                              </div>
                            </div>
                          )}
                          
                          {msg.geminiAnalysis && (
                            <div className="mt-3 p-3 bg-blue-900/20 rounded-lg border border-blue-500/20">
                              <p className="text-sm font-semibold text-blue-300 mb-2">Gemini Analysis:</p>
                              <div className="text-xs text-gray-300">
                                {typeof msg.geminiAnalysis === 'object' ? (
                                  <div>
                                    {msg.geminiAnalysis.analysis && (
                                      <p><strong>Analysis:</strong> {JSON.stringify(msg.geminiAnalysis.analysis)}</p>
                                    )}
                                    {msg.geminiAnalysis.styling_tips && (
                                      <div>
                                        <p><strong>Styling Tips:</strong></p>
                                        <ul className="list-disc list-inside ml-2">
                                          {msg.geminiAnalysis.styling_tips.map((tip: string, i: number) => (
                                            <li key={i}>{tip}</li>
                                          ))}
                                        </ul>
                                      </div>
                                    )}
                                  </div>
                                ) : (
                                  <p>{msg.geminiAnalysis}</p>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {/* Thinking Process */}
                    {msg.type === 'thinking' && msg.thinkingSteps && (
                      <div className="flex justify-start">
                        <div className="max-w-2xl px-4 py-3 rounded-lg bg-gradient-to-r from-purple-900/50 to-blue-900/50 text-white border border-purple-500/20">
                          <div className="flex items-center mb-3">
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            <span className="text-sm font-medium">AI Thinking Process</span>
                          </div>
                          <div className="space-y-2">
                            {msg.thinkingSteps.map((step, stepIndex) => (
                              <div key={stepIndex} className="flex items-start space-x-2">
                                <div className={`mt-1 ${getStepColor(step.status)}`}>
                                  {step.status === 'processing' ? (
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                  ) : step.status === 'completed' ? (
                                    <div className="w-3 h-3 bg-green-400 rounded-full" />
                                  ) : (
                                    <div className="w-3 h-3 bg-gray-400 rounded-full" />
                                  )}
                                </div>
                                <div className="flex-1">
                                  <div className="flex items-center space-x-2">
                                    {getStepIcon(step.type)}
                                    <span className={`text-sm font-medium ${getStepColor(step.status)}`}>
                                      {step.title}
                                    </span>
                                  </div>
                                  <p className="text-xs text-gray-300 mt-1">{step.content}</p>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
            
            {/* Input Area */}
            <div className="flex space-x-2">
              <input
                type="text"
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                placeholder="Ask about fashion recommendations..."
                className="flex-1 px-4 py-2 bg-white/10 border border-purple-500/20 rounded-lg text-white placeholder-purple-300 focus:outline-none focus:border-purple-500"
                disabled={isProcessing}
              />
              
              {/* Image Upload Button */}
              <label className="cursor-pointer">
                <input
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                  disabled={isProcessing}
                />
                <Button
                  variant="outline"
                  className="border-purple-500 text-purple-300 hover:bg-purple-500 hover:text-white"
                  disabled={isProcessing}
                >
                  📷
                </Button>
              </label>
              
              <Button
                onClick={sendMessage}
                className="bg-purple-600 hover:bg-purple-700"
                disabled={isProcessing}
              >
                {isProcessing ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Favorites Dialog */}
      <Dialog open={favoritesOpen} onOpenChange={setFavoritesOpen}>
        <DialogContent className="max-w-6xl h-[600px] bg-black/90 backdrop-blur-sm border-purple-500/20">
          <DialogHeader>
            <DialogTitle className="text-white flex items-center justify-between">
              <span className="flex items-center">
                <Heart className="w-5 h-5 mr-2" />
                My Favorites ({favorites.length})
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setFavoritesOpen(false)}
                className="text-purple-300 hover:text-white"
              >
                <X className="w-4 h-4" />
              </Button>
            </DialogTitle>
          </DialogHeader>
          
          <div className="flex-1 overflow-y-auto">
            {favorites.length === 0 ? (
              <div className="text-center text-purple-300 py-8">
                <Heart className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>No favorites yet!</p>
                <p className="text-sm opacity-70 mt-1">Add items to your favorites to see them here</p>
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                {clothingItems
                  .filter(item => favorites.includes(item.id))
                  .map((item) => (
                    <Card 
                      key={item.id}
                      className="clothing-card group cursor-pointer overflow-hidden bg-white/10 backdrop-blur-sm border-purple-500/20 hover:bg-white/20 transition-all duration-300"
                      onClick={() => handleItemClick(item)}
                    >
                      <div className="relative">
                        <div className="w-full h-40 flex items-center justify-center pt-6">
                          <div className="w-full h-full flex items-center justify-center">
                            <img
                              src={item.clothImage}
                              alt={item.name}
                              className="max-h-32 max-w-full object-contain transition-transform duration-300 group-hover:scale-105"
                            />
                          </div>
                        </div>
                        <div className="absolute inset-0 w-full h-full flex items-center justify-center opacity-0 hover-image pt-6">
                          <div className="w-full h-full flex items-center justify-center">
                            <img
                              src={item.personImage}
                              alt={`${item.name} on person`}
                              className="max-h-32 max-w-full object-contain"
                            />
                          </div>
                        </div>
                      </div>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm text-white">{item.name}</CardTitle>
                      </CardHeader>
                      <CardFooter className="pt-0 pb-3">
                        <div className="flex gap-2 w-full items-center justify-center">
                          <Button 
                            variant="outline" 
                            size="sm" 
                            className="flex-1 border-purple-500 text-black bg-white hover:bg-purple-500 hover:text-white text-xs h-8 px-2"
                            onClick={(e) => {
                              e.stopPropagation()
                              toggleRecommender(item.id)
                            }}
                          >
                            <Plus className="w-3 h-3 mr-1" />
                            {recommenderItems.includes(item.id) ? 'Remove' : 'Add to Recommender'}
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            className="w-8 h-8 border-purple-500 text-black bg-white hover:bg-purple-500 hover:text-white px-0"
                            onClick={(e) => {
                              e.stopPropagation()
                              toggleFavorite(item.id)
                            }}
                          >
                            <Heart 
                              className={`w-3 h-3 fill-red-500 text-red-500`} 
                            />
                          </Button>
                        </div>
                      </CardFooter>
                    </Card>
                  ))}
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
} 