import React, { useState, useEffect, useRef } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';
import { 
  VirtualDeskInterface, 
  Sidebar, 
  LoginForm, 
  AdminDashboard,
  ServiceCard,
  BookingInterface,
  LanguageSelector
} from './components';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Virtual Front Desk Services Configuration
const availableServices = [
  {
    id: 'health-card-renewal',
    name: { en: 'Health Card Renewal', ar: 'تجديد البطاقة الصحية' },
    category: 'government',
    description: { en: 'Renew your health insurance card', ar: 'تجديد بطاقة التأمين الصحي' },
    estimatedTime: 30,
    requiresAppointment: true,
    icon: '🏥',
    workingHours: { start: '08:00', end: '16:00' },
    availableDays: ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
  },
  {
    id: 'id-card-replacement',
    name: { en: 'ID Card Replacement', ar: 'استبدال بطاقة الهوية' },
    category: 'government',
    description: { en: 'Replace lost or damaged ID card', ar: 'استبدال بطاقة الهوية المفقودة أو التالفة' },
    estimatedTime: 45,
    requiresAppointment: true,
    icon: '🆔',
    workingHours: { start: '08:00', end: '15:00' },
    availableDays: ['sunday', 'tuesday', 'thursday']
  },
  {
    id: 'medical-consultation',
    name: { en: 'Medical Consultation', ar: 'استشارة طبية' },
    category: 'medical',
    description: { en: 'Book appointment with doctor', ar: 'حجز موعد مع الطبيب' },
    estimatedTime: 20,
    requiresAppointment: true,
    icon: '👩‍⚕️',
    workingHours: { start: '09:00', end: '17:00' },
    availableDays: ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday']
  },
  {
    id: 'student-enrollment',
    name: { en: 'Student Enrollment', ar: 'تسجيل الطلاب' },
    category: 'education',
    description: { en: 'Enroll in courses and programs', ar: 'التسجيل في الدورات والبرامج' },
    estimatedTime: 60,
    requiresAppointment: true,
    icon: '🎓',
    workingHours: { start: '08:00', end: '14:00' },
    availableDays: ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday']
  },
  {
    id: 'general-inquiry',
    name: { en: 'General Inquiry', ar: 'استفسار عام' },
    category: 'general',
    description: { en: 'Ask any question or get information', ar: 'اطرح أي سؤال أو احصل على معلومات' },
    estimatedTime: 10,
    requiresAppointment: false,
    icon: '💬',
    workingHours: { start: '00:00', end: '23:59' },
    availableDays: ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']
  }
];

// Mock conversation data for virtual front desk
const mockConversations = [
  {
    id: '1',
    title: { en: 'Health Card Renewal', ar: 'تجديد البطاقة الصحية' },
    type: 'service_request',
    status: 'completed',
    service: 'health-card-renewal',
    language: 'en',
    messages: [
      {
        id: '1',
        role: 'user',
        content: 'I need to renew my health card',
        timestamp: new Date('2025-01-15T10:30:00'),
        language: 'en'
      },
      {
        id: '2',
        role: 'assistant',
        content: 'I can help you with your health card renewal. I\'ll need to book an appointment for you.\n\n📋 **Required Documents:**\n• Current health card\n• Valid ID\n• Recent photo\n\n⏰ **Estimated Time:** 30 minutes\n\nWould you like to schedule an appointment?',
        timestamp: new Date('2025-01-15T10:30:15'),
        intent: 'health_card_renewal',
        confidence: 0.95
      }
    ],
    createdAt: new Date('2025-01-15T10:30:00'),
    appointment: {
      id: 'apt_001',
      date: '2025-01-20',
      time: '10:00',
      status: 'confirmed'
    }
  },
  {
    id: '2',
    title: { en: 'Medical Consultation', ar: 'استشارة طبية' },
    type: 'service_request',
    status: 'pending',
    service: 'medical-consultation',
    language: 'ar',
    messages: [
      {
        id: '3',
        role: 'user',
        content: 'أريد حجز موعد مع طبيب',
        timestamp: new Date('2025-01-14T14:20:00'),
        language: 'ar'
      },
      {
        id: '4',
        role: 'assistant',
        content: 'مرحبا! يمكنني مساعدتك في حجز موعد مع الطبيب.\n\n🏥 **تفاصيل الاستشارة:**\n• وقت الاستشارة: 20 دقيقة\n• متوفر: الأحد إلى الخميس\n• ساعات العمل: 9:00 صباحا - 5:00 مساء\n\nما هو التخصص المطلوب؟',
        timestamp: new Date('2025-01-14T14:20:18'),
        intent: 'medical_consultation',
        confidence: 0.92,
        language: 'ar'
      }
    ],
    createdAt: new Date('2025-01-14T14:20:00')
  }
];

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [conversations, setConversations] = useState(mockConversations);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [currentLanguage, setCurrentLanguage] = useState('en');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [currentView, setCurrentView] = useState('chat'); // 'chat', 'admin', 'services'
  const [sessionData, setSessionData] = useState({
    step: 'greeting', // 'greeting', 'intent_detection', 'service_selection', 'booking', 'confirmation'
    selectedService: null,
    collectedInfo: {},
    intent: null,
    confidence: 0
  });

  useEffect(() => {
    // Check authentication
    const savedUser = localStorage.getItem('mind14_user');
    if (savedUser) {
      setCurrentUser(JSON.parse(savedUser));
      setIsAuthenticated(true);
    }

    // Load language preference
    const savedLanguage = localStorage.getItem('mind14_language');
    if (savedLanguage) {
      setCurrentLanguage(savedLanguage);
    }
  }, []);

  const handleLogin = (credentials) => {
    const user = {
      id: '1',
      name: credentials.fullName || credentials.email.split('@')[0],
      email: credentials.email,
      avatar: 'https://images.pexels.com/photos/7658539/pexels-photo-7658539.jpeg',
      phoneNumber: credentials.phoneNumber || null,
      role: credentials.email.includes('admin') ? 'admin' : 'user'
    };
    setCurrentUser(user);
    setIsAuthenticated(true);
    localStorage.setItem('mind14_user', JSON.stringify(user));
  };

  const handleLogout = () => {
    setCurrentUser(null);
    setIsAuthenticated(false);
    localStorage.removeItem('mind14_user');
    setActiveConversationId(null);
    setSessionData({
      step: 'greeting',
      selectedService: null,
      collectedInfo: {},
      intent: null,
      confidence: 0
    });
  };

  const handleLanguageChange = (language) => {
    setCurrentLanguage(language);
    localStorage.setItem('mind14_language', language);
  };

  const createNewConversation = () => {
    const newConversation = {
      id: Date.now().toString(),
      title: { en: 'New Chat', ar: 'محادثة جديدة' },
      type: 'general_inquiry',
      status: 'active',
      service: null,
      language: currentLanguage,
      messages: [],
      createdAt: new Date()
    };
    setConversations(prev => [newConversation, ...prev]);
    setActiveConversationId(newConversation.id);
    setSessionData({
      step: 'greeting',
      selectedService: null,
      collectedInfo: {},
      intent: null,
      confidence: 0
    });
  };

  const sendMessage = async (content, attachments = []) => {
    if (!activeConversationId && !content.trim()) {
      createNewConversation();
      return;
    }

    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content,
      attachments,
      timestamp: new Date(),
      language: currentLanguage
    };

    // Add user message to local state immediately
    setConversations(prev => prev.map(conv => 
      conv.id === activeConversationId 
        ? { ...conv, messages: [...conv.messages, userMessage] }
        : conv
    ));

    try {
      // Call the real backend API
      const response = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: content,
          conversation_id: activeConversationId,
          language: currentLanguage,
          attachments: attachments.map(file => file.name || file)
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const aiResponse = await response.json();
      
      const aiMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: aiResponse.message,
        timestamp: new Date(),
        intent: aiResponse.intent,
        confidence: aiResponse.confidence,
        language: currentLanguage
      };

      // Update conversation with AI response
      setConversations(prev => prev.map(conv => 
        conv.id === aiResponse.conversation_id
          ? { 
              ...conv, 
              messages: conv.id === activeConversationId 
                ? [...conv.messages, aiMessage]
                : [...conv.messages, userMessage, aiMessage],
              type: aiResponse.intent || conv.type,
              service: aiResponse.session_data?.selected_service || conv.service
            }
          : conv
      ));

      // Update session data
      setSessionData(aiResponse.session_data);

      // Update active conversation ID if it was a new conversation
      if (!activeConversationId) {
        setActiveConversationId(aiResponse.conversation_id);
      }

      // Update conversation title if it's the first message
      if (aiResponse.session_data?.step === 'service_selection' || aiResponse.intent !== 'general_inquiry') {
        const titleUpdate = aiResponse.intent === 'health_card_renewal' 
          ? { en: 'Health Card Renewal', ar: 'تجديد البطاقة الصحية' }
          : aiResponse.intent === 'id_card_replacement'
          ? { en: 'ID Card Replacement', ar: 'استبدال بطاقة الهوية' }
          : aiResponse.intent === 'medical_consultation'
          ? { en: 'Medical Consultation', ar: 'استشارة طبية' }
          : aiResponse.intent === 'student_enrollment'
          ? { en: 'Student Enrollment', ar: 'تسجيل الطلاب' }
          : { en: content.slice(0, 30) + (content.length > 30 ? '...' : ''), ar: content.slice(0, 30) + (content.length > 30 ? '...' : '') };

        setConversations(prev => prev.map(conv => 
          conv.id === aiResponse.conversation_id
            ? { ...conv, title: titleUpdate }
            : conv
        ));
      }

    } catch (error) {
      console.error('Error calling backend API:', error);
      
      // Fallback to mock response if backend fails
      const fallbackMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: currentLanguage === 'ar' 
          ? 'عذراً، أواجه مشكلة في الاتصال. يرجى المحاولة مرة أخرى.'
          : 'Sorry, I\'m having connection issues. Please try again.',
        timestamp: new Date(),
        language: currentLanguage
      };

      setConversations(prev => prev.map(conv => 
        conv.id === activeConversationId 
          ? { ...conv, messages: [...conv.messages, fallbackMessage] }
          : conv
      ));
    }
  };

  // Load conversations from backend on startup
  useEffect(() => {
    const loadConversations = async () => {
      try {
        const response = await fetch(`${API}/conversations?user_id=demo_user`);
        if (response.ok) {
          const backendConversations = await response.json();
          if (backendConversations.length > 0) {
            setConversations(backendConversations);
          }
        }
      } catch (error) {
        console.log('Using mock conversations, backend not available:', error);
        // Keep mock conversations as fallback
      }
    };

    if (isAuthenticated) {
      loadConversations();
    }
  }, [isAuthenticated]);

  // Trigger n8n webhook for booking automation
  const triggerN8nWebhook = async (bookingData) => {
    try {
      // This would normally send to your n8n webhook
      console.log('Triggering n8n webhook with booking data:', bookingData);
      
      // Simulate n8n webhook call
      const webhookResponse = await fetch('/api/n8n/book-appointment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(bookingData)
      });
      
      if (webhookResponse.ok) {
        console.log('n8n webhook triggered successfully');
      }
    } catch (error) {
      console.error('Failed to trigger n8n webhook:', error);
    }
  };

  const deleteConversation = (id) => {
    setConversations(prev => prev.filter(conv => conv.id !== id));
    if (activeConversationId === id) {
      setActiveConversationId(null);
    }
  };

  if (!isAuthenticated) {
    return (
      <BrowserRouter>
        <div className="min-h-screen bg-gray-900">
          <LoginForm onLogin={handleLogin} />
        </div>
      </BrowserRouter>
    );
  }

  return (
    <BrowserRouter>
      <div className="App min-h-screen bg-gray-900 flex">
        <Sidebar
          conversations={conversations}
          activeConversationId={activeConversationId}
          onSelectConversation={setActiveConversationId}
          onNewConversation={createNewConversation}
          onDeleteConversation={deleteConversation}
          onLogout={handleLogout}
          currentUser={currentUser}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
          currentView={currentView}
          onViewChange={setCurrentView}
          language={currentLanguage}
        />
        
        <div className="flex-1 flex flex-col">
          <Routes>
            <Route 
              path="/" 
              element={
                currentView === 'admin' && currentUser?.role === 'admin' ? (
                  <AdminDashboard
                    conversations={conversations}
                    services={availableServices}
                    language={currentLanguage}
                    onLanguageChange={handleLanguageChange}
                  />
                ) : (
                  <VirtualDeskInterface
                    conversation={conversations.find(conv => conv.id === activeConversationId)}
                    onSendMessage={sendMessage}
                    availableServices={availableServices}
                    currentUser={currentUser}
                    sidebarCollapsed={sidebarCollapsed}
                    language={currentLanguage}
                    onLanguageChange={handleLanguageChange}
                    sessionData={sessionData}
                  />
                )
              } 
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;