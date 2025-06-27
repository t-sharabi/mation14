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
    if (!activeConversationId) {
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

    // Add user message
    setConversations(prev => prev.map(conv => 
      conv.id === activeConversationId 
        ? { ...conv, messages: [...conv.messages, userMessage] }
        : conv
    ));

    // Process with AI (simulate Mistral intent detection)
    setTimeout(async () => {
      const response = await processWithMistral(content, sessionData, currentLanguage);
      
      const aiMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.message,
        timestamp: new Date(),
        intent: response.intent,
        confidence: response.confidence,
        language: currentLanguage,
        actions: response.actions || []
      };

      setConversations(prev => prev.map(conv => 
        conv.id === activeConversationId 
          ? { 
              ...conv, 
              messages: [...conv.messages, aiMessage],
              type: response.intent || conv.type,
              service: response.service || conv.service
            }
          : conv
      ));

      // Update session data
      setSessionData(response.sessionData);

      // Update conversation title
      if (response.title) {
        setConversations(prev => prev.map(conv => 
          conv.id === activeConversationId
            ? { ...conv, title: response.title }
            : conv
        ));
      }

      // Trigger n8n webhook if booking is completed
      if (response.triggerWebhook && response.bookingData) {
        await triggerN8nWebhook(response.bookingData);
      }
    }, 1000 + Math.random() * 2000);
  };

  // Simulate Mistral 7B intent detection and response generation
  const processWithMistral = async (userInput, currentSession, language) => {
    // This would normally call Ollama/Mistral API
    // For now, simulating intent detection logic
    
    const intents = detectIntent(userInput, language);
    const service = detectService(userInput, intents);
    
    let response = {
      message: '',
      intent: intents.primary,
      confidence: intents.confidence,
      service: service?.id,
      sessionData: { ...currentSession },
      actions: []
    };

    // Generate response based on step and intent
    if (currentSession.step === 'greeting' || !currentSession.intent) {
      response = handleGreeting(userInput, intents, service, language);
    } else if (currentSession.step === 'service_selection') {
      response = handleServiceSelection(userInput, service, currentSession, language);
    } else if (currentSession.step === 'booking') {
      response = handleBooking(userInput, currentSession, language);
    } else {
      response = handleGeneral(userInput, intents, language);
    }

    return response;
  };

  // Intent detection logic (simplified)
  const detectIntent = (input, language) => {
    const lowercaseInput = input.toLowerCase();
    
    const intentPatterns = {
      health_card_renewal: {
        en: ['health card', 'renew', 'renewal', 'health insurance', 'medical card'],
        ar: ['بطاقة صحية', 'تجديد', 'تأمين صحي', 'بطاقة طبية']
      },
      id_card_replacement: {
        en: ['id card', 'identity', 'replace', 'lost id', 'damaged id'],
        ar: ['بطاقة هوية', 'استبدال', 'هوية مفقودة', 'بطاقة تالفة']
      },
      medical_consultation: {
        en: ['doctor', 'appointment', 'medical', 'consultation', 'doctor visit'],
        ar: ['طبيب', 'موعد', 'استشارة', 'طبية', 'زيارة طبيب']
      },
      student_enrollment: {
        en: ['enroll', 'student', 'course', 'register', 'education'],
        ar: ['تسجيل', 'طالب', 'دورة', 'تعليم', 'التحاق']
      }
    };

    let maxScore = 0;
    let detectedIntent = 'general_inquiry';

    Object.entries(intentPatterns).forEach(([intent, patterns]) => {
      const words = patterns[language] || patterns.en;
      const score = words.reduce((acc, word) => {
        return acc + (lowercaseInput.includes(word.toLowerCase()) ? 1 : 0);
      }, 0);
      
      if (score > maxScore) {
        maxScore = score;
        detectedIntent = intent;
      }
    });

    return {
      primary: detectedIntent,
      confidence: Math.min(0.5 + (maxScore * 0.2), 0.98)
    };
  };

  const detectService = (input, intents) => {
    if (intents.primary === 'general_inquiry') return null;
    return availableServices.find(service => service.id === intents.primary.replace('_', '-'));
  };

  const handleGreeting = (input, intents, service, language) => {
    const responses = {
      en: {
        greeting: "Hello! I'm MIND14, your AI virtual assistant. I can help you with various services including health card renewal, ID replacement, medical consultations, and student enrollment.\n\nHow can I assist you today?",
        withService: `I understand you need help with **${service?.name[language]}**. This service typically takes about ${service?.estimatedTime} minutes.\n\n${service?.requiresAppointment ? '📅 This service requires an appointment.' : '💬 This is a general inquiry service.'}\n\nWould you like me to help you with this?`
      },
      ar: {
        greeting: "مرحباً! أنا MIND14، مساعدك الافتراضي الذكي. يمكنني مساعدتك في خدمات متنوعة تشمل تجديد البطاقة الصحية، استبدال الهوية، الاستشارات الطبية، وتسجيل الطلاب.\n\nكيف يمكنني مساعدتك اليوم؟",
        withService: `أفهم أنك تحتاج مساعدة في **${service?.name[language]}**. تستغرق هذه الخدمة عادة حوالي ${service?.estimatedTime} دقيقة.\n\n${service?.requiresAppointment ? '📅 تتطلب هذه الخدمة حجز موعد.' : '💬 هذه خدمة استفسار عام.'}\n\nهل تريد مني مساعدتك في هذا؟`
      }
    };

    const sessionData = {
      step: service ? 'service_selection' : 'intent_detection',
      selectedService: service?.id || null,
      collectedInfo: {},
      intent: intents.primary,
      confidence: intents.confidence
    };

    return {
      message: service ? responses[language].withService : responses[language].greeting,
      intent: intents.primary,
      confidence: intents.confidence,
      service: service?.id,
      sessionData,
      title: service ? service.name : { en: 'General Inquiry', ar: 'استفسار عام' }
    };
  };

  const handleServiceSelection = (input, service, currentSession, language) => {
    const confirmationWords = {
      en: ['yes', 'sure', 'ok', 'okay', 'proceed', 'continue'],
      ar: ['نعم', 'موافق', 'حسنا', 'متابعة', 'استمر']
    };

    const isConfirming = confirmationWords[language].some(word => 
      input.toLowerCase().includes(word)
    );

    if (isConfirming && currentSession.selectedService) {
      const selectedService = availableServices.find(s => s.id === currentSession.selectedService);
      
      if (selectedService?.requiresAppointment) {
        const responses = {
          en: `Great! I'll help you book an appointment for **${selectedService.name[language]}**.\n\n📋 **What I need from you:**\n• Your full name\n• Phone number\n• Preferred date and time\n\n⏰ **Available hours:** ${selectedService.workingHours.start} - ${selectedService.workingHours.end}\n\nPlease provide your full name first.`,
          ar: `ممتاز! سأساعدك في حجز موعد لـ **${selectedService.name[language]}**.\n\n📋 **ما أحتاجه منك:**\n• اسمك الكامل\n• رقم الهاتف\n• التاريخ والوقت المفضل\n\n⏰ **ساعات العمل:** ${selectedService.workingHours.start} - ${selectedService.workingHours.end}\n\nيرجى تقديم اسمك الكامل أولاً.`
        };

        return {
          message: responses[language],
          intent: currentSession.intent,
          confidence: currentSession.confidence,
          service: selectedService.id,
          sessionData: {
            ...currentSession,
            step: 'booking',
            bookingStep: 'name'
          }
        };
      } else {
        const responses = {
          en: `I'm here to help with your **${selectedService.name[language]}**. This is a general inquiry service, so please feel free to ask me any questions you have about this topic.`,
          ar: `أنا هنا لمساعدتك في **${selectedService.name[language]}**. هذه خدمة استفسار عام، لذا لا تتردد في طرح أي أسئلة لديك حول هذا الموضوع.`
        };

        return {
          message: responses[language],
          intent: currentSession.intent,
          confidence: currentSession.confidence,
          service: selectedService.id,
          sessionData: {
            ...currentSession,
            step: 'general_inquiry'
          }
        };
      }
    }

    // If not confirming, show service options
    const responses = {
      en: "I can help you with several services:\n\n" + 
           availableServices.map(s => `${s.icon} **${s.name.en}** - ${s.description.en}`).join('\n') +
           "\n\nWhich service interests you?",
      ar: "يمكنني مساعدتك في عدة خدمات:\n\n" +
           availableServices.map(s => `${s.icon} **${s.name.ar}** - ${s.description.ar}`).join('\n') +
           "\n\nأي خدمة تهمك؟"
    };

    return {
      message: responses[language],
      intent: 'service_selection',
      confidence: 0.8,
      sessionData: {
        ...currentSession,
        step: 'service_selection'
      }
    };
  };

  const handleBooking = (input, currentSession, language) => {
    const bookingStep = currentSession.bookingStep || 'name';
    const collectedInfo = { ...currentSession.collectedInfo };

    if (bookingStep === 'name') {
      collectedInfo.name = input;
      const responses = {
        en: `Thank you, ${input}! Now I need your phone number for appointment confirmation.`,
        ar: `شكراً، ${input}! الآن أحتاج رقم هاتفك لتأكيد الموعد.`
      };

      return {
        message: responses[language],
        intent: currentSession.intent,
        confidence: currentSession.confidence,
        service: currentSession.selectedService,
        sessionData: {
          ...currentSession,
          collectedInfo,
          bookingStep: 'phone'
        }
      };
    } else if (bookingStep === 'phone') {
      collectedInfo.phone = input;
      const responses = {
        en: "Perfect! Now please tell me your preferred date and time. For example: 'January 25th at 2:00 PM' or 'Tomorrow morning'.",
        ar: "ممتاز! الآن أخبرني بالتاريخ والوقت المفضل. مثال: '25 يناير في الساعة 2:00 مساءً' أو 'غداً صباحاً'."
      };

      return {
        message: responses[language],
        intent: currentSession.intent,
        confidence: currentSession.confidence,
        service: currentSession.selectedService,
        sessionData: {
          ...currentSession,
          collectedInfo,
          bookingStep: 'datetime'
        }
      };
    } else if (bookingStep === 'datetime') {
      collectedInfo.preferredDateTime = input;
      
      // Generate appointment confirmation
      const appointmentId = 'APT' + Date.now();
      const selectedService = availableServices.find(s => s.id === currentSession.selectedService);
      
      const responses = {
        en: `🎉 **Appointment Booked Successfully!**\n\n📅 **Appointment Details:**\n• Service: ${selectedService?.name.en}\n• Name: ${collectedInfo.name}\n• Phone: ${collectedInfo.phone}\n• Preferred Time: ${collectedInfo.preferredDateTime}\n• Appointment ID: ${appointmentId}\n\n✅ You will receive a confirmation via SMS and email shortly.\n📞 If you need to reschedule, please call us or start a new chat.\n\nIs there anything else I can help you with?`,
        ar: `🎉 **تم حجز الموعد بنجاح!**\n\n📅 **تفاصيل الموعد:**\n• الخدمة: ${selectedService?.name.ar}\n• الاسم: ${collectedInfo.name}\n• الهاتف: ${collectedInfo.phone}\n• الوقت المفضل: ${collectedInfo.preferredDateTime}\n• رقم الموعد: ${appointmentId}\n\n✅ ستتلقى تأكيداً عبر الرسائل النصية والبريد الإلكتروني قريباً.\n📞 إذا كنت بحاجة لإعادة الجدولة، يرجى الاتصال بنا أو بدء محادثة جديدة.\n\nهل هناك أي شيء آخر يمكنني مساعدتك فيه؟`
      };

      return {
        message: responses[language],
        intent: currentSession.intent,
        confidence: currentSession.confidence,
        service: currentSession.selectedService,
        sessionData: {
          ...currentSession,
          step: 'completed',
          collectedInfo,
          appointmentId
        },
        triggerWebhook: true,
        bookingData: {
          appointmentId,
          service: selectedService,
          customerInfo: collectedInfo,
          language,
          timestamp: new Date().toISOString()
        }
      };
    }

    return {
      message: "I need more information to complete your booking.",
      intent: currentSession.intent,
      confidence: currentSession.confidence,
      sessionData: currentSession
    };
  };

  const handleGeneral = (input, intents, language) => {
    const responses = {
      en: "I understand your question. As your AI virtual assistant, I'm here to help with various services. If you need specific assistance with health card renewal, ID replacement, medical appointments, or student enrollment, please let me know!",
      ar: "أفهم سؤالك. كمساعدك الافتراضي الذكي، أنا هنا لمساعدتك في خدمات متنوعة. إذا كنت تحتاج مساعدة محددة في تجديد البطاقة الصحية، استبدال الهوية، المواعيد الطبية، أو تسجيل الطلاب، يرجى إخباري!"
    };

    return {
      message: responses[language],
      intent: intents.primary,
      confidence: intents.confidence,
      sessionData: {
        step: 'general_inquiry',
        intent: intents.primary,
        confidence: intents.confidence
      }
    };
  };

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