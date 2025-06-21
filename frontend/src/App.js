import React, { useState, useEffect, useRef } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';
import { 
  ChatInterface, 
  Sidebar, 
  LoginForm, 
  MessageBubble, 
  FileUpload,
  ModelSelector,
  LoadingDots
} from './components';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Mock data for conversations
const mockConversations = [
  {
    id: '1',
    title: 'React Development Tips',
    messages: [
      {
        id: '1',
        role: 'user',
        content: 'How can I optimize React performance?',
        timestamp: new Date('2025-01-15T10:30:00')
      },
      {
        id: '2',
        role: 'assistant',
        content: 'Here are some key strategies to optimize React performance:\n\n1. **Use React.memo()** for component memoization\n2. **Implement useMemo()** for expensive calculations\n3. **Use useCallback()** to prevent unnecessary re-renders\n4. **Code splitting** with React.lazy() and Suspense\n5. **Optimize bundle size** by removing unused dependencies\n\nWould you like me to elaborate on any of these techniques?',
        timestamp: new Date('2025-01-15T10:30:15'),
        model: 'Mistral'
      }
    ],
    model: 'Mistral',
    createdAt: new Date('2025-01-15T10:30:00')
  },
  {
    id: '2',
    title: 'Python Data Analysis',
    messages: [
      {
        id: '3',
        role: 'user',
        content: 'Best libraries for data analysis in Python?',
        timestamp: new Date('2025-01-14T14:20:00')
      },
      {
        id: '4',
        role: 'assistant',
        content: 'For Python data analysis, I recommend these essential libraries:\n\n📊 **Core Libraries:**\n• **Pandas** - Data manipulation and analysis\n• **NumPy** - Numerical computing\n• **Matplotlib** - Basic plotting and visualization\n• **Seaborn** - Statistical data visualization\n\n🔬 **Advanced Analytics:**\n• **SciPy** - Scientific computing\n• **Scikit-learn** - Machine learning\n• **Plotly** - Interactive visualizations\n\n💡 **Getting Started Tip:** Start with Pandas and NumPy, then add visualization with Matplotlib or Seaborn based on your needs.',
        timestamp: new Date('2025-01-14T14:20:18'),
        model: 'DeepSeek Coder'
      }
    ],
    model: 'DeepSeek Coder',
    createdAt: new Date('2025-01-14T14:20:00')
  },
  {
    id: '3',
    title: 'JavaScript ES6 Features',
    messages: [
      {
        id: '5',
        role: 'user',
        content: 'What are the most important ES6 features?',
        timestamp: new Date('2025-01-13T09:15:00')
      },
      {
        id: '6',
        role: 'assistant',
        content: '🚀 **Essential ES6 Features:**\n\n**1. Arrow Functions**\n```javascript\nconst add = (a, b) => a + b;\n```\n\n**2. Destructuring**\n```javascript\nconst {name, age} = person;\nconst [first, second] = array;\n```\n\n**3. Template Literals**\n```javascript\nconst message = `Hello ${name}!`;\n```\n\n**4. Let/Const**\n```javascript\nlet variable = "changeable";\nconst constant = "unchangeable";\n```\n\n**5. Modules**\n```javascript\nimport { function } from "./module";\nexport default MyComponent;\n```\n\nThese features make JavaScript more expressive and easier to work with!',
        timestamp: new Date('2025-01-13T09:15:22'),
        model: 'Qwen'
      }
    ],
    model: 'Qwen',
    createdAt: new Date('2025-01-13T09:15:00')
  }
];

// Mock models
const availableModels = [
  { id: 'mistral', name: 'Mistral', description: 'General purpose AI model' },
  { id: 'deepseek-coder', name: 'DeepSeek Coder', description: 'Specialized for coding tasks' },
  { id: 'qwen', name: 'Qwen', description: 'Advanced reasoning model' },
  { id: 'llama3', name: 'Llama 3', description: 'Meta\'s latest language model' }
];

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [conversations, setConversations] = useState(mockConversations);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [selectedModel, setSelectedModel] = useState('mistral');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  useEffect(() => {
    // Mock authentication check
    const savedUser = localStorage.getItem('mind14_user');
    if (savedUser) {
      setCurrentUser(JSON.parse(savedUser));
      setIsAuthenticated(true);
    }
  }, []);

  const handleLogin = (credentials) => {
    // Mock login - accept any credentials
    const user = {
      id: '1',
      name: credentials.fullName || credentials.email.split('@')[0],
      email: credentials.email,
      avatar: 'https://images.pexels.com/photos/7658539/pexels-photo-7658539.jpeg',
      phoneNumber: credentials.phoneNumber || null
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
  };

  const createNewConversation = () => {
    const newConversation = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [],
      model: selectedModel,
      createdAt: new Date()
    };
    setConversations(prev => [newConversation, ...prev]);
    setActiveConversationId(newConversation.id);
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
      timestamp: new Date()
    };

    // Add user message
    setConversations(prev => prev.map(conv => 
      conv.id === activeConversationId 
        ? { ...conv, messages: [...conv.messages, userMessage] }
        : conv
    ));

    // Simulate AI response
    setTimeout(() => {
      const responses = [
        "I understand your question. Let me help you with that...",
        "That's a great question! Here's what I think...",
        "Based on your query, I can provide the following insights...",
        "Let me break this down for you step by step...",
        "Here's a comprehensive answer to your question..."
      ];
      
      const aiMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: responses[Math.floor(Math.random() * responses.length)] + 
                 `\n\nI'm using the **${availableModels.find(m => m.id === selectedModel)?.name}** model to process your request. This is a demo response to show the chat functionality.`,
        timestamp: new Date(),
        model: availableModels.find(m => m.id === selectedModel)?.name
      };

      setConversations(prev => prev.map(conv => 
        conv.id === activeConversationId 
          ? { ...conv, messages: [...conv.messages, aiMessage] }
          : conv
      ));

      // Update conversation title if it's the first message
      setConversations(prev => prev.map(conv => 
        conv.id === activeConversationId && conv.title === 'New Chat'
          ? { ...conv, title: content.slice(0, 30) + (content.length > 30 ? '...' : '') }
          : conv
      ));
    }, 1000 + Math.random() * 2000);
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
        />
        
        <div className="flex-1 flex flex-col">
          <Routes>
            <Route 
              path="/" 
              element={
                <ChatInterface
                  conversation={conversations.find(conv => conv.id === activeConversationId)}
                  onSendMessage={sendMessage}
                  selectedModel={selectedModel}
                  onModelChange={setSelectedModel}
                  availableModels={availableModels}
                  currentUser={currentUser}
                  sidebarCollapsed={sidebarCollapsed}
                />
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