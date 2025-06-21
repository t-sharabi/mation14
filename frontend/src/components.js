import React, { useState, useRef, useEffect } from 'react';

// Icons (using SVG)
const Icons = {
  Menu: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
    </svg>
  ),
  Plus: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
    </svg>
  ),
  Send: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
    </svg>
  ),
  Paperclip: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
    </svg>
  ),
  Trash: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
    </svg>
  ),
  LogOut: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
    </svg>
  ),
  ChevronLeft: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
    </svg>
  ),
  ChevronDown: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
    </svg>
  ),
  Bot: () => (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
    </svg>
  ),
  Settings: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  ),
  X: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  ),
  Image: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
  ),
  Document: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  )
};

// Loading dots animation
export const LoadingDots = () => (
  <div className="flex space-x-1">
    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
  </div>
);

// Login Form Component
export const LoginForm = ({ onLogin }) => {
  const [isSignUp, setIsSignUp] = useState(false);
  const [step, setStep] = useState('form'); // 'form', 'otp', 'success'
  
  // Sign In Form
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  
  // Sign Up Form
  const [fullName, setFullName] = useState('');
  const [signUpEmail, setSignUpEmail] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [signUpPassword, setSignUpPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  
  // OTP
  const [otp, setOtp] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSignIn = (e) => {
    e.preventDefault();
    if (email && password) {
      onLogin({ email, password });
    }
  };

  const handleSignUp = async (e) => {
    e.preventDefault();
    if (!fullName || !signUpEmail || !phoneNumber || !signUpPassword || !confirmPassword) {
      alert('Please fill in all fields');
      return;
    }
    
    if (signUpPassword !== confirmPassword) {
      alert('Passwords do not match');
      return;
    }

    if (phoneNumber.length < 10) {
      alert('Please enter a valid phone number');
      return;
    }

    setIsLoading(true);
    
    // Simulate sending OTP
    setTimeout(() => {
      setIsLoading(false);
      setStep('otp');
    }, 2000);
  };

  const handleOtpVerification = (e) => {
    e.preventDefault();
    if (otp.length !== 6) {
      alert('Please enter a valid 6-digit OTP');
      return;
    }

    setIsLoading(true);
    
    // Simulate OTP verification
    setTimeout(() => {
      setIsLoading(false);
      if (otp === '123456') {
        setStep('success');
        // Auto login after successful signup
        setTimeout(() => {
          onLogin({ 
            email: signUpEmail, 
            password: signUpPassword,
            fullName: fullName,
            phoneNumber: phoneNumber 
          });
        }, 2000);
      } else {
        alert('Invalid OTP. Please try 123456 for demo');
      }
    }, 1500);
  };

  const resetForm = () => {
    setStep('form');
    setFullName('');
    setSignUpEmail('');
    setPhoneNumber('');
    setSignUpPassword('');
    setConfirmPassword('');
    setOtp('');
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
      <div className="max-w-md w-full space-y-8 p-8">
        <div className="text-center">
          {/* MIND14 Logo */}
          <div className="flex items-center justify-center mb-8">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <div className="w-12 h-12 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center">
                  <div className="w-8 h-8 relative">
                    <div className="absolute top-0 left-2 w-2 h-2 bg-white rounded-full"></div>
                    <div className="absolute bottom-0 right-2 w-2 h-2 bg-white rounded-full"></div>
                    <div className="absolute top-2 right-0 w-2 h-2 bg-white rounded-full"></div>
                    <div className="absolute bottom-2 left-0 w-2 h-2 bg-white rounded-full"></div>
                    <div className="absolute top-1 left-4 w-0.5 h-6 bg-white rounded rotate-45"></div>
                    <div className="absolute top-1 right-4 w-0.5 h-6 bg-white rounded -rotate-45"></div>
                  </div>
                </div>
              </div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                MIND14
              </h1>
            </div>
          </div>
          
          {step === 'form' && (
            <>
              <h2 className="text-3xl font-bold text-white">
                {isSignUp ? 'Create your account' : 'Welcome back'}
              </h2>
              <p className="mt-2 text-gray-400">
                {isSignUp ? 'Sign up to get started' : 'Sign in to your account'}
              </p>
            </>
          )}
          
          {step === 'otp' && (
            <>
              <h2 className="text-3xl font-bold text-white">Verify your phone</h2>
              <p className="mt-2 text-gray-400">
                We sent a 6-digit code to {phoneNumber}
              </p>
            </>
          )}
          
          {step === 'success' && (
            <>
              <h2 className="text-3xl font-bold text-white">Account created!</h2>
              <p className="mt-2 text-gray-400">
                Welcome to MIND14, {fullName}
              </p>
            </>
          )}
        </div>
        
        {step === 'form' && !isSignUp && (
          <form className="mt-8 space-y-6" onSubmit={handleSignIn}>
            <div className="space-y-4">
              <div>
                <label htmlFor="email" className="sr-only">Email address</label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="Email address"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
              <div>
                <label htmlFor="password" className="sr-only">Password</label>
                <input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="current-password"
                  required
                  className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
            </div>

            <div>
              <button
                type="submit"
                className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-lg text-white bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 transition duration-200"
              >
                Sign in
              </button>
            </div>
            
            <div className="text-center space-y-2">
              <p className="text-sm text-gray-400">
                📧 Demo: Use any email and password to sign in
              </p>
              <p className="text-sm text-gray-400">
                Don't have an account?{' '}
                <button
                  onClick={() => setIsSignUp(true)}
                  className="text-purple-400 hover:text-purple-300 underline"
                >
                  Sign up
                </button>
              </p>
            </div>
          </form>
        )}

        {step === 'form' && isSignUp && (
          <form className="mt-8 space-y-6" onSubmit={handleSignUp}>
            <div className="space-y-4">
              <div>
                <label htmlFor="fullName" className="sr-only">Full Name</label>
                <input
                  id="fullName"
                  name="fullName"
                  type="text"
                  autoComplete="name"
                  required
                  className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="Full Name"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                />
              </div>
              <div>
                <label htmlFor="signUpEmail" className="sr-only">Email address</label>
                <input
                  id="signUpEmail"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="Email address"
                  value={signUpEmail}
                  onChange={(e) => setSignUpEmail(e.target.value)}
                />
              </div>
              <div>
                <label htmlFor="phoneNumber" className="sr-only">Phone Number</label>
                <input
                  id="phoneNumber"
                  name="phone"
                  type="tel"
                  autoComplete="tel"
                  required
                  className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="Phone Number (+1234567890)"
                  value={phoneNumber}
                  onChange={(e) => setPhoneNumber(e.target.value)}
                />
              </div>
              <div>
                <label htmlFor="signUpPassword" className="sr-only">Password</label>
                <input
                  id="signUpPassword"
                  name="password"
                  type="password"
                  autoComplete="new-password"
                  required
                  className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="Password"
                  value={signUpPassword}
                  onChange={(e) => setSignUpPassword(e.target.value)}
                />
              </div>
              <div>
                <label htmlFor="confirmPassword" className="sr-only">Confirm Password</label>
                <input
                  id="confirmPassword"
                  name="confirmPassword"
                  type="password"
                  autoComplete="new-password"
                  required
                  className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="Confirm Password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                />
              </div>
            </div>

            <div>
              <button
                type="submit"
                disabled={isLoading}
                className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-lg text-white bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 transition duration-200 disabled:opacity-50"
              >
                {isLoading ? (
                  <LoadingDots />
                ) : (
                  'Create Account & Send OTP'
                )}
              </button>
            </div>
            
            <div className="text-center">
              <p className="text-sm text-gray-400">
                Already have an account?{' '}
                <button
                  onClick={() => setIsSignUp(false)}
                  className="text-purple-400 hover:text-purple-300 underline"
                >
                  Sign in
                </button>
              </p>
            </div>
          </form>
        )}

        {step === 'otp' && (
          <form className="mt-8 space-y-6" onSubmit={handleOtpVerification}>
            <div className="space-y-4">
              <div>
                <label htmlFor="otp" className="sr-only">OTP Code</label>
                <input
                  id="otp"
                  name="otp"
                  type="text"
                  maxLength="6"
                  required
                  className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-center text-2xl tracking-widest"
                  placeholder="000000"
                  value={otp}
                  onChange={(e) => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                />
              </div>
            </div>

            <div className="space-y-3">
              <button
                type="submit"
                disabled={isLoading || otp.length !== 6}
                className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-lg text-white bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 transition duration-200 disabled:opacity-50"
              >
                {isLoading ? <LoadingDots /> : 'Verify OTP'}
              </button>
              
              <button
                type="button"
                onClick={resetForm}
                className="w-full text-sm text-gray-400 hover:text-gray-300 underline"
              >
                Back to sign up
              </button>
            </div>
            
            <div className="text-center space-y-2">
              <p className="text-sm text-gray-400">
                📱 Demo: Use OTP code <span className="font-mono text-purple-400">123456</span>
              </p>
              <p className="text-xs text-gray-500">
                Didn't receive the code? Check your phone or try again
              </p>
            </div>
          </form>
        )}

        {step === 'success' && (
          <div className="mt-8 text-center space-y-6">
            <div className="w-16 h-16 mx-auto bg-gradient-to-r from-green-500 to-emerald-500 rounded-full flex items-center justify-center">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <p className="text-gray-400">
              Your account has been created successfully. Redirecting to MIND14...
            </p>
            <LoadingDots />
          </div>
        )}
      </div>
    </div>
  );
};

// Model Selector Component
export const ModelSelector = ({ selectedModel, onModelChange, availableModels }) => {
  const [isOpen, setIsOpen] = useState(false);
  const selectedModelData = availableModels.find(m => m.id === selectedModel);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg border border-gray-700 text-white transition-colors"
      >
        <Icons.Bot />
        <span className="text-sm font-medium">{selectedModelData?.name}</span>
        <Icons.ChevronDown />
      </button>
      
      {isOpen && (
        <div className="absolute top-full left-0 mt-2 w-64 bg-gray-800 border border-gray-700 rounded-lg shadow-lg z-50">
          {availableModels.map((model) => (
            <button
              key={model.id}
              onClick={() => {
                onModelChange(model.id);
                setIsOpen(false);
              }}
              className={`w-full text-left px-4 py-3 hover:bg-gray-700 transition-colors ${
                selectedModel === model.id ? 'bg-gray-700' : ''
              }`}
            >
              <div className="font-medium text-white">{model.name}</div>
              <div className="text-sm text-gray-400">{model.description}</div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

// File Upload Component
export const FileUpload = ({ onFileSelect }) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    onFileSelect(files);
  };

  const handleFileInput = (e) => {
    const files = Array.from(e.target.files);
    onFileSelect(files);
  };

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
        isDragging 
          ? 'border-purple-500 bg-purple-500/10' 
          : 'border-gray-600 hover:border-gray-500'
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple
        onChange={handleFileInput}
        className="hidden"
        accept=".txt,.md,.json,.js,.ts,.png,.jpg,.jpeg,.gif,.webp,.pdf,.doc,.docx,.csv"
      />
      
      <div className="space-y-2">
        <Icons.Paperclip className="mx-auto text-gray-400" />
        <p className="text-gray-400">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="text-purple-400 hover:text-purple-300 underline"
          >
            Click to upload
          </button>
          {' '}or drag and drop
        </p>
        <p className="text-xs text-gray-500">
          Supports: Text, Images, Documents
        </p>
      </div>
    </div>
  );
};

// Message Bubble Component
export const MessageBubble = ({ message, isUser, currentUser }) => {
  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatContent = (content) => {
    // Simple markdown-like formatting
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`(.*?)`/g, '<code class="bg-gray-700 px-1 py-0.5 rounded text-sm">$1</code>')
      .replace(/\n/g, '<br>');
  };

  return (
    <div className={`flex items-start space-x-3 ${isUser ? 'flex-row-reverse space-x-reverse' : ''} mb-6`}>
      <div className="flex-shrink-0">
        <img
          src={isUser ? currentUser?.avatar : 'https://images.pexels.com/photos/8728386/pexels-photo-8728386.jpeg'}
          alt={isUser ? 'User' : 'AI'}
          className="w-8 h-8 rounded-full object-cover"
        />
      </div>
      
      <div className={`flex-1 max-w-3xl ${isUser ? 'text-right' : ''}`}>
        <div className="flex items-center space-x-2 mb-1">
          <span className="text-sm font-medium text-gray-300">
            {isUser ? 'You' : `MIND14 ${message.model ? `(${message.model})` : ''}`}
          </span>
          <span className="text-xs text-gray-500">
            {formatTime(message.timestamp)}
          </span>
        </div>
        
        <div className={`inline-block px-4 py-3 rounded-2xl max-w-full ${
          isUser 
            ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white' 
            : 'bg-gray-800 text-gray-100 border border-gray-700'
        }`}>
          {message.attachments && message.attachments.length > 0 && (
            <div className="mb-2 space-y-1">
              {message.attachments.map((file, index) => (
                <div key={index} className="flex items-center space-x-2 text-sm">
                  <Icons.Paperclip className="w-3 h-3" />
                  <span>{file.name}</span>
                </div>
              ))}
            </div>
          )}
          
          <div 
            className="whitespace-pre-wrap"
            dangerouslySetInnerHTML={{ __html: formatContent(message.content) }}
          />
        </div>
      </div>
    </div>
  );
};

// Sidebar Component
export const Sidebar = ({ 
  conversations, 
  activeConversationId, 
  onSelectConversation, 
  onNewConversation,
  onDeleteConversation,
  onLogout,
  currentUser,
  collapsed,
  onToggleCollapse
}) => {
  return (
    <div className={`bg-gray-950 border-r border-gray-800 flex flex-col transition-all duration-300 ${
      collapsed ? 'w-16' : 'w-80'
    }`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between">
          {!collapsed && (
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                <div className="w-5 h-5 relative">
                  <div className="absolute top-0 left-1 w-1 h-1 bg-white rounded-full"></div>
                  <div className="absolute bottom-0 right-1 w-1 h-1 bg-white rounded-full"></div>
                  <div className="absolute top-1 right-0 w-1 h-1 bg-white rounded-full"></div>
                  <div className="absolute bottom-1 left-0 w-1 h-1 bg-white rounded-full"></div>
                </div>
              </div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                MIND14
              </h1>
            </div>
          )}
          <button
            onClick={onToggleCollapse}
            className="p-2 hover:bg-gray-800 rounded-lg transition-colors text-gray-400 hover:text-white"
          >
            <Icons.Menu />
          </button>
        </div>
      </div>

      {/* New Chat Button */}
      <div className="p-4">
        <button
          onClick={onNewConversation}
          className={`w-full flex items-center space-x-3 px-4 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg transition-colors ${
            collapsed ? 'justify-center' : ''
          }`}
        >
          <Icons.Plus />
          {!collapsed && <span className="font-medium">New Chat</span>}
        </button>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto px-4">
        <div className="space-y-2">
          {conversations.map((conversation) => (
            <div
              key={conversation.id}
              className={`group relative flex items-center space-x-3 px-3 py-3 rounded-lg cursor-pointer transition-colors ${
                activeConversationId === conversation.id
                  ? 'bg-gray-800 text-white'
                  : 'hover:bg-gray-800 text-gray-300 hover:text-white'
              } ${collapsed ? 'justify-center' : ''}`}
              onClick={() => onSelectConversation(conversation.id)}
            >
              {collapsed ? (
                <div className="w-6 h-6 bg-gradient-to-r from-purple-500 to-pink-500 rounded text-white flex items-center justify-center text-xs font-bold">
                  {conversation.title.charAt(0)}
                </div>
              ) : (
                <>
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-medium truncate">{conversation.title}</h3>
                    <p className="text-xs text-gray-500 truncate">
                      {conversation.messages.length} messages • {conversation.model}
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteConversation(conversation.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-700 rounded transition-opacity"
                  >
                    <Icons.Trash />
                  </button>
                </>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* User Profile */}
      <div className="p-4 border-t border-gray-800">
        <div className={`flex items-center space-x-3 ${collapsed ? 'justify-center' : ''}`}>
          <img
            src={currentUser?.avatar}
            alt="Profile"
            className="w-8 h-8 rounded-full object-cover"
          />
          {!collapsed && (
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-white truncate">{currentUser?.name}</p>
              <p className="text-xs text-gray-500 truncate">{currentUser?.email}</p>
            </div>
          )}
          <button
            onClick={onLogout}
            className="p-2 hover:bg-gray-800 rounded-lg transition-colors text-gray-400 hover:text-white"
            title="Logout"
          >
            <Icons.LogOut />
          </button>
        </div>
      </div>
    </div>
  );
};

// Main Chat Interface Component
export const ChatInterface = ({ 
  conversation, 
  onSendMessage, 
  selectedModel, 
  onModelChange, 
  availableModels,
  currentUser,
  sidebarCollapsed 
}) => {
  const [message, setMessage] = useState('');
  const [attachments, setAttachments] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversation?.messages]);

  const handleSend = async () => {
    if (!message.trim() && attachments.length === 0) return;
    
    setIsLoading(true);
    await onSendMessage(message, attachments);
    setMessage('');
    setAttachments([]);
    setIsLoading(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileSelect = (files) => {
    setAttachments(prev => [...prev, ...files]);
  };

  const removeAttachment = (index) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900">
      {/* Header */}
      <div className="border-b border-gray-800 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h2 className="text-lg font-semibold text-white">
              {conversation?.title || 'MIND14 Chat'}
            </h2>
            {conversation && (
              <span className="text-sm text-gray-400">
                {conversation.messages.length} messages
              </span>
            )}
          </div>
          <ModelSelector
            selectedModel={selectedModel}
            onModelChange={onModelChange}
            availableModels={availableModels}
          />
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6">
        {!conversation ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="max-w-md space-y-6">
              <img
                src="https://images.unsplash.com/photo-1639759032532-c7f288e9ef4f?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTY2NjZ8MHwxfHNlYXJjaHwyfHxjaGF0JTIwaW50ZXJmYWNlfGVufDB8fHxwdXJwbGV8MTc1MDUwNTU3MXww&ixlib=rb-4.1.0&q=85"
                alt="MIND14 Chat"
                className="w-32 h-32 mx-auto rounded-2xl object-cover opacity-50"
              />
              <div className="space-y-4">
                <h3 className="text-2xl font-bold text-white">Welcome to MIND14</h3>
                <p className="text-gray-400">
                  Your AI-powered chat assistant. Start a conversation by typing a message below
                  or create a new chat to begin.
                </p>
                <div className="flex flex-wrap gap-2 justify-center">
                  {['Ask about coding', 'Data analysis help', 'Creative writing', 'General questions'].map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => setMessage(suggestion)}
                      className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-full text-sm transition-colors"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto">
            {conversation.messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                isUser={msg.role === 'user'}
                currentUser={currentUser}
              />
            ))}
            {isLoading && (
              <div className="flex items-start space-x-3 mb-6">
                <img
                  src="https://images.pexels.com/photos/8728386/pexels-photo-8728386.jpeg"
                  alt="AI"
                  className="w-8 h-8 rounded-full object-cover"
                />
                <div className="bg-gray-800 px-4 py-3 rounded-2xl border border-gray-700">
                  <LoadingDots />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-800 p-4">
        <div className="max-w-4xl mx-auto">
          {/* Attachments Preview */}
          {attachments.length > 0 && (
            <div className="mb-4 flex flex-wrap gap-2">
              {attachments.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center space-x-2 bg-gray-800 px-3 py-2 rounded-lg text-sm"
                >
                  <Icons.Paperclip className="w-3 h-3 text-gray-400" />
                  <span className="text-gray-300">{file.name}</span>
                  <button
                    onClick={() => removeAttachment(index)}
                    className="text-gray-400 hover:text-red-400 ml-2"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}

          <div className="flex items-end space-x-4">
            <div className="flex-1">
              <div className="relative">
                <textarea
                  ref={textareaRef}
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message..."
                  className="w-full px-4 py-3 pr-12 bg-gray-800 border border-gray-700 rounded-2xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none min-h-[50px] max-h-32"
                  rows={1}
                  style={{ height: 'auto' }}
                  onInput={(e) => {
                    e.target.style.height = 'auto';
                    e.target.style.height = e.target.scrollHeight + 'px';
                  }}
                />
                
                <div className="absolute right-3 bottom-3 flex items-center space-x-2">
                  <FileUpload onFileSelect={handleFileSelect} />
                  <button
                    onClick={handleSend}
                    disabled={(!message.trim() && attachments.length === 0) || isLoading}
                    className="p-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-600 text-white rounded-full transition-colors disabled:cursor-not-allowed"
                  >
                    <Icons.Send />
                  </button>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-2 text-xs text-gray-500 text-center">
            MIND14 can make mistakes. Consider checking important information.
          </div>
        </div>
      </div>
    </div>
  );
};