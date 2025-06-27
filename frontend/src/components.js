import React, { useState, useRef, useEffect } from 'react';

// Icons (using SVG)
const Icons = {
  Menu: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
    </svg>
  ),
  BarChart: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  ),
  TrendingUp: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
    </svg>
  ),
  Users: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-.5a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z" />
    </svg>
  ),
  Server: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
    </svg>
  ),
  Activity: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  ),
  Eye: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
    </svg>
  ),
  Download: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
    </svg>
  ),
  Filter: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.707A1 1 0 013 7V4z" />
    </svg>
  ),
  Search: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
    </svg>
  ),
  RefreshCw: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
    </svg>
  ),
  AlertTriangle: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.464 0L4.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
    </svg>
  ),
  DollarSign: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
    </svg>
  ),
  Target: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
    </svg>
  ),
  Management: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
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
  Globe: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
    </svg>
  ),
  Dashboard: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  ),
  Chat: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
    </svg>
  ),
  Calendar: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
  ),
  Clock: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  CheckCircle: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  ExclamationCircle: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
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

// Language Selector Component
export const LanguageSelector = ({ currentLanguage, onLanguageChange }) => {
  const [isOpen, setIsOpen] = useState(false);
  
  const languages = [
    { code: 'en', name: 'English', flag: '🇺🇸' },
    { code: 'ar', name: 'العربية', flag: '🇸🇦' }
  ];

  const currentLang = languages.find(lang => lang.code === currentLanguage);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg border border-gray-700 text-white transition-colors"
      >
        <Icons.Globe />
        <span className="text-sm">{currentLang?.flag}</span>
        <span className="text-sm font-medium">{currentLang?.name}</span>
      </button>
      
      {isOpen && (
        <div className="absolute top-full right-0 mt-2 w-48 bg-gray-800 border border-gray-700 rounded-lg shadow-lg z-50">
          {languages.map((lang) => (
            <button
              key={lang.code}
              onClick={() => {
                onLanguageChange(lang.code);
                setIsOpen(false);
              }}
              className={`w-full text-left px-4 py-3 hover:bg-gray-700 transition-colors flex items-center space-x-3 ${
                currentLanguage === lang.code ? 'bg-gray-700' : ''
              }`}
            >
              <span className="text-lg">{lang.flag}</span>
              <span className="font-medium text-white">{lang.name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

// Login Form Component (Enhanced for Virtual Front Desk)
export const LoginForm = ({ onLogin }) => {
  const [isSignUp, setIsSignUp] = useState(false);
  const [step, setStep] = useState('form');
  const [language, setLanguage] = useState('en');
  
  // Form states
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [signUpEmail, setSignUpEmail] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [signUpPassword, setSignUpPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [otp, setOtp] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const text = {
    en: {
      title: 'MIND14 Virtual Front Desk',
      subtitle: 'AI-Powered Service Assistant',
      welcome: 'Welcome back',
      welcomeSignUp: 'Create your account',
      signInDesc: 'Sign in to access virtual services',
      signUpDesc: 'Sign up to get started with AI assistance',
      signIn: 'Sign in',
      signUp: 'Sign up',
      createAccount: 'Create Account & Send OTP',
      verifyOTP: 'Verify OTP',
      demoHint: '📧 Demo: Use any email and password to sign in',
      demoHintAdmin: '💼 Try admin@mind14.com for admin access',
      noAccount: "Don't have an account?",
      hasAccount: 'Already have an account?',
      verifyPhone: 'Verify your phone',
      sentCode: 'We sent a 6-digit code to',
      accountCreated: 'Account created!',
      welcomeUser: 'Welcome to MIND14',
      demoOTP: 'Demo: Use OTP code',
      backToSignUp: 'Back to sign up'
    },
    ar: {
      title: 'مكتب الاستقبال الافتراضي MIND14',
      subtitle: 'مساعد الخدمات المدعوم بالذكاء الاصطناعي',
      welcome: 'مرحباً بعودتك',
      welcomeSignUp: 'إنشاء حسابك',
      signInDesc: 'سجل الدخول للوصول إلى الخدمات الافتراضية',
      signUpDesc: 'سجل للبدء مع المساعدة بالذكاء الاصطناعي',
      signIn: 'تسجيل الدخول',
      signUp: 'إنشاء حساب',
      createAccount: 'إنشاء حساب وإرسال رمز التحقق',
      verifyOTP: 'تحقق من الرمز',
      demoHint: '📧 تجريبي: استخدم أي بريد إلكتروني وكلمة مرور للدخول',
      demoHintAdmin: '💼 جرب admin@mind14.com للوصول كمشرف',
      noAccount: 'ليس لديك حساب؟',
      hasAccount: 'لديك حساب بالفعل؟',
      verifyPhone: 'تحقق من هاتفك',
      sentCode: 'أرسلنا رمز مكون من 6 أرقام إلى',
      accountCreated: 'تم إنشاء الحساب!',
      welcomeUser: 'مرحباً بك في MIND14',
      demoOTP: 'تجريبي: استخدم رمز التحقق',
      backToSignUp: 'العودة إلى التسجيل'
    }
  };

  const currentText = text[language];

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

    setIsLoading(true);
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
    setTimeout(() => {
      setIsLoading(false);
      if (otp === '123456') {
        setStep('success');
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

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
      <div className="max-w-md w-full space-y-8 p-8">
        {/* Language Selector */}
        <div className="flex justify-end">
          <LanguageSelector currentLanguage={language} onLanguageChange={setLanguage} />
        </div>

        <div className="text-center">
          {/* MIND14 Logo */}
          <div className="flex flex-col items-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mb-4">
              <div className="w-10 h-10 relative">
                <div className="absolute top-1 left-2 w-2 h-2 bg-white rounded-full animate-pulse"></div>
                <div className="absolute bottom-1 right-2 w-2 h-2 bg-white rounded-full animate-pulse" style={{ animationDelay: '0.5s' }}></div>
                <div className="absolute top-2 right-1 w-2 h-2 bg-white rounded-full animate-pulse" style={{ animationDelay: '1s' }}></div>
                <div className="absolute bottom-2 left-1 w-2 h-2 bg-white rounded-full animate-pulse" style={{ animationDelay: '1.5s' }}></div>
              </div>
            </div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              {currentText.title}
            </h1>
            <p className="text-gray-400 mt-2 text-sm">{currentText.subtitle}</p>
          </div>
          
          {step === 'form' && (
            <>
              <h2 className="text-2xl font-bold text-white">
                {isSignUp ? currentText.welcomeSignUp : currentText.welcome}
              </h2>
              <p className="mt-2 text-gray-400 text-sm">
                {isSignUp ? currentText.signUpDesc : currentText.signInDesc}
              </p>
            </>
          )}
          
          {step === 'otp' && (
            <>
              <h2 className="text-2xl font-bold text-white">{currentText.verifyPhone}</h2>
              <p className="mt-2 text-gray-400 text-sm">
                {currentText.sentCode} {phoneNumber}
              </p>
            </>
          )}
          
          {step === 'success' && (
            <>
              <h2 className="text-2xl font-bold text-white">{currentText.accountCreated}</h2>
              <p className="mt-2 text-gray-400 text-sm">
                {currentText.welcomeUser}, {fullName}
              </p>
            </>
          )}
        </div>
        
        {step === 'form' && !isSignUp && (
          <form className="mt-8 space-y-6" onSubmit={handleSignIn}>
            <div className="space-y-4">
              <input
                type="email"
                required
                className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder="Email address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
              <input
                type="password"
                required
                className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>

            <button
              type="submit"
              className="w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-lg text-white bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 transition duration-200"
            >
              {currentText.signIn}
            </button>
            
            <div className="text-center space-y-2">
              <p className="text-xs text-gray-400">{currentText.demoHint}</p>
              <p className="text-xs text-gray-400">{currentText.demoHintAdmin}</p>
              <p className="text-sm text-gray-400">
                {currentText.noAccount}{' '}
                <button
                  onClick={() => setIsSignUp(true)}
                  className="text-purple-400 hover:text-purple-300 underline"
                >
                  {currentText.signUp}
                </button>
              </p>
            </div>
          </form>
        )}

        {step === 'form' && isSignUp && (
          <form className="mt-8 space-y-6" onSubmit={handleSignUp}>
            <div className="space-y-4">
              <input
                type="text"
                required
                className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder="Full Name"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
              />
              <input
                type="email"
                required
                className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder="Email address"
                value={signUpEmail}
                onChange={(e) => setSignUpEmail(e.target.value)}
              />
              <input
                type="tel"
                required
                className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder="Phone Number (+1234567890)"
                value={phoneNumber}
                onChange={(e) => setPhoneNumber(e.target.value)}
              />
              <input
                type="password"
                required
                className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder="Password"
                value={signUpPassword}
                onChange={(e) => setSignUpPassword(e.target.value)}
              />
              <input
                type="password"
                required
                className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                placeholder="Confirm Password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-lg text-white bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 transition duration-200 disabled:opacity-50"
            >
              {isLoading ? <LoadingDots /> : currentText.createAccount}
            </button>
            
            <div className="text-center">
              <p className="text-sm text-gray-400">
                {currentText.hasAccount}{' '}
                <button
                  onClick={() => setIsSignUp(false)}
                  className="text-purple-400 hover:text-purple-300 underline"
                >
                  {currentText.signIn}
                </button>
              </p>
            </div>
          </form>
        )}

        {step === 'otp' && (
          <form className="mt-8 space-y-6" onSubmit={handleOtpVerification}>
            <input
              type="text"
              maxLength="6"
              required
              className="appearance-none rounded-lg relative block w-full px-3 py-3 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-center text-2xl tracking-widest"
              placeholder="000000"
              value={otp}
              onChange={(e) => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
            />

            <div className="space-y-3">
              <button
                type="submit"
                disabled={isLoading || otp.length !== 6}
                className="w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-lg text-white bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 transition duration-200 disabled:opacity-50"
              >
                {isLoading ? <LoadingDots /> : currentText.verifyOTP}
              </button>
              
              <button
                type="button"
                onClick={() => setStep('form')}
                className="w-full text-sm text-gray-400 hover:text-gray-300 underline"
              >
                {currentText.backToSignUp}
              </button>
            </div>
            
            <div className="text-center space-y-2">
              <p className="text-sm text-gray-400">
                {currentText.demoOTP} <span className="font-mono text-purple-400">123456</span>
              </p>
            </div>
          </form>
        )}

        {step === 'success' && (
          <div className="mt-8 text-center space-y-6">
            <div className="w-16 h-16 mx-auto bg-gradient-to-r from-green-500 to-emerald-500 rounded-full flex items-center justify-center">
              <Icons.CheckCircle className="w-8 h-8 text-white" />
            </div>
            <p className="text-gray-400">
              {currentText.welcomeUser}...
            </p>
            <LoadingDots />
          </div>
        )}
      </div>
    </div>
  );
};

// Service Card Component
export const ServiceCard = ({ service, language, onClick }) => {
  const isRTL = language === 'ar';
  
  return (
    <div 
      onClick={() => onClick(service)}
      className="bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-xl p-6 cursor-pointer transition-all duration-200 hover:scale-105 hover:border-purple-500"
    >
      <div className={`flex items-start space-x-4 ${isRTL ? 'flex-row-reverse space-x-reverse' : ''}`}>
        <div className="text-4xl">{service.icon}</div>
        <div className="flex-1">
          <h3 className={`text-lg font-semibold text-white mb-2 ${isRTL ? 'text-right' : 'text-left'}`}>
            {service.name[language]}
          </h3>
          <p className={`text-gray-400 text-sm mb-3 ${isRTL ? 'text-right' : 'text-left'}`}>
            {service.description[language]}
          </p>
          <div className={`flex items-center space-x-4 text-xs text-gray-500 ${isRTL ? 'flex-row-reverse space-x-reverse' : ''}`}>
            <div className="flex items-center space-x-1">
              <Icons.Clock />
              <span>{service.estimatedTime} min</span>
            </div>
            {service.requiresAppointment && (
              <div className="flex items-center space-x-1">
                <Icons.Calendar />
                <span>{language === 'ar' ? 'موعد مطلوب' : 'Appointment'}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Message Bubble Component (Enhanced for Virtual Front Desk)
export const MessageBubble = ({ message, isUser, currentUser, language }) => {
  const isRTL = language === 'ar';
  
  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString(language === 'ar' ? 'ar-SA' : 'en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatContent = (content) => {
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`(.*?)`/g, '<code class="bg-gray-700 px-1 py-0.5 rounded text-sm">$1</code>')
      .replace(/\n/g, '<br>');
  };

  return (
    <div className={`flex items-start space-x-4 ${isUser ? 'flex-row-reverse space-x-reverse' : ''} mb-8`}>
      <div className="flex-shrink-0">
        <img
          src={isUser ? currentUser?.avatar : 'https://images.pexels.com/photos/8728386/pexels-photo-8728386.jpeg'}
          alt={isUser ? 'User' : 'AI'}
          className="w-10 h-10 rounded-full object-cover ring-2 ring-gray-700"
        />
      </div>
      
      <div className={`flex-1 max-w-4xl ${isUser ? 'text-right' : 'text-left'}`}>
        <div className={`flex items-center space-x-2 mb-2 ${isUser ? 'justify-end' : 'justify-start'}`}>
          <span className="text-sm font-medium text-gray-300">
            {isUser ? (language === 'ar' ? 'أنت' : 'You') : 'MIND14'}
          </span>
          <span className="text-xs text-gray-500">
            {formatTime(message.timestamp)}
          </span>
          {message.confidence && (
            <span className="text-xs text-purple-400">
              ({Math.round(message.confidence * 100)}%)
            </span>
          )}
        </div>
        
        <div className={`${isUser ? 'flex justify-end' : 'flex justify-start'}`}>
          <div className={`max-w-full ${
            isUser 
              ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-3xl rounded-tr-lg px-6 py-4' 
              : 'bg-gray-800 text-gray-100 border border-gray-700 rounded-3xl rounded-tl-lg px-6 py-4'
          }`}>
            {message.attachments && message.attachments.length > 0 && (
              <div className="mb-3 space-y-2">
                {message.attachments.map((file, index) => (
                  <div key={index} className="flex items-center space-x-2 text-sm opacity-90">
                    <Icons.Paperclip className="w-3 h-3" />
                    <span>{file.name}</span>
                  </div>
                ))}
              </div>
            )}
            
            <div 
              className={`whitespace-pre-wrap leading-relaxed ${isUser ? 'text-right' : (isRTL ? 'text-right' : 'text-left')}`}
              dangerouslySetInnerHTML={{ __html: formatContent(message.content) }}
              dir={isRTL ? 'rtl' : 'ltr'}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

// Sidebar Component (Enhanced for Virtual Front Desk)
export const Sidebar = ({ 
  conversations, 
  activeConversationId, 
  onSelectConversation, 
  onNewConversation,
  onDeleteConversation,
  onLogout,
  currentUser,
  collapsed,
  onToggleCollapse,
  currentView,
  onViewChange,
  language
}) => {
  const [showUserMenu, setShowUserMenu] = useState(false);
  const isRTL = language === 'ar';

  const text = {
    en: {
      newChat: 'New Chat',
      admin: 'Admin Panel',
      chat: 'Chat Interface'
    },
    ar: {
      newChat: 'محادثة جديدة',
      admin: 'لوحة الإدارة',
      chat: 'واجهة المحادثة'
    }
  };

  const currentText = text[language];

  return (
    <div className={`bg-gray-950 border-r border-gray-800 flex flex-col transition-all duration-300 ${
      collapsed ? 'w-16' : 'w-80'
    }`} dir={isRTL ? 'rtl' : 'ltr'}>
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

      {/* View Toggle */}
      {!collapsed && (currentUser?.role === 'admin' || currentUser?.role === 'manager') && (
        <div className="p-4 space-y-2">
          <button
            onClick={() => onViewChange('chat')}
            className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
              currentView === 'chat' ? 'bg-gray-800 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
          >
            <Icons.Chat />
            <span className="text-sm">{currentText.chat}</span>
          </button>
          
          {currentUser?.role === 'admin' && (
            <button
              onClick={() => onViewChange('admin')}
              className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                currentView === 'admin' ? 'bg-gray-800 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              <Icons.Dashboard />
              <span className="text-sm">{currentText.admin}</span>
            </button>
          )}
          
          {(currentUser?.role === 'admin' || currentUser?.role === 'manager') && (
            <button
              onClick={() => onViewChange('management')}
              className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                currentView === 'management' ? 'bg-gray-800 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              <Icons.Management />
              <span className="text-sm">{currentText.management}</span>
            </button>
          )}
        </div>
      )}

      {/* New Chat Button */}
      {currentView === 'chat' && (
        <div className="p-4">
          <button
            onClick={onNewConversation}
            className={`w-full flex items-center space-x-3 px-4 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg transition-colors ${
              collapsed ? 'justify-center' : 'justify-start'
            }`}
          >
            <Icons.Plus />
            {!collapsed && <span className="font-medium">{currentText.newChat}</span>}
          </button>
        </div>
      )}

      {/* Conversations List */}
      {currentView === 'chat' && (
        <div className="flex-1 overflow-y-auto px-3">
          <div className="space-y-1">
            {conversations.map((conversation) => (
              <div
                key={conversation.id}
                className={`group relative flex items-center px-3 py-3 rounded-lg cursor-pointer transition-colors ${
                  activeConversationId === conversation.id
                    ? 'bg-gray-800 text-white'
                    : 'hover:bg-gray-800 text-gray-300 hover:text-white'
                } ${collapsed ? 'justify-center' : ''}`}
                onClick={() => onSelectConversation(conversation.id)}
              >
                {collapsed ? (
                  <div className="w-6 h-6 bg-gradient-to-r from-purple-500 to-pink-500 rounded text-white flex items-center justify-center text-xs font-bold">
                    {conversation.title[language]?.charAt(0) || conversation.title.en?.charAt(0)}
                  </div>
                ) : (
                  <>
                    <div className="flex-1 min-w-0 text-left">
                      <h3 className="text-sm font-medium truncate text-left leading-5 mb-1">
                        {conversation.title[language] || conversation.title.en}
                      </h3>
                      <div className="flex items-center space-x-2 text-xs text-gray-500">
                        <span>{conversation.messages.length} {language === 'ar' ? 'رسائل' : 'messages'}</span>
                        <span>•</span>
                        <span className={`px-2 py-1 rounded text-xs ${
                          conversation.status === 'completed' ? 'bg-green-900 text-green-300' :
                          conversation.status === 'pending' ? 'bg-yellow-900 text-yellow-300' :
                          'bg-blue-900 text-blue-300'
                        }`}>
                          {conversation.status}
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteConversation(conversation.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-gray-700 rounded transition-all flex-shrink-0 ml-2"
                    >
                      <Icons.Trash />
                    </button>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* User Profile */}
      <div className="p-3 border-t border-gray-800">
        <div className={`${collapsed ? 'flex justify-center' : ''}`}>
          <div className="relative">
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className={`flex items-center p-2 hover:bg-gray-800 rounded-lg transition-colors w-full ${
                collapsed ? 'justify-center' : 'justify-start text-left'
              }`}
            >
              <img
                src={currentUser?.avatar}
                alt="Profile"
                className="w-8 h-8 rounded-full object-cover flex-shrink-0"
              />
              {!collapsed && (
                <div className="flex-1 min-w-0 ml-3 text-left">
                  <p className="text-sm font-medium text-white truncate text-left">
                    {currentUser?.name}
                  </p>
                  <p className="text-xs text-gray-500 truncate text-left">
                    {currentUser?.role === 'admin' ? (language === 'ar' ? 'مشرف' : 'Admin') : (language === 'ar' ? 'مستخدم' : 'User')}
                  </p>
                </div>
              )}
              {!collapsed && (
                <Icons.Settings className="text-gray-400 flex-shrink-0 ml-2" />
              )}
            </button>
            
            {/* User Menu Dropdown */}
            {showUserMenu && !collapsed && (
              <div className="absolute bottom-full left-0 mb-2 w-full bg-gray-800 border border-gray-700 rounded-lg shadow-lg">
                <div className="p-2">
                  <button
                    onClick={() => {
                      setShowUserMenu(false);
                      alert('Settings panel coming soon!');
                    }}
                    className="flex items-center space-x-3 w-full px-3 py-2 text-left hover:bg-gray-700 rounded-lg transition-colors text-gray-300 hover:text-white"
                  >
                    <Icons.Settings />
                    <span className="text-sm">{language === 'ar' ? 'الإعدادات' : 'Settings'}</span>
                  </button>
                  <button
                    onClick={() => {
                      setShowUserMenu(false);
                      onLogout();
                    }}
                    className="flex items-center space-x-3 w-full px-3 py-2 text-left hover:bg-gray-700 rounded-lg transition-colors text-gray-300 hover:text-white"
                  >
                    <Icons.LogOut />
                    <span className="text-sm">{language === 'ar' ? 'تسجيل الخروج' : 'Sign out'}</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Virtual Front Desk Interface Component
export const VirtualDeskInterface = ({ 
  conversation, 
  onSendMessage, 
  availableServices,
  currentUser,
  sidebarCollapsed,
  language,
  onLanguageChange,
  sessionData
}) => {
  const [message, setMessage] = useState('');
  const [attachments, setAttachments] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showFileUpload, setShowFileUpload] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const isRTL = language === 'ar';

  const text = {
    en: {
      title: 'Virtual Front Desk',
      subtitle: 'AI-Powered Service Assistant',
      welcomeTitle: 'Welcome to MIND14 Virtual Front Desk',
      welcomeDesc: 'Your AI-powered virtual assistant for government, medical, and educational services. Get help with appointments, renewals, consultations, and more.',
      services: 'Available Services',
      placeholder: 'Type your message...',
      disclaimer: 'MIND14 can make mistakes. Consider checking important information.',
      quickActions: {
        healthCard: 'Renew Health Card',
        idCard: 'Replace ID Card',
        medical: 'Book Medical Appointment',
        student: 'Student Enrollment'
      }
    },
    ar: {
      title: 'مكتب الاستقبال الافتراضي',
      subtitle: 'مساعد الخدمات المدعوم بالذكاء الاصطناعي',
      welcomeTitle: 'مرحباً بك في مكتب الاستقبال الافتراضي MIND14',
      welcomeDesc: 'مساعدك الافتراضي المدعوم بالذكاء الاصطناعي للخدمات الحكومية والطبية والتعليمية. احصل على المساعدة في المواعيد والتجديدات والاستشارات وأكثر.',
      services: 'الخدمات المتاحة',
      placeholder: 'اكتب رسالتك...',
      disclaimer: 'MIND14 قد يرتكب أخطاء. فكر في التحقق من المعلومات المهمة.',
      quickActions: {
        healthCard: 'تجديد البطاقة الصحية',
        idCard: 'استبدال بطاقة الهوية',
        medical: 'حجز موعد طبي',
        student: 'تسجيل الطلاب'
      }
    }
  };

  const currentText = text[language];

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
    setShowFileUpload(false);
    setIsLoading(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleServiceClick = (service) => {
    setMessage(`I need help with ${service.name[language]}`);
  };

  const handleQuickAction = (action) => {
    const messages = {
      healthCard: language === 'ar' ? 'أريد تجديد بطاقتي الصحية' : 'I need to renew my health card',
      idCard: language === 'ar' ? 'أريد استبدال بطاقة الهوية' : 'I need to replace my ID card',
      medical: language === 'ar' ? 'أريد حجز موعد طبي' : 'I want to book a medical appointment',
      student: language === 'ar' ? 'أريد المساعدة في تسجيل الطلاب' : 'I need help with student enrollment'
    };
    setMessage(messages[action]);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900" dir={isRTL ? 'rtl' : 'ltr'}>
      {/* Header */}
      <div className="border-b border-gray-800 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h2 className="text-lg font-semibold text-white">
              {conversation?.title?.[language] || conversation?.title?.en || currentText.title}
            </h2>
            {conversation && (
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-400">
                  {conversation.messages.length} {language === 'ar' ? 'رسائل' : 'messages'}
                </span>
                {conversation.status && (
                  <span className={`px-2 py-1 rounded text-xs ${
                    conversation.status === 'completed' ? 'bg-green-900 text-green-300' :
                    conversation.status === 'pending' ? 'bg-yellow-900 text-yellow-300' :
                    'bg-blue-900 text-blue-300'
                  }`}>
                    {conversation.status}
                  </span>
                )}
              </div>
            )}
          </div>
          <LanguageSelector currentLanguage={language} onLanguageChange={onLanguageChange} />
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6">
        {!conversation ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="max-w-4xl space-y-8">
              {/* MIND14 Logo Hero */}
              <div className="relative">
                <div className="w-32 h-32 mx-auto bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-3xl backdrop-blur-sm border border-purple-500/20 flex items-center justify-center">
                  <div className="w-20 h-20 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center">
                    <div className="w-12 h-12 relative">
                      <div className="absolute top-1 left-3 w-2 h-2 bg-white rounded-full animate-pulse"></div>
                      <div className="absolute bottom-1 right-3 w-2 h-2 bg-white rounded-full animate-pulse" style={{ animationDelay: '0.5s' }}></div>
                      <div className="absolute top-3 right-1 w-2 h-2 bg-white rounded-full animate-pulse" style={{ animationDelay: '1s' }}></div>
                      <div className="absolute bottom-3 left-1 w-2 h-2 bg-white rounded-full animate-pulse" style={{ animationDelay: '1.5s' }}></div>
                      <div className="absolute top-2 left-6 w-0.5 h-8 bg-white rounded rotate-45 opacity-80"></div>
                      <div className="absolute top-2 right-6 w-0.5 h-8 bg-white rounded -rotate-45 opacity-80"></div>
                    </div>
                  </div>
                </div>
                {/* Animated background elements */}
                <div className="absolute inset-0 -z-10">
                  <div className="absolute top-4 left-4 w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
                  <div className="absolute top-8 right-8 w-1 h-1 bg-pink-400 rounded-full animate-bounce" style={{ animationDelay: '0.3s' }}></div>
                  <div className="absolute bottom-6 left-8 w-1.5 h-1.5 bg-purple-300 rounded-full animate-bounce" style={{ animationDelay: '0.6s' }}></div>
                  <div className="absolute bottom-4 right-4 w-1 h-1 bg-pink-300 rounded-full animate-bounce" style={{ animationDelay: '0.9s' }}></div>
                </div>
              </div>
              
              <div className="space-y-6">
                <h3 className="text-3xl font-bold text-white">{currentText.welcomeTitle}</h3>
                <p className="text-gray-400 leading-relaxed text-lg max-w-2xl mx-auto">
                  {currentText.welcomeDesc}
                </p>
                
                {/* Quick Action Buttons */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto">
                  {[
                    { key: 'healthCard', icon: '🏥' },
                    { key: 'idCard', icon: '🆔' },
                    { key: 'medical', icon: '👩‍⚕️' },
                    { key: 'student', icon: '🎓' }
                  ].map((action) => (
                    <button
                      key={action.key}
                      onClick={() => handleQuickAction(action.key)}
                      className="flex flex-col items-center space-y-2 p-4 bg-gray-800 hover:bg-gradient-to-r hover:from-purple-600 hover:to-pink-600 text-gray-300 hover:text-white rounded-xl text-sm transition-all duration-200 transform hover:scale-105"
                    >
                      <span className="text-2xl">{action.icon}</span>
                      <span className="text-center leading-tight">{currentText.quickActions[action.key]}</span>
                    </button>
                  ))}
                </div>

                {/* Available Services */}
                <div className="space-y-4">
                  <h4 className="text-xl font-semibold text-white">{currentText.services}</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-4xl mx-auto">
                    {availableServices.map((service) => (
                      <ServiceCard
                        key={service.id}
                        service={service}
                        language={language}
                        onClick={handleServiceClick}
                      />
                    ))}
                  </div>
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
                language={language}
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
          <div className="relative">
            <div className="flex items-end space-x-3">
              <div className="flex-1 relative">
                <textarea
                  ref={textareaRef}
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={currentText.placeholder}
                  className="w-full pl-4 pr-16 py-3 bg-gray-800 border border-gray-700 rounded-2xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none min-h-[50px] max-h-32"
                  rows={1}
                  style={{ height: 'auto' }}
                  onInput={(e) => {
                    e.target.style.height = 'auto';
                    e.target.style.height = e.target.scrollHeight + 'px';
                  }}
                  dir={isRTL ? 'rtl' : 'ltr'}
                />
                
                {/* Attachment Icon Inside Input */}
                <button
                  onClick={() => setShowFileUpload(!showFileUpload)}
                  className={`absolute ${isRTL ? 'left-3' : 'right-3'} bottom-3 p-2 rounded-lg transition-colors ${
                    showFileUpload || attachments.length > 0
                      ? 'text-purple-400 bg-purple-400/10' 
                      : 'text-gray-400 hover:text-purple-400 hover:bg-gray-700'
                  }`}
                  title="Attach files"
                >
                  <Icons.Paperclip />
                </button>
              </div>
              
              {/* Always Visible Send Button */}
              <button
                onClick={handleSend}
                disabled={(!message.trim() && attachments.length === 0) || isLoading}
                className="p-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-600 text-white rounded-xl transition-all duration-200 transform hover:scale-105 disabled:cursor-not-allowed disabled:transform-none flex-shrink-0"
                title="Send message"
              >
                <Icons.Send />
              </button>
            </div>
            
            {/* File Upload (Expandable) */}
            {showFileUpload && (
              <div className="absolute bottom-full left-0 right-0 mb-2 p-4 bg-gray-800 rounded-lg border border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">
                    {language === 'ar' ? 'إرفاق ملفات' : 'Attach Files'}
                  </span>
                  <button
                    onClick={() => setShowFileUpload(false)}
                    className="text-gray-400 hover:text-gray-300"
                  >
                    <Icons.X />
                  </button>
                </div>
                
                <div
                  className="border-2 border-dashed rounded-lg p-4 text-center transition-colors border-gray-600 hover:border-gray-500"
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    const files = Array.from(e.dataTransfer.files);
                    setAttachments(prev => [...prev, ...files]);
                    setShowFileUpload(false);
                  }}
                >
                  <input
                    type="file"
                    multiple
                    onChange={(e) => {
                      setAttachments(prev => [...prev, ...Array.from(e.target.files)]);
                      setShowFileUpload(false);
                    }}
                    className="hidden"
                    accept=".txt,.md,.json,.js,.ts,.png,.jpg,.jpeg,.gif,.webp,.pdf,.doc,.docx,.csv"
                    id="file-upload"
                  />
                  
                  <div className="space-y-2">
                    <Icons.Paperclip className="mx-auto text-gray-400" />
                    <p className="text-gray-400 text-sm">
                      <label
                        htmlFor="file-upload"
                        className="text-purple-400 hover:text-purple-300 underline cursor-pointer"
                      >
                        {language === 'ar' ? 'انقر للرفع' : 'Click to upload'}
                      </label>
                      {' '}{language === 'ar' ? 'أو اسحب وأفلت' : 'or drag and drop'}
                    </p>
                    <p className="text-xs text-gray-500">
                      {language === 'ar' ? 'يدعم: النصوص، الصور، المستندات' : 'Supports: Text, Images, Documents'}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          <div className="mt-2 text-xs text-gray-500 text-center">
            {currentText.disclaimer}
          </div>
        </div>
      </div>
    </div>
  );
};

// Admin Dashboard Component
export const AdminDashboard = ({ conversations, services, language, onLanguageChange }) => {
  const [selectedTab, setSelectedTab] = useState('overview');
  const isRTL = language === 'ar';

  const text = {
    en: {
      title: 'Admin Dashboard',
      overview: 'Overview',
      conversations: 'Conversations',
      services: 'Services',
      analytics: 'Analytics',
      totalConversations: 'Total Conversations',
      activeChats: 'Active Chats',
      completedServices: 'Completed Services',
      avgResponseTime: 'Avg Response Time',
      recentActivity: 'Recent Activity',
      serviceUsage: 'Service Usage',
      intentAccuracy: 'Intent Detection Accuracy'
    },
    ar: {
      title: 'لوحة الإدارة',
      overview: 'نظرة عامة',
      conversations: 'المحادثات',
      services: 'الخدمات',
      analytics: 'التحليلات',
      totalConversations: 'إجمالي المحادثات',
      activeChats: 'المحادثات النشطة',
      completedServices: 'الخدمات المكتملة',
      avgResponseTime: 'متوسط وقت الاستجابة',
      recentActivity: 'النشاط الحديث',
      serviceUsage: 'استخدام الخدمات',
      intentAccuracy: 'دقة كشف النوايا'
    }
  };

  const currentText = text[language];

  const stats = {
    totalConversations: conversations.length,
    activeChats: conversations.filter(c => c.status === 'active').length,
    completedServices: conversations.filter(c => c.status === 'completed').length,
    avgResponseTime: '2.3s'
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900" dir={isRTL ? 'rtl' : 'ltr'}>
      {/* Header */}
      <div className="border-b border-gray-800 p-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-white">{currentText.title}</h1>
          <LanguageSelector currentLanguage={language} onLanguageChange={onLanguageChange} />
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-800">
        <div className="flex space-x-8 px-6">
          {[
            { id: 'overview', label: currentText.overview },
            { id: 'conversations', label: currentText.conversations },
            { id: 'services', label: currentText.services },
            { id: 'analytics', label: currentText.analytics }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedTab(tab.id)}
              className={`py-4 px-2 border-b-2 font-medium text-sm ${
                selectedTab === tab.id
                  ? 'border-purple-500 text-purple-600'
                  : 'border-transparent text-gray-500 hover:text-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {selectedTab === 'overview' && (
          <div className="space-y-6">
            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[
                { label: currentText.totalConversations, value: stats.totalConversations, icon: Icons.Chat, color: 'blue' },
                { label: currentText.activeChats, value: stats.activeChats, icon: Icons.ExclamationCircle, color: 'yellow' },
                { label: currentText.completedServices, value: stats.completedServices, icon: Icons.CheckCircle, color: 'green' },
                { label: currentText.avgResponseTime, value: stats.avgResponseTime, icon: Icons.Clock, color: 'purple' }
              ].map((stat, index) => (
                <div key={index} className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                  <div className="flex items-center">
                    <div className={`p-2 rounded-lg bg-${stat.color}-900`}>
                      <stat.icon className={`w-6 h-6 text-${stat.color}-400`} />
                    </div>
                    <div className="ml-4">
                      <p className="text-sm font-medium text-gray-400">{stat.label}</p>
                      <p className="text-2xl font-bold text-white">{stat.value}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Recent Activity */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">{currentText.recentActivity}</h3>
              <div className="space-y-4">
                {conversations.slice(0, 5).map((conv) => (
                  <div key={conv.id} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <div>
                        <p className="text-white font-medium">{conv.title[language] || conv.title.en}</p>
                        <p className="text-gray-400 text-sm">{conv.messages.length} messages</p>
                      </div>
                    </div>
                    <span className={`px-2 py-1 rounded text-xs ${
                      conv.status === 'completed' ? 'bg-green-900 text-green-300' :
                      conv.status === 'pending' ? 'bg-yellow-900 text-yellow-300' :
                      'bg-blue-900 text-blue-300'
                    }`}>
                      {conv.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'conversations' && (
          <div className="space-y-6">
            <div className="bg-gray-800 rounded-lg border border-gray-700">
              <div className="p-6 border-b border-gray-700">
                <h3 className="text-lg font-semibold text-white">{currentText.conversations}</h3>
              </div>
              <div className="divide-y divide-gray-700">
                {conversations.map((conv) => (
                  <div key={conv.id} className="p-6 hover:bg-gray-750">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <h4 className="text-white font-medium">{conv.title[language] || conv.title.en}</h4>
                        <div className="mt-1 flex items-center space-x-4 text-sm text-gray-400">
                          <span>{conv.messages.length} messages</span>
                          <span>•</span>
                          <span>{conv.type}</span>
                          <span>•</span>
                          <span>{new Date(conv.createdAt).toLocaleDateString()}</span>
                        </div>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-xs ${
                        conv.status === 'completed' ? 'bg-green-900 text-green-300' :
                        conv.status === 'pending' ? 'bg-yellow-900 text-yellow-300' :
                        'bg-blue-900 text-blue-300'
                      }`}>
                        {conv.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'services' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {services.map((service) => (
                <div key={service.id} className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                  <div className="flex items-start space-x-4">
                    <div className="text-3xl">{service.icon}</div>
                    <div className="flex-1">
                      <h3 className="text-white font-semibold">{service.name[language]}</h3>
                      <p className="text-gray-400 text-sm mt-1">{service.description[language]}</p>
                      <div className="mt-3 space-y-1">
                        <div className="flex items-center text-xs text-gray-500">
                          <Icons.Clock className="w-3 h-3 mr-1" />
                          <span>{service.estimatedTime} min</span>
                        </div>
                        <div className="flex items-center text-xs text-gray-500">
                          <Icons.Calendar className="w-3 h-3 mr-1" />
                          <span>{service.requiresAppointment ? 'Appointment required' : 'No appointment'}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {selectedTab === 'analytics' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4">{currentText.serviceUsage}</h3>
                <div className="space-y-4">
                  {services.map((service, index) => (
                    <div key={service.id} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-300">{service.name[language]}</span>
                        <span className="text-gray-400">{Math.floor(Math.random() * 100)}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full" 
                          style={{ width: `${Math.floor(Math.random() * 100)}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4">{currentText.intentAccuracy}</h3>
                <div className="text-center">
                  <div className="text-4xl font-bold text-green-400 mb-2">94.2%</div>
                  <p className="text-gray-400">Average intent detection accuracy</p>
                  <div className="mt-4 space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Health Card Renewal</span>
                      <span className="text-green-400">96%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Medical Consultation</span>
                      <span className="text-green-400">95%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">ID Card Replacement</span>
                      <span className="text-yellow-400">92%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Student Enrollment</span>
                      <span className="text-green-400">94%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Export placeholders for unused original components
export const ModelSelector = () => null;
export const FileUpload = () => null;
export const FilePreview = () => null;
export const BookingInterface = () => null;