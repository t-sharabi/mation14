from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import json
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import httpx
import ollama
from enum import Enum

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="MIND14 Virtual Front Desk API", version="1.0.0")
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enums
class ConversationStatus(str, Enum):
    ACTIVE = "active"
    PENDING = "pending" 
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class ServiceCategory(str, Enum):
    GOVERNMENT = "government"
    MEDICAL = "medical"
    EDUCATION = "education"
    GENERAL = "general"

# Data Models
class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    language: str = "en"
    intent: Optional[str] = None
    confidence: Optional[float] = None
    attachments: List[str] = []

class ServiceInfo(BaseModel):
    id: str
    name: Dict[str, str]  # {"en": "English name", "ar": "Arabic name"}
    category: ServiceCategory
    description: Dict[str, str]
    estimated_time: int  # in minutes
    requires_appointment: bool
    icon: str
    working_hours: Dict[str, str]  # {"start": "08:00", "end": "16:00"}
    available_days: List[str]

class SessionData(BaseModel):
    step: str = "greeting"
    selected_service: Optional[str] = None
    collected_info: Dict[str, Any] = {}
    intent: Optional[str] = None
    confidence: float = 0.0
    booking_step: Optional[str] = None
    appointment_id: Optional[str] = None

class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: Dict[str, str] = {"en": "New Chat", "ar": "محادثة جديدة"}
    type: str = "general_inquiry"
    status: ConversationStatus = ConversationStatus.ACTIVE
    service: Optional[str] = None
    language: str = "en"
    messages: List[Message] = []
    session_data: SessionData = Field(default_factory=SessionData)
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    phone_number: Optional[str] = None
    role: str = "user"  # "user" or "admin"
    avatar: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    language: str = "en"
    attachments: List[str] = []

class ChatResponse(BaseModel):
    message: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    conversation_id: str
    session_data: SessionData
    actions: List[str] = []

class BookingData(BaseModel):
    appointment_id: str
    service: ServiceInfo
    customer_info: Dict[str, str]
    language: str
    timestamp: str

# Available Services Configuration
AVAILABLE_SERVICES = [
    ServiceInfo(
        id="health-card-renewal",
        name={"en": "Health Card Renewal", "ar": "تجديد البطاقة الصحية"},
        category=ServiceCategory.GOVERNMENT,
        description={"en": "Renew your health insurance card", "ar": "تجديد بطاقة التأمين الصحي"},
        estimated_time=30,
        requires_appointment=True,
        icon="🏥",
        working_hours={"start": "08:00", "end": "16:00"},
        available_days=["monday", "tuesday", "wednesday", "thursday", "friday"]
    ),
    ServiceInfo(
        id="id-card-replacement",
        name={"en": "ID Card Replacement", "ar": "استبدال بطاقة الهوية"},
        category=ServiceCategory.GOVERNMENT,
        description={"en": "Replace lost or damaged ID card", "ar": "استبدال بطاقة الهوية المفقودة أو التالفة"},
        estimated_time=45,
        requires_appointment=True,
        icon="🆔",
        working_hours={"start": "08:00", "end": "15:00"},
        available_days=["sunday", "tuesday", "thursday"]
    ),
    ServiceInfo(
        id="medical-consultation",
        name={"en": "Medical Consultation", "ar": "استشارة طبية"},
        category=ServiceCategory.MEDICAL,
        description={"en": "Book appointment with doctor", "ar": "حجز موعد مع الطبيب"},
        estimated_time=20,
        requires_appointment=True,
        icon="👩‍⚕️",
        working_hours={"start": "09:00", "end": "17:00"},
        available_days=["sunday", "monday", "tuesday", "wednesday", "thursday"]
    ),
    ServiceInfo(
        id="student-enrollment",
        name={"en": "Student Enrollment", "ar": "تسجيل الطلاب"},
        category=ServiceCategory.EDUCATION,
        description={"en": "Enroll in courses and programs", "ar": "التسجيل في الدورات والبرامج"},
        estimated_time=60,
        requires_appointment=True,
        icon="🎓",
        working_hours={"start": "08:00", "end": "14:00"},
        available_days=["sunday", "monday", "tuesday", "wednesday", "thursday"]
    ),
    ServiceInfo(
        id="general-inquiry",
        name={"en": "General Inquiry", "ar": "استفسار عام"},
        category=ServiceCategory.GENERAL,
        description={"en": "Ask any question or get information", "ar": "اطرح أي سؤال أو احصل على معلومات"},
        estimated_time=10,
        requires_appointment=False,
        icon="💬",
        working_hours={"start": "00:00", "end": "23:59"},
        available_days=["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
    )
]

# Mistral AI Integration
class MistralService:
    def __init__(self):
        self.model_name = "mistral:7b-instruct-q4_0"  # or q5_0 for better quality
        
    async def ensure_model_available(self):
        """Ensure Mistral model is available - fallback to rule-based for demo"""
        try:
            # Try to check if Ollama is available
            import subprocess
            result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
            if result.returncode == 0:
                models = await asyncio.to_thread(ollama.list)
                model_names = [model['name'] for model in models['models']]
                
                if self.model_name not in model_names:
                    logger.info(f"Pulling Mistral model: {self.model_name}")
                    await asyncio.to_thread(ollama.pull, self.model_name)
                    logger.info("Mistral model pulled successfully")
                
                return True
            else:
                logger.warning("Ollama not available, using fallback AI system")
                return False
        except Exception as e:
            logger.warning(f"Ollama not available ({e}), using fallback AI system")
            return False

    async def classify_intent(self, user_input: str, language: str = "en") -> Dict[str, Any]:
        """Classify user intent - with fallback to rule-based system"""
        try:
            # Try Mistral first
            if await self.ensure_model_available():
                return await self._classify_with_mistral(user_input, language)
            else:
                # Fallback to enhanced rule-based system
                return self._fallback_intent_classification(user_input, language)
            
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return self._fallback_intent_classification(user_input, language)

    async def _classify_with_mistral(self, user_input: str, language: str) -> Dict[str, Any]:
        """Classify using actual Mistral model"""
        system_prompt = self._get_intent_classification_prompt(language)
        user_prompt = f"User input: {user_input}"
        
        response = await asyncio.to_thread(
            ollama.generate,
            model=self.model_name,
            prompt=f"{system_prompt}\n\n{user_prompt}",
            stream=False
        )
        
        result = self._parse_intent_response(response['response'])
        logger.info(f"Mistral intent classification result: {result}")
        return result

    def _get_intent_classification_prompt(self, language: str) -> str:
        """Get the system prompt for intent classification"""
        if language == "ar":
            return """أنت مساعد ذكي لمكتب الاستقبال الافتراضي. مهمتك تصنيف نوايا المستخدمين بناءً على رسائلهم.

الخدمات المتاحة:
1. health_card_renewal - تجديد البطاقة الصحية
2. id_card_replacement - استبدال بطاقة الهوية
3. medical_consultation - استشارة طبية
4. student_enrollment - تسجيل الطلاب
5. general_inquiry - استفسار عام

قم بتحليل رسالة المستخدم وأعد النتيجة بصيغة JSON:
{
  "intent": "اسم النية",
  "confidence": نسبة الثقة (0.0-1.0),
  "service_id": "معرف الخدمة أو null",
  "entities": {"كيانات مستخرجة"}
}"""
        else:
            return """You are an AI assistant for a virtual front desk. Your task is to classify user intents based on their messages.

Available services:
1. health_card_renewal - Renew health insurance card
2. id_card_replacement - Replace lost or damaged ID card  
3. medical_consultation - Book doctor appointment
4. student_enrollment - Enroll in courses and programs
5. general_inquiry - General questions and information

Analyze the user's message and return the result in JSON format:
{
  "intent": "intent_name",
  "confidence": confidence_score (0.0-1.0),
  "service_id": "service_id or null",
  "entities": {"extracted_entities"}
}"""

    def _parse_intent_response(self, response: str) -> Dict[str, Any]:
        """Parse Mistral's intent classification response"""
        try:
            # Try to extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Validate and normalize the result
                return {
                    "intent": result.get("intent", "general_inquiry"),
                    "confidence": min(max(float(result.get("confidence", 0.5)), 0.0), 1.0),
                    "service_id": result.get("service_id"),
                    "entities": result.get("entities", {})
                }
            else:
                # Fallback to rule-based classification
                return self._fallback_intent_classification(response)
                
        except Exception as e:
            logger.error(f"Error parsing intent response: {e}")
            return self._fallback_intent_classification(response)

    def _fallback_intent_classification(self, text: str) -> Dict[str, Any]:
        """Fallback rule-based intent classification"""
        text_lower = text.lower()
        
        intent_patterns = {
            "health_card_renewal": ["health card", "renew", "renewal", "health insurance", "medical card", "بطاقة صحية", "تجديد"],
            "id_card_replacement": ["id card", "identity", "replace", "lost id", "damaged id", "بطاقة هوية", "استبدال"],
            "medical_consultation": ["doctor", "appointment", "medical", "consultation", "طبيب", "موعد", "استشارة"],
            "student_enrollment": ["enroll", "student", "course", "register", "education", "تسجيل", "طالب", "دورة"]
        }
        
        max_score = 0
        detected_intent = "general_inquiry"
        
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > max_score:
                max_score = score
                detected_intent = intent
        
        confidence = min(0.5 + (max_score * 0.2), 0.95)
        service_id = detected_intent.replace("_", "-") if detected_intent != "general_inquiry" else None
        
        return {
            "intent": detected_intent,
            "confidence": confidence,
            "service_id": service_id,
            "entities": {}
        }

    async def generate_response(self, user_input: str, session_data: SessionData, language: str = "en") -> Dict[str, Any]:
        """Generate AI response based on conversation context"""
        try:
            system_prompt = self._get_response_generation_prompt(session_data, language)
            user_prompt = f"User input: {user_input}"
            
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.model_name,
                prompt=f"{system_prompt}\n\n{user_prompt}",
                stream=False
            )
            
            return {
                "message": response['response'].strip(),
                "session_data": session_data
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "message": "I apologize, but I'm having trouble processing your request. Please try again.",
                "session_data": session_data
            }

    def _get_response_generation_prompt(self, session_data: SessionData, language: str) -> str:
        """Get system prompt for response generation based on session context"""
        if language == "ar":
            base_prompt = """أنت MIND14، مساعد ذكي لمكتب الاستقبال الافتراضي. أنت مهذب ومفيد ومحترف.

مهمتك مساعدة المستخدمين في:
- تجديد البطاقة الصحية
- استبدال بطاقة الهوية  
- حجز المواعيد الطبية
- تسجيل الطلاب
- الاستفسارات العامة

قم بالرد بشكل ودود ومهني باللغة العربية."""
        else:
            base_prompt = """You are MIND14, an AI assistant for a virtual front desk. You are polite, helpful, and professional.

Your role is to assist users with:
- Health card renewals
- ID card replacements
- Medical appointments
- Student enrollment
- General inquiries

Respond in a friendly and professional manner."""

        # Add context based on session step
        if session_data.step == "booking":
            if language == "ar":
                base_prompt += "\n\nأنت حالياً في عملية حجز موعد. اجمع المعلومات المطلوبة بشكل منظم."
            else:
                base_prompt += "\n\nYou are currently in a booking process. Collect required information systematically."
        
        return base_prompt

# Initialize Mistral service
mistral_service = MistralService()

# API Routes
@api_router.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting MIND14 Virtual Front Desk API...")
    await mistral_service.ensure_model_available()
    logger.info("API startup completed")

@api_router.get("/")
async def root():
    return {"message": "MIND14 Virtual Front Desk API", "version": "1.0.0"}

@api_router.get("/services", response_model=List[ServiceInfo])
async def get_services():
    """Get available services"""
    return AVAILABLE_SERVICES

@api_router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with Mistral AI integration"""
    try:
        # Get or create conversation
        conversation = None
        if request.conversation_id:
            conversation_data = await db.conversations.find_one({"id": request.conversation_id})
            if conversation_data:
                conversation = Conversation(**conversation_data)
        
        if not conversation:
            # Create new conversation
            conversation = Conversation(
                language=request.language,
                user_id="demo_user"  # In production, get from auth
            )
            await db.conversations.insert_one(conversation.dict())

        # Add user message
        user_message = Message(
            role=MessageRole.USER,
            content=request.message,
            language=request.language,
            attachments=request.attachments
        )
        conversation.messages.append(user_message)

        # Classify intent using Mistral
        intent_result = await mistral_service.classify_intent(request.message, request.language)
        
        # Generate AI response
        ai_response = await process_conversation(
            request.message, 
            conversation.session_data, 
            intent_result,
            request.language
        )

        # Add AI message
        ai_message = Message(
            role=MessageRole.ASSISTANT,
            content=ai_response["message"],
            language=request.language,
            intent=intent_result["intent"],
            confidence=intent_result["confidence"]
        )
        conversation.messages.append(ai_message)
        conversation.session_data = ai_response["session_data"]
        conversation.updated_at = datetime.utcnow()

        # Update title if needed
        if len(conversation.messages) == 2:  # First user message + AI response
            conversation.title = generate_conversation_title(request.message, request.language)

        # Save conversation
        await db.conversations.replace_one(
            {"id": conversation.id}, 
            conversation.dict()
        )

        # Trigger n8n webhook if booking completed
        if ai_response.get("trigger_webhook") and ai_response.get("booking_data"):
            await trigger_n8n_webhook(ai_response["booking_data"])

        return ChatResponse(
            message=ai_response["message"],
            intent=intent_result["intent"],
            confidence=intent_result["confidence"],
            conversation_id=conversation.id,
            session_data=ai_response["session_data"],
            actions=ai_response.get("actions", [])
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@api_router.get("/conversations", response_model=List[Conversation])
async def get_conversations(user_id: str = "demo_user"):
    """Get user's conversations"""
    conversations_data = await db.conversations.find(
        {"user_id": user_id}
    ).sort("updated_at", -1).to_list(100)
    
    return [Conversation(**conv) for conv in conversations_data]

@api_router.post("/n8n/book-appointment")
async def n8n_booking_webhook(booking_data: BookingData):
    """n8n webhook endpoint for booking automation"""
    try:
        logger.info(f"n8n booking webhook triggered: {booking_data.appointment_id}")
        
        # Here you would trigger your n8n workflow
        # For now, we'll just log and return success
        
        # Example: Send to n8n webhook
        # async with httpx.AsyncClient() as client:
        #     await client.post(
        #         "YOUR_N8N_WEBHOOK_URL",
        #         json=booking_data.dict()
        #     )
        
        return {"status": "success", "message": "Booking automation triggered"}
        
    except Exception as e:
        logger.error(f"Error in n8n webhook: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")

# Helper Functions
async def process_conversation(user_input: str, session_data: SessionData, intent_result: Dict, language: str) -> Dict[str, Any]:
    """Process conversation based on current session state"""
    
    # Update session with intent information
    session_data.intent = intent_result["intent"]
    session_data.confidence = intent_result["confidence"]
    
    if intent_result["service_id"]:
        session_data.selected_service = intent_result["service_id"]
    
    # Process based on current step
    if session_data.step == "greeting" or not session_data.intent:
        return handle_greeting(user_input, intent_result, session_data, language)
    elif session_data.step == "service_selection":
        return handle_service_selection(user_input, session_data, language)
    elif session_data.step == "booking":
        return handle_booking(user_input, session_data, language)
    else:
        return handle_general(user_input, session_data, language)

def handle_greeting(user_input: str, intent_result: Dict, session_data: SessionData, language: str) -> Dict[str, Any]:
    """Handle initial greeting and service identification"""
    service = None
    if intent_result["service_id"]:
        service = next((s for s in AVAILABLE_SERVICES if s.id == intent_result["service_id"]), None)
    
    if language == "ar":
        if service:
            message = f"""مرحباً! أفهم أنك تحتاج مساعدة في **{service.name[language]}**. 

🕒 **تفاصيل الخدمة:**
• المدة المقدرة: {service.estimated_time} دقيقة
• {service.icon} {service.description[language]}

{'📅 تتطلب هذه الخدمة حجز موعد.' if service.requires_appointment else '💬 هذه خدمة استفسار عام.'}

هل تريد المتابعة مع هذه الخدمة؟"""
            session_data.step = "service_selection"
        else:
            message = """مرحباً! أنا MIND14، مساعدك الافتراضي الذكي.

🏛️ **يمكنني مساعدتك في:**
• 🏥 تجديد البطاقة الصحية
• 🆔 استبدال بطاقة الهوية
• 👩‍⚕️ حجز المواعيد الطبية
• 🎓 تسجيل الطلاب
• 💬 الاستفسارات العامة

كيف يمكنني مساعدتك اليوم؟"""
            session_data.step = "intent_detection"
    else:
        if service:
            message = f"""Hello! I understand you need help with **{service.name[language]}**.

🕒 **Service Details:**
• Estimated time: {service.estimated_time} minutes
• {service.icon} {service.description[language]}

{'📅 This service requires an appointment.' if service.requires_appointment else '💬 This is a general inquiry service.'}

Would you like to proceed with this service?"""
            session_data.step = "service_selection"
        else:
            message = """Hello! I'm MIND14, your AI virtual assistant.

🏛️ **I can help you with:**
• 🏥 Health card renewal
• 🆔 ID card replacement
• 👩‍⚕️ Medical appointments
• 🎓 Student enrollment
• 💬 General inquiries

How can I assist you today?"""
            session_data.step = "intent_detection"
    
    return {
        "message": message,
        "session_data": session_data,
        "actions": []
    }

def handle_service_selection(user_input: str, session_data: SessionData, language: str) -> Dict[str, Any]:
    """Handle service confirmation and proceed to booking"""
    confirmation_words = {
        "en": ["yes", "sure", "ok", "okay", "proceed", "continue", "confirm"],
        "ar": ["نعم", "موافق", "حسنا", "متابعة", "استمر", "أكد", "موافقة"]
    }
    
    is_confirming = any(word in user_input.lower() for word in confirmation_words[language])
    
    if is_confirming and session_data.selected_service:
        service = next((s for s in AVAILABLE_SERVICES if s.id == session_data.selected_service), None)
        
        if service and service.requires_appointment:
            if language == "ar":
                message = f"""ممتاز! سأساعدك في حجز موعد لـ **{service.name[language]}**.

📋 **المعلومات المطلوبة:**
• الاسم الكامل
• رقم الهاتف
• التاريخ والوقت المفضل

⏰ **ساعات العمل:** {service.working_hours['start']} - {service.working_hours['end']}

لنبدأ - ما هو اسمك الكامل؟"""
            else:
                message = f"""Great! I'll help you book an appointment for **{service.name[language]}**.

📋 **Required Information:**
• Full name
• Phone number
• Preferred date and time

⏰ **Working hours:** {service.working_hours['start']} - {service.working_hours['end']}

Let's start - what's your full name?"""
            
            session_data.step = "booking"
            session_data.booking_step = "name"
        else:
            if language == "ar":
                message = f"""أنا هنا لمساعدتك في **{service.name[language]}**. 

هذه خدمة استفسار عام، لذا يمكنك طرح أي أسئلة تريدها حول هذا الموضوع."""
            else:
                message = f"""I'm here to help with **{service.name[language]}**. 

This is a general inquiry service, so feel free to ask any questions you have about this topic."""
            
            session_data.step = "general_inquiry"
    else:
        # Show service options
        if language == "ar":
            services_text = "\n".join([f"{s.icon} **{s.name['ar']}** - {s.description['ar']}" for s in AVAILABLE_SERVICES])
            message = f"يمكنني مساعدتك في الخدمات التالية:\n\n{services_text}\n\nأي خدمة تهمك؟"
        else:
            services_text = "\n".join([f"{s.icon} **{s.name['en']}** - {s.description['en']}" for s in AVAILABLE_SERVICES])
            message = f"I can help you with these services:\n\n{services_text}\n\nWhich service interests you?"
    
    return {
        "message": message,
        "session_data": session_data,
        "actions": []
    }

def handle_booking(user_input: str, session_data: SessionData, language: str) -> Dict[str, Any]:
    """Handle multi-step booking process"""
    booking_step = session_data.booking_step or "name"
    
    if booking_step == "name":
        session_data.collected_info["name"] = user_input
        session_data.booking_step = "phone"
        
        if language == "ar":
            message = f"شكراً، {user_input}! الآن أحتاج رقم هاتفك لتأكيد الموعد."
        else:
            message = f"Thank you, {user_input}! Now I need your phone number for appointment confirmation."
            
    elif booking_step == "phone":
        session_data.collected_info["phone"] = user_input
        session_data.booking_step = "datetime"
        
        if language == "ar":
            message = "ممتاز! الآن أخبرني بالتاريخ والوقت المفضل. مثال: '25 يناير في الساعة 2:00 مساءً'"
        else:
            message = "Perfect! Now please tell me your preferred date and time. Example: 'January 25th at 2:00 PM'"
            
    elif booking_step == "datetime":
        session_data.collected_info["preferred_datetime"] = user_input
        
        # Generate appointment confirmation
        appointment_id = f"APT{datetime.now().strftime('%Y%m%d%H%M%S')}"
        session_data.appointment_id = appointment_id
        
        service = next((s for s in AVAILABLE_SERVICES if s.id == session_data.selected_service), None)
        
        if language == "ar":
            message = f"""🎉 **تم حجز الموعد بنجاح!**

📅 **تفاصيل الموعد:**
• الخدمة: {service.name['ar'] if service else 'غير محدد'}
• الاسم: {session_data.collected_info.get('name')}
• الهاتف: {session_data.collected_info.get('phone')}
• الوقت المفضل: {session_data.collected_info.get('preferred_datetime')}
• رقم الموعد: {appointment_id}

✅ ستتلقى تأكيداً عبر الرسائل النصية والبريد الإلكتروني قريباً.

هل تحتاج مساعدة في أي شيء آخر؟"""
        else:
            message = f"""🎉 **Appointment Booked Successfully!**

📅 **Appointment Details:**
• Service: {service.name['en'] if service else 'Not specified'}
• Name: {session_data.collected_info.get('name')}
• Phone: {session_data.collected_info.get('phone')}
• Preferred Time: {session_data.collected_info.get('preferred_datetime')}
• Appointment ID: {appointment_id}

✅ You will receive confirmation via SMS and email shortly.

Is there anything else I can help you with?"""
        
        session_data.step = "completed"
        
        # Prepare booking data for n8n webhook
        booking_data = BookingData(
            appointment_id=appointment_id,
            service=service,
            customer_info=session_data.collected_info,
            language=language,
            timestamp=datetime.now().isoformat()
        )
        
        return {
            "message": message,
            "session_data": session_data,
            "actions": ["booking_completed"],
            "trigger_webhook": True,
            "booking_data": booking_data
        }
    
    return {
        "message": message,
        "session_data": session_data,
        "actions": []
    }

def handle_general(user_input: str, session_data: SessionData, language: str) -> Dict[str, Any]:
    """Handle general inquiries"""
    if language == "ar":
        message = "أفهم سؤالك. كمساعدك الافتراضي، أنا هنا لمساعدتك في خدمات متنوعة. إذا كنت تحتاج مساعدة محددة، يرجى إخباري!"
    else:
        message = "I understand your question. As your virtual assistant, I'm here to help with various services. If you need specific assistance, please let me know!"
    
    return {
        "message": message,
        "session_data": session_data,
        "actions": []
    }

def generate_conversation_title(first_message: str, language: str) -> Dict[str, str]:
    """Generate conversation title based on first message"""
    title_en = first_message[:30] + ("..." if len(first_message) > 30 else "")
    title_ar = first_message[:30] + ("..." if len(first_message) > 30 else "")
    
    return {"en": title_en, "ar": title_ar}

async def trigger_n8n_webhook(booking_data: BookingData):
    """Trigger n8n webhook for booking automation"""
    try:
        logger.info(f"Triggering n8n webhook for booking: {booking_data.appointment_id}")
        
        # Here you would call your actual n8n webhook
        # Example:
        # async with httpx.AsyncClient() as client:
        #     response = await client.post(
        #         "https://your-n8n-instance.com/webhook/booking",
        #         json=booking_data.dict(),
        #         headers={"Authorization": "Bearer YOUR_TOKEN"}
        #     )
        
        # For now, just log the booking data
        logger.info(f"Booking data: {booking_data.dict()}")
        
    except Exception as e:
        logger.error(f"Error triggering n8n webhook: {e}")

# Include the API router
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
