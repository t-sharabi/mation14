"""
Universal Notification Service
Supports Email (SendGrid), SMS/WhatsApp (Twilio), and demo modes
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
import json

# Notification service imports
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class NotificationMessage:
    """Universal notification message representation"""
    recipient: str  # Email, phone number, or WhatsApp number
    subject: str = ""
    message: str = ""
    message_type: str = "text"  # text, html
    channel: str = "email"  # email, sms, whatsapp
    template_id: str = ""
    template_data: Dict = None
    appointment_id: str = ""
    language: str = "en"
    
    def __post_init__(self):
        if self.template_data is None:
            self.template_data = {}

class NotificationProvider(ABC):
    """Abstract base class for notification providers"""
    
    @abstractmethod
    async def send_notification(self, message: NotificationMessage) -> Dict[str, Any]:
        """Send a notification"""
        pass
    
    @abstractmethod
    async def send_bulk_notifications(self, messages: List[NotificationMessage]) -> Dict[str, Any]:
        """Send multiple notifications"""
        pass

class EmailProvider(NotificationProvider):
    """Email notification provider using SendGrid"""
    
    def __init__(self, api_key: str = None, from_email: str = None):
        self.api_key = api_key or os.getenv('SENDGRID_API_KEY')
        self.from_email = from_email or os.getenv('SENDGRID_FROM_EMAIL', 'noreply@mind14.com')
        self.client = None
        
        if self.api_key and SENDGRID_AVAILABLE:
            self.client = SendGridAPIClient(api_key=self.api_key)
            logger.info("SendGrid email provider initialized")
        else:
            logger.info("Email provider running in demo mode")
    
    async def send_notification(self, message: NotificationMessage) -> Dict[str, Any]:
        """Send email notification"""
        
        if not self.client:
            # Demo mode response
            return {
                "success": True,
                "demo_mode": True,
                "message_id": f"email_demo_{message.appointment_id}",
                "message": f"Demo: Would send email to {message.recipient}",
                "subject": message.subject,
                "content": message.message[:100] + "..." if len(message.message) > 100 else message.message
            }
        
        try:
            mail = Mail(
                from_email=self.from_email,
                to_emails=message.recipient,
                subject=message.subject,
                html_content=message.message if message.message_type == "html" else None,
                plain_text_content=message.message if message.message_type == "text" else None
            )
            
            response = self.client.send(mail)
            
            return {
                "success": True,
                "message_id": response.headers.get('X-Message-Id'),
                "status_code": response.status_code,
                "message": "Email sent successfully"
            }
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to send email"
            }
    
    async def send_bulk_notifications(self, messages: List[NotificationMessage]) -> Dict[str, Any]:
        """Send multiple email notifications"""
        results = []
        for message in messages:
            result = await self.send_notification(message)
            results.append(result)
        
        return {
            "total_sent": len(messages),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", True)),
            "results": results
        }

class SMSProvider(NotificationProvider):
    """SMS notification provider using Twilio"""
    
    def __init__(self, account_sid: str = None, auth_token: str = None, phone_number: str = None):
        self.account_sid = account_sid or os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = auth_token or os.getenv('TWILIO_AUTH_TOKEN')
        self.phone_number = phone_number or os.getenv('TWILIO_PHONE_NUMBER')
        self.client = None
        
        if self.account_sid and self.auth_token and TWILIO_AVAILABLE:
            self.client = TwilioClient(self.account_sid, self.auth_token)
            logger.info("Twilio SMS provider initialized")
        else:
            logger.info("SMS provider running in demo mode")
    
    async def send_notification(self, message: NotificationMessage) -> Dict[str, Any]:
        """Send SMS notification"""
        
        if not self.client:
            # Demo mode response
            return {
                "success": True,
                "demo_mode": True,
                "message_id": f"sms_demo_{message.appointment_id}",
                "message": f"Demo: Would send SMS to {message.recipient}",
                "content": message.message[:100] + "..." if len(message.message) > 100 else message.message
            }
        
        try:
            sms = self.client.messages.create(
                body=message.message,
                from_=self.phone_number,
                to=message.recipient
            )
            
            return {
                "success": True,
                "message_id": sms.sid,
                "status": sms.status,
                "message": "SMS sent successfully"
            }
            
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to send SMS"
            }
    
    async def send_bulk_notifications(self, messages: List[NotificationMessage]) -> Dict[str, Any]:
        """Send multiple SMS notifications"""
        results = []
        for message in messages:
            result = await self.send_notification(message)
            results.append(result)
        
        return {
            "total_sent": len(messages),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", True)),
            "results": results
        }

class WhatsAppProvider(NotificationProvider):
    """WhatsApp notification provider using Twilio WhatsApp Business API"""
    
    def __init__(self, account_sid: str = None, auth_token: str = None, whatsapp_number: str = None):
        self.account_sid = account_sid or os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = auth_token or os.getenv('TWILIO_AUTH_TOKEN')
        self.whatsapp_number = whatsapp_number or os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')  # Twilio Sandbox
        self.client = None
        
        if self.account_sid and self.auth_token and TWILIO_AVAILABLE:
            self.client = TwilioClient(self.account_sid, self.auth_token)
            logger.info("Twilio WhatsApp provider initialized")
        else:
            logger.info("WhatsApp provider running in demo mode")
    
    async def send_notification(self, message: NotificationMessage) -> Dict[str, Any]:
        """Send WhatsApp notification"""
        
        if not self.client:
            # Demo mode response
            return {
                "success": True,
                "demo_mode": True,
                "message_id": f"whatsapp_demo_{message.appointment_id}",
                "message": f"Demo: Would send WhatsApp to {message.recipient}",
                "content": message.message[:100] + "..." if len(message.message) > 100 else message.message
            }
        
        try:
            # Format WhatsApp number
            to_number = message.recipient
            if not to_number.startswith('whatsapp:'):
                to_number = f'whatsapp:{to_number}'
            
            whatsapp_msg = self.client.messages.create(
                body=message.message,
                from_=self.whatsapp_number,
                to=to_number
            )
            
            return {
                "success": True,
                "message_id": whatsapp_msg.sid,
                "status": whatsapp_msg.status,
                "message": "WhatsApp message sent successfully"
            }
            
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to send WhatsApp message"
            }
    
    async def send_bulk_notifications(self, messages: List[NotificationMessage]) -> Dict[str, Any]:
        """Send multiple WhatsApp notifications"""
        results = []
        for message in messages:
            result = await self.send_notification(message)
            results.append(result)
        
        return {
            "total_sent": len(messages),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", True)),
            "results": results
        }

class UniversalNotificationService:
    """Universal notification service that manages multiple channels"""
    
    def __init__(self):
        self.providers = {}
        self._initialize_providers()
        
        # Notification templates
        self.templates = {
            "appointment_confirmation": {
                "en": {
                    "email_subject": "Appointment Confirmation - {service_name}",
                    "email_body": """
Dear {customer_name},

Your appointment has been successfully booked!

📅 **Appointment Details:**
• Service: {service_name}
• Date & Time: {appointment_datetime}
• Appointment ID: {appointment_id}
• Estimated Duration: {duration} minutes
• Location: MIND14 Virtual Front Desk

📞 **Contact Information:**
If you need to reschedule or cancel, please contact us with your appointment ID.

Thank you for choosing MIND14!

Best regards,
MIND14 Virtual Front Desk Team
""",
                    "sms_body": "MIND14 Appointment Confirmed!\nService: {service_name}\nDate: {appointment_datetime}\nID: {appointment_id}\nLocation: MIND14 Virtual Front Desk",
                    "whatsapp_body": "🎉 Your MIND14 appointment is confirmed!\n\n📅 Service: {service_name}\n🕒 Date: {appointment_datetime}\n🆔 ID: {appointment_id}\n📍 Location: MIND14 Virtual Front Desk\n\nThank you for choosing MIND14! 👍"
                },
                "ar": {
                    "email_subject": "تأكيد الموعد - {service_name}",
                    "email_body": """
عزيزي {customer_name}،

تم حجز موعدك بنجاح!

📅 **تفاصيل الموعد:**
• الخدمة: {service_name}
• التاريخ والوقت: {appointment_datetime}
• رقم الموعد: {appointment_id}
• المدة المقدرة: {duration} دقيقة
• الموقع: مكتب الاستقبال الافتراضي MIND14

📞 **معلومات الاتصال:**
إذا كنت بحاجة لإعادة الجدولة أو الإلغاء، يرجى الاتصال بنا مع رقم الموعد.

شكراً لاختيارك MIND14!

مع أطيب التحيات،
فريق مكتب الاستقبال الافتراضي MIND14
""",
                    "sms_body": "تأكيد موعد MIND14!\nالخدمة: {service_name}\nالتاريخ: {appointment_datetime}\nالرقم: {appointment_id}\nالموقع: مكتب الاستقبال الافتراضي MIND14",
                    "whatsapp_body": "🎉 تم تأكيد موعدك مع MIND14!\n\n📅 الخدمة: {service_name}\n🕒 التاريخ: {appointment_datetime}\n🆔 الرقم: {appointment_id}\n📍 الموقع: مكتب الاستقبال الافتراضي MIND14\n\nشكراً لاختيارك MIND14! 👍"
                }
            },
            "appointment_reminder": {
                "en": {
                    "email_subject": "Reminder: Your appointment tomorrow - {service_name}",
                    "sms_body": "Reminder: Your MIND14 appointment is tomorrow at {appointment_time}. Service: {service_name}, ID: {appointment_id}",
                    "whatsapp_body": "⏰ Reminder: Your MIND14 appointment is tomorrow!\n\n📅 Service: {service_name}\n🕒 Time: {appointment_time}\n🆔 ID: {appointment_id}\n\nSee you soon! 👋"
                },
                "ar": {
                    "email_subject": "تذكير: موعدك غداً - {service_name}",
                    "sms_body": "تذكير: موعدك مع MIND14 غداً في {appointment_time}. الخدمة: {service_name}، الرقم: {appointment_id}",
                    "whatsapp_body": "⏰ تذكير: موعدك مع MIND14 غداً!\n\n📅 الخدمة: {service_name}\n🕒 الوقت: {appointment_time}\n🆔 الرقم: {appointment_id}\n\nنراك قريباً! 👋"
                }
            }
        }
    
    def _initialize_providers(self):
        """Initialize all notification providers"""
        
        # Email provider
        self.providers['email'] = EmailProvider()
        
        # SMS provider
        self.providers['sms'] = SMSProvider()
        
        # WhatsApp provider
        self.providers['whatsapp'] = WhatsAppProvider()
        
        logger.info("All notification providers initialized")
    
    async def send_appointment_confirmation(self, appointment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send appointment confirmation across all channels"""
        
        # Extract appointment information
        service_info = appointment_data.get('service', {})
        customer_info = appointment_data.get('customer_info', {})
        appointment_id = appointment_data.get('appointment_id')
        language = appointment_data.get('language', 'en')
        
        # Get template data
        template_data = {
            'customer_name': customer_info.get('name', 'Valued Customer'),
            'service_name': service_info.get('name', {}).get(language, 'Service'),
            'appointment_datetime': customer_info.get('preferred_datetime', 'TBD'),
            'appointment_id': appointment_id,
            'duration': service_info.get('estimated_time', 30)
        }
        
        results = {}
        
        # Prepare notifications for different channels
        notifications = []
        
        # Email notification (if we had customer email)
        email_template = self.templates['appointment_confirmation'][language]
        email_notification = NotificationMessage(
            recipient="customer@example.com",  # In real scenario, get from customer data
            subject=email_template['email_subject'].format(**template_data),
            message=email_template['email_body'].format(**template_data),
            message_type="text",
            channel="email",
            appointment_id=appointment_id,
            language=language
        )
        notifications.append(('email', email_notification))
        
        # SMS notification (using customer phone)
        if customer_info.get('phone'):
            sms_notification = NotificationMessage(
                recipient=customer_info['phone'],
                message=email_template['sms_body'].format(**template_data),
                channel="sms",
                appointment_id=appointment_id,
                language=language
            )
            notifications.append(('sms', sms_notification))
        
        # WhatsApp notification (using customer phone with WhatsApp format)
        if customer_info.get('phone'):
            whatsapp_notification = NotificationMessage(
                recipient=customer_info['phone'],
                message=email_template['whatsapp_body'].format(**template_data),
                channel="whatsapp",
                appointment_id=appointment_id,
                language=language
            )
            notifications.append(('whatsapp', whatsapp_notification))
        
        # Send notifications
        for channel, notification in notifications:
            try:
                provider = self.providers[channel]
                result = await provider.send_notification(notification)
                results[channel] = result
                logger.info(f"Notification sent via {channel}: {result}")
            except Exception as e:
                logger.error(f"Error sending {channel} notification: {e}")
                results[channel] = {
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to send {channel} notification"
                }
        
        return {
            "appointment_id": appointment_id,
            "notifications_sent": results,
            "total_channels": len(notifications),
            "successful_channels": sum(1 for r in results.values() if r.get("success", False)),
            "template_used": "appointment_confirmation",
            "language": language
        }
    
    async def schedule_reminder(self, appointment_data: Dict[str, Any], reminder_time: datetime) -> Dict[str, Any]:
        """Schedule appointment reminder (would integrate with task queue in production)"""
        
        # In a real implementation, this would schedule a task using Celery, Redis, or similar
        # For demo, we'll return a confirmation that the reminder is "scheduled"
        
        appointment_id = appointment_data.get('appointment_id')
        customer_info = appointment_data.get('customer_info', {})
        
        return {
            "success": True,
            "demo_mode": True,
            "appointment_id": appointment_id,
            "reminder_scheduled": reminder_time.isoformat(),
            "reminder_channels": ["email", "sms", "whatsapp"],
            "customer_phone": customer_info.get('phone', 'Not provided'),
            "message": f"Demo: Reminder scheduled for {reminder_time.strftime('%Y-%m-%d %H:%M')}"
        }
    
    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all notification providers"""
        status = {}
        
        for channel, provider in self.providers.items():
            if channel == 'email':
                status[channel] = {
                    "available": provider.client is not None,
                    "demo_mode": provider.client is None,
                    "type": "SendGrid Email",
                    "from_email": provider.from_email
                }
            elif channel == 'sms':
                status[channel] = {
                    "available": provider.client is not None,
                    "demo_mode": provider.client is None,
                    "type": "Twilio SMS",
                    "from_number": provider.phone_number or "Not configured"
                }
            elif channel == 'whatsapp':
                status[channel] = {
                    "available": provider.client is not None,
                    "demo_mode": provider.client is None,
                    "type": "Twilio WhatsApp Business",
                    "from_number": provider.whatsapp_number
                }
        
        return {
            "providers": status,
            "total_channels": len(self.providers),
            "available_templates": list(self.templates.keys()),
            "supported_languages": ["en", "ar"]
        }

# Global notification service instance
notification_service = UniversalNotificationService()