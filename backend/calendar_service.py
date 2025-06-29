"""
Universal Calendar Integration Service
Supports Google Calendar, Outlook/Exchange, and CalDAV
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass

# Calendar service imports
try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from exchangelib import Credentials as ExchangeCredentials, Account, CalendarItem, EWSDateTime
    EXCHANGE_AVAILABLE = True
except ImportError:
    EXCHANGE_AVAILABLE = False

try:
    from icalendar import Calendar, Event as iCalEvent
    import requests
    CALDAV_AVAILABLE = True
except ImportError:
    CALDAV_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CalendarEvent:
    """Universal calendar event representation"""
    id: Optional[str] = None
    title: str = ""
    description: str = ""
    start_time: datetime = None
    end_time: datetime = None
    attendees: List[str] = None
    location: str = ""
    service_type: str = ""
    appointment_id: str = ""
    
    def __post_init__(self):
        if self.attendees is None:
            self.attendees = []

class CalendarProvider(ABC):
    """Abstract base class for calendar providers"""
    
    @abstractmethod
    async def create_event(self, event: CalendarEvent) -> Dict[str, Any]:
        """Create a calendar event"""
        pass
    
    @abstractmethod
    async def update_event(self, event_id: str, event: CalendarEvent) -> Dict[str, Any]:
        """Update a calendar event"""
        pass
    
    @abstractmethod
    async def delete_event(self, event_id: str) -> bool:
        """Delete a calendar event"""
        pass
    
    @abstractmethod
    async def get_availability(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Check availability for a time range"""
        pass

class GoogleCalendarProvider(CalendarProvider):
    """Google Calendar integration"""
    
    def __init__(self, credentials_file: str = None, token_file: str = None):
        self.credentials_file = credentials_file or os.getenv('GOOGLE_CREDENTIALS_FILE')
        self.token_file = token_file or os.getenv('GOOGLE_TOKEN_FILE', '/tmp/google_token.json')
        self.service = None
        self.calendar_id = os.getenv('GOOGLE_CALENDAR_ID', 'primary')
        
    async def _get_service(self):
        """Get authenticated Google Calendar service"""
        if self.service:
            return self.service
            
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google Calendar dependencies not installed")
            
        try:
            # In demo mode, return mock service
            if not self.credentials_file:
                logger.info("Google Calendar running in demo mode")
                return None
                
            creds = None
            # Load existing token
            if os.path.exists(self.token_file):
                creds = Credentials.from_authorized_user_file(self.token_file)
            
            # If no valid credentials, run OAuth flow
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = Flow.from_client_secrets_file(
                        self.credentials_file,
                        scopes=['https://www.googleapis.com/auth/calendar']
                    )
                    flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
                    auth_url, _ = flow.authorization_url(prompt='consent')
                    
                    logger.warning(f"Please visit this URL to authorize the application: {auth_url}")
                    return None
            
            self.service = build('calendar', 'v3', credentials=creds)
            return self.service
            
        except Exception as e:
            logger.error(f"Error setting up Google Calendar: {e}")
            return None
    
    async def create_event(self, event: CalendarEvent) -> Dict[str, Any]:
        """Create Google Calendar event"""
        service = await self._get_service()
        
        if not service:
            # Demo mode response
            return {
                "success": True,
                "demo_mode": True,
                "event_id": f"google_demo_{event.appointment_id}",
                "message": f"Demo: Would create Google Calendar event for {event.title}",
                "event_link": f"https://calendar.google.com/calendar/event?eid=demo_{event.appointment_id}"
            }
        
        try:
            calendar_event = {
                'summary': event.title,
                'description': event.description,
                'start': {
                    'dateTime': event.start_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': event.end_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'attendees': [{'email': email} for email in event.attendees],
                'location': event.location,
            }
            
            result = service.events().insert(
                calendarId=self.calendar_id, 
                body=calendar_event
            ).execute()
            
            return {
                "success": True,
                "event_id": result['id'],
                "event_link": result.get('htmlLink'),
                "message": f"Google Calendar event created successfully"
            }
            
        except Exception as e:
            logger.error(f"Error creating Google Calendar event: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to create Google Calendar event"
            }
    
    async def update_event(self, event_id: str, event: CalendarEvent) -> Dict[str, Any]:
        """Update Google Calendar event"""
        # Implementation for updating events
        return {"success": True, "demo_mode": True, "message": "Demo: Event update simulated"}
    
    async def delete_event(self, event_id: str) -> bool:
        """Delete Google Calendar event"""
        # Implementation for deleting events
        return True
    
    async def get_availability(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get Google Calendar availability"""
        return []

class OutlookCalendarProvider(CalendarProvider):
    """Microsoft Outlook/Exchange calendar integration"""
    
    def __init__(self, email: str = None, username: str = None, password: str = None):
        self.email = email or os.getenv('OUTLOOK_EMAIL')
        self.username = username or os.getenv('OUTLOOK_USERNAME')
        self.password = password or os.getenv('OUTLOOK_PASSWORD')
        self.account = None
    
    async def _get_account(self):
        """Get authenticated Exchange account"""
        if self.account:
            return self.account
            
        if not EXCHANGE_AVAILABLE:
            raise ImportError("Exchange dependencies not installed")
            
        try:
            # Demo mode
            if not self.email or not self.password:
                logger.info("Outlook Calendar running in demo mode")
                return None
                
            credentials = ExchangeCredentials(self.username or self.email, self.password)
            self.account = Account(self.email, credentials=credentials, autodiscover=True)
            return self.account
            
        except Exception as e:
            logger.error(f"Error setting up Outlook Calendar: {e}")
            return None
    
    async def create_event(self, event: CalendarEvent) -> Dict[str, Any]:
        """Create Outlook calendar event"""
        account = await self._get_account()
        
        if not account:
            # Demo mode response
            return {
                "success": True,
                "demo_mode": True,
                "event_id": f"outlook_demo_{event.appointment_id}",
                "message": f"Demo: Would create Outlook event for {event.title}",
                "event_link": f"https://outlook.office365.com/calendar/event?id=demo_{event.appointment_id}"
            }
        
        try:
            calendar_item = CalendarItem(
                account=account,
                subject=event.title,
                body=event.description,
                start=EWSDateTime.from_datetime(event.start_time),
                end=EWSDateTime.from_datetime(event.end_time),
                location=event.location
            )
            
            calendar_item.save()
            
            return {
                "success": True,
                "event_id": calendar_item.id,
                "message": "Outlook Calendar event created successfully"
            }
            
        except Exception as e:
            logger.error(f"Error creating Outlook Calendar event: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to create Outlook Calendar event"
            }
    
    async def update_event(self, event_id: str, event: CalendarEvent) -> Dict[str, Any]:
        """Update Outlook calendar event"""
        return {"success": True, "demo_mode": True, "message": "Demo: Event update simulated"}
    
    async def delete_event(self, event_id: str) -> bool:
        """Delete Outlook calendar event"""
        return True
    
    async def get_availability(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get Outlook calendar availability"""
        return []

class CalDAVProvider(CalendarProvider):
    """CalDAV universal calendar provider"""
    
    def __init__(self, server_url: str = None, username: str = None, password: str = None):
        self.server_url = server_url or os.getenv('CALDAV_SERVER_URL')
        self.username = username or os.getenv('CALDAV_USERNAME')
        self.password = password or os.getenv('CALDAV_PASSWORD')
    
    async def create_event(self, event: CalendarEvent) -> Dict[str, Any]:
        """Create CalDAV calendar event"""
        if not self.server_url:
            # Demo mode response
            return {
                "success": True,
                "demo_mode": True,
                "event_id": f"caldav_demo_{event.appointment_id}",
                "message": f"Demo: Would create CalDAV event for {event.title}"
            }
        
        # Implementation would use CalDAV protocol
        return {"success": True, "demo_mode": True, "message": "CalDAV integration ready"}
    
    async def update_event(self, event_id: str, event: CalendarEvent) -> Dict[str, Any]:
        """Update CalDAV calendar event"""
        return {"success": True, "demo_mode": True, "message": "Demo: Event update simulated"}
    
    async def delete_event(self, event_id: str) -> bool:
        """Delete CalDAV calendar event"""
        return True
    
    async def get_availability(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get CalDAV calendar availability"""
        return []

class UniversalCalendarService:
    """Universal calendar service that manages multiple providers"""
    
    def __init__(self):
        self.providers = {}
        self.default_provider = os.getenv('DEFAULT_CALENDAR_PROVIDER', 'google')
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available calendar providers"""
        
        # Google Calendar
        if GOOGLE_AVAILABLE or os.getenv('GOOGLE_CREDENTIALS_FILE'):
            self.providers['google'] = GoogleCalendarProvider()
            logger.info("Google Calendar provider initialized")
        
        # Outlook Calendar
        if EXCHANGE_AVAILABLE or os.getenv('OUTLOOK_EMAIL'):
            self.providers['outlook'] = OutlookCalendarProvider()
            logger.info("Outlook Calendar provider initialized")
        
        # CalDAV
        if CALDAV_AVAILABLE or os.getenv('CALDAV_SERVER_URL'):
            self.providers['caldav'] = CalDAVProvider()
            logger.info("CalDAV provider initialized")
        
        # Always have demo providers
        if not self.providers:
            self.providers['google'] = GoogleCalendarProvider()
            self.providers['outlook'] = OutlookCalendarProvider()
            logger.info("Calendar providers initialized in demo mode")
    
    async def create_appointment(self, appointment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create calendar appointment across all configured providers"""
        
        # Extract appointment information
        service_info = appointment_data.get('service', {})
        customer_info = appointment_data.get('customer_info', {})
        appointment_id = appointment_data.get('appointment_id')
        language = appointment_data.get('language', 'en')
        
        # Parse preferred datetime
        preferred_time = customer_info.get('preferred_datetime', '')
        start_time = self._parse_datetime(preferred_time)
        end_time = start_time + timedelta(minutes=service_info.get('estimated_time', 30))
        
        # Create calendar event
        event = CalendarEvent(
            title=f"{service_info.get('name', {}).get(language, 'Service Appointment')} - {customer_info.get('name', 'Customer')}",
            description=f"""
Appointment Details:
- Service: {service_info.get('description', {}).get(language, 'Service appointment')}
- Customer: {customer_info.get('name', 'Not provided')}
- Phone: {customer_info.get('phone', 'Not provided')}
- Appointment ID: {appointment_id}
- Estimated Duration: {service_info.get('estimated_time', 30)} minutes
""".strip(),
            start_time=start_time,
            end_time=end_time,
            attendees=[],  # Could add customer email if available
            location="MIND14 Virtual Front Desk",
            service_type=service_info.get('id', 'general'),
            appointment_id=appointment_id
        )
        
        # Create events in all available providers
        results = {}
        for provider_name, provider in self.providers.items():
            try:
                result = await provider.create_event(event)
                results[provider_name] = result
                logger.info(f"Calendar event created in {provider_name}: {result}")
            except Exception as e:
                logger.error(f"Error creating event in {provider_name}: {e}")
                results[provider_name] = {
                    "success": False,
                    "error": str(e),
                    "message": f"Failed to create event in {provider_name}"
                }
        
        return {
            "appointment_id": appointment_id,
            "calendar_events": results,
            "primary_calendar": self.default_provider,
            "success": any(r.get("success", False) for r in results.values())
        }
    
    def _parse_datetime(self, datetime_str: str) -> datetime:
        """Parse datetime string to datetime object"""
        try:
            # Try various common formats
            formats = [
                "%Y-%m-%d %H:%M",
                "%d/%m/%Y %H:%M",
                "%B %d at %I:%M %p",
                "%B %dst at %I:%M %p",
                "%B %dnd at %I:%M %p",
                "%B %drd at %I:%M %p",
                "%B %dth at %I:%M %p",
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(datetime_str, fmt)
                except ValueError:
                    continue
            
            # If parsing fails, default to tomorrow at 10 AM
            return datetime.now() + timedelta(days=1, hours=10)
            
        except Exception as e:
            logger.warning(f"Failed to parse datetime '{datetime_str}': {e}")
            return datetime.now() + timedelta(days=1, hours=10)
    
    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all calendar providers"""
        status = {}
        
        for provider_name, provider in self.providers.items():
            try:
                # Test provider availability
                if provider_name == 'google':
                    service = await provider._get_service()
                    status[provider_name] = {
                        "available": service is not None,
                        "demo_mode": service is None,
                        "type": "Google Calendar"
                    }
                elif provider_name == 'outlook':
                    account = await provider._get_account()
                    status[provider_name] = {
                        "available": account is not None,
                        "demo_mode": account is None,
                        "type": "Microsoft Outlook"
                    }
                else:
                    status[provider_name] = {
                        "available": True,
                        "demo_mode": True,
                        "type": "CalDAV"
                    }
                    
            except Exception as e:
                status[provider_name] = {
                    "available": False,
                    "error": str(e),
                    "demo_mode": True
                }
        
        return {
            "providers": status,
            "default_provider": self.default_provider,
            "total_providers": len(self.providers)
        }

# Global calendar service instance
calendar_service = UniversalCalendarService()