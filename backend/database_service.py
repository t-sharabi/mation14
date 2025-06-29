"""
Enhanced Database Service for MIND14 Virtual Front Desk
Provides persistent storage for appointments, customers, and automation workflows
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING, DESCENDING
from pydantic import BaseModel, Field
import json
from enum import Enum

logger = logging.getLogger(__name__)

class AppointmentStatus(str, Enum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    REMINDED = "reminded"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"

class NotificationStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Enhanced database models
class Customer(BaseModel):
    id: str = Field(default_factory=lambda: str(datetime.utcnow().timestamp()).replace('.', ''))
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    language: str = "en"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    total_appointments: int = 0
    last_appointment: Optional[datetime] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)

class Appointment(BaseModel):
    id: str = Field(default_factory=lambda: f"APT{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
    customer_id: str
    service_id: str
    service_name: Dict[str, str]  # {"en": "name", "ar": "name"}
    status: AppointmentStatus = AppointmentStatus.SCHEDULED
    scheduled_datetime: datetime
    estimated_duration: int  # minutes
    customer_info: Dict[str, Any]
    language: str = "en"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Calendar integration
    calendar_events: Dict[str, str] = Field(default_factory=dict)  # provider -> event_id
    
    # Notification tracking
    notifications_sent: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Workflow tracking
    workflow_executions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Notes and metadata
    notes: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NotificationLog(BaseModel):
    id: str = Field(default_factory=lambda: str(datetime.utcnow().timestamp()).replace('.', ''))
    appointment_id: str
    customer_id: str
    channel: str  # email, sms, whatsapp
    status: NotificationStatus = NotificationStatus.PENDING
    message_type: str  # confirmation, reminder, follow_up
    recipient: str
    message_content: str
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    provider_id: Optional[str] = None  # SendGrid message ID, Twilio SID, etc.
    created_at: datetime = Field(default_factory=datetime.utcnow)

class WorkflowExecution(BaseModel):
    id: str = Field(default_factory=lambda: str(datetime.utcnow().timestamp()).replace('.', ''))
    appointment_id: str
    workflow_name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    trigger_type: str
    execution_data: Dict[str, Any]
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    n8n_execution_id: Optional[str] = None
    steps_completed: List[str] = Field(default_factory=list)

class EnhancedDatabaseService:
    """Enhanced database service with comprehensive appointment management"""
    
    def __init__(self):
        self.mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
        self.db_name = os.environ.get('DB_NAME', 'mind14_db')
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        
        # Collection names
        self.collections = {
            'customers': 'customers',
            'appointments': 'appointments',
            'conversations': 'conversations',
            'notifications': 'notification_logs',
            'workflows': 'workflow_executions',
            'analytics': 'analytics_events'
        }
    
    async def initialize(self):
        """Initialize database connection and create indexes"""
        try:
            self.client = AsyncIOMotorClient(self.mongo_url)
            self.db = self.client[self.db_name]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {self.db_name}")
            
            # Create indexes for better performance
            await self._create_indexes()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    async def _create_indexes(self):
        """Create database indexes for optimal performance"""
        
        # Appointments indexes
        appointments_indexes = [
            IndexModel([("customer_id", ASCENDING)]),
            IndexModel([("status", ASCENDING)]),
            IndexModel([("scheduled_datetime", ASCENDING)]),
            IndexModel([("service_id", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("customer_id", ASCENDING), ("status", ASCENDING)]),
            IndexModel([("scheduled_datetime", ASCENDING), ("status", ASCENDING)])
        ]
        
        # Customers indexes
        customers_indexes = [
            IndexModel([("email", ASCENDING)], unique=True, sparse=True),
            IndexModel([("phone", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("last_appointment", DESCENDING)])
        ]
        
        # Notifications indexes
        notifications_indexes = [
            IndexModel([("appointment_id", ASCENDING)]),
            IndexModel([("customer_id", ASCENDING)]),
            IndexModel([("channel", ASCENDING)]),
            IndexModel([("status", ASCENDING)]),
            IndexModel([("sent_at", DESCENDING)])
        ]
        
        # Workflows indexes
        workflows_indexes = [
            IndexModel([("appointment_id", ASCENDING)]),
            IndexModel([("workflow_name", ASCENDING)]),
            IndexModel([("status", ASCENDING)]),
            IndexModel([("started_at", DESCENDING)])
        ]
        
        # Create indexes
        try:
            await self.db[self.collections['appointments']].create_indexes(appointments_indexes)
            await self.db[self.collections['customers']].create_indexes(customers_indexes)
            await self.db[self.collections['notifications']].create_indexes(notifications_indexes)
            await self.db[self.collections['workflows']].create_indexes(workflows_indexes)
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    async def create_customer(self, customer_data: Dict[str, Any]) -> Customer:
        """Create a new customer record"""
        
        # Check if customer already exists (by phone or email)
        existing_customer = None
        if customer_data.get('phone'):
            existing_customer = await self.get_customer_by_phone(customer_data['phone'])
        elif customer_data.get('email'):
            existing_customer = await self.get_customer_by_email(customer_data['email'])
        
        if existing_customer:
            return existing_customer
        
        # Create new customer
        customer = Customer(**customer_data)
        
        try:
            await self.db[self.collections['customers']].insert_one(customer.dict())
            logger.info(f"Customer created: {customer.id}")
            return customer
            
        except Exception as e:
            logger.error(f"Error creating customer: {e}")
            raise
    
    async def get_customer_by_phone(self, phone: str) -> Optional[Customer]:
        """Get customer by phone number"""
        try:
            customer_data = await self.db[self.collections['customers']].find_one({"phone": phone})
            if customer_data:
                return Customer(**customer_data)
            return None
        except Exception as e:
            logger.error(f"Error getting customer by phone: {e}")
            return None
    
    async def get_customer_by_email(self, email: str) -> Optional[Customer]:
        """Get customer by email"""
        try:
            customer_data = await self.db[self.collections['customers']].find_one({"email": email})
            if customer_data:
                return Customer(**customer_data)
            return None
        except Exception as e:
            logger.error(f"Error getting customer by email: {e}")
            return None
    
    async def create_appointment(self, appointment_data: Dict[str, Any]) -> Appointment:
        """Create a new appointment"""
        
        # Create or get customer
        customer_info = appointment_data.get('customer_info', {})
        customer = await self.create_customer({
            'name': customer_info.get('name', 'Unknown'),
            'phone': customer_info.get('phone'),
            'email': customer_info.get('email'),
            'language': appointment_data.get('language', 'en')
        })
        
        # Parse appointment datetime
        preferred_datetime = customer_info.get('preferred_datetime', '')
        scheduled_datetime = self._parse_datetime(preferred_datetime)
        
        # Create appointment
        appointment = Appointment(
            id=appointment_data.get('appointment_id', ''),
            customer_id=customer.id,
            service_id=appointment_data.get('service', {}).get('id', 'general'),
            service_name=appointment_data.get('service', {}).get('name', {}),
            scheduled_datetime=scheduled_datetime,
            estimated_duration=appointment_data.get('service', {}).get('estimated_time', 30),
            customer_info=customer_info,
            language=appointment_data.get('language', 'en')
        )
        
        try:
            await self.db[self.collections['appointments']].insert_one(appointment.dict())
            
            # Update customer statistics
            await self.update_customer_stats(customer.id)
            
            logger.info(f"Appointment created: {appointment.id}")
            return appointment
            
        except Exception as e:
            logger.error(f"Error creating appointment: {e}")
            raise
    
    async def update_appointment_status(self, appointment_id: str, status: AppointmentStatus, notes: str = "") -> bool:
        """Update appointment status"""
        try:
            update_data = {
                "status": status.value,
                "updated_at": datetime.utcnow()
            }
            
            if notes:
                update_data["notes"] = notes
            
            result = await self.db[self.collections['appointments']].update_one(
                {"id": appointment_id},
                {"$set": update_data}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating appointment status: {e}")
            return False
    
    async def add_calendar_event(self, appointment_id: str, provider: str, event_id: str) -> bool:
        """Add calendar event ID to appointment"""
        try:
            result = await self.db[self.collections['appointments']].update_one(
                {"id": appointment_id},
                {
                    "$set": {
                        f"calendar_events.{provider}": event_id,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error adding calendar event: {e}")
            return False
    
    async def log_notification(self, notification_data: Dict[str, Any]) -> NotificationLog:
        """Log a notification attempt"""
        
        notification = NotificationLog(**notification_data)
        
        try:
            await self.db[self.collections['notifications']].insert_one(notification.dict())
            
            # Also add to appointment record
            await self.db[self.collections['appointments']].update_one(
                {"id": notification.appointment_id},
                {
                    "$push": {
                        "notifications_sent": {
                            "channel": notification.channel,
                            "status": notification.status.value,
                            "sent_at": notification.sent_at,
                            "message_type": notification.message_type
                        }
                    },
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            
            logger.info(f"Notification logged: {notification.id}")
            return notification
            
        except Exception as e:
            logger.error(f"Error logging notification: {e}")
            raise
    
    async def log_workflow_execution(self, workflow_data: Dict[str, Any]) -> WorkflowExecution:
        """Log a workflow execution"""
        
        workflow = WorkflowExecution(**workflow_data)
        
        try:
            await self.db[self.collections['workflows']].insert_one(workflow.dict())
            
            # Also add to appointment record
            await self.db[self.collections['appointments']].update_one(
                {"id": workflow.appointment_id},
                {
                    "$push": {
                        "workflow_executions": {
                            "workflow_name": workflow.workflow_name,
                            "status": workflow.status.value,
                            "started_at": workflow.started_at,
                            "n8n_execution_id": workflow.n8n_execution_id
                        }
                    },
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            
            logger.info(f"Workflow execution logged: {workflow.id}")
            return workflow
            
        except Exception as e:
            logger.error(f"Error logging workflow execution: {e}")
            raise
    
    async def get_appointment(self, appointment_id: str) -> Optional[Appointment]:
        """Get appointment by ID"""
        try:
            appointment_data = await self.db[self.collections['appointments']].find_one({"id": appointment_id})
            if appointment_data:
                return Appointment(**appointment_data)
            return None
        except Exception as e:
            logger.error(f"Error getting appointment: {e}")
            return None
    
    async def get_customer_appointments(self, customer_id: str, limit: int = 50) -> List[Appointment]:
        """Get appointments for a customer"""
        try:
            cursor = self.db[self.collections['appointments']].find(
                {"customer_id": customer_id}
            ).sort("scheduled_datetime", DESCENDING).limit(limit)
            
            appointments = []
            async for appointment_data in cursor:
                appointments.append(Appointment(**appointment_data))
            
            return appointments
            
        except Exception as e:
            logger.error(f"Error getting customer appointments: {e}")
            return []
    
    async def get_upcoming_appointments(self, hours_ahead: int = 24) -> List[Appointment]:
        """Get upcoming appointments for reminders"""
        try:
            now = datetime.utcnow()
            future_time = now + timedelta(hours=hours_ahead)
            
            cursor = self.db[self.collections['appointments']].find({
                "scheduled_datetime": {"$gte": now, "$lte": future_time},
                "status": {"$in": [AppointmentStatus.SCHEDULED.value, AppointmentStatus.CONFIRMED.value]}
            }).sort("scheduled_datetime", ASCENDING)
            
            appointments = []
            async for appointment_data in cursor:
                appointments.append(Appointment(**appointment_data))
            
            return appointments
            
        except Exception as e:
            logger.error(f"Error getting upcoming appointments: {e}")
            return []
    
    async def update_customer_stats(self, customer_id: str):
        """Update customer statistics"""
        try:
            # Count total appointments
            total_appointments = await self.db[self.collections['appointments']].count_documents(
                {"customer_id": customer_id}
            )
            
            # Get last appointment
            last_appointment_data = await self.db[self.collections['appointments']].find_one(
                {"customer_id": customer_id},
                sort=[("scheduled_datetime", DESCENDING)]
            )
            
            update_data = {
                "total_appointments": total_appointments,
                "updated_at": datetime.utcnow()
            }
            
            if last_appointment_data:
                update_data["last_appointment"] = last_appointment_data["scheduled_datetime"]
            
            await self.db[self.collections['customers']].update_one(
                {"id": customer_id},
                {"$set": update_data}
            )
            
        except Exception as e:
            logger.error(f"Error updating customer stats: {e}")
    
    async def get_analytics_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get analytics data for dashboard"""
        try:
            # Total appointments
            total_appointments = await self.db[self.collections['appointments']].count_documents({
                "created_at": {"$gte": start_date, "$lte": end_date}
            })
            
            # Appointments by status
            status_pipeline = [
                {"$match": {"created_at": {"$gte": start_date, "$lte": end_date}}},
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ]
            status_cursor = self.db[self.collections['appointments']].aggregate(status_pipeline)
            status_data = {doc["_id"]: doc["count"] async for doc in status_cursor}
            
            # Appointments by service
            service_pipeline = [
                {"$match": {"created_at": {"$gte": start_date, "$lte": end_date}}},
                {"$group": {"_id": "$service_id", "count": {"$sum": 1}}}
            ]
            service_cursor = self.db[self.collections['appointments']].aggregate(service_pipeline)
            service_data = {doc["_id"]: doc["count"] async for doc in service_cursor}
            
            # Total customers
            total_customers = await self.db[self.collections['customers']].count_documents({
                "created_at": {"$gte": start_date, "$lte": end_date}
            })
            
            # Notification stats
            notification_pipeline = [
                {"$match": {"created_at": {"$gte": start_date, "$lte": end_date}}},
                {"$group": {
                    "_id": {"channel": "$channel", "status": "$status"},
                    "count": {"$sum": 1}
                }}
            ]
            notification_cursor = self.db[self.collections['notifications']].aggregate(notification_pipeline)
            notification_data = {}
            async for doc in notification_cursor:
                key = f"{doc['_id']['channel']}_{doc['_id']['status']}"
                notification_data[key] = doc["count"]
            
            return {
                "period": {"start": start_date, "end": end_date},
                "total_appointments": total_appointments,
                "total_customers": total_customers,
                "appointments_by_status": status_data,
                "appointments_by_service": service_data,
                "notification_stats": notification_data,
                "generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics data: {e}")
            return {}
    
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
            return datetime.utcnow().replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=1)
            
        except Exception as e:
            logger.warning(f"Failed to parse datetime '{datetime_str}': {e}")
            return datetime.utcnow().replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health and return status"""
        try:
            # Test database connection
            await self.client.admin.command('ping')
            
            # Get collection stats
            collections_status = {}
            for name, collection_name in self.collections.items():
                try:
                    count = await self.db[collection_name].count_documents({})
                    collections_status[name] = {"count": count, "status": "healthy"}
                except Exception as e:
                    collections_status[name] = {"status": "error", "error": str(e)}
            
            return {
                "database_connected": True,
                "database_name": self.db_name,
                "collections": collections_status,
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "database_connected": False,
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }

# Global database service instance
enhanced_db = EnhancedDatabaseService()