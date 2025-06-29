"""
n8n Workflow Integration Service
Handles automation workflows for appointment processing
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
import json
import httpx
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WorkflowTrigger:
    """Workflow trigger data"""
    workflow_id: str
    trigger_type: str  # webhook, schedule, event
    data: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class N8NWorkflowService:
    """n8n workflow integration service"""
    
    def __init__(self):
        self.n8n_base_url = os.getenv('N8N_BASE_URL', 'http://localhost:5678')
        self.n8n_api_key = os.getenv('N8N_API_KEY')
        self.webhook_base_url = os.getenv('N8N_WEBHOOK_URL', f'{self.n8n_base_url}/webhook')
        
        # Pre-defined workflow webhooks
        self.workflow_hooks = {
            'appointment_booking': 'appointment-booking',
            'appointment_reminder': 'appointment-reminder',
            'appointment_cancellation': 'appointment-cancellation',
            'customer_follow_up': 'customer-follow-up'
        }
        
        logger.info(f"n8n Service initialized - Base URL: {self.n8n_base_url}")
    
    async def trigger_appointment_workflow(self, appointment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger the main appointment booking workflow"""
        
        workflow_data = {
            'trigger_type': 'appointment_booking',
            'appointment_id': appointment_data.get('appointment_id'),
            'timestamp': datetime.utcnow().isoformat(),
            'customer_info': appointment_data.get('customer_info', {}),
            'service_info': appointment_data.get('service', {}),
            'language': appointment_data.get('language', 'en'),
            'calendar_events': [],  # Will be populated by calendar service
            'notifications': []     # Will be populated by notification service
        }
        
        # In demo mode, simulate n8n webhook call
        if not self._is_n8n_available():
            return await self._simulate_workflow(workflow_data)
        
        # Send to actual n8n webhook
        return await self._send_webhook('appointment_booking', workflow_data)
    
    async def trigger_reminder_workflow(self, appointment_data: Dict[str, Any], reminder_type: str = '24h') -> Dict[str, Any]:
        """Trigger appointment reminder workflow"""
        
        workflow_data = {
            'trigger_type': 'appointment_reminder',
            'reminder_type': reminder_type,  # 24h, 1h, 15min
            'appointment_id': appointment_data.get('appointment_id'),
            'timestamp': datetime.utcnow().isoformat(),
            'customer_info': appointment_data.get('customer_info', {}),
            'service_info': appointment_data.get('service', {}),
            'language': appointment_data.get('language', 'en')
        }
        
        if not self._is_n8n_available():
            return await self._simulate_workflow(workflow_data)
        
        return await self._send_webhook('appointment_reminder', workflow_data)
    
    async def trigger_follow_up_workflow(self, appointment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger customer follow-up workflow after appointment"""
        
        workflow_data = {
            'trigger_type': 'customer_follow_up',
            'appointment_id': appointment_data.get('appointment_id'),
            'timestamp': datetime.utcnow().isoformat(),
            'customer_info': appointment_data.get('customer_info', {}),
            'service_info': appointment_data.get('service', {}),
            'language': appointment_data.get('language', 'en'),
            'follow_up_type': 'satisfaction_survey'
        }
        
        if not self._is_n8n_available():
            return await self._simulate_workflow(workflow_data)
        
        return await self._send_webhook('customer_follow_up', workflow_data)
    
    async def _send_webhook(self, workflow_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send data to n8n webhook"""
        
        webhook_id = self.workflow_hooks.get(workflow_name)
        if not webhook_id:
            return {
                "success": False,
                "error": f"Unknown workflow: {workflow_name}",
                "message": "Workflow not configured"
            }
        
        webhook_url = f"{self.webhook_base_url}/{webhook_id}"
        
        try:
            async with httpx.AsyncClient() as client:
                headers = {}
                if self.n8n_api_key:
                    headers['Authorization'] = f'Bearer {self.n8n_api_key}'
                
                response = await client.post(
                    webhook_url,
                    json=data,
                    headers=headers,
                    timeout=30.0
                )
                
                response.raise_for_status()
                
                return {
                    "success": True,
                    "workflow": workflow_name,
                    "webhook_url": webhook_url,
                    "response_status": response.status_code,
                    "execution_id": response.headers.get('x-n8n-execution-id'),
                    "message": "Workflow triggered successfully"
                }
                
        except httpx.HTTPError as e:
            logger.error(f"Error calling n8n webhook {webhook_url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow": workflow_name,
                "webhook_url": webhook_url,
                "message": "Failed to trigger workflow"
            }
        except Exception as e:
            logger.error(f"Unexpected error in n8n webhook call: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Unexpected error in workflow trigger"
            }
    
    async def _simulate_workflow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate n8n workflow execution in demo mode"""
        
        trigger_type = data.get('trigger_type')
        appointment_id = data.get('appointment_id')
        
        # Simulate different workflow types
        if trigger_type == 'appointment_booking':
            return {
                "success": True,
                "demo_mode": True,
                "workflow": "appointment_booking",
                "execution_id": f"demo_exec_{appointment_id}",
                "simulated_steps": [
                    "✅ Customer data validated",
                    "✅ Calendar event created (Google, Outlook)",
                    "✅ Confirmation email sent",
                    "✅ SMS notification sent",
                    "✅ WhatsApp message sent",
                    "✅ 24h reminder scheduled",
                    "✅ 1h reminder scheduled",
                    "✅ Follow-up survey scheduled (+1 day)"
                ],
                "estimated_completion": "2-5 minutes",
                "message": f"Demo: Appointment booking workflow executed for {appointment_id}"
            }
        
        elif trigger_type == 'appointment_reminder':
            reminder_type = data.get('reminder_type', '24h')
            return {
                "success": True,
                "demo_mode": True,
                "workflow": "appointment_reminder",
                "execution_id": f"demo_reminder_{appointment_id}",
                "reminder_type": reminder_type,
                "simulated_steps": [
                    f"✅ {reminder_type} reminder triggered",
                    "✅ Customer contact info verified",
                    "✅ Reminder email sent",
                    "✅ Reminder SMS sent",
                    "✅ WhatsApp reminder sent",
                    "✅ Delivery status tracked"
                ],
                "message": f"Demo: {reminder_type} reminder workflow executed for {appointment_id}"
            }
        
        elif trigger_type == 'customer_follow_up':
            return {
                "success": True,
                "demo_mode": True,
                "workflow": "customer_follow_up",
                "execution_id": f"demo_followup_{appointment_id}",
                "simulated_steps": [
                    "✅ Appointment completion verified",
                    "✅ Customer satisfaction survey sent",
                    "✅ Service rating request sent",
                    "✅ Follow-up email scheduled (+3 days)",
                    "✅ CRM data updated"
                ],
                "message": f"Demo: Follow-up workflow executed for {appointment_id}"
            }
        
        else:
            return {
                "success": True,
                "demo_mode": True,
                "workflow": "generic",
                "execution_id": f"demo_generic_{appointment_id}",
                "message": f"Demo: Generic workflow executed for trigger type: {trigger_type}"
            }
    
    def _is_n8n_available(self) -> bool:
        """Check if n8n is available"""
        # In demo mode, we assume n8n is not available unless explicitly configured
        return bool(self.n8n_api_key and self.n8n_base_url != 'http://localhost:5678')
    
    async def get_workflow_status(self, execution_id: str) -> Dict[str, Any]:
        """Get status of a workflow execution"""
        
        if not self._is_n8n_available():
            return {
                "success": True,
                "demo_mode": True,
                "execution_id": execution_id,
                "status": "completed",
                "message": "Demo: Workflow execution completed successfully"
            }
        
        try:
            async with httpx.AsyncClient() as client:
                headers = {}
                if self.n8n_api_key:
                    headers['Authorization'] = f'Bearer {self.n8n_api_key}'
                
                response = await client.get(
                    f"{self.n8n_base_url}/api/v1/executions/{execution_id}",
                    headers=headers,
                    timeout=10.0
                )
                
                response.raise_for_status()
                execution_data = response.json()
                
                return {
                    "success": True,
                    "execution_id": execution_id,
                    "status": execution_data.get('status'),
                    "finished": execution_data.get('finished'),
                    "mode": execution_data.get('mode'),
                    "workflow_id": execution_data.get('workflowId')
                }
                
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_id": execution_id,
                "message": "Failed to get workflow status"
            }
    
    async def get_available_workflows(self) -> Dict[str, Any]:
        """Get list of available workflows"""
        
        workflows = {
            "appointment_booking": {
                "name": "Appointment Booking Workflow",
                "description": "Complete appointment booking process with calendar integration and notifications",
                "webhook_id": self.workflow_hooks['appointment_booking'],
                "triggers": ["appointment_created"],
                "actions": [
                    "Validate customer data",
                    "Create calendar events (Google, Outlook, CalDAV)",
                    "Send confirmation notifications (Email, SMS, WhatsApp)",
                    "Schedule reminders",
                    "Update database",
                    "Log to analytics"
                ]
            },
            "appointment_reminder": {
                "name": "Appointment Reminder Workflow",
                "description": "Automated appointment reminders at 24h, 1h, and 15min intervals",
                "webhook_id": self.workflow_hooks['appointment_reminder'],
                "triggers": ["scheduled_time"],
                "actions": [
                    "Check appointment status",
                    "Send reminder notifications",
                    "Track delivery status",
                    "Reschedule if needed"
                ]
            },
            "appointment_cancellation": {
                "name": "Appointment Cancellation Workflow",
                "description": "Handle appointment cancellations and rescheduling",
                "webhook_id": self.workflow_hooks['appointment_cancellation'],
                "triggers": ["cancellation_request"],
                "actions": [
                    "Cancel calendar events",
                    "Send cancellation confirmations",
                    "Update availability",
                    "Process refunds if applicable"
                ]
            },
            "customer_follow_up": {
                "name": "Customer Follow-up Workflow",
                "description": "Post-appointment customer satisfaction and follow-up",
                "webhook_id": self.workflow_hooks['customer_follow_up'],
                "triggers": ["appointment_completed"],
                "actions": [
                    "Send satisfaction survey",
                    "Request service rating",
                    "Schedule follow-up calls",
                    "Update customer records"
                ]
            }
        }
        
        return {
            "available_workflows": workflows,
            "total_workflows": len(workflows),
            "n8n_status": "available" if self._is_n8n_available() else "demo_mode",
            "webhook_base_url": self.webhook_base_url
        }
    
    def generate_n8n_import_templates(self) -> Dict[str, Any]:
        """Generate n8n workflow templates for easy import"""
        
        # This would contain actual n8n workflow JSON templates
        # For brevity, returning metadata about the templates
        
        templates = {
            "appointment_booking_workflow": {
                "name": "MIND14 Appointment Booking",
                "description": "Complete booking workflow with calendar and notifications",
                "nodes": [
                    "Webhook Trigger",
                    "Data Validation",
                    "Google Calendar Event",
                    "Outlook Calendar Event",
                    "SendGrid Email",
                    "Twilio SMS",
                    "Twilio WhatsApp",
                    "MongoDB Insert",
                    "Schedule Reminders"
                ],
                "estimated_setup_time": "15-20 minutes"
            },
            "reminder_workflow": {
                "name": "MIND14 Appointment Reminders",
                "description": "Automated reminder system",
                "nodes": [
                    "Cron Trigger",
                    "MongoDB Query",
                    "Filter Upcoming Appointments",
                    "Send Notifications",
                    "Update Status"
                ],
                "estimated_setup_time": "10-15 minutes"
            }
        }
        
        return {
            "templates": templates,
            "import_instructions": [
                "1. Open n8n and go to Workflows",
                "2. Click 'Import from URL' or 'Import from JSON'",
                "3. Use the provided templates",
                "4. Configure your API keys in the nodes",
                "5. Test each workflow",
                "6. Activate the workflows"
            ],
            "required_credentials": [
                "Google Calendar OAuth2",
                "Microsoft Graph OAuth2",
                "SendGrid API Key",
                "Twilio Account SID & Auth Token",
                "MongoDB Connection String"
            ]
        }

# Global n8n service instance
n8n_service = N8NWorkflowService()