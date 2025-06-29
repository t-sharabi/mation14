import requests
import sys
import json
from datetime import datetime, timedelta

class MIND14APITester:
    def __init__(self, base_url="https://7d6b7df0-b5d7-44c5-8846-e57f2dc71e63.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.conversation_id = None
        self.appointment_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    return success, response.json()
                except:
                    return success, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    print(f"Response: {response.text}")
                    return False, response.json() if response.text else {}
                except:
                    return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        success, response = self.run_test(
            "Root API Endpoint",
            "GET",
            "",
            200
        )
        if success:
            print(f"API Version: {response.get('version')}")
            print(f"AI Provider: {response.get('ai_provider')}")
            print(f"AI Available: {response.get('ai_available')}")
        return success

    def test_ai_status(self):
        """Test the AI status endpoint"""
        success, response = self.run_test(
            "AI Status Endpoint",
            "GET",
            "ai-status",
            200
        )
        if success:
            print(f"AI Provider: {response.get('provider')}")
            print(f"AI Model: {response.get('model')}")
            print(f"AI Available: {response.get('available')}")
        return success

    def test_automation_status(self):
        """Test the automation status endpoint"""
        success, response = self.run_test(
            "Automation Status Endpoint",
            "GET",
            "automation-status",
            200
        )
        if success:
            print(f"Automation Available: {response.get('automation_available')}")
            print(f"Calendar Service: {response.get('calendar_service', {}).get('status')}")
            print(f"Notification Service: {response.get('notification_service', {}).get('status')}")
            print(f"Workflow Service: {response.get('workflow_service', {}).get('status')}")
            print(f"Database Service: {response.get('database_service', {}).get('status')}")
        return success

    def test_calendar_status(self):
        """Test the calendar status endpoint"""
        success, response = self.run_test(
            "Calendar Status Endpoint",
            "GET",
            "calendar/status",
            200
        )
        if success:
            print(f"Default Provider: {response.get('default_provider')}")
            print(f"Total Providers: {response.get('total_providers')}")
            providers = response.get('providers', {})
            for provider, status in providers.items():
                print(f"- {provider}: {'Demo Mode' if status.get('demo_mode') else 'Live'}")
        return success

    def test_notifications_status(self):
        """Test the notifications status endpoint"""
        success, response = self.run_test(
            "Notifications Status Endpoint",
            "GET",
            "notifications/status",
            200
        )
        if success:
            print(f"Total Channels: {response.get('total_channels')}")
            print(f"Available Templates: {', '.join(response.get('available_templates', []))}")
            print(f"Supported Languages: {', '.join(response.get('supported_languages', []))}")
            providers = response.get('providers', {})
            for channel, status in providers.items():
                print(f"- {channel}: {'Demo Mode' if status.get('demo_mode') else 'Live'}")
        return success

    def test_workflows_available(self):
        """Test the available workflows endpoint"""
        success, response = self.run_test(
            "Available Workflows Endpoint",
            "GET",
            "workflows/available",
            200
        )
        if success:
            print(f"n8n Status: {response.get('n8n_status')}")
            print(f"Total Workflows: {response.get('total_workflows')}")
            workflows = response.get('available_workflows', {})
            for workflow_id, workflow in workflows.items():
                print(f"- {workflow.get('name')}: {workflow.get('description')[:50]}...")
        return success

    def test_appointment_details(self):
        """Test the appointment details endpoint"""
        if not self.appointment_id:
            print("❌ No appointment ID available for testing")
            return False
            
        success, response = self.run_test(
            "Appointment Details Endpoint",
            "GET",
            f"appointments/{self.appointment_id}",
            200
        )
        if success:
            print(f"Appointment ID: {response.get('id')}")
            print(f"Service: {response.get('service_name', {}).get('en')}")
            print(f"Status: {response.get('status')}")
            print(f"Calendar Events: {len(response.get('calendar_events', {}))}")
            print(f"Notifications Sent: {len(response.get('notifications_sent', []))}")
            print(f"Workflow Executions: {len(response.get('workflow_executions', []))}")
        return success

    def test_analytics_dashboard(self):
        """Test the analytics dashboard endpoint"""
        # Set date range for last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        date_params = f"start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}"
        
        success, response = self.run_test(
            "Analytics Dashboard Endpoint",
            "GET",
            f"analytics/dashboard?{date_params}",
            200
        )
        if success:
            print(f"Total Appointments: {response.get('total_appointments')}")
            print(f"Total Customers: {response.get('total_customers')}")
            print(f"Appointments by Status: {response.get('appointments_by_status')}")
            print(f"Notification Stats: {response.get('notification_stats')}")
        return success

    def test_services_list(self):
        """Test the services list endpoint"""
        success, response = self.run_test(
            "Services List Endpoint",
            "GET",
            "services",
            200
        )
        if success:
            print(f"Number of services: {len(response)}")
            for service in response:
                print(f"- {service['name']['en']} ({service['id']})")
        return success

    def test_chat_endpoint(self, message, language="en"):
        """Test the chat endpoint with a specific message"""
        data = {
            "message": message,
            "conversation_id": self.conversation_id,
            "language": language
        }
        
        success, response = self.run_test(
            f"Chat Endpoint ({message})",
            "POST",
            "chat",
            200,
            data=data
        )
        
        if success:
            print(f"AI Response: {response.get('message')[:100]}...")
            print(f"Intent: {response.get('intent')}")
            print(f"Confidence: {response.get('confidence')}")
            
            # Save conversation ID for future tests
            if 'conversation_id' in response and not self.conversation_id:
                self.conversation_id = response['conversation_id']
                print(f"Conversation ID: {self.conversation_id}")
                
            # Print session data
            if 'session_data' in response:
                print(f"Session Step: {response['session_data'].get('step')}")
                print(f"Selected Service: {response['session_data'].get('selected_service')}")
                
            # Save appointment ID if available
            if response.get('session_data', {}).get('appointment_id'):
                self.appointment_id = response['session_data']['appointment_id']
                print(f"Appointment ID: {self.appointment_id}")
                
            # Check for booking data and automation triggers
            if 'booking_data' in response:
                print(f"Booking Data: {response['booking_data'].get('service', {}).get('id')} for {response['booking_data'].get('customer_info', {}).get('name')}")
                
            if 'trigger_webhook' in response and response['trigger_webhook']:
                print("🔄 Automation workflow triggered")
        
        return success, response

    def test_conversations_endpoint(self):
        """Test the conversations endpoint"""
        success, response = self.run_test(
            "Conversations Endpoint",
            "GET",
            "conversations?user_id=demo_user",
            200
        )
        if success:
            print(f"Number of conversations: {len(response)}")
        return success

    def test_n8n_webhook(self):
        """Test the n8n webhook endpoint"""
        if not self.appointment_id:
            print("❌ No appointment ID available for testing")
            return False
            
        # Create test booking data
        booking_data = {
            "appointment_id": self.appointment_id,
            "service": {
                "id": "medical-consultation",
                "name": {"en": "Medical Consultation", "ar": "استشارة طبية"},
                "estimated_time": 20
            },
            "customer_info": {
                "name": "Test User",
                "phone": "555-123-4567",
                "preferred_datetime": "Next Monday at 10:00 AM"
            },
            "language": "en",
            "timestamp": datetime.now().isoformat()
        }
        
        success, response = self.run_test(
            "n8n Webhook Endpoint",
            "POST",
            "n8n/book-appointment",
            200,
            data=booking_data
        )
        
        if success:
            print(f"Workflow Success: {response.get('success')}")
            print(f"Workflow: {response.get('workflow')}")
            print(f"Demo Mode: {response.get('demo_mode')}")
            print(f"Execution ID: {response.get('execution_id')}")
            
            # Print simulated steps if in demo mode
            if response.get('demo_mode') and 'simulated_steps' in response:
                print("Simulated Steps:")
                for step in response['simulated_steps']:
                    print(f"  {step}")
        
        return success

    def test_intent_classification(self):
        """Test intent classification with various inputs"""
        test_messages = [
            ("Hello, I need help with health card renewal", "en"),
            ("My ID card is lost, need replacement", "en"),
            ("Book doctor appointment", "en"),
            ("Student enrollment information", "en"),
            ("مرحبا، أحتاج تجديد البطاقة الصحية", "ar")
        ]
        
        results = []
        for message, language in test_messages:
            print(f"\n🔍 Testing intent classification for: '{message}' ({language})")
            success, response = self.test_chat_endpoint(message, language)
            if success:
                results.append({
                    "message": message,
                    "language": language,
                    "intent": response.get("intent"),
                    "confidence": response.get("confidence"),
                    "service": response.get("session_data", {}).get("selected_service")
                })
        
        print("\n📊 Intent Classification Results:")
        for result in results:
            print(f"- '{result['message']}' ({result['language']}): {result['intent']} (confidence: {result['confidence']})")
        
        return len(results) > 0

    def test_multi_step_booking(self):
        """Test a multi-step booking flow with automation integration"""
        # Start a new conversation
        self.conversation_id = None
        self.appointment_id = None
        
        # Step 1: Initial contact
        print("\n🔍 Testing complete booking flow with automation")
        success1, response1 = self.test_chat_endpoint("I need to book a medical consultation")
        
        if not success1:
            return False
        
        # Step 2: Service confirmation
        success2, response2 = self.test_chat_endpoint("Yes, I want to proceed")
        
        if not success2:
            return False
        
        # Step 3: Provide name
        success3, response3 = self.test_chat_endpoint("My name is John Smith")
        
        if not success3:
            return False
        
        # Step 4: Provide phone
        success4, response4 = self.test_chat_endpoint("My phone is 555-123-4567")
        
        if not success4:
            return False
        
        # Step 5: Provide date/time
        success5, response5 = self.test_chat_endpoint("Next Monday at 10:00 AM")
        
        if not success5 or not self.appointment_id:
            return False
            
        print("\n✅ Multi-step booking flow completed successfully")
        print(f"Appointment ID: {self.appointment_id}")
        
        # Test appointment details
        appointment_success = self.test_appointment_details()
        
        # Test n8n webhook
        webhook_success = self.test_n8n_webhook()
        
        return appointment_success and webhook_success

    def test_multilingual_booking(self):
        """Test booking flow in Arabic"""
        # Start a new conversation
        self.conversation_id = None
        self.appointment_id = None
        
        # Step 1: Initial contact in Arabic
        print("\n🔍 Testing Arabic booking flow")
        success1, response1 = self.test_chat_endpoint("أحتاج حجز موعد طبي", "ar")
        
        if not success1:
            return False
        
        # Step 2: Service confirmation
        success2, response2 = self.test_chat_endpoint("نعم، أريد المتابعة", "ar")
        
        if not success2:
            return False
        
        # Step 3: Provide name
        success3, response3 = self.test_chat_endpoint("اسمي محمد أحمد", "ar")
        
        if not success3:
            return False
        
        # Step 4: Provide phone
        success4, response4 = self.test_chat_endpoint("رقم هاتفي 555-123-4567", "ar")
        
        if not success4:
            return False
        
        # Step 5: Provide date/time
        success5, response5 = self.test_chat_endpoint("يوم الاثنين القادم الساعة 10:00 صباحاً", "ar")
        
        if not success5 or not self.appointment_id:
            return False
            
        print("\n✅ Arabic booking flow completed successfully")
        print(f"Appointment ID: {self.appointment_id}")
        
        return True

def main():
    # Get the backend URL from command line or use default
    backend_url = "https://7d6b7df0-b5d7-44c5-8846-e57f2dc71e63.preview.emergentagent.com/api"
    
    # Setup tester
    tester = MIND14APITester(backend_url)
    
    # Run tests
    print("🚀 Starting MIND14 Virtual Front Desk API Tests - Phase 3: Automation Integration")
    print(f"Backend URL: {backend_url}")
    
    # Basic health checks
    root_success = tester.test_root_endpoint()
    ai_status_success = tester.test_ai_status()
    
    # Automation service status checks
    automation_success = tester.test_automation_status()
    calendar_success = tester.test_calendar_status()
    notifications_success = tester.test_notifications_status()
    workflows_success = tester.test_workflows_available()
    
    # Service and conversation checks
    services_success = tester.test_services_list()
    conversations_success = tester.test_conversations_endpoint()
    
    # Intent classification tests
    intent_success = tester.test_intent_classification()
    
    # Complete booking flow with automation
    booking_success = tester.test_multi_step_booking()
    
    # Multilingual test
    multilingual_success = tester.test_multilingual_booking()
    
    # Analytics dashboard
    analytics_success = tester.test_analytics_dashboard()
    
    # Print results
    print(f"\n📊 Tests passed: {tester.tests_passed}/{tester.tests_run}")
    
    # Return success status
    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    sys.exit(main())