import requests
import sys
import json
from datetime import datetime

class MIND14APITester:
    def __init__(self, base_url="https://7d6b7df0-b5d7-44c5-8846-e57f2dc71e63.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.conversation_id = None

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

    def test_services_list(self):
        """Test the services list endpoint"""
        success, response = self.run_test(
            "Services List Endpoint",
            "GET",
            "services",
            200
        )
        if success and 'services' in response:
            print(f"Number of services: {len(response['services'])}")
            for service in response['services']:
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
        """Test a multi-step booking flow"""
        # Start a new conversation
        self.conversation_id = None
        
        # Step 1: Initial request
        print("\n🔍 Testing multi-step booking flow")
        success1, response1 = self.test_chat_endpoint("I need to book a doctor appointment")
        
        if not success1:
            return False
        
        # Step 2: Confirm service
        success2, response2 = self.test_chat_endpoint("Yes, I want to book an appointment")
        
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
        
        if not success5:
            return False
            
        print("\n✅ Multi-step booking flow completed successfully")
        return True

def main():
    # Get the backend URL from command line or use default
    backend_url = "https://7d6b7df0-b5d7-44c5-8846-e57f2dc71e63.preview.emergentagent.com/api"
    
    # Setup tester
    tester = MIND14APITester(backend_url)
    
    # Run tests
    print("🚀 Starting MIND14 Virtual Front Desk API Tests")
    print(f"Backend URL: {backend_url}")
    
    # Basic health checks
    root_success = tester.test_root_endpoint()
    ai_status_success = tester.test_ai_status()
    services_success = tester.test_services_list()
    conversations_success = tester.test_conversations_endpoint()
    
    # Intent classification tests
    intent_success = tester.test_intent_classification()
    
    # Multi-step booking flow
    booking_success = tester.test_multi_step_booking()
    
    # Print results
    print(f"\n📊 Tests passed: {tester.tests_passed}/{tester.tests_run}")
    
    # Return success status
    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    sys.exit(main())