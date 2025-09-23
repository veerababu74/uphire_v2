# How to Get Session ID - Complete Guide

The session ID is the key to tracking and monitoring your resume processing operations. Here are all the different ways to obtain and use session IDs:

## 🔍 **1. Getting Session ID from Direct Processing Functions**

### Excel Processing
```python
from excel_resume_parser.enhanced_excel_parser_with_tracking import enhanced_excel_parser

# Process Excel file and get session ID from result
result = await enhanced_excel_parser.process_excel_file_with_tracking(
    file_path="your_file.xlsx",
    base_user_id="user_123",
    base_username="john_doe",
    cleanup_file=True
)

# Extract session ID from result
session_id = result.get('session_id')
print(f"Session ID: {session_id}")
```

### Multiple Resume Processing
```python
from multipleresumepraser.enhanced_multiple_resume_parser_with_tracking import enhanced_multiple_resume_parser

# Process multiple resumes and get session ID
result = await enhanced_multiple_resume_parser.process_multiple_resumes_with_tracking(
    resume_files=["resume1.pdf", "resume2.docx"],
    base_user_id="user_456",
    base_username="jane_doe",
    cleanup_files=True
)

# Extract session ID from result
session_id = result.get('session_id')
print(f"Session ID: {session_id}")
```

## 🌐 **2. Getting Session ID from API Responses**

### Excel Upload API
```python
import requests

# Upload Excel file via API
files = {'file': open('data.xlsx', 'rb')}
params = {
    'base_user_id': 'api_user_123',
    'base_username': 'api_user'
}

response = requests.post(
    'http://localhost:8000/enhanced-excel-parser/upload-and-process',
    files=files,
    params=params
)

result = response.json()
session_id = result['session_id']
print(f"API Session ID: {session_id}")

# API response structure:
{
    "status": "accepted",
    "message": "Excel file uploaded and processing started",
    "session_id": "12345678-1234-1234-1234-123456789abc",
    "file_name": "data.xlsx",
    "file_size": 1024000,
    "tracking_endpoints": {
        "status": "/enhanced-resume-processing/status/12345678-1234-1234-1234-123456789abc",
        "errors": "/enhanced-resume-processing/errors/12345678-1234-1234-1234-123456789abc",
        "live_updates": "/enhanced-resume-processing/live-updates/12345678-1234-1234-1234-123456789abc"
    }
}
```

### Multiple Resume Upload API
```python
import requests

# Upload multiple resume files
files = [
    ('files', open('resume1.pdf', 'rb')),
    ('files', open('resume2.docx', 'rb'))
]
params = {
    'base_user_id': 'api_user_456',
    'base_username': 'api_user_resumes'
}

response = requests.post(
    'http://localhost:8000/enhanced-multiple-resume-parser/upload-and-process',
    files=files,
    params=params
)

result = response.json()
session_id = result['session_id']
print(f"Resume API Session ID: {session_id}")
```

## 📋 **3. Getting Session IDs from Active Sessions List**

### Using Progress Tracker
```python
from core.progress_tracker import progress_tracker

# Get all active sessions
active_sessions = progress_tracker.list_active_sessions()

for session in active_sessions:
    session_id = session['session_id']
    operation_type = session['operation_type']
    status = session['status']
    username = session['username']
    
    print(f"Session ID: {session_id}")
    print(f"Operation: {operation_type}")
    print(f"Status: {status}")
    print(f"User: {username}")
    print("-" * 40)
```

### Using API Endpoint
```python
import requests

# Get active sessions via API
response = requests.get('http://localhost:8000/enhanced-resume-processing/active-sessions')
data = response.json()

for session in data['sessions']:
    session_id = session['session_id']
    print(f"Active Session ID: {session_id}")
```

### Filter by User
```python
# Get sessions for specific user
user_sessions = progress_tracker.list_active_sessions(user_id="user_123")

for session in user_sessions:
    session_id = session['session_id']
    print(f"User Session ID: {session_id}")
```

## 🎯 **4. Creating Session ID Manually**

### Direct Session Creation
```python
from core.progress_tracker import progress_tracker, OperationType

# Create a new session manually
session_id = progress_tracker.create_session(
    operation_type=OperationType.EXCEL_PARSING,
    user_id="manual_user_123",
    username="manual_user",
    file_name="manual_file.xlsx",
    total_items=1000,
    configuration={
        "batch_size": 50,
        "timeout": 120
    }
)

print(f"Manually Created Session ID: {session_id}")
```

## 📊 **5. Using Session ID for Monitoring**

### Real-time Status Monitoring
```python
from core.progress_tracker import progress_tracker
import time

def monitor_session(session_id):
    """Monitor a session in real-time."""
    print(f"Monitoring Session: {session_id}")
    
    while True:
        status = progress_tracker.get_session_status(session_id)
        
        if not status:
            print("Session not found!")
            break
        
        metrics = status['metrics']
        current_status = status['status']
        
        print(f"Progress: {metrics['completion_percentage']:.1f}%")
        print(f"Status: {current_status}")
        print(f"Processed: {metrics['processed_items']}/{metrics['total_items']}")
        print(f"Success Rate: {metrics['success_rate']:.1f}%")
        print("-" * 30)
        
        if current_status in ['completed', 'failed', 'cancelled']:
            print(f"Session {current_status}!")
            break
        
        time.sleep(5)  # Check every 5 seconds

# Use the monitor function
session_id = "your-session-id-here"
monitor_session(session_id)
```

### Get Error Details
```python
def get_session_errors(session_id):
    """Get detailed error information for a session."""
    errors = progress_tracker.get_session_errors(session_id, limit=20)
    
    if not errors:
        print("No errors found for this session.")
        return
    
    print(f"Found {len(errors)} errors for session {session_id}:")
    
    for i, error in enumerate(errors, 1):
        print(f"\nError {i}:")
        print(f"  Type: {error['error_type']}")
        print(f"  Message: {error['error_message']}")
        print(f"  Severity: {error['severity']}")
        print(f"  Time: {error['timestamp']}")
        
        if error.get('item_index') is not None:
            print(f"  Item Index: {error['item_index']}")

# Use the function
session_id = "your-session-id-here"
get_session_errors(session_id)
```

## 🔄 **6. Session Recovery and Resumption**

### Resume Interrupted Session
```python
def resume_processing_session(session_id):
    """Resume an interrupted processing session."""
    from excel_resume_parser.enhanced_excel_parser_with_tracking import enhanced_excel_parser
    from multipleresumepraser.enhanced_multiple_resume_parser_with_tracking import enhanced_multiple_resume_parser
    
    # Get session status first
    status = progress_tracker.get_session_status(session_id)
    
    if not status:
        print(f"Session {session_id} not found!")
        return
    
    if status['status'] not in ['paused', 'failed']:
        print(f"Cannot resume session with status: {status['status']}")
        return
    
    operation_type = status['operation_type']
    
    try:
        if operation_type == 'excel_parsing':
            resume_data = enhanced_excel_parser.resume_processing(session_id)
        elif operation_type == 'multiple_resume_parsing':
            resume_data = enhanced_multiple_resume_parser.resume_processing(session_id)
        else:
            print(f"Unknown operation type: {operation_type}")
            return
        
        if resume_data:
            print(f"Session {session_id} resumed successfully!")
            print(f"Resuming from item: {resume_data.get('last_processed_index', 0)}")
        else:
            print(f"Failed to resume session {session_id}")
            
    except Exception as e:
        print(f"Error resuming session: {str(e)}")

# Use the function
session_id = "your-paused-session-id"
resume_processing_session(session_id)
```

## 🌐 **7. API-based Session Management**

### Complete API Example
```python
import requests
import time
import json

class SessionManager:
    """Manage sessions via API calls."""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def start_excel_processing(self, file_path, user_id, username):
        """Start Excel processing and return session ID."""
        files = {'file': open(file_path, 'rb')}
        params = {
            'base_user_id': user_id,
            'base_username': username
        }
        
        response = requests.post(
            f"{self.base_url}/enhanced-excel-parser/upload-and-process",
            files=files,
            params=params
        )
        
        result = response.json()
        return result['session_id']
    
    def get_session_status(self, session_id):
        """Get current session status."""
        response = requests.get(
            f"{self.base_url}/enhanced-resume-processing/status/{session_id}"
        )
        return response.json()
    
    def get_session_errors(self, session_id, limit=50):
        """Get session errors."""
        response = requests.get(
            f"{self.base_url}/enhanced-resume-processing/errors/{session_id}",
            params={'limit': limit}
        )
        return response.json()
    
    def resume_session(self, session_id):
        """Resume a paused session."""
        response = requests.post(
            f"{self.base_url}/enhanced-resume-processing/resume-session/{session_id}"
        )
        return response.json()
    
    def stop_session(self, session_id):
        """Stop a running session."""
        response = requests.post(
            f"{self.base_url}/enhanced-resume-processing/stop-session/{session_id}"
        )
        return response.json()
    
    def monitor_session(self, session_id, check_interval=5):
        """Monitor session progress in real-time."""
        print(f"Monitoring session: {session_id}")
        
        while True:
            try:
                status = self.get_session_status(session_id)
                
                metrics = status['metrics']
                current_status = status['status']
                
                print(f"Progress: {metrics['completion_percentage']:.1f}%")
                print(f"Status: {current_status}")
                print(f"Rate: {metrics['processing_rate']:.2f} items/sec")
                print("-" * 30)
                
                if current_status in ['completed', 'failed', 'cancelled']:
                    print(f"Session {current_status}!")
                    break
                
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"Monitoring error: {str(e)}")
                break

# Usage example
manager = SessionManager()

# Start processing and get session ID
session_id = manager.start_excel_processing(
    file_path="data.xlsx",
    user_id="api_user_123",
    username="api_user"
)

print(f"Started processing with session ID: {session_id}")

# Monitor the session
manager.monitor_session(session_id)
```

## 🎯 **8. Best Practices for Session ID Management**

### Store Session IDs
```python
import json
from datetime import datetime

class SessionTracker:
    """Track and manage session IDs."""
    
    def __init__(self, storage_file="session_tracker.json"):
        self.storage_file = storage_file
        self.sessions = self.load_sessions()
    
    def load_sessions(self):
        """Load sessions from storage."""
        try:
            with open(self.storage_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_sessions(self):
        """Save sessions to storage."""
        with open(self.storage_file, 'w') as f:
            json.dump(self.sessions, f, indent=2)
    
    def add_session(self, session_id, operation_type, user_id, file_name=None):
        """Add a new session to tracking."""
        self.sessions[session_id] = {
            'operation_type': operation_type,
            'user_id': user_id,
            'file_name': file_name,
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
        self.save_sessions()
    
    def update_session_status(self, session_id, status):
        """Update session status."""
        if session_id in self.sessions:
            self.sessions[session_id]['status'] = status
            self.sessions[session_id]['updated_at'] = datetime.now().isoformat()
            self.save_sessions()
    
    def get_user_sessions(self, user_id):
        """Get all sessions for a user."""
        return {
            sid: data for sid, data in self.sessions.items()
            if data['user_id'] == user_id
        }
    
    def get_active_sessions(self):
        """Get all active sessions."""
        return {
            sid: data for sid, data in self.sessions.items()
            if data.get('status', 'active') == 'active'
        }

# Usage
tracker = SessionTracker()

# Add session when starting processing
session_id = "new-session-id"
tracker.add_session(session_id, "excel_parsing", "user_123", "data.xlsx")

# Update when completed
tracker.update_session_status(session_id, "completed")

# Get user's sessions
user_sessions = tracker.get_user_sessions("user_123")
print(f"User sessions: {user_sessions}")
```

### Session ID Validation
```python
import re
import uuid

def is_valid_session_id(session_id):
    """Validate if a string is a valid session ID."""
    if not session_id or not isinstance(session_id, str):
        return False
    
    # Check if it's a valid UUID format
    try:
        uuid.UUID(session_id)
        return True
    except ValueError:
        return False

def validate_and_get_session(session_id):
    """Validate session ID and get session data."""
    if not is_valid_session_id(session_id):
        return None, "Invalid session ID format"
    
    status = progress_tracker.get_session_status(session_id)
    if not status:
        return None, "Session not found"
    
    return status, "Valid session"

# Usage
session_id = "12345678-1234-1234-1234-123456789abc"
session_data, message = validate_and_get_session(session_id)

if session_data:
    print(f"Valid session: {session_data['status']}")
else:
    print(f"Error: {message}")
```

## 📱 **9. Frontend/JavaScript Integration**

### JavaScript Example for Web Applications
```javascript
class ResumeProcessingClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    // Start Excel processing
    async startExcelProcessing(file, userId, username) {
        const formData = new FormData();
        formData.append('file', file);
        
        const params = new URLSearchParams({
            base_user_id: userId,
            base_username: username
        });
        
        const response = await fetch(
            `${this.baseUrl}/enhanced-excel-parser/upload-and-process?${params}`,
            {
                method: 'POST',
                body: formData
            }
        );
        
        const result = await response.json();
        return result.session_id;
    }
    
    // Get session status
    async getSessionStatus(sessionId) {
        const response = await fetch(
            `${this.baseUrl}/enhanced-resume-processing/status/${sessionId}`
        );
        return await response.json();
    }
    
    // Monitor session with Server-Sent Events
    monitorSession(sessionId, onUpdate, onComplete, onError) {
        const eventSource = new EventSource(
            `${this.baseUrl}/enhanced-resume-processing/live-updates/${sessionId}`
        );
        
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            onUpdate(data);
            
            if (data.status === 'completed') {
                onComplete(data);
                eventSource.close();
            }
        };
        
        eventSource.onerror = function(event) {
            onError(event);
            eventSource.close();
        };
        
        return eventSource;
    }
}

// Usage in web application
const client = new ResumeProcessingClient();

// Start processing
document.getElementById('uploadForm').onsubmit = async function(e) {
    e.preventDefault();
    
    const file = document.getElementById('fileInput').files[0];
    const sessionId = await client.startExcelProcessing(file, 'web_user_123', 'web_user');
    
    console.log('Session ID:', sessionId);
    
    // Store session ID for later use
    localStorage.setItem('currentSessionId', sessionId);
    
    // Start monitoring
    client.monitorSession(
        sessionId,
        (data) => {
            // Update progress bar
            const progress = data.metrics.completion_percentage;
            document.getElementById('progressBar').style.width = `${progress}%`;
            document.getElementById('progressText').textContent = `${progress.toFixed(1)}%`;
        },
        (data) => {
            // Processing completed
            alert('Processing completed successfully!');
        },
        (error) => {
            // Handle error
            console.error('Monitoring error:', error);
        }
    );
};
```

This comprehensive guide covers all the ways to obtain and use session IDs in the enhanced resume processing system. The session ID is your key to tracking, monitoring, and managing processing operations effectively.