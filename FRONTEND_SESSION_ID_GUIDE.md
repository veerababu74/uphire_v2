# Frontend Session ID Integration Guide

This guide shows frontend developers exactly how to get and use session IDs from the resume processing APIs.

## 🚀 **Quick Start for Frontend Developers**

### **Step 1: Upload File and Get Session ID**

```javascript
// Upload Excel file and get session ID
async function uploadExcelFile(file, userId, username) {
    const formData = new FormData();
    formData.append('file', file);
    
    const params = new URLSearchParams({
        base_user_id: userId,
        base_username: username
    });
    
    try {
        const response = await fetch(`/enhanced-excel-parser/upload-and-process?${params}`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        // ✅ SESSION ID IS HERE!
        const sessionId = result.session_id;
        
        console.log('Session ID:', sessionId);
        return {
            sessionId: sessionId,
            fileName: result.file_name,
            fileSize: result.file_size,
            trackingEndpoints: result.tracking_endpoints
        };
        
    } catch (error) {
        console.error('Upload failed:', error);
        throw error;
    }
}

// Upload multiple resume files and get session ID
async function uploadMultipleResumes(files, userId, username) {
    const formData = new FormData();
    
    // Add all files to FormData
    files.forEach(file => {
        formData.append('files', file);
    });
    
    const params = new URLSearchParams({
        base_user_id: userId,
        base_username: username
    });
    
    try {
        const response = await fetch(`/enhanced-multiple-resume-parser/upload-and-process?${params}`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        // ✅ SESSION ID IS HERE!
        const sessionId = result.session_id;
        
        return {
            sessionId: sessionId,
            fileCount: result.file_count,
            totalSize: result.total_size,
            trackingEndpoints: result.tracking_endpoints
        };
        
    } catch (error) {
        console.error('Upload failed:', error);
        throw error;
    }
}
```

### **Step 2: Store Session ID for Tracking**

```javascript
// Store session ID in browser storage
function storeSessionId(sessionId, metadata = {}) {
    const sessionData = {
        sessionId: sessionId,
        timestamp: new Date().toISOString(),
        ...metadata
    };
    
    // Store in localStorage for persistence
    localStorage.setItem(`session_${sessionId}`, JSON.stringify(sessionData));
    
    // Store in sessionStorage for current tab only
    sessionStorage.setItem('currentSession', sessionId);
    
    // Store in global variable for immediate access
    window.currentSessionId = sessionId;
}

// Retrieve stored session ID
function getStoredSessionId(sessionId = null) {
    if (sessionId) {
        // Get specific session
        const data = localStorage.getItem(`session_${sessionId}`);
        return data ? JSON.parse(data) : null;
    } else {
        // Get current session
        return sessionStorage.getItem('currentSession') || window.currentSessionId;
    }
}
```

## 📱 **Complete React Component Example**

```jsx
import React, { useState, useEffect } from 'react';

const ResumeUploadComponent = () => {
    const [sessionId, setSessionId] = useState(null);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [processingStatus, setProcessingStatus] = useState('idle');
    const [uploadedFile, setUploadedFile] = useState(null);

    // Step 1: Handle file upload and get session ID
    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        setUploadedFile(file);
        setProcessingStatus('uploading');

        try {
            // Upload file and get session ID
            const uploadResult = await uploadExcelFile(
                file, 
                'frontend_user_123', 
                'frontend_user'
            );

            // ✅ GOT SESSION ID FROM UPLOAD!
            const newSessionId = uploadResult.sessionId;
            setSessionId(newSessionId);
            
            // Store session ID
            storeSessionId(newSessionId, {
                fileName: file.name,
                fileSize: file.size,
                uploadTime: new Date().toISOString()
            });

            setProcessingStatus('processing');

            // Start monitoring with the session ID
            startMonitoring(newSessionId);

        } catch (error) {
            console.error('Upload failed:', error);
            setProcessingStatus('error');
        }
    };

    // Step 2: Monitor progress using session ID
    const startMonitoring = (sessionId) => {
        // Option 1: Server-Sent Events for real-time updates
        const eventSource = new EventSource(
            `/enhanced-resume-processing/live-updates/${sessionId}`
        );

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            setUploadProgress(data.metrics.completion_percentage);
            setProcessingStatus(data.status);

            if (data.status === 'completed') {
                eventSource.close();
                console.log('Processing completed!');
            }
        };

        eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
            eventSource.close();
            
            // Fallback to polling
            startPolling(sessionId);
        };
    };

    // Fallback polling method
    const startPolling = (sessionId) => {
        const pollInterval = setInterval(async () => {
            try {
                const response = await fetch(
                    `/enhanced-resume-processing/status/${sessionId}`
                );
                const status = await response.json();

                setUploadProgress(status.metrics.completion_percentage);
                setProcessingStatus(status.status);

                if (status.status === 'completed' || status.status === 'failed') {
                    clearInterval(pollInterval);
                }

            } catch (error) {
                console.error('Polling error:', error);
                clearInterval(pollInterval);
            }
        }, 2000); // Poll every 2 seconds
    };

    // Get session errors
    const getSessionErrors = async () => {
        if (!sessionId) return;

        try {
            const response = await fetch(
                `/enhanced-resume-processing/errors/${sessionId}`
            );
            const errors = await response.json();
            console.log('Session errors:', errors);
            return errors;
        } catch (error) {
            console.error('Failed to get errors:', error);
        }
    };

    return (
        <div className="resume-upload">
            <h2>Resume Processing</h2>
            
            {/* File upload */}
            <input 
                type="file" 
                accept=".xlsx,.xls,.pdf,.doc,.docx"
                onChange={handleFileUpload}
                disabled={processingStatus === 'processing'}
            />

            {/* Display session ID */}
            {sessionId && (
                <div className="session-info">
                    <p><strong>Session ID:</strong> {sessionId}</p>
                    <p><strong>Status:</strong> {processingStatus}</p>
                </div>
            )}

            {/* Progress bar */}
            {processingStatus === 'processing' && (
                <div className="progress-container">
                    <div className="progress-bar">
                        <div 
                            className="progress-fill"
                            style={{ width: `${uploadProgress}%` }}
                        ></div>
                    </div>
                    <p>{uploadProgress.toFixed(1)}% Complete</p>
                </div>
            )}

            {/* Action buttons */}
            {sessionId && (
                <div className="session-actions">
                    <button onClick={getSessionErrors}>
                        View Errors
                    </button>
                    <button onClick={() => navigator.clipboard.writeText(sessionId)}>
                        Copy Session ID
                    </button>
                </div>
            )}
        </div>
    );
};

export default ResumeUploadComponent;
```

## 🔄 **Vue.js Example**

```vue
<template>
  <div class="resume-processor">
    <h2>Resume Processing</h2>
    
    <!-- File Upload -->
    <input 
      type="file" 
      @change="handleFileUpload"
      :disabled="isProcessing"
      multiple
    />
    
    <!-- Session Information -->
    <div v-if="sessionId" class="session-info">
      <p><strong>Session ID:</strong> {{ sessionId }}</p>
      <p><strong>Status:</strong> {{ processingStatus }}</p>
      <p><strong>Progress:</strong> {{ progress }}%</p>
    </div>
    
    <!-- Progress Bar -->
    <div v-if="isProcessing" class="progress-bar">
      <div class="progress-fill" :style="{ width: progress + '%' }"></div>
    </div>
    
    <!-- Session Actions -->
    <div v-if="sessionId" class="actions">
      <button @click="getErrors">View Errors</button>
      <button @click="resumeSession">Resume</button>
      <button @click="stopSession">Stop</button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ResumeProcessor',
  data() {
    return {
      sessionId: null,
      processingStatus: 'idle',
      progress: 0,
      isProcessing: false,
      eventSource: null
    }
  },
  methods: {
    async handleFileUpload(event) {
      const files = Array.from(event.target.files);
      if (files.length === 0) return;

      this.isProcessing = true;
      this.processingStatus = 'uploading';

      try {
        // Upload files and get session ID
        const result = await this.uploadFiles(files);
        
        // ✅ SESSION ID FROM UPLOAD RESPONSE
        this.sessionId = result.sessionId;
        
        // Store session ID
        this.$store.commit('setCurrentSession', this.sessionId);
        
        // Start monitoring
        this.startRealTimeMonitoring();
        
      } catch (error) {
        console.error('Upload failed:', error);
        this.processingStatus = 'error';
        this.isProcessing = false;
      }
    },

    async uploadFiles(files) {
      const formData = new FormData();
      files.forEach(file => formData.append('files', file));
      
      const params = new URLSearchParams({
        base_user_id: this.$store.state.user.id,
        base_username: this.$store.state.user.username
      });

      const response = await fetch(
        `/enhanced-multiple-resume-parser/upload-and-process?${params}`,
        {
          method: 'POST',
          body: formData
        }
      );

      const result = await response.json();
      return {
        sessionId: result.session_id,
        fileCount: result.file_count
      };
    },

    startRealTimeMonitoring() {
      // Close existing connection
      if (this.eventSource) {
        this.eventSource.close();
      }

      // Start new SSE connection
      this.eventSource = new EventSource(
        `/enhanced-resume-processing/live-updates/${this.sessionId}`
      );

      this.eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        this.progress = data.metrics.completion_percentage;
        this.processingStatus = data.status;

        if (data.status === 'completed') {
          this.isProcessing = false;
          this.eventSource.close();
          this.$emit('processing-complete', data);
        }
      };

      this.eventSource.onerror = (error) => {
        console.error('SSE Error:', error);
        this.eventSource.close();
        // Fallback to polling
        this.startPolling();
      };
    },

    async getErrors() {
      if (!this.sessionId) return;

      try {
        const response = await fetch(
          `/enhanced-resume-processing/errors/${this.sessionId}`
        );
        const errors = await response.json();
        
        // Show errors in modal or component
        this.$emit('show-errors', errors);
        
      } catch (error) {
        console.error('Failed to get errors:', error);
      }
    },

    async resumeSession() {
      if (!this.sessionId) return;

      try {
        const response = await fetch(
          `/enhanced-resume-processing/resume-session/${this.sessionId}`,
          { method: 'POST' }
        );
        
        if (response.ok) {
          this.startRealTimeMonitoring();
          this.isProcessing = true;
        }
        
      } catch (error) {
        console.error('Failed to resume session:', error);
      }
    },

    async stopSession() {
      if (!this.sessionId) return;

      try {
        await fetch(
          `/enhanced-resume-processing/stop-session/${this.sessionId}`,
          { method: 'POST' }
        );
        
        this.isProcessing = false;
        if (this.eventSource) {
          this.eventSource.close();
        }
        
      } catch (error) {
        console.error('Failed to stop session:', error);
      }
    }
  },

  beforeUnmount() {
    if (this.eventSource) {
      this.eventSource.close();
    }
  }
}
</script>
```

## 📋 **API Response Examples**

### **Excel Upload Response**
```json
{
  "status": "accepted",
  "message": "Excel file uploaded and processing started",
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "file_name": "resumes_data.xlsx",
  "file_size": 2048000,
  "estimated_processing_time": "5-10 minutes",
  "tracking_endpoints": {
    "status": "/enhanced-resume-processing/status/a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "errors": "/enhanced-resume-processing/errors/a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "live_updates": "/enhanced-resume-processing/live-updates/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  }
}
```

### **Multiple Resume Upload Response**
```json
{
  "status": "accepted",
  "message": "Multiple resume files uploaded and processing started",
  "session_id": "b2c3d4e5-f6g7-8901-bcde-f23456789012",
  "file_count": 15,
  "total_size": 15728640,
  "files": [
    "resume_john_doe.pdf",
    "resume_jane_smith.docx",
    "..."
  ],
  "tracking_endpoints": {
    "status": "/enhanced-resume-processing/status/b2c3d4e5-f6g7-8901-bcde-f23456789012",
    "errors": "/enhanced-resume-processing/errors/b2c3d4e5-f6g7-8901-bcde-f23456789012",
    "live_updates": "/enhanced-resume-processing/live-updates/b2c3d4e5-f6g7-8901-bcde-f23456789012"
  }
}
```

## 🛠 **Utility Functions for Frontend Teams**

```javascript
// Session management utility class
class SessionManager {
    constructor(baseURL = '') {
        this.baseURL = baseURL;
        this.sessions = new Map();
    }

    // Store session with metadata
    storeSession(sessionId, metadata) {
        const sessionData = {
            id: sessionId,
            createdAt: new Date(),
            ...metadata
        };
        
        this.sessions.set(sessionId, sessionData);
        localStorage.setItem(`session_${sessionId}`, JSON.stringify(sessionData));
        
        return sessionData;
    }

    // Get session by ID
    getSession(sessionId) {
        // Check memory first
        if (this.sessions.has(sessionId)) {
            return this.sessions.get(sessionId);
        }
        
        // Check localStorage
        const stored = localStorage.getItem(`session_${sessionId}`);
        if (stored) {
            const sessionData = JSON.parse(stored);
            this.sessions.set(sessionId, sessionData);
            return sessionData;
        }
        
        return null;
    }

    // Get all stored sessions
    getAllSessions() {
        const allSessions = [];
        
        // Get from localStorage
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key.startsWith('session_')) {
                const sessionData = JSON.parse(localStorage.getItem(key));
                allSessions.push(sessionData);
            }
        }
        
        return allSessions;
    }

    // Monitor session with callbacks
    monitorSession(sessionId, callbacks = {}) {
        const {
            onProgress = () => {},
            onComplete = () => {},
            onError = () => {},
            onStatusChange = () => {}
        } = callbacks;

        // Server-Sent Events
        const eventSource = new EventSource(
            `${this.baseURL}/enhanced-resume-processing/live-updates/${sessionId}`
        );

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            onProgress(data.metrics);
            onStatusChange(data.status);

            if (data.status === 'completed') {
                onComplete(data);
                eventSource.close();
            } else if (data.status === 'failed') {
                onError(data);
                eventSource.close();
            }
        };

        eventSource.onerror = (error) => {
            onError(error);
            eventSource.close();
        };

        return eventSource;
    }

    // Get session status
    async getSessionStatus(sessionId) {
        try {
            const response = await fetch(
                `${this.baseURL}/enhanced-resume-processing/status/${sessionId}`
            );
            return await response.json();
        } catch (error) {
            console.error('Failed to get session status:', error);
            return null;
        }
    }

    // Resume a paused session
    async resumeSession(sessionId) {
        try {
            const response = await fetch(
                `${this.baseURL}/enhanced-resume-processing/resume-session/${sessionId}`,
                { method: 'POST' }
            );
            return await response.json();
        } catch (error) {
            console.error('Failed to resume session:', error);
            return null;
        }
    }
}

// Export for use in components
export default SessionManager;

// Usage example:
const sessionManager = new SessionManager();

// After upload
const sessionId = uploadResult.session_id;
sessionManager.storeSession(sessionId, {
    fileName: file.name,
    uploadTime: new Date()
});

// Monitor session
sessionManager.monitorSession(sessionId, {
    onProgress: (metrics) => {
        updateProgressBar(metrics.completion_percentage);
    },
    onComplete: (data) => {
        showSuccessMessage('Processing completed!');
    },
    onError: (error) => {
        showErrorMessage('Processing failed');
    }
});
```

## 🎯 **Key Takeaways for Frontend Teams**

1. **Session ID Location**: Always in the upload response as `session_id`
2. **Immediate Storage**: Store the session ID immediately after upload
3. **Real-time Monitoring**: Use Server-Sent Events for live updates
4. **Fallback Strategy**: Implement polling as fallback for SSE
5. **Error Handling**: Always handle network errors and API failures
6. **User Experience**: Show progress, status, and allow session management

The session ID is your frontend's key to tracking and managing resume processing operations in real-time!