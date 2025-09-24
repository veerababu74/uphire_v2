/**
 * Real-time Resume Processing Client
 * 
 * Industry-ready JavaScript client for tracking bulk resume processing
 * with live progress updates and comprehensive error handling.
 */

class ResumeProcessingClient {
    constructor(baseUrl = '', options = {}) {
        this.baseUrl = baseUrl.replace(/\/$/, ''); // Remove trailing slash
        this.options = {
            pollInterval: 2000, // 2 seconds
            maxRetries: 3,
            retryDelay: 1000,
            onProgress: null,
            onComplete: null,
            onError: null,
            ...options
        };

        // Active polling jobs
        this.activePollingJobs = new Map();

        // Event listeners
        this.listeners = {
            progress: [],
            complete: [],
            error: [],
            statusChange: []
        };
    }

    /**
     * Upload Excel file for processing with real-time tracking
     */
    async uploadExcelFile(file, userId, username, sheetName = null) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('user_id', userId);
            formData.append('username', username);
            if (sheetName) {
                formData.append('sheet_name', sheetName);
            }

            const response = await fetch(`${this.baseUrl}/enhanced-excel-parser/upload-async`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            if (result.status === 'accepted') {
                // Start polling for progress
                this.startPolling(result.job_id, 'excel');
                return {
                    success: true,
                    jobId: result.job_id,
                    message: result.message,
                    pollUrl: result.poll_url,
                    fileInfo: result.file_info
                };
            } else {
                throw new Error(result.message || 'Upload failed');
            }
        } catch (error) {
            this.emit('error', { type: 'upload', error: error.message });
            throw error;
        }
    }

    /**
     * Upload multiple resume files for processing with real-time tracking
     */
    async uploadMultipleResumes(files, duplicateCheck = true) {
        try {
            const formData = new FormData();

            // Add all files
            for (let i = 0; i < files.length; i++) {
                formData.append('files', files[i]);
            }

            formData.append('duplicate_check', duplicateCheck.toString());

            const response = await fetch(`${this.baseUrl}/enhanced-bulk-parser/upload-async`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            if (result.status === 'accepted') {
                // Start polling for progress
                this.startPolling(result.job_id, 'bulk');
                return {
                    success: true,
                    jobId: result.job_id,
                    message: result.message,
                    pollUrl: result.poll_url,
                    processingInfo: result.processing_info
                };
            } else {
                throw new Error(result.message || 'Upload failed');
            }
        } catch (error) {
            this.emit('error', { type: 'upload', error: error.message });
            throw error;
        }
    }

    /**
     * Start polling for job progress
     */
    startPolling(jobId, jobType) {
        if (this.activePollingJobs.has(jobId)) {
            return; // Already polling this job
        }

        const pollData = {
            jobId,
            jobType,
            retryCount: 0,
            intervalId: null
        };

        const poll = async () => {
            try {
                const status = await this.getJobStatus(jobId, jobType);

                if (status) {
                    this.emit('progress', {
                        jobId,
                        jobType,
                        status,
                        timestamp: new Date().toISOString()
                    });

                    // Check if job is complete
                    if (['completed', 'failed', 'cancelled'].includes(status.status)) {
                        this.stopPolling(jobId);

                        if (status.status === 'completed') {
                            // Get final results
                            const results = await this.getJobResults(jobId, jobType);
                            this.emit('complete', {
                                jobId,
                                jobType,
                                status,
                                results,
                                timestamp: new Date().toISOString()
                            });
                        } else {
                            this.emit('error', {
                                jobId,
                                jobType,
                                status,
                                error: `Job ${status.status}`,
                                timestamp: new Date().toISOString()
                            });
                        }
                    }
                }

                pollData.retryCount = 0; // Reset retry count on success

            } catch (error) {
                pollData.retryCount++;

                if (pollData.retryCount >= this.options.maxRetries) {
                    this.stopPolling(jobId);
                    this.emit('error', {
                        jobId,
                        jobType,
                        error: `Polling failed after ${this.options.maxRetries} retries: ${error.message}`,
                        timestamp: new Date().toISOString()
                    });
                } else {
                    // Exponential backoff
                    const delay = this.options.retryDelay * Math.pow(2, pollData.retryCount - 1);
                    setTimeout(() => {
                        if (this.activePollingJobs.has(jobId)) {
                            poll();
                        }
                    }, delay);
                    return;
                }
            }
        };

        // Start polling
        pollData.intervalId = setInterval(poll, this.options.pollInterval);
        this.activePollingJobs.set(jobId, pollData);

        // Initial poll
        poll();
    }

    /**
     * Stop polling for a specific job
     */
    stopPolling(jobId) {
        const pollData = this.activePollingJobs.get(jobId);
        if (pollData && pollData.intervalId) {
            clearInterval(pollData.intervalId);
            this.activePollingJobs.delete(jobId);
        }
    }

    /**
     * Stop all active polling
     */
    stopAllPolling() {
        for (const jobId of this.activePollingJobs.keys()) {
            this.stopPolling(jobId);
        }
    }

    /**
     * Get job status
     */
    async getJobStatus(jobId, jobType) {
        try {
            const endpoint = jobType === 'excel'
                ? `/enhanced-excel-parser/status/${jobId}`
                : `/enhanced-bulk-parser/status/${jobId}`;

            const response = await fetch(`${this.baseUrl}${endpoint}`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            return result.job_status;
        } catch (error) {
            throw new Error(`Failed to get job status: ${error.message}`);
        }
    }

    /**
     * Get job results
     */
    async getJobResults(jobId, jobType) {
        try {
            const endpoint = jobType === 'excel'
                ? `/enhanced-excel-parser/results/${jobId}`
                : `/enhanced-bulk-parser/results/${jobId}`;

            const response = await fetch(`${this.baseUrl}${endpoint}`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            return result;
        } catch (error) {
            throw new Error(`Failed to get job results: ${error.message}`);
        }
    }

    /**
     * Control job execution (pause, resume, cancel)
     */
    async controlJob(jobId, jobType, action) {
        try {
            const endpoint = jobType === 'excel'
                ? `/enhanced-excel-parser/control/${jobId}/${action}`
                : `/enhanced-bulk-parser/control/${jobId}/${action}`;

            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            // Update polling based on action
            if (action === 'cancel') {
                this.stopPolling(jobId);
            }

            return result;
        } catch (error) {
            throw new Error(`Failed to ${action} job: ${error.message}`);
        }
    }

    /**
     * Get all jobs for a user
     */
    async getAllJobs(userId = null, activeOnly = false, jobType = 'excel') {
        try {
            const endpoint = jobType === 'excel'
                ? '/enhanced-excel-parser/jobs'
                : '/enhanced-bulk-parser/jobs';

            const params = new URLSearchParams();
            if (userId) params.append('user_id', userId);
            if (activeOnly) params.append('active_only', activeOnly.toString());

            const url = `${this.baseUrl}${endpoint}${params.toString() ? '?' + params.toString() : ''}`;
            const response = await fetch(url);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            return result.jobs;
        } catch (error) {
            throw new Error(`Failed to get jobs: ${error.message}`);
        }
    }

    /**
     * Get processing statistics
     */
    async getStatistics(jobType = 'excel') {
        try {
            const endpoint = jobType === 'excel'
                ? '/enhanced-excel-parser/statistics'
                : '/enhanced-bulk-parser/statistics';

            const response = await fetch(`${this.baseUrl}${endpoint}`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            return result;
        } catch (error) {
            throw new Error(`Failed to get statistics: ${error.message}`);
        }
    }

    /**
     * Event listener management
     */
    on(event, callback) {
        if (this.listeners[event]) {
            this.listeners[event].push(callback);
        }
    }

    off(event, callback) {
        if (this.listeners[event]) {
            const index = this.listeners[event].indexOf(callback);
            if (index > -1) {
                this.listeners[event].splice(index, 1);
            }
        }
    }

    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in ${event} listener:`, error);
                }
            });
        }
    }

    /**
     * Utility method to format progress for display
     */
    formatProgress(status) {
        const percentage = status.progress_percentage || 0;
        const processed = status.processed_items || 0;
        const total = status.total_items || 0;
        const successful = status.successful_items || 0;
        const failed = status.failed_items || 0;
        const skipped = status.skipped_items || 0;

        const estimatedTime = status.estimated_remaining_time;
        const estimatedTimeStr = estimatedTime
            ? `${Math.ceil(estimatedTime / 60)} min remaining`
            : 'Calculating...';

        return {
            percentage: Math.round(percentage * 100) / 100,
            progressText: `${processed}/${total} items processed`,
            statusText: `${successful} successful, ${failed} failed${skipped > 0 ? `, ${skipped} skipped` : ''}`,
            estimatedTime: estimatedTimeStr,
            currentItem: status.current_item || '',
            isComplete: ['completed', 'failed', 'cancelled'].includes(status.status),
            isSuccess: status.status === 'completed',
            hasErrors: failed > 0 || status.error_messages?.length > 0
        };
    }
}

// Export for both CommonJS and ES modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ResumeProcessingClient;
} else if (typeof window !== 'undefined') {
    window.ResumeProcessingClient = ResumeProcessingClient;
}

/**
 * Usage Examples:
 * 
 * // Initialize client
 * const client = new ResumeProcessingClient('http://localhost:8000', {
 *     pollInterval: 2000,
 *     onProgress: (data) => console.log('Progress:', data),
 *     onComplete: (data) => console.log('Complete:', data),
 *     onError: (data) => console.error('Error:', data)
 * });
 * 
 * // Upload Excel file
 * const excelFile = document.getElementById('excel-file').files[0];
 * try {
 *     const result = await client.uploadExcelFile(excelFile, 'user123', 'john_doe', 'Sheet1');
 *     console.log('Upload started:', result.jobId);
 * } catch (error) {
 *     console.error('Upload failed:', error.message);
 * }
 * 
 * // Upload multiple resumes
 * const resumeFiles = Array.from(document.getElementById('resume-files').files);
 * try {
 *     const result = await client.uploadMultipleResumes(resumeFiles, true);
 *     console.log('Bulk upload started:', result.jobId);
 * } catch (error) {
 *     console.error('Bulk upload failed:', error.message);
 * }
 * 
 * // Listen for events
 * client.on('progress', (data) => {
 *     const progress = client.formatProgress(data.status);
 *     updateProgressBar(progress.percentage);
 *     updateProgressText(progress.progressText);
 * });
 * 
 * client.on('complete', (data) => {
 *     showSuccessMessage('Processing completed successfully!');
 *     displayResults(data.results);
 * });
 * 
 * client.on('error', (data) => {
 *     showErrorMessage(`Processing failed: ${data.error}`);
 * });
 * 
 * // Control job
 * await client.controlJob(jobId, 'excel', 'pause');
 * await client.controlJob(jobId, 'excel', 'resume');
 * await client.controlJob(jobId, 'excel', 'cancel');
 */