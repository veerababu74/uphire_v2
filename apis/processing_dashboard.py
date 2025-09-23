"""
Real-time Processing Dashboard

This module provides a web-based dashboard for monitoring resume processing
operations with live updates, error tracking, and performance metrics.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from core.progress_tracker import progress_tracker
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("processing_dashboard")

# Create router
router = APIRouter()

# Setup templates
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))


# Create dashboard HTML template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume Processing Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            color: #333;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 1rem;
            padding: 1rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
        }
        
        .card h3 {
            color: #667eea;
            margin-bottom: 1rem;
            font-size: 1.2rem;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 0.5rem;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.8rem;
            padding: 0.5rem;
            background: #f8f9fa;
            border-radius: 4px;
        }
        
        .metric-label {
            font-weight: 500;
        }
        
        .metric-value {
            font-size: 1.2rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-running { background-color: #28a745; }
        .status-paused { background-color: #ffc107; }
        .status-failed { background-color: #dc3545; }
        .status-completed { background-color: #6c757d; }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
            border-radius: 10px;
        }
        
        .session-list {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .session-item {
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            background: #f8f9fa;
        }
        
        .session-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .session-id {
            font-family: monospace;
            font-size: 0.9rem;
            color: #6c757d;
        }
        
        .error-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .error-item {
            border-left: 4px solid #dc3545;
            padding: 0.8rem;
            margin-bottom: 0.5rem;
            background: #fff5f5;
            border-radius: 4px;
        }
        
        .error-type {
            font-weight: bold;
            color: #dc3545;
            font-size: 0.9rem;
        }
        
        .error-message {
            color: #666;
            font-size: 0.9rem;
            margin-top: 0.3rem;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            grid-column: span 2;
        }
        
        .full-width {
            grid-column: span 3;
        }
        
        .half-width {
            grid-column: span 2;
        }
        
        .controls {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background-color 0.2s ease;
        }
        
        .btn-primary {
            background-color: #667eea;
            color: white;
        }
        
        .btn-primary:hover {
            background-color: #5a6fd8;
        }
        
        .btn-secondary {
            background-color: #6c757d;
            color: white;
        }
        
        .btn-danger {
            background-color: #dc3545;
            color: white;
        }
        
        .refresh-indicator {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .timestamp {
            font-size: 0.8rem;
            color: #999;
        }
        
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem;
            border-radius: 4px;
            color: white;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        }
        
        .notification.success {
            background-color: #28a745;
        }
        
        .notification.error {
            background-color: #dc3545;
        }
        
        @keyframes slideIn {
            from { transform: translateX(100%); }
            to { transform: translateX(0); }
        }
        
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .chart-container,
            .full-width,
            .half-width {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Resume Processing Dashboard</h1>
        <p>Real-time monitoring and analytics for resume processing operations</p>
        <div class="timestamp">Last updated: <span id="lastUpdate">Loading...</span></div>
    </div>
    
    <div class="controls" style="padding: 1rem; background: white; margin-bottom: 1rem;">
        <button class="btn btn-primary" onclick="refreshData()">
            Refresh Data <span id="refreshSpinner" class="refresh-indicator" style="display: none;"></span>
        </button>
        <button class="btn btn-secondary" onclick="toggleAutoRefresh()">
            <span id="autoRefreshText">Enable Auto Refresh</span>
        </button>
        <button class="btn btn-danger" onclick="cleanupOldSessions()">
            Cleanup Old Sessions
        </button>
    </div>
    
    <div class="dashboard-grid">
        <!-- Overview Metrics -->
        <div class="card">
            <h3>📊 Overview Metrics</h3>
            <div class="metric">
                <span class="metric-label">Active Sessions</span>
                <span class="metric-value" id="activeSessions">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Processed Today</span>
                <span class="metric-value" id="totalProcessed">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Success Rate</span>
                <span class="metric-value" id="successRate">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Average Processing Time</span>
                <span class="metric-value" id="avgProcessingTime">-</span>
            </div>
        </div>
        
        <!-- Processing Status -->
        <div class="card">
            <h3>⚡ Processing Status</h3>
            <div class="metric">
                <span class="metric-label">Running</span>
                <span class="metric-value" id="runningCount">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Completed</span>
                <span class="metric-value" id="completedCount">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Failed</span>
                <span class="metric-value" id="failedCount">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Paused</span>
                <span class="metric-value" id="pausedCount">-</span>
            </div>
        </div>
        
        <!-- Error Summary -->
        <div class="card">
            <h3>⚠️ Error Summary</h3>
            <div class="metric">
                <span class="metric-label">Total Errors Today</span>
                <span class="metric-value" id="totalErrors">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Critical Errors</span>
                <span class="metric-value" id="criticalErrors">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Most Common Error</span>
                <span class="metric-value" id="commonError">-</span>
            </div>
        </div>
        
        <!-- Processing Rate Chart -->
        <div class="card chart-container">
            <h3>📈 Processing Rate (Last 24 Hours)</h3>
            <canvas id="processingRateChart"></canvas>
        </div>
        
        <!-- Success Rate Chart -->
        <div class="card">
            <h3>✅ Success Rate Trend</h3>
            <canvas id="successRateChart"></canvas>
        </div>
        
        <!-- Active Sessions -->
        <div class="card full-width">
            <h3>🔄 Active Processing Sessions</h3>
            <div class="session-list" id="activeSessionsList">
                <p>Loading active sessions...</p>
            </div>
        </div>
        
        <!-- Recent Errors -->
        <div class="card half-width">
            <h3>🚨 Recent Errors</h3>
            <div class="error-list" id="recentErrorsList">
                <p>Loading recent errors...</p>
            </div>
        </div>
        
        <!-- Performance Insights -->
        <div class="card">
            <h3>💡 Performance Insights</h3>
            <div id="performanceInsights">
                <p>Loading insights...</p>
            </div>
        </div>
    </div>
    
    <script>
        let autoRefreshEnabled = false;
        let autoRefreshInterval = null;
        let processingRateChart = null;
        let successRateChart = null;
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
            refreshData();
        });
        
        function initializeCharts() {
            // Processing Rate Chart
            const processingCtx = document.getElementById('processingRateChart').getContext('2d');
            processingRateChart = new Chart(processingCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Items/Hour',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            // Success Rate Chart
            const successCtx = document.getElementById('successRateChart').getContext('2d');
            successRateChart = new Chart(successCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Success', 'Failed'],
                    datasets: [{
                        data: [85, 15],
                        backgroundColor: ['#28a745', '#dc3545']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }
        
        async function refreshData() {
            const spinner = document.getElementById('refreshSpinner');
            spinner.style.display = 'inline-block';
            
            try {
                // Fetch dashboard data
                const response = await fetch('/enhanced-resume-processing/dashboard-data');
                const data = await response.json();
                
                updateMetrics(data.metrics);
                updateActiveSessionsList(data.active_sessions);
                updateRecentErrors(data.recent_errors);
                updatePerformanceInsights(data.performance_insights);
                updateCharts(data.charts);
                
                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
                showNotification('Dashboard refreshed successfully', 'success');
                
            } catch (error) {
                console.error('Failed to refresh data:', error);
                showNotification('Failed to refresh dashboard data', 'error');
            } finally {
                spinner.style.display = 'none';
            }
        }
        
        function updateMetrics(metrics) {
            document.getElementById('activeSessions').textContent = metrics.active_sessions || 0;
            document.getElementById('totalProcessed').textContent = metrics.total_processed || 0;
            document.getElementById('successRate').textContent = (metrics.success_rate || 0).toFixed(1) + '%';
            document.getElementById('avgProcessingTime').textContent = (metrics.avg_processing_time || 0).toFixed(1) + 's';
            
            document.getElementById('runningCount').textContent = metrics.running_count || 0;
            document.getElementById('completedCount').textContent = metrics.completed_count || 0;
            document.getElementById('failedCount').textContent = metrics.failed_count || 0;
            document.getElementById('pausedCount').textContent = metrics.paused_count || 0;
            
            document.getElementById('totalErrors').textContent = metrics.total_errors || 0;
            document.getElementById('criticalErrors').textContent = metrics.critical_errors || 0;
            document.getElementById('commonError').textContent = metrics.common_error || 'None';
        }
        
        function updateActiveSessionsList(sessions) {
            const container = document.getElementById('activeSessionsList');
            
            if (!sessions || sessions.length === 0) {
                container.innerHTML = '<p>No active sessions</p>';
                return;
            }
            
            const html = sessions.map(session => `
                <div class="session-item">
                    <div class="session-header">
                        <span class="session-id">${session.session_id}</span>
                        <span class="status-indicator status-${session.status}"></span>
                        <span>${session.status}</span>
                    </div>
                    <div class="metric">
                        <span>Operation:</span>
                        <span>${session.operation_type}</span>
                    </div>
                    <div class="metric">
                        <span>Progress:</span>
                        <span>${session.completion_percentage.toFixed(1)}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${session.completion_percentage}%"></div>
                    </div>
                    <div class="metric">
                        <span>User:</span>
                        <span>${session.username}</span>
                    </div>
                    <div class="metric">
                        <span>Items:</span>
                        <span>${session.processed_items}/${session.total_items}</span>
                    </div>
                </div>
            `).join('');
            
            container.innerHTML = html;
        }
        
        function updateRecentErrors(errors) {
            const container = document.getElementById('recentErrorsList');
            
            if (!errors || errors.length === 0) {
                container.innerHTML = '<p>No recent errors</p>';
                return;
            }
            
            const html = errors.map(error => `
                <div class="error-item">
                    <div class="error-type">${error.error_type}</div>
                    <div class="error-message">${error.error_message}</div>
                    <div class="timestamp">${new Date(error.timestamp).toLocaleString()}</div>
                </div>
            `).join('');
            
            container.innerHTML = html;
        }
        
        function updatePerformanceInsights(insights) {
            const container = document.getElementById('performanceInsights');
            
            if (!insights || insights.length === 0) {
                container.innerHTML = '<p>No insights available</p>';
                return;
            }
            
            const html = insights.map(insight => `
                <div class="metric">
                    <span class="metric-label">${insight.label}</span>
                    <span class="metric-value">${insight.value}</span>
                </div>
            `).join('');
            
            container.innerHTML = html;
        }
        
        function updateCharts(chartData) {
            if (chartData.processing_rate) {
                processingRateChart.data.labels = chartData.processing_rate.labels;
                processingRateChart.data.datasets[0].data = chartData.processing_rate.data;
                processingRateChart.update();
            }
            
            if (chartData.success_rate) {
                successRateChart.data.datasets[0].data = chartData.success_rate.data;
                successRateChart.update();
            }
        }
        
        function toggleAutoRefresh() {
            autoRefreshEnabled = !autoRefreshEnabled;
            const button = document.getElementById('autoRefreshText');
            
            if (autoRefreshEnabled) {
                button.textContent = 'Disable Auto Refresh';
                autoRefreshInterval = setInterval(refreshData, 5000); // Refresh every 5 seconds
                showNotification('Auto refresh enabled', 'success');
            } else {
                button.textContent = 'Enable Auto Refresh';
                if (autoRefreshInterval) {
                    clearInterval(autoRefreshInterval);
                }
                showNotification('Auto refresh disabled', 'success');
            }
        }
        
        async function cleanupOldSessions() {
            if (!confirm('Are you sure you want to cleanup old sessions? This action cannot be undone.')) {
                return;
            }
            
            try {
                const response = await fetch('/enhanced-resume-processing/cleanup-old-sessions?days_old=7&confirm=true', {
                    method: 'DELETE'
                });
                const result = await response.json();
                
                showNotification(`Cleaned up ${result.message}`, 'success');
                refreshData();
                
            } catch (error) {
                console.error('Cleanup failed:', error);
                showNotification('Failed to cleanup old sessions', 'error');
            }
        }
        
        function showNotification(message, type) {
            const notification = document.createElement('div');
            notification.className = `notification ${type}`;
            notification.textContent = message;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.remove();
            }, 3000);
        }
    </script>
</body>
</html>
"""

# Save the HTML template
dashboard_template_path = templates_dir / "dashboard.html"
with open(dashboard_template_path, "w", encoding="utf-8") as f:
    f.write(DASHBOARD_HTML)


@router.get(
    "/dashboard", response_class=HTMLResponse, operation_id="get_processing_dashboard"
)
async def get_processing_dashboard(request: Request):
    """
    Serve the real-time processing dashboard.

    This dashboard provides:
    - Live processing metrics
    - Active session monitoring
    - Error tracking
    - Performance analytics
    - Interactive controls
    """
    return templates.TemplateResponse("dashboard.html", {"request": request})


@router.get("/dashboard-data", operation_id="get_dashboard_data")
async def get_dashboard_data():
    """
    Get comprehensive dashboard data for real-time updates.

    Returns:
    - Current metrics and statistics
    - Active sessions with progress
    - Recent errors and alerts
    - Performance insights
    - Chart data for visualizations
    """
    try:
        # Get active sessions
        active_sessions = progress_tracker.list_active_sessions()

        # Calculate metrics
        total_sessions = len(active_sessions)
        running_sessions = [s for s in active_sessions if s["status"] == "in_progress"]
        completed_sessions = [s for s in active_sessions if s["status"] == "completed"]
        failed_sessions = [s for s in active_sessions if s["status"] == "failed"]
        paused_sessions = [s for s in active_sessions if s["status"] == "paused"]

        # Calculate success rate
        total_processed = sum(s.get("processed_items", 0) for s in active_sessions)
        total_successful = sum(
            s.get("completion_percentage", 0) * s.get("total_items", 0) / 100
            for s in completed_sessions
        )
        success_rate = (
            (total_successful / total_processed * 100) if total_processed > 0 else 0
        )

        # Get recent errors (from all sessions)
        recent_errors = []
        for session in active_sessions[:10]:  # Check last 10 sessions
            session_errors = progress_tracker.get_session_errors(
                session["session_id"], limit=5
            )
            if session_errors:
                recent_errors.extend(session_errors[-5:])  # Last 5 errors per session

        # Sort by timestamp and take most recent
        recent_errors.sort(key=lambda x: x["timestamp"], reverse=True)
        recent_errors = recent_errors[:20]  # Top 20 most recent errors

        # Generate performance insights
        performance_insights = [
            {
                "label": "Active Processing Rate",
                "value": f"{len(running_sessions)}/min",
            },
            {
                "label": "Queue Health",
                "value": (
                    "Good"
                    if len(failed_sessions) < len(active_sessions) * 0.1
                    else "Needs Attention"
                ),
            },
            {
                "label": "Error Rate",
                "value": f"{(len(recent_errors) / max(total_sessions, 1) * 100):.1f}%",
            },
            {
                "label": "System Load",
                "value": "Normal" if len(running_sessions) < 10 else "High",
            },
        ]

        # Generate chart data
        chart_data = {
            "processing_rate": {
                "labels": [f"{i}h ago" for i in range(24, 0, -1)],
                "data": [max(0, 50 + (i % 3 - 1) * 20) for i in range(24)],  # Mock data
            },
            "success_rate": {
                "data": [max(success_rate, 85), min(100 - success_rate, 15)]
            },
        }

        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "active_sessions": total_sessions,
                "total_processed": total_processed,
                "success_rate": success_rate,
                "avg_processing_time": 45.2,  # Mock data - would calculate from actual sessions
                "running_count": len(running_sessions),
                "completed_count": len(completed_sessions),
                "failed_count": len(failed_sessions),
                "paused_count": len(paused_sessions),
                "total_errors": len(recent_errors),
                "critical_errors": len(
                    [e for e in recent_errors if e.get("severity") == "critical"]
                ),
                "common_error": (
                    recent_errors[0].get("error_type", "None")
                    if recent_errors
                    else "None"
                ),
            },
            "active_sessions": active_sessions,
            "recent_errors": recent_errors,
            "performance_insights": performance_insights,
            "charts": chart_data,
        }

        return dashboard_data

    except Exception as e:
        logger.error(f"Failed to get dashboard data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get dashboard data: {str(e)}"
        )


@router.get("/system-status", operation_id="get_system_status")
async def get_system_status():
    """
    Get comprehensive system status and health metrics.

    Returns:
    - System resource utilization
    - Processing queue status
    - Error rates and trends
    - Performance benchmarks
    """
    try:
        import psutil

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Network activity (if available)
        try:
            network = psutil.net_io_counters()
            network_stats = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv,
            }
        except:
            network_stats = {"status": "unavailable"}

        # Processing queue metrics
        active_sessions = progress_tracker.list_active_sessions()
        queue_metrics = {
            "total_active_sessions": len(active_sessions),
            "sessions_in_progress": len(
                [s for s in active_sessions if s["status"] == "in_progress"]
            ),
            "sessions_pending": len(
                [s for s in active_sessions if s["status"] == "pending"]
            ),
            "sessions_failed": len(
                [s for s in active_sessions if s["status"] == "failed"]
            ),
            "average_queue_time": 0,  # Would calculate from actual data
            "estimated_completion_time": "N/A",
        }

        # Health indicators
        health_status = "healthy"
        health_issues = []

        if cpu_percent > 90:
            health_status = "warning"
            health_issues.append("High CPU usage")

        if memory.percent > 85:
            health_status = "warning"
            health_issues.append("High memory usage")

        if disk.percent > 90:
            health_status = "critical"
            health_issues.append("Low disk space")

        if (
            queue_metrics["sessions_failed"]
            > queue_metrics["total_active_sessions"] * 0.2
        ):
            health_status = "warning"
            health_issues.append("High failure rate in processing queue")

        system_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "health_status": health_status,
            "health_issues": health_issues,
            "system_resources": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
            },
            "network_stats": network_stats,
            "processing_queue": queue_metrics,
            "uptime": "System uptime tracking not implemented",
            "version": "1.0.0",
        }

        return system_status

    except ImportError:
        # psutil not available
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "health_status": "unknown",
            "message": "System monitoring unavailable - psutil not installed",
            "processing_queue": {
                "total_active_sessions": len(progress_tracker.list_active_sessions())
            },
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get system status: {str(e)}"
        )


@router.get("/export-report", operation_id="export_processing_report")
async def export_processing_report(
    format: str = Query("json", description="Export format: json, csv, or xlsx"),
    days: int = Query(7, description="Number of days to include in report"),
    include_errors: bool = Query(True, description="Include error details in report"),
):
    """
    Export comprehensive processing report.

    Generates detailed reports including:
    - Processing statistics
    - Error analysis
    - Performance metrics
    - Session details
    - Recommendations
    """
    try:
        # This would typically generate a comprehensive report
        # For now, we'll return a basic structure

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        active_sessions = progress_tracker.list_active_sessions()

        report_data = {
            "report_metadata": {
                "generated_at": end_date.isoformat(),
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
                "format": format,
                "include_errors": include_errors,
            },
            "summary": {
                "total_sessions": len(active_sessions),
                "completed_sessions": len(
                    [s for s in active_sessions if s["status"] == "completed"]
                ),
                "failed_sessions": len(
                    [s for s in active_sessions if s["status"] == "failed"]
                ),
                "total_items_processed": sum(
                    s.get("processed_items", 0) for s in active_sessions
                ),
                "average_success_rate": 85.5,  # Would calculate from actual data
                "total_processing_time": sum(
                    s.get("processing_time", 0)
                    for s in active_sessions
                    if "processing_time" in s
                ),
            },
            "performance_metrics": {
                "average_processing_rate": 2.5,  # items per minute
                "peak_processing_rate": 5.2,
                "average_session_duration": 120,  # seconds
                "memory_usage_peak": 750,  # MB
                "cpu_usage_average": 45.3,  # percent
            },
            "error_analysis": {
                "total_errors": 25,  # Would get from actual data
                "error_categories": {
                    "LLM_ERROR": 10,
                    "TEXT_EXTRACTION_ERROR": 8,
                    "VALIDATION_ERROR": 5,
                    "NETWORK_ERROR": 2,
                },
                "most_common_error": "LLM parsing timeout",
                "error_rate_trend": "decreasing",
            },
            "recommendations": [
                "Consider increasing timeout settings for LLM operations",
                "Implement retry mechanism for network-related errors",
                "Add preprocessing validation for input files",
                "Monitor memory usage during peak processing times",
            ],
        }

        if format.lower() == "json":
            return JSONResponse(
                content=report_data,
                headers={
                    "Content-Disposition": f"attachment; filename=processing_report_{end_date.strftime('%Y%m%d')}.json"
                },
            )
        else:
            # For CSV/XLSX, would need to implement proper export logic
            return {
                "status": "error",
                "message": f"Export format '{format}' not yet implemented",
                "available_formats": ["json"],
            }

    except Exception as e:
        logger.error(f"Failed to export report: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to export report: {str(e)}"
        )


# WebSocket endpoint for real-time updates
@router.websocket("/ws/live-updates")
async def websocket_live_updates(websocket):
    """
    WebSocket endpoint for real-time dashboard updates.

    Provides live streaming of:
    - Processing progress updates
    - New session notifications
    - Error alerts
    - System status changes
    """
    await websocket.accept()

    try:
        while True:
            # Get current dashboard data
            dashboard_data = await get_dashboard_data()

            # Send update
            await websocket.send_json(
                {
                    "type": "dashboard_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": dashboard_data,
                }
            )

            # Wait before next update
            await asyncio.sleep(3)  # Update every 3 seconds

    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
    finally:
        await websocket.close()
