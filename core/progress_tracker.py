"""
Progress Tracker for Resume Parsing Operations

This module provides comprehensive tracking and monitoring capabilities for both
Excel and multiple resume parsing operations with real-time progress updates,
error handling, and recovery mechanisms.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from threading import Lock
import pickle

from core.custom_logger import CustomLogger
from core.database import get_database

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("progress_tracker")


class OperationType(Enum):
    """Types of parsing operations."""

    EXCEL_PARSING = "excel_parsing"
    MULTIPLE_RESUME_PARSING = "multiple_resume_parsing"
    BULK_RESUME_PARSING = "bulk_resume_parsing"


class ProcessingStatus(Enum):
    """Status of processing operations."""

    PENDING = "pending"
    INITIALIZING = "initializing"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"


class JobStatus(Enum):
    """Job status enumeration for enhanced APIs."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProcessingError:
    """Represents a processing error."""

    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    item_index: Optional[int] = None
    item_identifier: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class ProgressMetrics:
    """Comprehensive metrics for progress tracking."""

    total_items: int
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.processed_items / self.total_items) * 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.processed_items == 0:
            return 100.0
        return (self.successful_items / self.processed_items) * 100

    @property
    def elapsed_time(self) -> Optional[timedelta]:
        """Calculate elapsed time."""
        if self.start_time is None:
            return None
        end_time = self.end_time or datetime.utcnow()
        return end_time - self.start_time


@dataclass
class ProcessingSession:
    """Represents a processing session with comprehensive tracking."""

    session_id: str
    operation_type: OperationType
    user_id: str
    username: str
    status: ProcessingStatus
    metrics: ProgressMetrics
    errors: List[ProcessingError] = None
    configuration: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.metrics.start_time is None:
            self.metrics.start_time = datetime.utcnow()


class ProgressTracker:
    """
    Advanced progress tracking system for resume parsing operations.

    Provides real-time monitoring, error tracking, recovery mechanisms,
    and comprehensive analytics for both Excel and multiple resume parsing.
    """

    def __init__(self):
        """Initialize the progress tracker."""
        self.active_sessions: Dict[str, ProcessingSession] = {}
        self.completed_sessions: Dict[str, ProcessingSession] = {}
        self.session_lock = Lock()
        self.db_collection = get_database()
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {
            operation_type.value: [] for operation_type in OperationType
        }
        self.error_callbacks: List[Callable[[str, ProcessingError], None]] = []
        self.progress_callbacks: List[Callable[[str, ProgressMetrics], None]] = []

    def create_session(
        self,
        operation_type: OperationType,
        user_id: str,
        username: str,
        total_items: int,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new processing session."""
        session_id = str(uuid.uuid4())
        metrics = ProgressMetrics(total_items=total_items)

        session = ProcessingSession(
            session_id=session_id,
            operation_type=operation_type,
            user_id=user_id,
            username=username,
            status=ProcessingStatus.PENDING,
            metrics=metrics,
            configuration=configuration or {},
        )

        with self.session_lock:
            self.active_sessions[session_id] = session

        logger.info(f"Created session {session_id} for {operation_type.value}")
        return session_id

    def start_session(self, session_id: str) -> bool:
        """Start a processing session."""
        with self.session_lock:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.status = ProcessingStatus.IN_PROGRESS
                session.metrics.start_time = datetime.utcnow()
                session.updated_at = datetime.utcnow()
                logger.info(f"Started session {session_id}")
                return True
        return False

    def update_progress(
        self,
        session_id: str,
        processed_items: Optional[int] = None,
        successful_items: Optional[int] = None,
        failed_items: Optional[int] = None,
        skipped_items: Optional[int] = None,
        current_item: Optional[str] = None,
    ) -> bool:
        """Update progress for a session."""
        with self.session_lock:
            if session_id not in self.active_sessions:
                return False

            session = self.active_sessions[session_id]
            metrics = session.metrics

            if processed_items is not None:
                metrics.processed_items = processed_items
            if successful_items is not None:
                metrics.successful_items = successful_items
            if failed_items is not None:
                metrics.failed_items = failed_items
            if skipped_items is not None:
                metrics.skipped_items = skipped_items

            session.updated_at = datetime.utcnow()

            # Trigger progress callbacks
            for callback in self.progress_callbacks:
                try:
                    callback(session_id, metrics)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")

            return True

    def add_error(
        self,
        session_id: str,
        error_type: str,
        error_message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        item_index: Optional[int] = None,
        item_identifier: Optional[str] = None,
        stack_trace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add an error to a session."""
        with self.session_lock:
            if session_id not in self.active_sessions:
                return False

            session = self.active_sessions[session_id]
            error = ProcessingError(
                timestamp=datetime.utcnow(),
                error_type=error_type,
                error_message=error_message,
                severity=severity,
                item_index=item_index,
                item_identifier=item_identifier,
                stack_trace=stack_trace,
                context=context,
            )

            session.errors.append(error)
            session.updated_at = datetime.utcnow()

            # Trigger error callbacks
            for callback in self.error_callbacks:
                try:
                    callback(session_id, error)
                except Exception as e:
                    logger.error(f"Error callback error: {e}")

            logger.warning(f"Added error to session {session_id}: {error_message}")
            return True

    def complete_session(
        self, session_id: str, status: ProcessingStatus = ProcessingStatus.COMPLETED
    ) -> bool:
        """Complete a processing session."""
        with self.session_lock:
            if session_id not in self.active_sessions:
                return False

            session = self.active_sessions[session_id]
            session.status = status
            session.metrics.end_time = datetime.utcnow()
            session.updated_at = datetime.utcnow()

            # Move to completed sessions
            self.completed_sessions[session_id] = session
            del self.active_sessions[session_id]

            # Record performance data
            self._record_performance(session)

            logger.info(f"Completed session {session_id} with status {status.value}")
            return True

    def pause_session(self, session_id: str) -> bool:
        """Pause a processing session."""
        with self.session_lock:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                if session.status == ProcessingStatus.IN_PROGRESS:
                    session.status = ProcessingStatus.PAUSED
                    session.updated_at = datetime.utcnow()
                    logger.info(f"Paused session {session_id}")
                    return True
        return False

    def resume_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Resume a paused session."""
        with self.session_lock:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                if session.status == ProcessingStatus.PAUSED:
                    session.status = ProcessingStatus.IN_PROGRESS
                    session.updated_at = datetime.utcnow()
                    logger.info(f"Resumed session {session_id}")
                    return self._session_to_dict(session)
        return None

    def cancel_session(self, session_id: str) -> bool:
        """Cancel a processing session."""
        return self.complete_session(session_id, ProcessingStatus.CANCELLED)

    def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a session."""
        with self.session_lock:
            # Check active sessions first
            if session_id in self.active_sessions:
                return self._session_to_dict(self.active_sessions[session_id])

            # Check completed sessions
            if session_id in self.completed_sessions:
                return self._session_to_dict(self.completed_sessions[session_id])

        return None

    def get_active_sessions(
        self, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all active sessions, optionally filtered by user."""
        sessions = []
        with self.session_lock:
            for session in self.active_sessions.values():
                if user_id is None or session.user_id == user_id:
                    sessions.append(self._session_to_dict(session))
        return sessions

    def get_user_sessions(self, user_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get all sessions for a specific user."""
        active = []
        completed = []

        with self.session_lock:
            for session in self.active_sessions.values():
                if session.user_id == user_id:
                    active.append(self._session_to_dict(session))

            for session in self.completed_sessions.values():
                if session.user_id == user_id:
                    completed.append(self._session_to_dict(session))

        return {"active": active, "completed": completed}

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        with self.session_lock:
            active_count = len(self.active_sessions)
            completed_count = len(self.completed_sessions)

            # Calculate success rates by operation type
            success_rates = {}
            for op_type in OperationType:
                completed_sessions = [
                    s
                    for s in self.completed_sessions.values()
                    if s.operation_type == op_type
                    and s.status == ProcessingStatus.COMPLETED
                ]
                total_sessions = [
                    s
                    for s in self.completed_sessions.values()
                    if s.operation_type == op_type
                ]

                if total_sessions:
                    success_rates[op_type.value] = (
                        len(completed_sessions) / len(total_sessions) * 100
                    )
                else:
                    success_rates[op_type.value] = 0.0

            return {
                "active_sessions": active_count,
                "completed_sessions": completed_count,
                "total_sessions": active_count + completed_count,
                "success_rates": success_rates,
                "performance_history_length": {
                    op_type.value: len(self.performance_history[op_type.value])
                    for op_type in OperationType
                },
            }

    def cleanup_old_sessions(self, days: int = 7) -> int:
        """Clean up old completed sessions."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        removed_count = 0

        with self.session_lock:
            sessions_to_remove = [
                session_id
                for session_id, session in self.completed_sessions.items()
                if session.updated_at < cutoff_date
            ]

            for session_id in sessions_to_remove:
                del self.completed_sessions[session_id]
                removed_count += 1

        logger.info(f"Cleaned up {removed_count} old sessions")
        return removed_count

    def _session_to_dict(self, session: ProcessingSession) -> Dict[str, Any]:
        """Convert session to dictionary format."""
        return {
            "session_id": session.session_id,
            "operation_type": session.operation_type.value,
            "user_id": session.user_id,
            "username": session.username,
            "status": session.status.value,
            "metrics": {
                "total_items": session.metrics.total_items,
                "processed_items": session.metrics.processed_items,
                "successful_items": session.metrics.successful_items,
                "failed_items": session.metrics.failed_items,
                "skipped_items": session.metrics.skipped_items,
                "completion_percentage": session.metrics.completion_percentage,
                "success_rate": session.metrics.success_rate,
                "start_time": (
                    session.metrics.start_time.isoformat()
                    if session.metrics.start_time
                    else None
                ),
                "end_time": (
                    session.metrics.end_time.isoformat()
                    if session.metrics.end_time
                    else None
                ),
            },
            "errors": [
                {
                    "timestamp": error.timestamp.isoformat(),
                    "error_type": error.error_type,
                    "error_message": error.error_message,
                    "severity": error.severity.value,
                    "item_index": error.item_index,
                    "item_identifier": error.item_identifier,
                    "context": error.context,
                }
                for error in session.errors
            ],
            "configuration": session.configuration,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        }

    def _record_performance(self, session: ProcessingSession):
        """Record performance data for analytics."""
        performance_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session.session_id,
            "total_items": session.metrics.total_items,
            "processing_time": (
                session.metrics.elapsed_time.total_seconds()
                if session.metrics.elapsed_time
                else 0
            ),
            "success_rate": session.metrics.success_rate,
            "error_count": len(session.errors),
            "status": session.status.value,
        }

        self.performance_history[session.operation_type.value].append(
            performance_record
        )

        # Keep only last 100 records per operation type
        if len(self.performance_history[session.operation_type.value]) > 100:
            self.performance_history[session.operation_type.value] = (
                self.performance_history[session.operation_type.value][-100:]
            )

    # Enhanced methods for the new APIs
    def create_job(
        self,
        job_type: "JobType",
        total_items: int,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new job for tracking (compatibility method for enhanced APIs).
        """
        # Convert JobType to OperationType
        operation_type = OperationType(job_type.value)

        # Create session using existing method
        session_id = self.create_session(
            operation_type=operation_type,
            user_id=user_id or "unknown",
            username=user_id or "unknown",
            total_items=total_items,
            configuration=metadata,
        )

        # Start the session immediately
        self.start_session(session_id)
        return session_id

    def update_job_status(self, job_id: str, status: JobStatus):
        """Update job status (compatibility method for enhanced APIs)."""
        with self.session_lock:
            if job_id in self.active_sessions:
                session = self.active_sessions[job_id]

                # Map JobStatus to ProcessingStatus
                if status == JobStatus.QUEUED:
                    session.status = ProcessingStatus.PENDING
                elif status == JobStatus.PROCESSING:
                    session.status = ProcessingStatus.IN_PROGRESS
                elif status == JobStatus.COMPLETED:
                    session.status = ProcessingStatus.COMPLETED
                elif status == JobStatus.FAILED:
                    session.status = ProcessingStatus.FAILED
                elif status == JobStatus.CANCELLED:
                    session.status = ProcessingStatus.CANCELLED
                elif status == JobStatus.PAUSED:
                    session.status = ProcessingStatus.PAUSED

                session.updated_at = datetime.utcnow()
                if status in [
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                ]:
                    session.metrics.end_time = datetime.utcnow()

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status in the format expected by enhanced APIs."""
        session = self.get_session_details(job_id)
        if not session:
            return None

        # Convert to the format expected by enhanced APIs
        job_status_map = {
            ProcessingStatus.PENDING: "queued",
            ProcessingStatus.INITIALIZING: "queued",
            ProcessingStatus.IN_PROGRESS: "processing",
            ProcessingStatus.PAUSED: "paused",
            ProcessingStatus.COMPLETED: "completed",
            ProcessingStatus.FAILED: "failed",
            ProcessingStatus.CANCELLED: "cancelled",
            ProcessingStatus.RECOVERING: "processing",
        }

        return {
            "job_id": job_id,
            "job_type": session["operation_type"],
            "status": job_status_map.get(
                ProcessingStatus(session["status"]), "unknown"
            ),
            "total_items": session["metrics"]["total_items"],
            "processed_items": session["metrics"]["processed_items"],
            "successful_items": session["metrics"]["successful_items"],
            "failed_items": session["metrics"]["failed_items"],
            "skipped_items": session["metrics"]["skipped_items"],
            "start_time": session["created_at"],
            "end_time": session["metrics"]["end_time"],
            "last_update": session["updated_at"],
            "current_item": session.get("current_item"),
            "error_messages": [
                {
                    "timestamp": error["timestamp"],
                    "message": error["error_message"],
                    "item": error.get("item_identifier"),
                }
                for error in session.get("errors", [])
            ],
            "metadata": session.get("configuration", {}),
            "user_id": session.get("user_id"),
            "session_id": session.get("session_id"),
            "progress_percentage": session["metrics"]["completion_percentage"],
            "elapsed_time": (
                (
                    datetime.utcnow() - datetime.fromisoformat(session["created_at"])
                ).total_seconds()
                if session.get("created_at")
                else 0
            ),
            "estimated_remaining_time": self._estimate_remaining_time(session),
        }

    def _estimate_remaining_time(self, session: Dict[str, Any]) -> Optional[float]:
        """Estimate remaining time for a session."""
        metrics = session["metrics"]
        if metrics["processed_items"] == 0 or session["status"] != "in_progress":
            return None

        elapsed_seconds = (
            datetime.utcnow() - datetime.fromisoformat(session["created_at"])
        ).total_seconds()
        rate = (
            metrics["processed_items"] / elapsed_seconds if elapsed_seconds > 0 else 0
        )
        remaining_items = metrics["total_items"] - metrics["processed_items"]

        return remaining_items / rate if rate > 0 else None

    def get_all_jobs(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all jobs, optionally filtered by user ID."""
        jobs = []
        with self.session_lock:
            for session in self.active_sessions.values():
                if user_id is None or session.user_id == user_id:
                    job_status = self.get_job_status(session.session_id)
                    if job_status:
                        jobs.append(job_status)
        return jobs

    def get_active_jobs(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active jobs, optionally filtered by user ID."""
        active_statuses = [
            ProcessingStatus.PENDING,
            ProcessingStatus.INITIALIZING,
            ProcessingStatus.IN_PROGRESS,
            ProcessingStatus.PAUSED,
            ProcessingStatus.RECOVERING,
        ]

        jobs = []
        with self.session_lock:
            for session in self.active_sessions.values():
                if session.status in active_statuses and (
                    user_id is None or session.user_id == user_id
                ):
                    job_status = self.get_job_status(session.session_id)
                    if job_status:
                        jobs.append(job_status)
        return jobs

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        return self.cancel_session(job_id)

    def pause_job(self, job_id: str) -> bool:
        """Pause a job."""
        return self.pause_session(job_id)

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        resume_data = self.resume_session(job_id)
        return resume_data is not None


# Alias for compatibility with enhanced APIs
JobType = OperationType

# Global instance for the progress tracker
progress_tracker = ProgressTracker()
