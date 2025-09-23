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
from core.database import AsyncDatabaseManager

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
    """Metrics for tracking progress."""

    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    duplicate_items: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None
    processing_rate: float = 0.0  # items per second

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.processed_items == 0:
            return 0.0
        return (self.successful_items / self.processed_items) * 100

    @property
    def elapsed_time(self) -> Optional[timedelta]:
        """Calculate elapsed time."""
        if not self.start_time:
            return None
        end_time = self.end_time or datetime.utcnow()
        return end_time - self.start_time


@dataclass
class ProcessingSession:
    """Represents a complete processing session."""

    session_id: str
    operation_type: OperationType
    status: ProcessingStatus
    user_id: str
    username: str
    file_name: Optional[str] = None
    metrics: ProgressMetrics = None
    errors: List[ProcessingError] = None
    checkpoints: List[Dict[str, Any]] = None
    resume_data: Optional[Dict[str, Any]] = None
    configuration: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ProgressMetrics()
        if self.errors is None:
            self.errors = []
        if self.checkpoints is None:
            self.checkpoints = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


class ProgressTracker:
    """
    Comprehensive progress tracking system for resume parsing operations.

    Features:
    - Real-time progress monitoring
    - Error tracking and recovery
    - Checkpoint system for resuming operations
    - Performance metrics
    - Session persistence
    """

    def __init__(self, storage_path: str = "data/progress_sessions"):
        """
        Initialize the progress tracker.

        Args:
            storage_path: Path to store session data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory session storage
        self.active_sessions: Dict[str, ProcessingSession] = {}
        self.session_lock = Lock()

        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}

        # Recovery settings
        self.auto_recovery_enabled = True
        self.max_recovery_attempts = 3
        self.recovery_delay = 5  # seconds

        # Checkpoint settings
        self.checkpoint_interval = 100  # items
        self.auto_checkpoint_enabled = True

        logger.info(
            f"Progress Tracker initialized with storage path: {self.storage_path}"
        )

    def create_session(
        self,
        operation_type: OperationType,
        user_id: str,
        username: str,
        file_name: Optional[str] = None,
        total_items: int = 0,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new processing session.

        Args:
            operation_type: Type of operation
            user_id: User ID
            username: Username
            file_name: Name of file being processed
            total_items: Total number of items to process
            configuration: Operation configuration

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())

        session = ProcessingSession(
            session_id=session_id,
            operation_type=operation_type,
            status=ProcessingStatus.PENDING,
            user_id=user_id,
            username=username,
            file_name=file_name,
            configuration=configuration or {},
        )

        session.metrics.total_items = total_items

        with self.session_lock:
            self.active_sessions[session_id] = session

        # Persist session
        self._save_session(session)

        logger.info(f"Created new session {session_id} for {operation_type.value}")
        return session_id

    def start_session(self, session_id: str) -> bool:
        """
        Start a processing session.

        Args:
            session_id: Session ID

        Returns:
            True if started successfully
        """
        with self.session_lock:
            session = self.active_sessions.get(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return False

            session.status = ProcessingStatus.IN_PROGRESS
            session.metrics.start_time = datetime.utcnow()
            session.updated_at = datetime.utcnow()

        self._save_session(session)
        logger.info(f"Started session {session_id}")
        return True

    def update_progress(
        self,
        session_id: str,
        processed_count: int = 1,
        successful_count: int = 0,
        failed_count: int = 0,
        skipped_count: int = 0,
        duplicate_count: int = 0,
        item_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update session progress.

        Args:
            session_id: Session ID
            processed_count: Number of items processed in this update
            successful_count: Number of successful items
            failed_count: Number of failed items
            skipped_count: Number of skipped items
            duplicate_count: Number of duplicate items
            item_data: Additional data about the processed item

        Returns:
            True if updated successfully
        """
        with self.session_lock:
            session = self.active_sessions.get(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return False

            # Update metrics
            session.metrics.processed_items += processed_count
            session.metrics.successful_items += successful_count
            session.metrics.failed_items += failed_count
            session.metrics.skipped_items += skipped_count
            session.metrics.duplicate_items += duplicate_count
            session.updated_at = datetime.utcnow()

            # Calculate processing rate
            if session.metrics.start_time:
                elapsed = (
                    datetime.utcnow() - session.metrics.start_time
                ).total_seconds()
                if elapsed > 0:
                    session.metrics.processing_rate = (
                        session.metrics.processed_items / elapsed
                    )

                    # Estimate completion time
                    remaining_items = (
                        session.metrics.total_items - session.metrics.processed_items
                    )
                    if session.metrics.processing_rate > 0:
                        remaining_seconds = (
                            remaining_items / session.metrics.processing_rate
                        )
                        session.metrics.estimated_completion_time = (
                            datetime.utcnow() + timedelta(seconds=remaining_seconds)
                        )

            # Auto-checkpoint if enabled
            if (
                self.auto_checkpoint_enabled
                and session.metrics.processed_items % self.checkpoint_interval == 0
            ):
                self._create_checkpoint(session, item_data)

        # Persist session periodically
        if session.metrics.processed_items % 50 == 0:  # Save every 50 items
            self._save_session(session)

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
        """
        Add an error to the session.

        Args:
            session_id: Session ID
            error_type: Type of error
            error_message: Error message
            severity: Error severity
            item_index: Index of the item that caused the error
            item_identifier: Identifier of the item
            stack_trace: Stack trace
            context: Additional context

        Returns:
            True if added successfully
        """
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

        with self.session_lock:
            session = self.active_sessions.get(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return False

            session.errors.append(error)
            session.updated_at = datetime.utcnow()

            # Check if session should be paused due to critical errors
            critical_errors = [
                e for e in session.errors if e.severity == ErrorSeverity.CRITICAL
            ]
            if len(critical_errors) >= 5:  # Pause after 5 critical errors
                session.status = ProcessingStatus.PAUSED
                logger.warning(f"Session {session_id} paused due to critical errors")

        logger.error(
            f"Added {severity.value} error to session {session_id}: {error_message}"
        )
        return True

    def complete_session(
        self, session_id: str, final_summary: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark a session as completed.

        Args:
            session_id: Session ID
            final_summary: Final processing summary

        Returns:
            True if completed successfully
        """
        with self.session_lock:
            session = self.active_sessions.get(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return False

            session.status = ProcessingStatus.COMPLETED
            session.metrics.end_time = datetime.utcnow()
            session.updated_at = datetime.utcnow()

            if final_summary:
                session.resume_data = final_summary

        # Save final session state
        self._save_session(session)

        # Record performance metrics
        self._record_performance(session)

        logger.info(f"Completed session {session_id}")
        return True

    def fail_session(self, session_id: str, failure_reason: str) -> bool:
        """
        Mark a session as failed.

        Args:
            session_id: Session ID
            failure_reason: Reason for failure

        Returns:
            True if marked as failed successfully
        """
        with self.session_lock:
            session = self.active_sessions.get(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return False

            session.status = ProcessingStatus.FAILED
            session.metrics.end_time = datetime.utcnow()
            session.updated_at = datetime.utcnow()

            # Add failure as critical error
            self.add_error(
                session_id=session_id,
                error_type="SESSION_FAILURE",
                error_message=failure_reason,
                severity=ErrorSeverity.CRITICAL,
                context={"failure_reason": failure_reason},
            )

        self._save_session(session)
        logger.error(f"Failed session {session_id}: {failure_reason}")
        return True

    def pause_session(self, session_id: str, reason: str = "Manual pause") -> bool:
        """
        Pause a session.

        Args:
            session_id: Session ID
            reason: Reason for pausing

        Returns:
            True if paused successfully
        """
        with self.session_lock:
            session = self.active_sessions.get(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return False

            session.status = ProcessingStatus.PAUSED
            session.updated_at = datetime.utcnow()

            # Create checkpoint before pausing
            self._create_checkpoint(session, {"pause_reason": reason})

        self._save_session(session)
        logger.info(f"Paused session {session_id}: {reason}")
        return True

    def resume_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Resume a paused or failed session.

        Args:
            session_id: Session ID

        Returns:
            Resume data with last checkpoint information
        """
        with self.session_lock:
            session = self.active_sessions.get(session_id)
            if not session:
                # Try to load from storage
                session = self._load_session(session_id)
                if session:
                    self.active_sessions[session_id] = session
                else:
                    logger.error(f"Session {session_id} not found")
                    return None

            if session.status not in [ProcessingStatus.PAUSED, ProcessingStatus.FAILED]:
                logger.warning(
                    f"Cannot resume session {session_id} with status {session.status.value}"
                )
                return None

            session.status = ProcessingStatus.RECOVERING
            session.updated_at = datetime.utcnow()

            # Get last checkpoint
            resume_data = {
                "session_id": session_id,
                "last_processed_index": session.metrics.processed_items,
                "successful_count": session.metrics.successful_items,
                "failed_count": session.metrics.failed_items,
                "total_items": session.metrics.total_items,
                "last_checkpoint": (
                    session.checkpoints[-1] if session.checkpoints else None
                ),
                "configuration": session.configuration,
            }

        logger.info(
            f"Resuming session {session_id} from item {resume_data['last_processed_index']}"
        )
        return resume_data

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current session status and metrics.

        Args:
            session_id: Session ID

        Returns:
            Session status information
        """
        with self.session_lock:
            session = self.active_sessions.get(session_id)
            if not session:
                # Try to load from storage
                session = self._load_session(session_id)
                if not session:
                    return None

            status_info = {
                "session_id": session.session_id,
                "operation_type": session.operation_type.value,
                "status": session.status.value,
                "user_id": session.user_id,
                "username": session.username,
                "file_name": session.file_name,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metrics": {
                    "total_items": session.metrics.total_items,
                    "processed_items": session.metrics.processed_items,
                    "successful_items": session.metrics.successful_items,
                    "failed_items": session.metrics.failed_items,
                    "skipped_items": session.metrics.skipped_items,
                    "duplicate_items": session.metrics.duplicate_items,
                    "completion_percentage": session.metrics.completion_percentage,
                    "success_rate": session.metrics.success_rate,
                    "processing_rate": session.metrics.processing_rate,
                    "elapsed_time": (
                        session.metrics.elapsed_time.total_seconds()
                        if session.metrics.elapsed_time
                        else 0
                    ),
                    "estimated_completion_time": (
                        session.metrics.estimated_completion_time.isoformat()
                        if session.metrics.estimated_completion_time
                        else None
                    ),
                },
                "error_summary": {
                    "total_errors": len(session.errors),
                    "critical_errors": len(
                        [
                            e
                            for e in session.errors
                            if e.severity == ErrorSeverity.CRITICAL
                        ]
                    ),
                    "high_errors": len(
                        [e for e in session.errors if e.severity == ErrorSeverity.HIGH]
                    ),
                    "medium_errors": len(
                        [
                            e
                            for e in session.errors
                            if e.severity == ErrorSeverity.MEDIUM
                        ]
                    ),
                    "low_errors": len(
                        [e for e in session.errors if e.severity == ErrorSeverity.LOW]
                    ),
                },
                "last_checkpoint": (
                    session.checkpoints[-1]["timestamp"].isoformat()
                    if session.checkpoints
                    else None
                ),
            }

        return status_info

    def get_session_errors(
        self, session_id: str, limit: int = 100
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get session errors.

        Args:
            session_id: Session ID
            limit: Maximum number of errors to return

        Returns:
            List of error information
        """
        with self.session_lock:
            session = self.active_sessions.get(session_id)
            if not session:
                session = self._load_session(session_id)
                if not session:
                    return None

            errors = session.errors[-limit:] if limit > 0 else session.errors
            return [
                {
                    "timestamp": error.timestamp.isoformat(),
                    "error_type": error.error_type,
                    "error_message": error.error_message,
                    "severity": error.severity.value,
                    "item_index": error.item_index,
                    "item_identifier": error.item_identifier,
                    "context": error.context,
                }
                for error in errors
            ]

    def list_active_sessions(
        self, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List active sessions.

        Args:
            user_id: Filter by user ID

        Returns:
            List of active session summaries
        """
        with self.session_lock:
            sessions = []
            for session in self.active_sessions.values():
                if user_id and session.user_id != user_id:
                    continue

                sessions.append(
                    {
                        "session_id": session.session_id,
                        "operation_type": session.operation_type.value,
                        "status": session.status.value,
                        "user_id": session.user_id,
                        "username": session.username,
                        "file_name": session.file_name,
                        "created_at": session.created_at.isoformat(),
                        "completion_percentage": session.metrics.completion_percentage,
                        "total_items": session.metrics.total_items,
                        "processed_items": session.metrics.processed_items,
                    }
                )

        return sessions

    def cleanup_old_sessions(self, days_old: int = 7) -> int:
        """
        Cleanup old completed sessions.

        Args:
            days_old: Age threshold in days

        Returns:
            Number of sessions cleaned up
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        cleaned_count = 0

        # Clean up from memory
        with self.session_lock:
            to_remove = []
            for session_id, session in self.active_sessions.items():
                if (
                    session.status
                    in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]
                    and session.updated_at < cutoff_date
                ):
                    to_remove.append(session_id)

            for session_id in to_remove:
                del self.active_sessions[session_id]
                cleaned_count += 1

        # Clean up from storage
        for session_file in self.storage_path.glob("*.pkl"):
            if session_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    session_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete session file {session_file}: {e}")

        logger.info(f"Cleaned up {cleaned_count} old sessions")
        return cleaned_count

    def _create_checkpoint(
        self, session: ProcessingSession, data: Optional[Dict[str, Any]] = None
    ):
        """Create a checkpoint for the session."""
        checkpoint = {
            "timestamp": datetime.utcnow(),
            "processed_items": session.metrics.processed_items,
            "successful_items": session.metrics.successful_items,
            "failed_items": session.metrics.failed_items,
            "data": data or {},
        }

        session.checkpoints.append(checkpoint)

        # Keep only last 10 checkpoints
        if len(session.checkpoints) > 10:
            session.checkpoints = session.checkpoints[-10:]

        logger.debug(f"Created checkpoint for session {session.session_id}")

    def _save_session(self, session: ProcessingSession):
        """Save session to persistent storage."""
        try:
            session_file = self.storage_path / f"{session.session_id}.pkl"
            with open(session_file, "wb") as f:
                pickle.dump(session, f)
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")

    def _load_session(self, session_id: str) -> Optional[ProcessingSession]:
        """Load session from persistent storage."""
        try:
            session_file = self.storage_path / f"{session_id}.pkl"
            if session_file.exists():
                with open(session_file, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
        return None

    def _record_performance(self, session: ProcessingSession):
        """Record performance metrics for analysis."""
        if session.operation_type.value not in self.performance_history:
            self.performance_history[session.operation_type.value] = []

        performance_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session.session_id,
            "total_items": session.metrics.total_items,
            "processing_time": (
                session.metrics.elapsed_time.total_seconds()
                if session.metrics.elapsed_time
                else 0
            ),
            "processing_rate": session.metrics.processing_rate,
            "success_rate": session.metrics.success_rate,
            "error_count": len(session.errors),
            "file_name": session.file_name,
        }

        self.performance_history[session.operation_type.value].append(
            performance_record
        )

        # Keep only last 100 records per operation type
        if len(self.performance_history[session.operation_type.value]) > 100:
            self.performance_history[session.operation_type.value] = (
                self.performance_history[session.operation_type.value][-100:]
            )


# Global instance
progress_tracker = ProgressTracker()
