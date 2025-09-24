"""
Enhanced Progress Tracker for Resume Processing

Industry-ready progress tracking system that provides real-time updates
for bulk resume processing operations.
"""

import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict

from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("enhanced_progress_tracker")


class JobStatus(Enum):
    """Job status enumeration."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class JobType(Enum):
    """Job type enumeration."""

    EXCEL_PARSING = "excel_parsing"
    BULK_RESUME_PARSING = "bulk_resume_parsing"
    SINGLE_RESUME_PARSING = "single_resume_parsing"


@dataclass
class JobProgress:
    """Job progress tracking data."""

    job_id: str
    job_type: JobType
    status: JobStatus
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    skipped_items: int
    start_time: datetime
    end_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    current_item: Optional[str] = None
    error_messages: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []
        if self.metadata is None:
            self.metadata = {}
        if self.last_update is None:
            self.last_update = datetime.now()

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100

    @property
    def is_active(self) -> bool:
        """Check if job is actively processing."""
        return self.status in [JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.PAUSED]

    @property
    def is_complete(self) -> bool:
        """Check if job is complete."""
        return self.status in [
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        ]

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        end_time = self.end_time or datetime.now()
        return (end_time - self.start_time).total_seconds()

    @property
    def estimated_remaining_time(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        if self.processed_items == 0 or self.status != JobStatus.PROCESSING:
            return None

        elapsed = self.elapsed_time
        rate = self.processed_items / elapsed if elapsed > 0 else 0
        remaining_items = self.total_items - self.processed_items

        return remaining_items / rate if rate > 0 else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["job_type"] = self.job_type.value
        data["status"] = self.status.value
        data["start_time"] = self.start_time.isoformat()
        data["end_time"] = self.end_time.isoformat() if self.end_time else None
        data["last_update"] = self.last_update.isoformat() if self.last_update else None
        data["progress_percentage"] = self.progress_percentage
        data["elapsed_time"] = self.elapsed_time
        data["estimated_remaining_time"] = self.estimated_remaining_time
        return data


class EnhancedProgressTracker:
    """
    Industry-ready progress tracker for bulk operations.

    Features:
    - Real-time progress updates
    - Error tracking and recovery
    - Job queuing and management
    - Automatic cleanup of old jobs
    - Thread-safe operations
    """

    def __init__(self, cleanup_interval: int = 3600):  # 1 hour
        """
        Initialize progress tracker.

        Args:
            cleanup_interval: Interval in seconds for cleaning up old jobs
        """
        self.jobs: Dict[str, JobProgress] = {}
        self.cleanup_interval = cleanup_interval
        self._lock = threading.Lock()
        self._cleanup_task = None
        self._start_cleanup_task()

        logger.info("Enhanced Progress Tracker initialized")

    def _start_cleanup_task(self):
        """Start background cleanup task."""

        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self._cleanup_old_jobs()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")

        self._cleanup_task = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_task.start()

    def _cleanup_old_jobs(self):
        """Clean up jobs older than 24 hours."""
        cutoff_time = datetime.now() - timedelta(hours=24)

        with self._lock:
            jobs_to_remove = []
            for job_id, job in self.jobs.items():
                if job.is_complete and job.start_time < cutoff_time:
                    jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                del self.jobs[job_id]
                logger.info(f"Cleaned up old job: {job_id}")

    def create_job(
        self,
        job_type: JobType,
        total_items: int,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new job for tracking.

        Args:
            job_type: Type of job
            total_items: Total number of items to process
            user_id: User ID associated with the job
            session_id: Session ID associated with the job
            metadata: Additional metadata

        Returns:
            Job ID for tracking
        """
        job_id = str(uuid.uuid4())

        job = JobProgress(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.QUEUED,
            total_items=total_items,
            processed_items=0,
            successful_items=0,
            failed_items=0,
            skipped_items=0,
            start_time=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
        )

        with self._lock:
            self.jobs[job_id] = job

        logger.info(
            f"Created job {job_id} for {job_type.value} with {total_items} items"
        )
        return job_id

    def update_job_status(self, job_id: str, status: JobStatus):
        """Update job status."""
        with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.status = status
                job.last_update = datetime.now()

                if status in [
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                ]:
                    job.end_time = datetime.now()

                logger.debug(f"Updated job {job_id} status to {status.value}")

    def update_progress(
        self,
        job_id: str,
        processed_items: Optional[int] = None,
        successful_items: Optional[int] = None,
        failed_items: Optional[int] = None,
        skipped_items: Optional[int] = None,
        current_item: Optional[str] = None,
    ):
        """
        Update job progress.

        Args:
            job_id: Job ID
            processed_items: Total processed items
            successful_items: Successfully processed items
            failed_items: Failed items
            skipped_items: Skipped items
            current_item: Current item being processed
        """
        with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]

                if processed_items is not None:
                    job.processed_items = processed_items
                if successful_items is not None:
                    job.successful_items = successful_items
                if failed_items is not None:
                    job.failed_items = failed_items
                if skipped_items is not None:
                    job.skipped_items = skipped_items
                if current_item is not None:
                    job.current_item = current_item

                job.last_update = datetime.now()

                # Auto-update status based on progress
                if (
                    job.processed_items >= job.total_items
                    and job.status == JobStatus.PROCESSING
                ):
                    job.status = JobStatus.COMPLETED
                    job.end_time = datetime.now()

    def add_error(
        self, job_id: str, error_message: str, item_identifier: Optional[str] = None
    ):
        """
        Add error to job tracking.

        Args:
            job_id: Job ID
            error_message: Error message
            item_identifier: Identifier of the item that failed
        """
        with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                error_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "message": error_message,
                    "item": item_identifier,
                }
                job.error_messages.append(error_entry)
                job.last_update = datetime.now()

                logger.warning(f"Added error to job {job_id}: {error_message}")

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and progress."""
        with self._lock:
            if job_id in self.jobs:
                return self.jobs[job_id].to_dict()
        return None

    def get_all_jobs(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all jobs, optionally filtered by user ID."""
        with self._lock:
            jobs = []
            for job in self.jobs.values():
                if user_id is None or job.user_id == user_id:
                    jobs.append(job.to_dict())
            return jobs

    def get_active_jobs(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active jobs, optionally filtered by user ID."""
        with self._lock:
            jobs = []
            for job in self.jobs.values():
                if job.is_active and (user_id is None or job.user_id == user_id):
                    jobs.append(job.to_dict())
            return jobs

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        with self._lock:
            if job_id in self.jobs and self.jobs[job_id].is_active:
                job = self.jobs[job_id]
                job.status = JobStatus.CANCELLED
                job.end_time = datetime.now()
                job.last_update = datetime.now()
                logger.info(f"Cancelled job {job_id}")
                return True
        return False

    def pause_job(self, job_id: str) -> bool:
        """Pause a job."""
        with self._lock:
            if job_id in self.jobs and self.jobs[job_id].status == JobStatus.PROCESSING:
                job = self.jobs[job_id]
                job.status = JobStatus.PAUSED
                job.last_update = datetime.now()
                logger.info(f"Paused job {job_id}")
                return True
        return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        with self._lock:
            if job_id in self.jobs and self.jobs[job_id].status == JobStatus.PAUSED:
                job = self.jobs[job_id]
                job.status = JobStatus.PROCESSING
                job.last_update = datetime.now()
                logger.info(f"Resumed job {job_id}")
                return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        with self._lock:
            total_jobs = len(self.jobs)
            active_jobs = sum(1 for job in self.jobs.values() if job.is_active)
            completed_jobs = sum(
                1 for job in self.jobs.values() if job.status == JobStatus.COMPLETED
            )
            failed_jobs = sum(
                1 for job in self.jobs.values() if job.status == JobStatus.FAILED
            )

            return {
                "total_jobs": total_jobs,
                "active_jobs": active_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "queued_jobs": sum(
                    1 for job in self.jobs.values() if job.status == JobStatus.QUEUED
                ),
                "processing_jobs": sum(
                    1
                    for job in self.jobs.values()
                    if job.status == JobStatus.PROCESSING
                ),
                "paused_jobs": sum(
                    1 for job in self.jobs.values() if job.status == JobStatus.PAUSED
                ),
                "cancelled_jobs": sum(
                    1 for job in self.jobs.values() if job.status == JobStatus.CANCELLED
                ),
                "timestamp": datetime.now().isoformat(),
            }


# Global enhanced progress tracker instance
enhanced_progress_tracker = EnhancedProgressTracker()

# Alias for backward compatibility with existing code
progress_tracker = enhanced_progress_tracker
