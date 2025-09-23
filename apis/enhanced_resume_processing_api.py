"""
Enhanced Resume Processing API with Live Tracking

This module provides API endpoints for resume processing with real-time progress tracking,
error handling, and recovery capabilities.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, File, UploadFile, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from excel_resume_parser.enhanced_excel_parser_with_tracking import (
    enhanced_excel_parser,
)
from multipleresumepraser.enhanced_multiple_resume_parser_with_tracking import (
    enhanced_multiple_resume_parser,
)
from core.progress_tracker import progress_tracker
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("enhanced_resume_processing_api")

# Create router
router = APIRouter()


# Pydantic models for request/response
class ProcessingStatusResponse(BaseModel):
    session_id: str
    status: str
    operation_type: str
    user_id: str
    username: str
    file_name: Optional[str]
    created_at: str
    updated_at: str
    metrics: Dict[str, Any]
    error_summary: Dict[str, Any]
    last_checkpoint: Optional[str]


class ErrorDetailsResponse(BaseModel):
    session_id: str
    total_errors: int
    errors: List[Dict[str, Any]]


class ActiveSessionsResponse(BaseModel):
    total_sessions: int
    sessions: List[Dict[str, Any]]


class ResumeProcessingRequest(BaseModel):
    base_user_id: str
    base_username: str
    cleanup_files: bool = True
    llm_provider: Optional[str] = None
    api_keys: Optional[List[str]] = None


# Temporary file storage
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)


@router.post(
    "/enhanced-excel-parser/upload-and-process",
    operation_id="upload_and_process_excel_enhanced",
)
async def upload_and_process_excel_with_tracking(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    base_user_id: str = Query(..., description="Base user ID for generated resumes"),
    base_username: str = Query(..., description="Base username for generated resumes"),
    sheet_name: Optional[str] = Query(None, description="Specific sheet to process"),
    cleanup_file: bool = Query(
        True, description="Whether to cleanup file after processing"
    ),
    llm_provider: Optional[str] = Query(None, description="LLM provider to use"),
    session_id: Optional[str] = Query(
        None, description="Existing session ID to resume"
    ),
):
    """
    Upload and process Excel file with comprehensive tracking.

    This endpoint provides:
    - Real-time progress tracking
    - Error handling and recovery
    - Detailed processing metrics
    - Resume capability
    """
    try:
        # Validate file
        if not file.filename.endswith((".xlsx", ".xls")):
            raise HTTPException(
                status_code=400, detail="Only Excel files (.xlsx, .xls) are supported"
            )

        # Save uploaded file temporarily
        temp_file_path = TEMP_DIR / f"{int(time.time())}_{file.filename}"

        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Excel file uploaded: {file.filename} ({len(content)} bytes)")

        # Start background processing
        async def process_excel_background():
            try:
                result = await enhanced_excel_parser.process_excel_file_with_tracking(
                    file_path=str(temp_file_path),
                    base_user_id=base_user_id,
                    base_username=base_username,
                    sheet_name=sheet_name,
                    cleanup_file=cleanup_file,
                    session_id=session_id,
                )
                logger.info(
                    f"Excel processing completed for session: {result.get('session_id')}"
                )
                return result
            except Exception as e:
                logger.error(f"Background Excel processing failed: {str(e)}")
                # Cleanup temp file on error
                if temp_file_path.exists():
                    temp_file_path.unlink()
                raise

        # Start processing in background
        background_tasks.add_task(process_excel_background)

        # Return immediate response with session info
        # Create session first to get session_id
        from core.progress_tracker import OperationType

        temp_session_id = progress_tracker.create_session(
            operation_type=OperationType.EXCEL_PARSING,
            user_id=base_user_id,
            username=base_username,
            file_name=file.filename,
            total_items=0,  # Will be updated when processing starts
            configuration={
                "sheet_name": sheet_name,
                "cleanup_file": cleanup_file,
                "llm_provider": llm_provider,
            },
        )

        return {
            "status": "accepted",
            "message": "Excel file uploaded and processing started",
            "session_id": temp_session_id,
            "file_name": file.filename,
            "file_size": len(content),
            "estimated_processing_time": "Will be calculated once processing starts",
            "tracking_endpoints": {
                "status": f"/enhanced-resume-processing/status/{temp_session_id}",
                "errors": f"/enhanced-resume-processing/errors/{temp_session_id}",
                "live_updates": f"/enhanced-resume-processing/live-updates/{temp_session_id}",
            },
        }

    except Exception as e:
        logger.error(f"Excel upload and process failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post(
    "/enhanced-multiple-resume-parser/upload-and-process",
    operation_id="upload_and_process_multiple_resumes_enhanced",
)
async def upload_and_process_multiple_resumes_with_tracking(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    base_user_id: str = Query(..., description="Base user ID for generated resumes"),
    base_username: str = Query(..., description="Base username for generated resumes"),
    cleanup_files: bool = Query(
        True, description="Whether to cleanup files after processing"
    ),
    llm_provider: Optional[str] = Query(None, description="LLM provider to use"),
    session_id: Optional[str] = Query(
        None, description="Existing session ID to resume"
    ),
):
    """
    Upload and process multiple resume files with comprehensive tracking.

    This endpoint provides:
    - Real-time progress tracking
    - Error handling and recovery
    - Detailed processing metrics
    - Resume capability
    - File validation
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        if len(files) > 1000:  # Limit to prevent abuse
            raise HTTPException(
                status_code=400, detail="Too many files. Maximum 1000 files allowed"
            )

        # Save uploaded files temporarily
        temp_file_paths = []
        total_size = 0

        for file in files:
            # Validate file type
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in {".pdf", ".doc", ".docx", ".txt"}:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_ext}. Supported: .pdf, .doc, .docx, .txt",
                )

            # Save file
            temp_file_path = (
                TEMP_DIR / f"{int(time.time())}_{len(temp_file_paths)}_{file.filename}"
            )

            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
                total_size += len(content)

            temp_file_paths.append(str(temp_file_path))

        logger.info(
            f"Multiple resume files uploaded: {len(files)} files ({total_size} bytes total)"
        )

        # Start background processing
        async def process_resumes_background():
            try:
                result = await enhanced_multiple_resume_parser.process_multiple_resumes_with_tracking(
                    resume_files=temp_file_paths,
                    base_user_id=base_user_id,
                    base_username=base_username,
                    cleanup_files=cleanup_files,
                    session_id=session_id,
                )
                logger.info(
                    f"Multiple resume processing completed for session: {result.get('session_id')}"
                )
                return result
            except Exception as e:
                logger.error(f"Background resume processing failed: {str(e)}")
                # Cleanup temp files on error
                for temp_path in temp_file_paths:
                    try:
                        Path(temp_path).unlink()
                    except:
                        pass
                raise

        # Start processing in background
        background_tasks.add_task(process_resumes_background)

        # Create session and return immediate response
        from core.progress_tracker import OperationType

        temp_session_id = progress_tracker.create_session(
            operation_type=OperationType.MULTIPLE_RESUME_PARSING,
            user_id=base_user_id,
            username=base_username,
            total_items=len(files),
            configuration={
                "cleanup_files": cleanup_files,
                "llm_provider": llm_provider,
                "file_count": len(files),
            },
        )

        return {
            "status": "accepted",
            "message": "Resume files uploaded and processing started",
            "session_id": temp_session_id,
            "total_files": len(files),
            "total_size": total_size,
            "estimated_processing_time": f"Approximately {len(files) * 30} seconds",
            "tracking_endpoints": {
                "status": f"/enhanced-resume-processing/status/{temp_session_id}",
                "errors": f"/enhanced-resume-processing/errors/{temp_session_id}",
                "live_updates": f"/enhanced-resume-processing/live-updates/{temp_session_id}",
            },
        }

    except Exception as e:
        logger.error(f"Multiple resume upload and process failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get(
    "/status/{session_id}",
    response_model=ProcessingStatusResponse,
    operation_id="get_processing_status_enhanced",
)
async def get_processing_status(session_id: str):
    """
    Get current processing status for a session.

    Returns comprehensive status including:
    - Processing progress
    - Performance metrics
    - Error summary
    - Time estimates
    """
    try:
        status = progress_tracker.get_session_status(session_id)

        if not status:
            raise HTTPException(
                status_code=404, detail=f"Session {session_id} not found"
            )

        return ProcessingStatusResponse(**status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get status failed for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get(
    "/errors/{session_id}",
    response_model=ErrorDetailsResponse,
    operation_id="get_processing_errors_enhanced",
)
async def get_processing_errors(
    session_id: str,
    limit: int = Query(50, description="Maximum number of errors to return"),
):
    """
    Get processing errors for a session.

    Returns detailed error information including:
    - Error messages and types
    - Error locations (row/file indices)
    - Error context and stack traces
    - Error categorization
    """
    try:
        errors = progress_tracker.get_session_errors(session_id, limit)

        if errors is None:
            raise HTTPException(
                status_code=404, detail=f"Session {session_id} not found"
            )

        return ErrorDetailsResponse(
            session_id=session_id, total_errors=len(errors), errors=errors
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get errors failed for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get errors: {str(e)}")


@router.get(
    "/live-updates/{session_id}", operation_id="get_live_processing_updates_enhanced"
)
async def get_live_processing_updates(session_id: str):
    """
    Get live processing updates via Server-Sent Events (SSE).

    Streams real-time updates including:
    - Progress updates
    - Processing milestones
    - Error notifications
    - Completion status
    """

    async def generate_updates():
        """Generate live updates for the session."""
        try:
            last_update_time = datetime.utcnow()

            while True:
                # Get current status
                status = progress_tracker.get_session_status(session_id)

                if not status:
                    yield f'event: error\ndata: {{"error": "Session not found"}}\n\n'
                    break

                # Check if session is completed or failed
                if status["status"] in ["completed", "failed", "cancelled"]:
                    yield f"event: final\ndata: {json.dumps(status)}\n\n"
                    break

                # Send update if there are changes
                current_time = datetime.fromisoformat(status["updated_at"])
                if current_time > last_update_time:
                    yield f"event: update\ndata: {json.dumps(status)}\n\n"
                    last_update_time = current_time

                # Send heartbeat
                yield f'event: heartbeat\ndata: {{"timestamp": "{datetime.utcnow().isoformat()}"}}\n\n'

                # Wait before next update
                await asyncio.sleep(2)  # Update every 2 seconds

        except Exception as e:
            logger.error(f"Live updates failed for session {session_id}: {str(e)}")
            yield f'event: error\ndata: {{"error": "{str(e)}"}}\n\n'

    return StreamingResponse(
        generate_updates(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )


@router.get(
    "/active-sessions",
    response_model=ActiveSessionsResponse,
    operation_id="get_active_sessions_enhanced",
)
async def get_active_sessions(
    user_id: Optional[str] = Query(None, description="Filter by user ID")
):
    """
    Get list of active processing sessions.

    Returns:
    - Session summaries
    - Processing status
    - Progress metrics
    - User information
    """
    try:
        sessions = progress_tracker.list_active_sessions(user_id)

        return ActiveSessionsResponse(total_sessions=len(sessions), sessions=sessions)

    except Exception as e:
        logger.error(f"Get active sessions failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get active sessions: {str(e)}"
        )


@router.post(
    "/resume-session/{session_id}", operation_id="resume_processing_session_enhanced"
)
async def resume_processing_session(session_id: str):
    """
    Resume a paused or failed processing session.

    This endpoint allows continuing processing from the last checkpoint,
    useful for handling interruptions and recovering from errors.
    """
    try:
        # Get session status first
        status = progress_tracker.get_session_status(session_id)
        if not status:
            raise HTTPException(
                status_code=404, detail=f"Session {session_id} not found"
            )

        if status["status"] not in ["paused", "failed"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot resume session with status: {status['status']}",
            )

        # Determine operation type and resume accordingly
        operation_type = status["operation_type"]

        if operation_type == "excel_parsing":
            # Resume Excel processing
            resume_data = enhanced_excel_parser.resume_processing(session_id)
        elif operation_type == "multiple_resume_parsing":
            # Resume multiple resume processing
            resume_data = enhanced_multiple_resume_parser.resume_processing(session_id)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown operation type: {operation_type}"
            )

        if not resume_data:
            raise HTTPException(
                status_code=400,
                detail="Cannot resume session - no resume data available",
            )

        return {
            "status": "resumed",
            "session_id": session_id,
            "message": "Session resumed successfully",
            "resume_from_index": resume_data.get("last_processed_index", 0),
            "remaining_items": resume_data.get("total_items", 0)
            - resume_data.get("last_processed_index", 0),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume session failed for {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to resume session: {str(e)}"
        )


@router.post(
    "/stop-session/{session_id}", operation_id="stop_processing_session_enhanced"
)
async def stop_processing_session(session_id: str):
    """
    Stop/pause a currently running processing session.

    This endpoint gracefully stops processing and creates a checkpoint
    for later resumption.
    """
    try:
        # Get session status first
        status = progress_tracker.get_session_status(session_id)
        if not status:
            raise HTTPException(
                status_code=404, detail=f"Session {session_id} not found"
            )

        if status["status"] != "in_progress":
            raise HTTPException(
                status_code=400,
                detail=f"Cannot stop session with status: {status['status']}",
            )

        # Determine operation type and stop accordingly
        operation_type = status["operation_type"]

        success = False
        if operation_type == "excel_parsing":
            success = enhanced_excel_parser.stop_processing(session_id)
        elif operation_type == "multiple_resume_parsing":
            success = enhanced_multiple_resume_parser.stop_processing(session_id)

        if success:
            return {
                "status": "stopped",
                "session_id": session_id,
                "message": "Session stopped successfully",
                "processed_items": status["metrics"]["processed_items"],
                "can_resume": True,
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to stop session")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stop session failed for {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop session: {str(e)}")


@router.get("/performance-analytics", operation_id="get_performance_analytics_enhanced")
async def get_performance_analytics(
    operation_type: Optional[str] = Query(None, description="Filter by operation type"),
    days: int = Query(7, description="Number of days to analyze"),
):
    """
    Get performance analytics and insights.

    Returns:
    - Processing performance metrics
    - Error analysis
    - Throughput statistics
    - Performance trends
    """
    try:
        # This would typically query a performance database
        # For now, we'll return basic analytics from the progress tracker

        analytics = {
            "time_period": f"Last {days} days",
            "operation_type": operation_type or "all",
            "summary": {
                "total_sessions": 0,
                "completed_sessions": 0,
                "failed_sessions": 0,
                "average_success_rate": 0.0,
                "average_processing_rate": 0.0,
                "total_items_processed": 0,
            },
            "performance_trends": {
                "daily_throughput": [],
                "error_trends": [],
                "success_rate_trends": [],
            },
            "recommendations": [
                "Monitor error rates and investigate common failure patterns",
                "Consider optimizing batch sizes based on performance data",
                "Implement alerting for sessions with high error rates",
            ],
        }

        return analytics

    except Exception as e:
        logger.error(f"Get performance analytics failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get analytics: {str(e)}"
        )


@router.delete("/cleanup-old-sessions", operation_id="cleanup_old_sessions_enhanced")
async def cleanup_old_sessions(
    days_old: int = Query(7, description="Delete sessions older than this many days"),
    confirm: bool = Query(False, description="Confirm deletion"),
):
    """
    Cleanup old completed sessions to free up storage.

    This endpoint removes old session data and logs to prevent
    storage from growing indefinitely.
    """
    try:
        if not confirm:
            return {
                "message": "Add ?confirm=true to actually delete old sessions",
                "would_delete": "Sessions older than {days_old} days",
            }

        cleaned_count = progress_tracker.cleanup_old_sessions(days_old)

        return {
            "status": "success",
            "message": f"Cleaned up {cleaned_count} old sessions",
            "days_old_threshold": days_old,
        }

    except Exception as e:
        logger.error(f"Cleanup old sessions failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to cleanup sessions: {str(e)}"
        )


# Health check endpoint
@router.get("/health", operation_id="health_check_enhanced_processing")
async def health_check():
    """Health check endpoint for the enhanced resume processing API."""
    return {
        "status": "healthy",
        "service": "Enhanced Resume Processing API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "Real-time progress tracking",
            "Error handling and recovery",
            "Session management",
            "Live updates via SSE",
            "Performance analytics",
            "Batch processing with checkpoints",
        ],
    }
