"""
Enhanced Excel Resume Parser API with Real-time Progress Tracking

Industry-ready background job processing with comprehensive progress tracking
for Excel resume parsing operations.
"""

import asyncio
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse

from excel_resume_parser.main import ExcelResumeParserManager
from core.custom_logger import CustomLogger
from core.enhanced_progress_tracker import progress_tracker, JobType, JobStatus
from core.llm_config import LLMConfigManager, LLMProvider
from core.config import AppConfig

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("enhanced_excel_resume_parser_api")

# Initialize LLM config manager
llm_manager = LLMConfigManager()

# Create router
router = APIRouter(
    tags=["Enhanced Excel Resume Parser"], prefix="/enhanced-excel-parser"
)

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=4)


async def process_excel_file_background(
    job_id: str,
    file_content: bytes,
    filename: str,
    base_user_id: str,
    base_username: str,
    sheet_name: Optional[str] = None,
    llm_provider: str = None,
):
    """
    Background task for processing Excel file with progress tracking.
    """
    try:
        logger.info(f"Starting background Excel processing for job {job_id}")

        # Update job status to processing
        progress_tracker.update_job_status(job_id, JobStatus.PROCESSING)

        # Initialize Excel Resume Parser Manager
        parser_manager = ExcelResumeParserManager(llm_provider=llm_provider)

        # Process the Excel file to get rows first
        excel_processor = parser_manager.excel_processor
        excel_data = excel_processor.process_excel_file_from_bytes(
            file_bytes=file_content, filename=filename, sheet_name=sheet_name
        )

        if not excel_data or not excel_data.get("rows"):
            progress_tracker.add_error(job_id, "No data found in Excel file")
            progress_tracker.update_job_status(job_id, JobStatus.FAILED)
            return

        rows = excel_data["rows"]
        total_rows = len(rows)

        # Update total items in progress tracker
        with progress_tracker._lock:
            if job_id in progress_tracker.jobs:
                progress_tracker.jobs[job_id].total_items = total_rows

        logger.info(f"Processing {total_rows} rows from Excel file")

        processed_count = 0
        successful_count = 0
        failed_count = 0
        results = []

        # Process each row with progress updates
        for index, row in enumerate(rows):
            try:
                # Check if job was cancelled
                with progress_tracker._lock:
                    if job_id in progress_tracker.jobs:
                        job = progress_tracker.jobs[job_id]
                        if job.status == JobStatus.CANCELLED:
                            logger.info(
                                f"Job {job_id} was cancelled, stopping processing"
                            )
                            return
                        elif job.status == JobStatus.PAUSED:
                            logger.info(f"Job {job_id} is paused, waiting...")
                            while job.status == JobStatus.PAUSED:
                                await asyncio.sleep(1)
                                if job.status == JobStatus.CANCELLED:
                                    return

                # Update current item
                current_item = f"Row {index + 1}/{total_rows}"
                progress_tracker.update_progress(job_id, current_item=current_item)

                # Process individual row
                row_result = await process_single_row(
                    parser_manager, row, index, base_user_id, base_username
                )

                if row_result.get("success"):
                    successful_count += 1
                    results.append(row_result)
                else:
                    failed_count += 1
                    error_msg = row_result.get("error", "Unknown error")
                    progress_tracker.add_error(
                        job_id, f"Row {index + 1}: {error_msg}", f"row_{index + 1}"
                    )

                processed_count += 1

                # Update progress
                progress_tracker.update_progress(
                    job_id,
                    processed_items=processed_count,
                    successful_items=successful_count,
                    failed_items=failed_count,
                )

                # Add small delay to prevent overwhelming the system
                if index % 10 == 0:
                    await asyncio.sleep(0.1)

            except Exception as e:
                failed_count += 1
                processed_count += 1
                error_msg = f"Error processing row {index + 1}: {str(e)}"
                logger.error(error_msg)
                progress_tracker.add_error(job_id, error_msg, f"row_{index + 1}")

                progress_tracker.update_progress(
                    job_id,
                    processed_items=processed_count,
                    successful_items=successful_count,
                    failed_items=failed_count,
                )

        # Mark job as completed
        progress_tracker.update_job_status(job_id, JobStatus.COMPLETED)

        # Store final results in job metadata
        with progress_tracker._lock:
            if job_id in progress_tracker.jobs:
                job = progress_tracker.jobs[job_id]
                job.metadata.update(
                    {
                        "final_results": {
                            "total_rows": total_rows,
                            "processed_rows": processed_count,
                            "successful_rows": successful_count,
                            "failed_rows": failed_count,
                            "results": results[:100],  # Store first 100 results
                            "file_info": {
                                "filename": filename,
                                "sheet_name": sheet_name,
                                "file_size": len(file_content),
                                "llm_provider": llm_provider,
                            },
                        }
                    }
                )

        logger.info(
            f"Completed Excel processing for job {job_id}: "
            f"{successful_count}/{total_rows} successful"
        )

    except Exception as e:
        logger.error(
            f"Critical error in background Excel processing for job {job_id}: {e}"
        )
        progress_tracker.add_error(job_id, f"Critical processing error: {str(e)}")
        progress_tracker.update_job_status(job_id, JobStatus.FAILED)


async def process_single_row(
    parser_manager: ExcelResumeParserManager,
    row: Dict[str, Any],
    index: int,
    base_user_id: str,
    base_username: str,
) -> Dict[str, Any]:
    """
    Process a single Excel row.
    """
    try:
        # Extract resume text from row
        resume_text = parser_manager.excel_processor._extract_resume_text_from_row(row)

        if not resume_text or len(resume_text.strip()) < 50:
            return {
                "success": False,
                "error": "Insufficient resume content",
                "row_index": index,
            }

        # Generate user ID and username for this row
        user_id = f"{base_user_id}_{index + 1:04d}"
        username = f"{base_username}_{index + 1:04d}"

        # Parse the resume
        parsed_result = await asyncio.get_event_loop().run_in_executor(
            executor,
            parser_manager.resume_parser.parse_resume,
            resume_text,
            user_id,
            username,
        )

        if parsed_result and parsed_result.get("success"):
            return {
                "success": True,
                "row_index": index,
                "user_id": user_id,
                "username": username,
                "parsed_data": parsed_result,
            }
        else:
            return {
                "success": False,
                "error": "Resume parsing failed",
                "row_index": index,
            }

    except Exception as e:
        return {"success": False, "error": str(e), "row_index": index}


@router.post("/upload-async", summary="Upload Excel file for asynchronous processing")
async def upload_excel_file_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    username: str = Form(...),
    sheet_name: Optional[str] = Form(None),
):
    """
    Upload and process an Excel file asynchronously with real-time progress tracking.

    This endpoint:
    1. Validates and accepts the Excel file
    2. Creates a background job for processing
    3. Returns a job ID immediately for progress tracking
    4. Processes the file in the background with live updates

    Use the job ID to poll `/status/{job_id}` for real-time progress.
    """
    try:
        logger.info(f"Starting async Excel upload for user {user_id}")

        # Validate file type
        if not file.filename.lower().endswith((".xlsx", ".xls", ".xlsm")):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only Excel files (.xlsx, .xls, .xlsm) are allowed.",
            )

        # Read file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Auto-detect LLM provider
        llm_provider = AppConfig.LLM_PROVIDER

        # Create job with initial estimate (will be updated when we know actual row count)
        job_id = progress_tracker.create_job(
            job_type=JobType.EXCEL_PARSING,
            total_items=0,  # Will be updated after Excel analysis
            user_id=user_id,
            metadata={
                "filename": file.filename,
                "sheet_name": sheet_name,
                "file_size": len(file_content),
                "llm_provider": llm_provider,
                "base_username": username,
            },
        )

        # Start background processing
        background_tasks.add_task(
            process_excel_file_background,
            job_id,
            file_content,
            file.filename,
            user_id,
            username,
            sheet_name,
            llm_provider,
        )

        logger.info(f"Created Excel processing job {job_id}")

        return JSONResponse(
            status_code=202,  # Accepted
            content={
                "status": "accepted",
                "message": "Excel file accepted for processing",
                "job_id": job_id,
                "poll_url": f"/enhanced-excel-parser/status/{job_id}",
                "estimated_processing_time": "varies based on file size",
                "file_info": {
                    "filename": file.filename,
                    "file_size": len(file_content),
                    "sheet_name": sheet_name,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting async Excel processing: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error starting Excel processing: {str(e)}"
        )


@router.get("/status/{job_id}", summary="Get real-time job status and progress")
async def get_job_status(job_id: str):
    """
    Get real-time status and progress of an Excel processing job.

    Response includes:
    - Current progress percentage
    - Items processed/total
    - Success/failure counts
    - Estimated remaining time
    - Any errors encountered
    - Current processing item
    """
    try:
        job_status = progress_tracker.get_job_status(job_id)

        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "status": "success",
            "job_status": job_status,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status for {job_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting job status: {str(e)}"
        )


@router.get("/results/{job_id}", summary="Get final processing results")
async def get_job_results(job_id: str):
    """
    Get the final results of a completed Excel processing job.

    Only available for completed jobs. Includes:
    - Final processing statistics
    - Parsed resume data
    - Error summary
    - Performance metrics
    """
    try:
        job_status = progress_tracker.get_job_status(job_id)

        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")

        if job_status["status"] not in ["completed", "failed"]:
            raise HTTPException(
                status_code=400,
                detail=f"Job is still {job_status['status']}. Results not available yet.",
            )

        # Get detailed results from metadata
        final_results = job_status.get("metadata", {}).get("final_results", {})

        return {
            "status": "success",
            "job_id": job_id,
            "job_status": job_status["status"],
            "results": final_results,
            "summary": {
                "total_rows": job_status.get("total_items", 0),
                "processed_rows": job_status.get("processed_items", 0),
                "successful_rows": job_status.get("successful_items", 0),
                "failed_rows": job_status.get("failed_items", 0),
                "success_rate": f"{(job_status.get('successful_items', 0) / max(job_status.get('processed_items', 1), 1)) * 100:.2f}%",
                "processing_time": f"{job_status.get('elapsed_time', 0):.2f} seconds",
            },
            "errors": job_status.get("error_messages", []),
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results for {job_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting job results: {str(e)}"
        )


@router.post("/control/{job_id}/{action}", summary="Control job execution")
async def control_job(job_id: str, action: str):
    """
    Control job execution (pause, resume, cancel).

    Actions:
    - pause: Pause the job temporarily
    - resume: Resume a paused job
    - cancel: Cancel the job permanently
    """
    try:
        if action not in ["pause", "resume", "cancel"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid action. Use 'pause', 'resume', or 'cancel'",
            )

        success = False
        if action == "pause":
            success = progress_tracker.pause_job(job_id)
        elif action == "resume":
            success = progress_tracker.resume_job(job_id)
        elif action == "cancel":
            success = progress_tracker.cancel_job(job_id)

        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot {action} job. Job may not exist or not in appropriate state.",
            )

        return {
            "status": "success",
            "message": f"Job {job_id} {action}d successfully",
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error controlling job {job_id} with action {action}: {e}")
        raise HTTPException(status_code=500, detail=f"Error controlling job: {str(e)}")


@router.get("/jobs", summary="Get all jobs for user")
async def get_user_jobs(user_id: Optional[str] = None, active_only: bool = False):
    """
    Get all jobs, optionally filtered by user ID and active status.
    """
    try:
        if active_only:
            jobs = progress_tracker.get_active_jobs(user_id)
        else:
            jobs = progress_tracker.get_all_jobs(user_id)

        return {
            "status": "success",
            "jobs": jobs,
            "count": len(jobs),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting user jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting jobs: {str(e)}")


@router.get("/statistics", summary="Get processing statistics")
async def get_processing_statistics():
    """
    Get overall processing statistics and system status.
    """
    try:
        stats = progress_tracker.get_statistics()

        return {
            "status": "success",
            "statistics": stats,
            "system_info": {
                "llm_provider": AppConfig.LLM_PROVIDER,
                "max_workers": executor._max_workers,
                "active_threads": (
                    executor._threads.__len__() if hasattr(executor, "_threads") else 0
                ),
            },
        }

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting statistics: {str(e)}"
        )
