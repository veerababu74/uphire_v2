"""
Enhanced Multiple Resume Parser API with Real-time Progress Tracking

Industry-ready background job processing with comprehensive progress tracking
for bulk resume parsing operations.
"""

import asyncio
import time
import uuid
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import APIRouter, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse

from multipleresumepraser.main import ResumeParser
from GroqcloudLLM.text_extraction import extract_and_clean_text, clean_text
from mangodatabase.operations import ResumeOperations, SkillsTitlesOperations
from mangodatabase.client import get_collection, get_skills_titles_collection
from embeddings.vectorizer import AddUserDataVectorizer
from schemas.add_user_schemas import ResumeData
from core.custom_logger import CustomLogger
from core.enhanced_progress_tracker import progress_tracker, JobType, JobStatus
from core.llm_config import LLMConfigManager, LLMProvider

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("enhanced_multiple_resume_parser")

# Initialize database connections and operations
collection = get_collection()
skills_titles_collection = get_skills_titles_collection()
skills_ops = SkillsTitlesOperations(skills_titles_collection)
add_user_vectorizer = AddUserDataVectorizer()
resume_ops = ResumeOperations(collection, add_user_vectorizer)

# Initialize LLM config manager
llm_manager = LLMConfigManager()

# Create router
router = APIRouter(
    tags=["Enhanced Multiple Resume Parser"], prefix="/enhanced-bulk-parser"
)

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=6)

# Directory configuration
BASE_FOLDER = "dummy_data_save"
TEMP_FOLDER = os.path.join(BASE_FOLDER, "temp_text_extract")
TEMP_DIR = Path(os.path.join(BASE_FOLDER, "temp_files"))

# Ensure directories exist
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


async def process_multiple_resumes_background(
    job_id: str,
    files: List[Dict[str, Any]],  # List of {filename, content} dicts
    duplicate_check: bool = True,
):
    """
    Background task for processing multiple resume files with progress tracking.
    """
    try:
        logger.info(f"Starting background bulk resume processing for job {job_id}")

        # Update job status to processing
        progress_tracker.update_job_status(job_id, JobStatus.PROCESSING)

        total_files = len(files)
        processed_count = 0
        successful_count = 0
        failed_count = 0
        skipped_count = 0
        results = []

        # Initialize resume parser
        resume_parser = ResumeParser()

        logger.info(f"Processing {total_files} resume files")

        # Process files in smaller batches to prevent memory issues
        batch_size = 10
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_files = files[batch_start:batch_end]

            # Check if job was cancelled
            with progress_tracker._lock:
                if job_id in progress_tracker.jobs:
                    job = progress_tracker.jobs[job_id]
                    if job.status == JobStatus.CANCELLED:
                        logger.info(f"Job {job_id} was cancelled, stopping processing")
                        return
                    elif job.status == JobStatus.PAUSED:
                        logger.info(f"Job {job_id} is paused, waiting...")
                        while job.status == JobStatus.PAUSED:
                            await asyncio.sleep(1)
                            if job.status == JobStatus.CANCELLED:
                                return

            # Process batch
            batch_results = await process_file_batch(
                job_id, batch_files, resume_parser, batch_start, duplicate_check
            )

            # Update counters
            for result in batch_results:
                processed_count += 1
                if result.get("success"):
                    if result.get("skipped"):
                        skipped_count += 1
                    else:
                        successful_count += 1
                else:
                    failed_count += 1

                results.append(result)

            # Update progress
            progress_tracker.update_progress(
                job_id,
                processed_items=processed_count,
                successful_items=successful_count,
                failed_items=failed_count,
                skipped_items=skipped_count,
                current_item=f"Processed batch {batch_start//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}",
            )

            # Small delay between batches
            await asyncio.sleep(0.5)

        # Mark job as completed
        progress_tracker.update_job_status(job_id, JobStatus.COMPLETED)

        # Store final results in job metadata
        with progress_tracker._lock:
            if job_id in progress_tracker.jobs:
                job = progress_tracker.jobs[job_id]
                job.metadata.update(
                    {
                        "final_results": {
                            "total_files": total_files,
                            "processed_files": processed_count,
                            "successful_files": successful_count,
                            "failed_files": failed_count,
                            "skipped_files": skipped_count,
                            "results": results[:50],  # Store first 50 results
                            "processing_settings": {
                                "duplicate_check": duplicate_check,
                                "batch_size": batch_size,
                            },
                        }
                    }
                )

        logger.info(
            f"Completed bulk resume processing for job {job_id}: "
            f"{successful_count}/{total_files} successful, {skipped_count} skipped"
        )

    except Exception as e:
        logger.error(
            f"Critical error in background bulk processing for job {job_id}: {e}"
        )
        progress_tracker.add_error(job_id, f"Critical processing error: {str(e)}")
        progress_tracker.update_job_status(job_id, JobStatus.FAILED)


async def process_file_batch(
    job_id: str,
    batch_files: List[Dict[str, Any]],
    resume_parser: ResumeParser,
    batch_offset: int,
    duplicate_check: bool,
) -> List[Dict[str, Any]]:
    """
    Process a batch of resume files concurrently.
    """
    batch_results = []

    # Create temporary files for this batch
    temp_files = []
    try:
        for i, file_data in enumerate(batch_files):
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(file_data["filename"]).suffix, dir=TEMP_DIR
            )
            temp_file.write(file_data["content"])
            temp_file.close()
            temp_files.append(
                {
                    "temp_path": temp_file.name,
                    "original_filename": file_data["filename"],
                    "index": batch_offset + i,
                }
            )

        # Process files concurrently within the batch
        tasks = []
        for temp_file_info in temp_files:
            task = process_single_resume_file(
                job_id,
                temp_file_info["temp_path"],
                temp_file_info["original_filename"],
                temp_file_info["index"],
                resume_parser,
                duplicate_check,
            )
            tasks.append(task)

        # Wait for all tasks in batch to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                error_result = {
                    "success": False,
                    "filename": temp_files[i]["original_filename"],
                    "file_index": temp_files[i]["index"],
                    "error": str(result),
                    "timestamp": datetime.now().isoformat(),
                }
                batch_results[i] = error_result
                progress_tracker.add_error(
                    job_id,
                    f"File {temp_files[i]['original_filename']}: {str(result)}",
                    temp_files[i]["original_filename"],
                )

    finally:
        # Cleanup temporary files
        for temp_file_info in temp_files:
            try:
                if os.path.exists(temp_file_info["temp_path"]):
                    os.unlink(temp_file_info["temp_path"])
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup temp file {temp_file_info['temp_path']}: {e}"
                )

    return batch_results


async def process_single_resume_file(
    job_id: str,
    file_path: str,
    original_filename: str,
    file_index: int,
    resume_parser: ResumeParser,
    duplicate_check: bool,
) -> Dict[str, Any]:
    """
    Process a single resume file.
    """
    try:
        # Update current item being processed
        progress_tracker.update_progress(
            job_id, current_item=f"Processing {original_filename} ({file_index + 1})"
        )

        # Extract text from file
        extracted_text = await asyncio.get_event_loop().run_in_executor(
            executor, extract_and_clean_text, file_path
        )

        if not extracted_text or len(extracted_text.strip()) < 50:
            return {
                "success": False,
                "filename": original_filename,
                "file_index": file_index,
                "error": "Insufficient resume content extracted",
                "timestamp": datetime.now().isoformat(),
            }

        # Generate user ID based on filename
        base_filename = Path(original_filename).stem
        user_id = f"bulk_{int(time.time())}_{file_index:04d}"
        username = f"user_{base_filename}_{file_index:04d}"

        # Check for duplicates if enabled
        if duplicate_check:
            is_duplicate = await check_duplicate_resume(extracted_text)
            if is_duplicate:
                return {
                    "success": True,
                    "skipped": True,
                    "filename": original_filename,
                    "file_index": file_index,
                    "reason": "Duplicate resume detected",
                    "timestamp": datetime.now().isoformat(),
                }

        # Parse the resume
        parsed_result = await asyncio.get_event_loop().run_in_executor(
            executor, resume_parser.parse_resume, extracted_text, user_id, username
        )

        if parsed_result and parsed_result.get("success"):
            return {
                "success": True,
                "filename": original_filename,
                "file_index": file_index,
                "user_id": user_id,
                "username": username,
                "parsed_data": parsed_result,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            return {
                "success": False,
                "filename": original_filename,
                "file_index": file_index,
                "error": "Resume parsing failed",
                "timestamp": datetime.now().isoformat(),
            }

    except Exception as e:
        return {
            "success": False,
            "filename": original_filename,
            "file_index": file_index,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def check_duplicate_resume(resume_text: str) -> bool:
    """
    Check if resume is a duplicate (simplified version).
    """
    try:
        # Generate a simple hash of the resume content for duplicate detection
        text_hash = hash(resume_text.lower().strip())

        # This is a simplified duplicate check
        # In production, you might want to use more sophisticated methods
        existing_resume = collection.find_one({"content_hash": text_hash})

        return existing_resume is not None

    except Exception as e:
        logger.warning(f"Error checking for duplicates: {e}")
        return False  # If duplicate check fails, proceed with processing


@router.post(
    "/upload-async", summary="Upload multiple resume files for asynchronous processing"
)
async def upload_multiple_resumes_async(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    duplicate_check: bool = True,
):
    """
    Upload and process multiple resume files asynchronously with real-time progress tracking.

    This endpoint:
    1. Validates and accepts multiple resume files
    2. Creates a background job for processing
    3. Returns a job ID immediately for progress tracking
    4. Processes files in the background with live updates

    Use the job ID to poll `/status/{job_id}` for real-time progress.
    """
    try:
        logger.info(f"Starting async bulk resume upload with {len(files)} files")

        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        if len(files) > 10000:  # Reasonable limit
            raise HTTPException(
                status_code=400,
                detail="Too many files. Maximum 10,000 files allowed per batch",
            )

        # Validate file types and read content
        valid_extensions = {".pdf", ".doc", ".docx", ".txt"}
        file_data = []
        total_size = 0

        for file in files:
            # Check file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in valid_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {file.filename}. Supported: PDF, DOC, DOCX, TXT",
                )

            # Read file content
            content = await file.read()
            if not content:
                logger.warning(f"Empty file skipped: {file.filename}")
                continue

            total_size += len(content)
            if total_size > 500 * 1024 * 1024:  # 500MB limit
                raise HTTPException(
                    status_code=400, detail="Total file size exceeds 500MB limit"
                )

            file_data.append({"filename": file.filename, "content": content})

        if not file_data:
            raise HTTPException(status_code=400, detail="No valid files to process")

        # Create job
        job_id = progress_tracker.create_job(
            job_type=JobType.BULK_RESUME_PARSING,
            total_items=len(file_data),
            metadata={
                "total_files": len(file_data),
                "total_size": total_size,
                "duplicate_check": duplicate_check,
                "file_types": list(
                    set(Path(fd["filename"]).suffix.lower() for fd in file_data)
                ),
            },
        )

        # Start background processing
        background_tasks.add_task(
            process_multiple_resumes_background, job_id, file_data, duplicate_check
        )

        logger.info(
            f"Created bulk resume processing job {job_id} with {len(file_data)} files"
        )

        return JSONResponse(
            status_code=202,  # Accepted
            content={
                "status": "accepted",
                "message": f"Bulk resume processing started for {len(file_data)} files",
                "job_id": job_id,
                "poll_url": f"/enhanced-bulk-parser/status/{job_id}",
                "estimated_processing_time": f"{len(file_data) * 2} seconds",
                "processing_info": {
                    "total_files": len(file_data),
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "duplicate_check": duplicate_check,
                    "supported_formats": list(valid_extensions),
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting async bulk resume processing: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error starting bulk processing: {str(e)}"
        )


@router.get("/status/{job_id}", summary="Get real-time job status and progress")
async def get_job_status(job_id: str):
    """
    Get real-time status and progress of a bulk resume processing job.

    Response includes:
    - Current progress percentage
    - Files processed/total
    - Success/failure/skipped counts
    - Estimated remaining time
    - Any errors encountered
    - Current processing file
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
    Get the final results of a completed bulk resume processing job.

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
                "total_files": job_status.get("total_items", 0),
                "processed_files": job_status.get("processed_items", 0),
                "successful_files": job_status.get("successful_items", 0),
                "failed_files": job_status.get("failed_items", 0),
                "skipped_files": job_status.get("skipped_items", 0),
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
                "max_workers": executor._max_workers,
                "active_threads": (
                    executor._threads.__len__() if hasattr(executor, "_threads") else 0
                ),
                "temp_directory": str(TEMP_DIR),
                "supported_formats": [".pdf", ".doc", ".docx", ".txt"],
            },
        }

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting statistics: {str(e)}"
        )
