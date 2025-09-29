"""
Unified Resume Parser API

Consolidated API that combines all resume parsing capabilities:
- Single resume parsing (PDF, DOC, DOCX, TXT)
- Multiple resume parsing with tracking
- Excel resume parsing
- Enhanced parsing with accuracy validation

This replaces:
- enhanced_resume_parser_api.py
- enhanced_multiple_resume_parser_with_tracking.py
- enhanced_excel_resume_parser_api.py
- single_resume_parser.py
- multiple_resume_parser_clean.py
"""

import asyncio
import time
import uuid
import os
import tempfile
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import (
    APIRouter,
    HTTPException,
    File,
    UploadFile,
    Form,
    Query,
    BackgroundTasks,
)
from fastapi.responses import JSONResponse

# Import existing components
from GroqcloudLLM.text_extraction import extract_and_clean_text, clean_text
from GroqcloudLLM.main import ResumeParser as LLMResumeParser
from core.enhanced_resume_parser import EnhancedResumeParser, create_enhanced_parser
from core.fixed_enhanced_resume_parser import (
    FixedEnhancedResumeParser,
    create_fixed_enhanced_parser,
)
from core.custom_logger import CustomLogger
from mangodatabase.operations import ResumeOperations, SkillsTitlesOperations
from mangodatabase.duplicate_detection import DuplicateDetectionOperations
from mangodatabase.client import get_collection, get_skills_titles_collection
from embeddings.vectorizer import AddUserDataVectorizer
from schemas.add_user_schemas import ResumeData
from core.enhanced_progress_tracker import progress_tracker, JobType, JobStatus
from core.llm_config import LLMConfigManager, LLMProvider
from excel_resume_parser.enhanced_excel_resume_parser import EnhancedExcelResumeParser
from excel_resume_parser.main import ExcelResumeParserManager
from excel_resume_parser.excel_resume_parser import ExcelResumeParser
from excel_resume_parser.fixed_excel_resume_parser import (
    FixedExcelResumeParser,
    create_fixed_excel_parser,
)
from excel_resume_parser.fixed_excel_parser_adapter import (
    FixedExcelParserAdapter,
    create_fixed_excel_parser_adapter,
)
from multipleresumepraser.main import ResumeParser

# Initialize components
logger_manager = CustomLogger()
logger = logger_manager.get_logger("unified_resume_parser_api")

collection = get_collection()
skills_titles_collection = get_skills_titles_collection()
skills_ops = SkillsTitlesOperations(skills_titles_collection)
duplicate_ops = DuplicateDetectionOperations(collection)
add_user_vectorizer = AddUserDataVectorizer()
resume_ops = ResumeOperations(collection, add_user_vectorizer)
llm_manager = LLMConfigManager()

# Create router
router = APIRouter(
    prefix="/resume-parser",
    tags=["Unified Resume Parser"],
)

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=6)

# Directory configuration
BASE_FOLDER = "dummy_data_save"
TEMP_FOLDER = os.path.join(BASE_FOLDER, "temp_text_extract")
TEMP_DIR = Path(os.path.join(BASE_FOLDER, "temp_files"))

# Ensure directories exist
os.makedirs(TEMP_FOLDER, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Global tracking for processing sessions
PROCESSING_SESSIONS = {}


class UnifiedAccuracyMetrics:
    """Track parsing accuracy metrics across all parser types"""

    def __init__(self):
        self.total_parsed = 0
        self.successful_parses = 0
        self.validation_failures = 0
        self.extraction_failures = 0

    def record_success(self):
        self.total_parsed += 1
        self.successful_parses += 1

    def record_validation_failure(self):
        self.total_parsed += 1
        self.validation_failures += 1

    def record_extraction_failure(self):
        self.total_parsed += 1
        self.extraction_failures += 1

    def get_accuracy_rate(self) -> float:
        if self.total_parsed == 0:
            return 0.0
        return (self.successful_parses / self.total_parsed) * 100

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_parsed": self.total_parsed,
            "successful_parses": self.successful_parses,
            "validation_failures": self.validation_failures,
            "extraction_failures": self.extraction_failures,
            "accuracy_rate": self.get_accuracy_rate(),
        }


# Global metrics tracker
accuracy_metrics = UnifiedAccuracyMetrics()


def validate_extracted_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive validation of extracted resume data
    Returns validation report with fixes applied
    """
    validation_report = {
        "validation_passed": True,
        "issues_found": [],
        "fixes_applied": [],
        "confidence_score": 100,
    }

    # Check required contact details
    contact = data.get("contact_details", {})

    if not contact.get("name") or contact.get("name") == "Name Not Found":
        validation_report["issues_found"].append("Name extraction failed")
        validation_report["confidence_score"] -= 20

    if not contact.get("email") or "@" not in contact.get("email", ""):
        validation_report["issues_found"].append("Invalid email address")
        validation_report["confidence_score"] -= 15

    if not contact.get("phone") or len(contact.get("phone", "")) < 10:
        validation_report["issues_found"].append("Invalid phone number")
        validation_report["confidence_score"] -= 10

    # Check experience data
    experiences = data.get("experience", [])
    if not experiences:
        validation_report["issues_found"].append("No work experience found")
        validation_report["confidence_score"] -= 25
    else:
        for i, exp in enumerate(experiences):
            if not exp.get("company"):
                validation_report["issues_found"].append(
                    f"Experience {i+1}: Missing company"
                )
                validation_report["confidence_score"] -= 5
            if not exp.get("title"):
                validation_report["issues_found"].append(
                    f"Experience {i+1}: Missing job title"
                )
                validation_report["confidence_score"] -= 5

    # Check skills
    skills = data.get("skills", [])
    if len(skills) < 3:
        validation_report["issues_found"].append("Insufficient skills extracted")
        validation_report["confidence_score"] -= 15

    # Set validation status
    if validation_report["confidence_score"] < 70:
        validation_report["validation_passed"] = False

    return validation_report


# =============================================================================
# 1. SINGLE RESUME PARSING
# =============================================================================


@router.post(
    "/single",
    summary="Parse Single Resume File",
    description="""
    Parse a single resume file with enhanced accuracy and validation.
    
    **Features:**
    - Multi-method extraction (Rule-based + NLP + LLM)
    - Comprehensive data validation 
    - Error correction and fallback mechanisms
    - Detailed confidence scoring
    - Enhanced contact information extraction
    - Improved experience calculation
    - Advanced skills categorization
    
    **Supported Formats:** PDF, DOC, DOCX, TXT
    """,
)
async def parse_single_resume(
    file: UploadFile = File(...),
    user_name: str = Form(..., description="Name of the user uploading the resume"),
    user_id: str = Form(..., description="Unique identifier for the user"),
):
    """Parse a single resume file with enhanced accuracy."""
    try:
        logger.info(
            f"Starting single resume parsing for user: {user_name} (ID: {user_id})"
        )

        # Set default values for simplified API
        validation_level = "standard"  # Always use standard validation
        save_to_database = True  # Always save to database
        detect_duplicates = True  # Always check for duplicates
        update_existing = False  # Don't update existing duplicates
        llm_provider = None  # Get from .env
        api_keys = None  # Get from .env

        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        file_ext = Path(file.filename).suffix.lower()
        valid_extensions = {".pdf", ".doc", ".docx", ".txt"}
        if file_ext not in valid_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(valid_extensions)}",
            )

        # Parse API keys if provided (will be None for simplified API)
        parsed_api_keys = None

        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)

        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Extract text
        extracted_text = extract_and_clean_text(temp_file_path)
        if not extracted_text or extracted_text.strip() == "":
            accuracy_metrics.record_extraction_failure()
            raise HTTPException(
                status_code=400, detail="Could not extract text from the uploaded file"
            )

        # Initialize enhanced parser (no LLM backup for simplified API - uses .env config)
        # Use FIXED enhanced parser for better accuracy
        enhanced_parser = create_fixed_enhanced_parser()

        # Parse resume using enhanced parser with LLM enabled
        parsed_data = enhanced_parser.parse_resume(extracted_text, use_llm=True)

        # Add user information to parsed data
        if "parsing_metadata" not in parsed_data:
            parsed_data["parsing_metadata"] = {}
        parsed_data["parsing_metadata"]["user_name"] = user_name
        parsed_data["parsing_metadata"]["user_id"] = user_id

        # Validate extracted data
        validation_report = validate_extracted_data(parsed_data)

        if validation_report["validation_passed"]:
            accuracy_metrics.record_success()
        else:
            accuracy_metrics.record_validation_failure()

        # Add metadata
        parsed_data["parsing_metadata"] = {
            "file_name": file.filename,
            "file_size": len(content),
            "parsing_time": datetime.now().isoformat(),
            "user_name": user_name,
            "user_id": user_id,
            "validation_report": validation_report,
            "parser_version": "unified_v1.0",
        }

        # Save to database (always enabled for simplified API)
        database_result = None

        # Check for duplicates (always enabled for simplified API)
        existing_resume = resume_ops.check_duplicate_resume(
            email=parsed_data.get("contact_details", {}).get("email"),
            phone=parsed_data.get("contact_details", {}).get("phone"),
            name=parsed_data.get("contact_details", {}).get("name"),
        )

        if existing_resume and not update_existing:
            return JSONResponse(
                status_code=409,
                content={
                    "message": "Duplicate resume found",
                    "existing_resume_id": str(existing_resume.get("_id")),
                    "parsed_data": parsed_data,
                    "duplicate_info": {
                        "email": existing_resume.get("contact_details", {}).get(
                            "email"
                        ),
                        "name": existing_resume.get("contact_details", {}).get("name"),
                    },
                },
            )

        # Convert to ResumeData schema and save
        resume_data = ResumeData(**parsed_data)
        result = resume_ops.add_user_data(resume_data)
        database_result = {
            "saved": True,
            "resume_id": (
                str(result.inserted_id) if hasattr(result, "inserted_id") else None
            ),
        }

        # Clean up temp file
        try:
            os.remove(temp_file_path)
            os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp file: {e}")

        return JSONResponse(
            status_code=200,
            content={
                "message": "Resume parsed successfully",
                "parsed_data": parsed_data,
                "database_result": database_result,
                "accuracy_metrics": accuracy_metrics.get_stats(),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in single resume parsing: {e}")
        accuracy_metrics.record_extraction_failure()
        raise HTTPException(status_code=500, detail=f"Resume parsing failed: {str(e)}")


# =============================================================================
# 2. MULTIPLE RESUME PARSING
# =============================================================================


async def process_multiple_resumes_background(
    job_id: str,
    files: List[Dict[str, Any]],
    user_name: str,
    user_id: str,
    validation_level: str = "standard",
    save_to_database: bool = True,
    detect_duplicates: bool = True,
    llm_provider: Optional[str] = None,
    api_keys: Optional[List[str]] = None,
):
    """Background task for processing multiple resume files with progress tracking."""
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

        # Initialize enhanced parser - always use LLM for better extraction in simplified API
        try:
            from multipleresumepraser.main import ResumeParser as MultiResumeParser

            llm_parser = MultiResumeParser(
                llm_provider="groq"
            )  # Use default groq provider
            logger.info(
                "Initialized LLM parser with groq provider for better extraction"
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize LLM parser: {e}. Falling back to enhanced parser without LLM."
            )
            llm_parser = None

        # Use FIXED enhanced parser for better accuracy
        enhanced_parser = create_fixed_enhanced_parser(llm_parser=llm_parser)

        logger.info(
            f"Processing {total_files} resume files for user: {user_name} (ID: {user_id})"
        )

        # Process files in smaller batches to prevent memory issues
        batch_size = 5
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
            for file_index, file_data in enumerate(batch_files):
                try:
                    filename = file_data["filename"]
                    content = file_data["content"]

                    # Create temp file
                    temp_dir = tempfile.mkdtemp()
                    temp_file_path = os.path.join(temp_dir, filename)

                    with open(temp_file_path, "wb") as buffer:
                        buffer.write(content)

                    # Extract text
                    extracted_text = extract_and_clean_text(temp_file_path)

                    if not extracted_text or extracted_text.strip() == "":
                        failed_count += 1
                        results.append(
                            {
                                "filename": filename,
                                "success": False,
                                "error": "Could not extract text from file",
                                "user_name": user_name,
                                "user_id": user_id,
                            }
                        )
                        continue

                    # Generate unique user_id for each resume in batch
                    resume_user_id = (
                        f"{user_id}_{int(time.time())}_{processed_count:04d}"
                    )

                    # Parse resume using enhanced parser with LLM for better extraction
                    parsed_data = enhanced_parser.parse_resume(
                        extracted_text,
                        use_llm=True,  # Always use LLM for better extraction
                    )

                    # Validate extracted data
                    validation_report = validate_extracted_data(parsed_data)

                    # Add metadata
                    if "parsing_metadata" not in parsed_data:
                        parsed_data["parsing_metadata"] = {}
                    parsed_data["parsing_metadata"].update(
                        {
                            "file_name": filename,
                            "file_size": len(content),
                            "parsing_time": datetime.now().isoformat(),
                            "user_name": user_name,
                            "user_id": resume_user_id,
                            "original_user_id": user_id,
                            "validation_report": validation_report,
                            "parser_version": "unified_v1.0",
                            "batch_index": processed_count,
                        }
                    )

                    # Save to database if requested
                    database_result = None
                    skipped = False

                    if save_to_database:
                        # Check for duplicates if requested
                        if detect_duplicates:
                            existing_resume = resume_ops.check_duplicate_resume(
                                email=parsed_data.get("contact_details", {}).get(
                                    "email"
                                ),
                                phone=parsed_data.get("contact_details", {}).get(
                                    "phone"
                                ),
                                name=parsed_data.get("contact_details", {}).get("name"),
                            )

                            if existing_resume:
                                skipped = True
                                skipped_count += 1
                                database_result = {
                                    "saved": False,
                                    "reason": "duplicate_found",
                                    "existing_resume_id": str(
                                        existing_resume.get("_id")
                                    ),
                                }
                            else:
                                # Save new resume
                                resume_data = ResumeData(**parsed_data)
                                result = resume_ops.add_user_data(resume_data)
                                database_result = {
                                    "saved": True,
                                    "resume_id": (
                                        str(result.inserted_id)
                                        if hasattr(result, "inserted_id")
                                        else None
                                    ),
                                }
                                successful_count += 1
                        else:
                            # Save without duplicate check
                            resume_data = ResumeData(**parsed_data)
                            result = resume_ops.add_user_data(resume_data)
                            database_result = {
                                "saved": True,
                                "resume_id": (
                                    str(result.inserted_id)
                                    if hasattr(result, "inserted_id")
                                    else None
                                ),
                            }
                            successful_count += 1
                    else:
                        successful_count += 1

                    results.append(
                        {
                            "filename": filename,
                            "success": True,
                            "skipped": skipped,
                            "user_name": user_name,
                            "user_id": resume_user_id,
                            "original_user_id": user_id,
                            "parsed_data": parsed_data,
                            "database_result": database_result,
                            "validation_report": validation_report,
                        }
                    )

                    # Clean up temp file
                    try:
                        os.remove(temp_file_path)
                        os.rmdir(temp_dir)
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file: {e}")

                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    failed_count += 1
                    results.append(
                        {
                            "filename": filename,
                            "success": False,
                            "error": str(e),
                            "user_name": user_name,
                            "user_id": user_id,
                        }
                    )

                processed_count += 1

                # Update progress
                progress_tracker.update_progress(
                    job_id,
                    processed_items=processed_count,
                    successful_items=successful_count,
                    failed_items=failed_count,
                    skipped_items=skipped_count,
                    current_item=f"Processing {filename}",
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
                        "results": results,
                        "summary": {
                            "total_files": total_files,
                            "processed_count": processed_count,
                            "successful_count": successful_count,
                            "failed_count": failed_count,
                            "skipped_count": skipped_count,
                            "accuracy_rate": (
                                (successful_count / total_files * 100)
                                if total_files > 0
                                else 0
                            ),
                        },
                        "user_info": {
                            "user_name": user_name,
                            "user_id": user_id,
                        },
                    }
                )

        logger.info(
            f"Completed bulk processing job {job_id}: {successful_count}/{total_files} successful"
        )

    except Exception as e:
        logger.error(f"Error in background bulk processing job {job_id}: {e}")
        progress_tracker.update_job_status(job_id, JobStatus.FAILED)
        with progress_tracker._lock:
            if job_id in progress_tracker.jobs:
                job = progress_tracker.jobs[job_id]
                job.metadata["error"] = str(e)


@router.post(
    "/multiple",
    summary="Parse Multiple Resume Files",
    description="""
    Upload and process multiple resume files asynchronously with real-time progress tracking.
    
    **Features:**
    - Asynchronous processing with job tracking
    - Real-time progress updates
    - Batch processing for performance
    - Duplicate detection and handling
    - Comprehensive error handling
    - User identification for each resume
    
    **Supported Formats:** PDF, DOC, DOCX, TXT
    
    **Usage:**
    1. Upload files and get a job_id
    2. Use job_id to check progress via /status/{job_id}
    3. Get final results via /results/{job_id}
    """,
)
async def parse_multiple_resumes(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user_name: str = Form(..., description="Name of the user uploading resumes"),
    user_id: str = Form(..., description="Unique identifier for the user"),
):
    """Upload and process multiple resume files asynchronously."""
    try:
        logger.info(
            f"Starting async bulk resume upload with {len(files)} files for user: {user_name} (ID: {user_id})"
        )

        # Set default values for simplified API
        validation_level = "standard"  # Always use standard validation
        save_to_database = True  # Always save to database
        detect_duplicates = True  # Always check for duplicates
        llm_provider = None  # Get from .env
        api_keys = None  # Get from .env

        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        if len(files) > 1000:  # Reasonable limit
            raise HTTPException(
                status_code=400,
                detail="Too many files. Maximum 1000 files allowed per batch",
            )

        # Parse API keys (will be None for simplified API)
        parsed_api_keys = None

        # Validate file types and read content
        valid_extensions = {".pdf", ".doc", ".docx", ".txt"}
        file_data = []
        total_size = 0

        for file in files:
            # Check file extension
            if not file.filename:
                continue

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
            if total_size > 200 * 1024 * 1024:  # 200MB limit
                raise HTTPException(
                    status_code=400, detail="Total file size exceeds 200MB limit"
                )

            file_data.append({"filename": file.filename, "content": content})

        if not file_data:
            raise HTTPException(status_code=400, detail="No valid files to process")

        # Create job
        job_id = progress_tracker.create_job(
            job_type=JobType.BULK_RESUME_PARSING,
            total_items=len(file_data),
            metadata={
                "user_name": user_name,
                "user_id": user_id,
                "file_count": len(file_data),
                "validation_level": validation_level,
                "save_to_database": save_to_database,
                "detect_duplicates": detect_duplicates,
                "total_size_mb": round(total_size / 1024 / 1024, 2),
            },
        )

        # Start background processing
        background_tasks.add_task(
            process_multiple_resumes_background,
            job_id,
            file_data,
            user_name,
            user_id,
            validation_level,
            save_to_database,
            detect_duplicates,
            llm_provider,
            parsed_api_keys,
        )

        return JSONResponse(
            status_code=202,
            content={
                "message": "Multiple resume processing started",
                "job_id": job_id,
                "status": "processing",
                "user_info": {
                    "user_name": user_name,
                    "user_id": user_id,
                },
                "files_count": len(file_data),
                "check_status_url": f"/resume-parser/status/{job_id}",
                "check_results_url": f"/resume-parser/results/{job_id}",
                "estimated_processing_time": f"Approximately {len(file_data) * 2} seconds",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating multiple resume processing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Multiple resume processing initiation failed: {str(e)}",
        )


# =============================================================================
# 3. EXCEL RESUME PARSING
# =============================================================================


async def process_excel_file_enhanced(
    session_id: str,
    file_path: str,
    user_name: str,
    user_id: str,
    sheet_name: Optional[str],
    validation_level: str,
    cleaning_aggressive: bool,
    include_quality_scores: bool,
    batch_size: int,
    save_to_database: bool,
    detect_duplicates: bool,
    update_existing: bool,
    export_report: bool,
    llm_provider: Optional[str],
    api_keys: Optional[List[str]],
    temp_dir: str,
):
    """Background task for processing Excel file with enhanced capabilities."""
    try:
        logger.info(
            f"Processing Excel file for session {session_id}, user: {user_name} (ID: {user_id})"
        )

        # Update session status
        PROCESSING_SESSIONS[session_id]["status"] = "processing"
        PROCESSING_SESSIONS[session_id][
            "progress"
        ] = "Initializing FIXED Excel parser for better accuracy"

        # Initialize FIXED Excel parser for 100% accuracy
        parser = create_fixed_excel_parser_adapter(
            llm_provider=llm_provider, api_keys=api_keys
        )

        # Update progress
        PROCESSING_SESSIONS[session_id][
            "progress"
        ] = "Processing Excel file with enhanced capabilities"

        # Process the Excel file
        processing_result = parser.process_excel_file(
            file_path=file_path,
            sheet_name=sheet_name,
            validation_level=validation_level,
            cleaning_aggressive=cleaning_aggressive,
            include_quality_scores=include_quality_scores,
            batch_size=batch_size,
            user_id=user_id,
            user_name=user_name,
        )

        # Update session with initial results
        PROCESSING_SESSIONS[session_id]["processing_result"] = processing_result
        PROCESSING_SESSIONS[session_id]["progress"] = "Excel processing completed"

        # FIXED: User information is now properly set in the parser
        # No need to override the user_id and user_name as they are correctly assigned
        if processing_result.get("parsed_resumes"):
            for i, resume in enumerate(processing_result["parsed_resumes"]):
                if "parsing_metadata" not in resume:
                    resume["parsing_metadata"] = {}

                # Add additional metadata without overriding user identification
                resume["parsing_metadata"].update(
                    {
                        "source_type": "excel",
                        "row_index": i + 1,
                        "upload_session_id": session_id,
                        "processing_timestamp": datetime.now().isoformat(),
                    }
                )

        # Save to database if requested
        database_result = None
        if save_to_database and processing_result.get("parsed_resumes"):
            PROCESSING_SESSIONS[session_id]["progress"] = "Saving to database"

            database_result = parser.save_parsed_resumes_to_database(
                parsed_resumes=processing_result["parsed_resumes"],
                detect_duplicates=detect_duplicates,
                update_existing=update_existing,
            )

            PROCESSING_SESSIONS[session_id]["database_result"] = database_result

        # Export report if requested
        report_path = None
        if export_report:
            PROCESSING_SESSIONS[session_id]["progress"] = "Generating processing report"

            report_filename = f"excel_processing_report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = os.path.join(temp_dir, report_filename)

            # Enhanced report with user information
            enhanced_report = {
                "session_id": session_id,
                "user_info": {
                    "user_name": user_name,
                    "user_id": user_id,
                },
                "processing_result": processing_result,
                "database_result": database_result,
                "processing_time": datetime.now().isoformat(),
            }

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(enhanced_report, f, indent=2, ensure_ascii=False)

            PROCESSING_SESSIONS[session_id]["report_path"] = report_path

        # Mark as completed
        PROCESSING_SESSIONS[session_id]["status"] = "completed"
        PROCESSING_SESSIONS[session_id]["end_time"] = datetime.now().isoformat()
        PROCESSING_SESSIONS[session_id]["user_info"] = {
            "user_name": user_name,
            "user_id": user_id,
        }

    except Exception as e:
        logger.error(f"Error processing Excel file for session {session_id}: {e}")
        PROCESSING_SESSIONS[session_id]["status"] = "failed"
        PROCESSING_SESSIONS[session_id]["error"] = str(e)
        PROCESSING_SESSIONS[session_id]["end_time"] = datetime.now().isoformat()


@router.post(
    "/excel",
    summary="Parse Excel Resume File",
    description="""
    Upload and parse Excel file containing multiple resumes with enhanced capabilities.
    
    **Features:**
    - Intelligent column mapping
    - Data validation and cleaning
    - Quality scoring
    - Batch processing
    - Duplicate detection
    - Processing reports
    - User identification for all resumes
    
    **Supported Formats:** XLSX, XLS
    
    **Usage:**
    1. Upload Excel file and get a session_id
    2. Use session_id to check progress via /status/{session_id}
    3. Get final results via /results/{session_id}
    """,
)
async def parse_excel_resume(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_name: str = Form(..., description="Name of the user uploading the Excel file"),
    user_id: str = Form(..., description="Unique identifier for the user"),
    sheet_name: Optional[str] = Form(
        None, description="Name or index of the Excel sheet to process (optional)"
    ),
    validation_level: str = Form(
        "standard", description="Level of validation: basic, standard, strict"
    ),
    cleaning_aggressive: bool = Form(
        False, description="Whether to apply aggressive data cleaning"
    ),
    include_quality_scores: bool = Form(
        True, description="Whether to calculate data quality scores"
    ),
):
    """Upload and parse Excel file with enhanced capabilities."""
    # Generate session ID
    session_id = str(uuid.uuid4())
    logger.info(
        f"Starting enhanced Excel processing session: {session_id} for user: {user_name} (ID: {user_id})"
    )

    # Set default values based on requirements
    save_to_database = True  # Always save to database
    detect_duplicates = True  # Always check duplicates (skip duplicates)
    update_existing = False  # Don't update duplicates, skip them
    export_report = False  # Don't export reports by default
    llm_provider = None  # Get from .env
    api_keys = None  # Get from .env

    # Auto-determine batch size based on file size
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)  # Convert to MB
    if file_size_mb < 1:
        batch_size = 5
    elif file_size_mb < 5:
        batch_size = 10
    elif file_size_mb < 10:
        batch_size = 20
    else:
        batch_size = 50

    # Validate parameters
    if validation_level not in ["basic", "standard", "strict"]:
        raise HTTPException(
            status_code=400,
            detail="validation_level must be 'basic', 'standard', or 'strict'",
        )

    # Validate file
    if not file.filename or not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=400, detail="File must be an Excel file (.xlsx or .xls)"
        )

    try:
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)

        # Save uploaded file (content already read above)
        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)

        # Initialize session tracking
        PROCESSING_SESSIONS[session_id] = {
            "status": "initializing",
            "start_time": datetime.now().isoformat(),
            "file_name": file.filename,
            "file_size": len(content),
            "user_info": {
                "user_name": user_name,
                "user_id": user_id,
            },
            "parameters": {
                "sheet_name": sheet_name,
                "validation_level": validation_level,
                "cleaning_aggressive": cleaning_aggressive,
                "include_quality_scores": include_quality_scores,
                "batch_size": batch_size,
                "save_to_database": save_to_database,
                "detect_duplicates": detect_duplicates,
                "update_existing": update_existing,
                "export_report": export_report,
                "auto_determined_batch_size": True,
            },
        }

        # Process in background
        background_tasks.add_task(
            process_excel_file_enhanced,
            session_id,
            temp_file_path,
            user_name,
            user_id,
            sheet_name,
            validation_level,
            cleaning_aggressive,
            include_quality_scores,
            batch_size,
            save_to_database,
            detect_duplicates,
            update_existing,
            export_report,
            llm_provider,
            None,  # api_keys - will be handled by .env
            temp_dir,
        )

        return JSONResponse(
            status_code=202,
            content={
                "message": "Excel file processing started",
                "session_id": session_id,
                "status": "processing",
                "user_info": {
                    "user_name": user_name,
                    "user_id": user_id,
                },
                "check_status_url": f"/resume-parser/status/{session_id}",
                "check_results_url": f"/resume-parser/results/{session_id}",
                "estimated_processing_time": "Processing time varies based on file size and complexity",
            },
        )

    except Exception as e:
        logger.error(f"Error initiating enhanced Excel processing: {e}")
        if session_id in PROCESSING_SESSIONS:
            PROCESSING_SESSIONS[session_id]["status"] = "failed"
            PROCESSING_SESSIONS[session_id]["error"] = str(e)

        raise HTTPException(
            status_code=500, detail=f"Excel processing initiation failed: {str(e)}"
        )


# =============================================================================
# 4. PROGRESS AND STATUS ENDPOINTS
# =============================================================================


@router.get(
    "/status/{job_or_session_id}",
    summary="Get Processing Status",
    description="""
    Get real-time status and progress for any processing job or session.
    
    Works for:
    - Multiple resume processing jobs (job_id)
    - Excel processing sessions (session_id)
    """,
)
async def get_processing_status(job_or_session_id: str):
    """Get real-time job status and progress."""
    try:
        # First check if it's a job ID (multiple resume processing)
        job_status = progress_tracker.get_job_status(job_or_session_id)
        if job_status:
            return JSONResponse(
                status_code=200,
                content={"type": "job", "job_id": job_or_session_id, **job_status},
            )

        # Check if it's a session ID (Excel processing)
        if job_or_session_id in PROCESSING_SESSIONS:
            session = PROCESSING_SESSIONS[job_or_session_id]
            return JSONResponse(
                status_code=200,
                content={"type": "session", "session_id": job_or_session_id, **session},
            )

        # Not found
        raise HTTPException(
            status_code=404, detail=f"Job or session not found: {job_or_session_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for {job_or_session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get(
    "/results/{job_or_session_id}",
    summary="Get Processing Results",
    description="""
    Get final processing results for any completed job or session.
    
    Works for:
    - Multiple resume processing jobs (job_id)
    - Excel processing sessions (session_id)
    """,
)
async def get_processing_results(job_or_session_id: str):
    """Get final processing results."""
    try:
        # First check if it's a job ID (multiple resume processing)
        job_status = progress_tracker.get_job_status(job_or_session_id)
        if job_status:
            if job_status["status"] != "completed":
                raise HTTPException(
                    status_code=400,
                    detail=f"Job not completed. Current status: {job_status['status']}",
                )

            return JSONResponse(
                status_code=200,
                content={
                    "type": "job",
                    "job_id": job_or_session_id,
                    "results": job_status.get("metadata", {}).get("results", []),
                    "summary": job_status.get("metadata", {}).get("summary", {}),
                    "user_info": job_status.get("metadata", {}).get("user_info", {}),
                    "metadata": job_status.get("metadata", {}),
                },
            )

        # Check if it's a session ID (Excel processing)
        if job_or_session_id in PROCESSING_SESSIONS:
            session = PROCESSING_SESSIONS[job_or_session_id]

            if session["status"] != "completed":
                raise HTTPException(
                    status_code=400,
                    detail=f"Session not completed. Current status: {session['status']}",
                )

            return JSONResponse(
                status_code=200,
                content={
                    "type": "session",
                    "session_id": job_or_session_id,
                    "processing_result": session.get("processing_result", {}),
                    "database_result": session.get("database_result", {}),
                    "user_info": session.get("user_info", {}),
                    "report_available": "report_path" in session,
                    "summary": session.get("processing_result", {}).get("summary", {}),
                },
            )

        # Not found
        raise HTTPException(
            status_code=404, detail=f"Job or session not found: {job_or_session_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting results for {job_or_session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")


@router.post(
    "/control/{job_id}/{action}",
    summary="Control Job Execution",
    description="""
    Control job execution for multiple resume processing jobs.
    
    **Available Actions:**
    - pause: Pause job execution
    - resume: Resume paused job
    - cancel: Cancel job execution
    """,
)
async def control_job_execution(
    job_id: str,
    action: str,
):
    """Control job execution (pause, resume, cancel)."""
    if action not in ["pause", "resume", "cancel"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid action. Must be 'pause', 'resume', or 'cancel'",
        )

    try:
        if action == "pause":
            result = progress_tracker.pause_job(job_id)
        elif action == "resume":
            result = progress_tracker.resume_job(job_id)
        else:  # cancel
            result = progress_tracker.cancel_job(job_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Job {action}{'d' if action != 'cancel' else 'led'} successfully",
                "job_id": job_id,
                "action": action,
                "timestamp": datetime.now().isoformat(),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error controlling job {job_id} with action {action}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to {action} job: {str(e)}")


@router.get(
    "/jobs",
    summary="Get All Jobs for User",
    description="""
    Get all processing jobs and sessions for a user.
    
    **Query Parameters:**
    - user_id: Filter by user ID (optional)
    - active_only: Show only active jobs (optional, default: false)
    """,
)
async def get_user_jobs(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    active_only: bool = Query(False, description="Show only active jobs"),
):
    """Get all jobs for user."""
    try:
        # Get jobs from progress tracker
        if active_only:
            jobs = progress_tracker.get_active_jobs(user_id)
        else:
            jobs = progress_tracker.get_all_jobs(user_id)

        # Get sessions from Excel processing
        sessions = []
        for session_id, session_data in PROCESSING_SESSIONS.items():
            session_user_id = session_data.get("user_info", {}).get("user_id")
            if user_id is None or session_user_id == user_id:
                if not active_only or session_data.get("status") in [
                    "processing",
                    "initializing",
                ]:
                    sessions.append(
                        {
                            "session_id": session_id,
                            "type": "excel_processing",
                            "user_info": session_data.get("user_info", {}),
                            "status": session_data.get("status"),
                            "start_time": session_data.get("start_time"),
                            "file_name": session_data.get("file_name"),
                        }
                    )

        return JSONResponse(
            status_code=200,
            content={
                "jobs": jobs,
                "sessions": sessions,
                "total_jobs": len(jobs),
                "total_sessions": len(sessions),
                "filter": {
                    "user_id": user_id,
                    "active_only": active_only,
                },
            },
        )

    except Exception as e:
        logger.error(f"Error getting user jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get jobs: {str(e)}")


@router.get(
    "/statistics",
    summary="Get Processing Statistics",
    description="Get overall processing statistics and metrics.",
)
async def get_processing_statistics():
    """Get processing statistics."""
    try:
        # Get job statistics
        job_stats = progress_tracker.get_statistics()

        # Get session statistics
        session_count = len(PROCESSING_SESSIONS)
        completed_sessions = len(
            [s for s in PROCESSING_SESSIONS.values() if s.get("status") == "completed"]
        )
        failed_sessions = len(
            [s for s in PROCESSING_SESSIONS.values() if s.get("status") == "failed"]
        )

        # Get accuracy metrics
        accuracy_stats = accuracy_metrics.get_stats()

        return JSONResponse(
            status_code=200,
            content={
                "job_statistics": job_stats,
                "session_statistics": {
                    "total_sessions": session_count,
                    "completed_sessions": completed_sessions,
                    "failed_sessions": failed_sessions,
                    "active_sessions": session_count
                    - completed_sessions
                    - failed_sessions,
                },
                "accuracy_metrics": accuracy_stats,
                "system_status": "operational",
                "last_updated": datetime.now().isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get statistics: {str(e)}"
        )


@router.get(
    "/health", summary="Health Check", description="Check API health and system status."
)
async def health_check():
    """Check API health."""
    try:
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "unified_v1.0",
                "features": [
                    "single_resume_parsing",
                    "multiple_resume_parsing",
                    "excel_resume_parsing",
                    "progress_tracking",
                    "duplicate_detection",
                    "validation",
                ],
                "supported_formats": ["pdf", "doc", "docx", "txt", "xlsx", "xls"],
                "accuracy_metrics": accuracy_metrics.get_stats(),
            },
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


# =============================================================================
# 5. CLEANUP AND MAINTENANCE
# =============================================================================


@router.delete(
    "/cleanup/{job_or_session_id}",
    summary="Clean Up Job or Session",
    description="Remove completed job or session data to free up memory.",
)
async def cleanup_job_or_session(job_or_session_id: str):
    """Clean up completed job or session data."""
    try:
        cleaned_up = False

        # Try to clean up job
        if progress_tracker.cleanup_job(job_or_session_id):
            cleaned_up = True

        # Try to clean up session
        if job_or_session_id in PROCESSING_SESSIONS:
            session = PROCESSING_SESSIONS[job_or_session_id]
            if session.get("status") in ["completed", "failed"]:
                # Clean up temp files if they exist
                if "report_path" in session:
                    try:
                        os.remove(session["report_path"])
                    except Exception as e:
                        logger.warning(f"Could not remove report file: {e}")

                del PROCESSING_SESSIONS[job_or_session_id]
                cleaned_up = True

        if not cleaned_up:
            raise HTTPException(
                status_code=404, detail=f"Job or session not found: {job_or_session_id}"
            )

        return JSONResponse(
            status_code=200,
            content={
                "message": "Job or session cleaned up successfully",
                "job_or_session_id": job_or_session_id,
                "timestamp": datetime.now().isoformat(),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up {job_or_session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clean up: {str(e)}")


if __name__ == "__main__":
    print("Unified Resume Parser API - Ready for integration")
    print("Features:")
    print("- Single resume parsing")
    print("- Multiple resume parsing with tracking")
    print("- Excel resume parsing")
    print("- Progress tracking and job control")
    print("- User identification for all parsers")
    print("- Comprehensive validation and error handling")
