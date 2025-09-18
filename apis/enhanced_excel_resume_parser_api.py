"""
Enhanced Excel Resume Parser API

API endpoints for processing Excel/XLSX files with advanced capabilities including
intelligent column mapping, data validation, cleaning, and quality scoring.
"""

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
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
import tempfile
import json
import uuid

from excel_resume_parser.enhanced_excel_resume_parser import EnhancedExcelResumeParser
from core.custom_logger import CustomLogger
from core.llm_config import LLMConfigManager, LLMProvider
from core.config import AppConfig

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("enhanced_excel_resume_parser_api")

# Initialize LLM config manager
llm_manager = LLMConfigManager()

# Create router
router = APIRouter()

# Global tracking for processing
PROCESSING_SESSIONS = {}


@router.post("/upload-and-parse-excel/enhanced")
async def upload_and_parse_excel_enhanced(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    validation_level: str = Form("standard"),
    cleaning_aggressive: bool = Form(False),
    include_quality_scores: bool = Form(True),
    batch_size: int = Form(10),
    save_to_database: bool = Form(True),
    detect_duplicates: bool = Form(True),
    update_existing: bool = Form(False),
    export_report: bool = Form(False),
    llm_provider: Optional[str] = Form(None),
    api_keys: Optional[str] = Form(None),
):
    """
    Upload and parse Excel file with enhanced capabilities.

    Args:
        file: Excel file to process (.xlsx or .xls)
        validation_level: Level of validation ("basic", "standard", "strict")
        cleaning_aggressive: Whether to apply aggressive data cleaning
        include_quality_scores: Whether to calculate data quality scores
        batch_size: Number of rows to process in each batch (1-50)
        save_to_database: Whether to save parsed resumes to database
        detect_duplicates: Whether to detect and handle duplicates
        update_existing: Whether to update existing duplicate records
        export_report: Whether to generate a processing report
        llm_provider: LLM provider to use (optional, auto-detected if not provided)
        api_keys: API keys for LLM provider (JSON string)

    Returns:
        JSON response with processing results and session information
    """
    # Generate session ID
    session_id = str(uuid.uuid4())
    logger.info(f"Starting enhanced Excel processing session: {session_id}")

    # Validate parameters
    if validation_level not in ["basic", "standard", "strict"]:
        raise HTTPException(
            status_code=400,
            detail="validation_level must be 'basic', 'standard', or 'strict'",
        )

    if batch_size < 1 or batch_size > 50:
        raise HTTPException(
            status_code=400, detail="batch_size must be between 1 and 50"
        )

    # Validate file
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=400, detail="File must be an Excel file (.xlsx or .xls)"
        )

    # Parse API keys if provided
    parsed_api_keys = None
    if api_keys:
        try:
            parsed_api_keys = json.loads(api_keys)
            if not isinstance(parsed_api_keys, list):
                parsed_api_keys = [parsed_api_keys]
        except json.JSONDecodeError:
            parsed_api_keys = [api_keys]  # Treat as single key

    try:
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)

        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Initialize session tracking
        PROCESSING_SESSIONS[session_id] = {
            "status": "initializing",
            "start_time": datetime.now().isoformat(),
            "file_name": file.filename,
            "file_size": len(content),
            "parameters": {
                "validation_level": validation_level,
                "cleaning_aggressive": cleaning_aggressive,
                "include_quality_scores": include_quality_scores,
                "batch_size": batch_size,
                "save_to_database": save_to_database,
                "detect_duplicates": detect_duplicates,
                "update_existing": update_existing,
                "export_report": export_report,
            },
        }

        # Process in background
        background_tasks.add_task(
            process_excel_file_enhanced,
            session_id,
            temp_file_path,
            validation_level,
            cleaning_aggressive,
            include_quality_scores,
            batch_size,
            save_to_database,
            detect_duplicates,
            update_existing,
            export_report,
            llm_provider,
            parsed_api_keys,
            temp_dir,
        )

        return JSONResponse(
            status_code=202,
            content={
                "message": "Excel file processing started",
                "session_id": session_id,
                "status": "processing",
                "check_status_url": f"/excel-parser/enhanced/status/{session_id}",
                "estimated_processing_time": "Processing time varies based on file size and complexity",
            },
        )

    except Exception as e:
        logger.error(f"Error initiating enhanced Excel processing: {e}")
        if session_id in PROCESSING_SESSIONS:
            PROCESSING_SESSIONS[session_id]["status"] = "failed"
            PROCESSING_SESSIONS[session_id]["error"] = str(e)

        raise HTTPException(
            status_code=500, detail=f"Processing initiation failed: {str(e)}"
        )


async def process_excel_file_enhanced(
    session_id: str,
    file_path: str,
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
        logger.info(f"Processing Excel file for session {session_id}")

        # Update session status
        PROCESSING_SESSIONS[session_id]["status"] = "processing"
        PROCESSING_SESSIONS[session_id]["progress"] = "Initializing enhanced parser"

        # Initialize enhanced parser
        parser = EnhancedExcelResumeParser(llm_provider=llm_provider, api_keys=api_keys)

        # Update progress
        PROCESSING_SESSIONS[session_id][
            "progress"
        ] = "Processing Excel file with enhanced capabilities"

        # Process the Excel file
        processing_result = parser.process_excel_file(
            file_path=file_path,
            validation_level=validation_level,
            cleaning_aggressive=cleaning_aggressive,
            include_quality_scores=include_quality_scores,
            batch_size=batch_size,
        )

        # Update session with initial results
        PROCESSING_SESSIONS[session_id]["processing_result"] = processing_result
        PROCESSING_SESSIONS[session_id]["progress"] = "Excel processing completed"

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

            parser.export_processing_report(report_path, processing_result)
            PROCESSING_SESSIONS[session_id]["report_path"] = report_path

        # Mark as completed
        PROCESSING_SESSIONS[session_id]["status"] = "completed"
        PROCESSING_SESSIONS[session_id]["end_time"] = datetime.now().isoformat()
        PROCESSING_SESSIONS[session_id][
            "progress"
        ] = "Processing completed successfully"

        # Add final summary
        PROCESSING_SESSIONS[session_id]["summary"] = {
            "total_rows_processed": processing_result.get("total_rows", 0),
            "successfully_parsed": processing_result.get("successfully_processed", 0),
            "failed_rows": len(processing_result.get("failed_rows", [])),
            "average_quality_score": processing_result.get(
                "processing_statistics", {}
            ).get("average_data_quality_score", 0),
            "processing_time_seconds": processing_result.get(
                "processing_statistics", {}
            ).get("total_processing_time_seconds", 0),
            "database_saves": (
                database_result.get("successfully_saved", 0) if database_result else 0
            ),
            "duplicates_found": (
                database_result.get("duplicates_found", 0) if database_result else 0
            ),
        }

        logger.info(f"Enhanced Excel processing completed for session {session_id}")

    except Exception as e:
        logger.error(
            f"Error in enhanced Excel processing for session {session_id}: {e}"
        )
        PROCESSING_SESSIONS[session_id]["status"] = "failed"
        PROCESSING_SESSIONS[session_id]["error"] = str(e)
        PROCESSING_SESSIONS[session_id]["end_time"] = datetime.now().isoformat()

    finally:
        # Clean up temporary file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Could not clean up temporary file {file_path}: {e}")


@router.get("/enhanced/status/{session_id}")
async def get_processing_status_enhanced(session_id: str):
    """
    Get the status of an enhanced Excel processing session.

    Args:
        session_id: Session ID from the upload request

    Returns:
        Current processing status and results
    """
    if session_id not in PROCESSING_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = PROCESSING_SESSIONS[session_id]

    response_data = {
        "session_id": session_id,
        "status": session_data["status"],
        "start_time": session_data["start_time"],
        "file_name": session_data["file_name"],
        "file_size": session_data["file_size"],
        "parameters": session_data["parameters"],
    }

    # Add progress information
    if "progress" in session_data:
        response_data["progress"] = session_data["progress"]

    if "end_time" in session_data:
        response_data["end_time"] = session_data["end_time"]

    # Add error information if failed
    if session_data["status"] == "failed":
        response_data["error"] = session_data.get("error", "Unknown error")

    # Add results if completed
    if session_data["status"] == "completed":
        response_data["summary"] = session_data.get("summary", {})

        # Add detailed results if available
        if "processing_result" in session_data:
            response_data["detailed_results"] = {
                "processing_statistics": session_data["processing_result"].get(
                    "processing_statistics", {}
                ),
                "data_quality_summary": session_data["processing_result"].get(
                    "data_quality_summary", {}
                ),
                "column_analysis": session_data["processing_result"].get(
                    "column_analysis", {}
                ),
                "failed_rows_count": len(
                    session_data["processing_result"].get("failed_rows", [])
                ),
            }

        if "database_result" in session_data:
            response_data["database_statistics"] = session_data["database_result"]

        if "report_path" in session_data:
            response_data["report_available"] = True
            response_data["report_download_url"] = (
                f"/excel-parser/enhanced/download-report/{session_id}"
            )

    return JSONResponse(content=response_data)


@router.get("/enhanced/download-report/{session_id}")
async def download_processing_report(session_id: str):
    """
    Download the processing report for a completed session.

    Args:
        session_id: Session ID from the upload request

    Returns:
        Processing report file
    """
    if session_id not in PROCESSING_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = PROCESSING_SESSIONS[session_id]

    if session_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")

    if "report_path" not in session_data or not os.path.exists(
        session_data["report_path"]
    ):
        raise HTTPException(status_code=404, detail="Report not found")

    try:
        with open(session_data["report_path"], "r", encoding="utf-8") as f:
            report_content = json.load(f)

        return JSONResponse(
            content=report_content,
            headers={
                "Content-Disposition": f"attachment; filename=processing_report_{session_id}.json"
            },
        )

    except Exception as e:
        logger.error(f"Error downloading report for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Error accessing report file")


@router.get("/enhanced/results/{session_id}")
async def get_detailed_results(
    session_id: str, include_failed_rows: bool = Query(False)
):
    """
    Get detailed processing results including parsed resumes.

    Args:
        session_id: Session ID from the upload request
        include_failed_rows: Whether to include details of failed rows

    Returns:
        Detailed processing results
    """
    if session_id not in PROCESSING_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = PROCESSING_SESSIONS[session_id]

    if session_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")

    if "processing_result" not in session_data:
        raise HTTPException(status_code=404, detail="Processing results not found")

    processing_result = session_data["processing_result"]

    detailed_results = {
        "session_info": {
            "session_id": session_id,
            "file_name": session_data["file_name"],
            "processing_parameters": session_data["parameters"],
            "start_time": session_data["start_time"],
            "end_time": session_data.get("end_time"),
        },
        "processing_summary": processing_result.get("processing_statistics", {}),
        "data_quality_analysis": processing_result.get("data_quality_summary", {}),
        "column_mapping_analysis": processing_result.get("column_analysis", {}),
        "parsed_resumes_count": len(processing_result.get("parsed_resumes", [])),
        "parsed_resumes": [
            {
                "resume_id": getattr(resume, "id", None),
                "name": getattr(resume, "name", None),
                "email": getattr(resume, "email", None),
                "phone": getattr(resume, "phone", None),
                "experience_years": getattr(resume, "experience_years", None),
                "current_role": getattr(resume, "current_role", None),
                "skills": getattr(resume, "skills", []),
                "processing_metadata": getattr(resume, "processing_metadata", {}),
            }
            for resume in processing_result.get("parsed_resumes", [])
        ],
    }

    # Add failed rows if requested
    if include_failed_rows:
        detailed_results["failed_rows"] = processing_result.get("failed_rows", [])
    else:
        detailed_results["failed_rows_count"] = len(
            processing_result.get("failed_rows", [])
        )

    # Add database results if available
    if "database_result" in session_data:
        detailed_results["database_operations"] = session_data["database_result"]

    return JSONResponse(content=detailed_results)


@router.delete("/enhanced/session/{session_id}")
async def cleanup_session(session_id: str):
    """
    Clean up a processing session and its temporary files.

    Args:
        session_id: Session ID to clean up

    Returns:
        Cleanup confirmation
    """
    if session_id not in PROCESSING_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        session_data = PROCESSING_SESSIONS[session_id]

        # Clean up report file if exists
        if "report_path" in session_data and os.path.exists(
            session_data["report_path"]
        ):
            os.remove(session_data["report_path"])

        # Remove session data
        del PROCESSING_SESSIONS[session_id]

        logger.info(f"Cleaned up session {session_id}")

        return JSONResponse(
            content={
                "message": "Session cleaned up successfully",
                "session_id": session_id,
            }
        )

    except Exception as e:
        logger.error(f"Error cleaning up session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")


@router.get("/enhanced/sessions")
async def list_active_sessions():
    """
    List all active processing sessions.

    Returns:
        List of active sessions with their status
    """
    sessions_info = []

    for session_id, session_data in PROCESSING_SESSIONS.items():
        session_info = {
            "session_id": session_id,
            "status": session_data["status"],
            "file_name": session_data["file_name"],
            "start_time": session_data["start_time"],
            "progress": session_data.get("progress", "Unknown"),
        }

        if "end_time" in session_data:
            session_info["end_time"] = session_data["end_time"]

        if "summary" in session_data:
            session_info["summary"] = session_data["summary"]

        sessions_info.append(session_info)

    return JSONResponse(
        content={"total_sessions": len(sessions_info), "sessions": sessions_info}
    )


@router.get("/enhanced/health")
async def health_check_enhanced():
    """Health check for enhanced Excel parser API."""
    try:
        # Test parser initialization
        parser = EnhancedExcelResumeParser()

        return JSONResponse(
            content={
                "status": "healthy",
                "service": "enhanced_excel_resume_parser",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "enhanced_parser": "operational",
                    "column_mapper": "operational",
                    "data_validator": "operational",
                    "text_formatter": "operational",
                    "field_extractor": "operational",
                },
                "active_sessions": len(PROCESSING_SESSIONS),
                "supported_features": [
                    "intelligent_column_mapping",
                    "data_validation_cleaning",
                    "quality_scoring",
                    "batch_processing",
                    "duplicate_detection",
                    "processing_reports",
                ],
            }
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            },
        )
