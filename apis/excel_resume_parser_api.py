"""
Excel Resume Parser API

API endpoints for processing Excel/XLSX files containing resume data.
Handles file upload, preprocessing, resume parsin    except Exception as e:
        logger.error(f"Error processing Excel file for session {session_id}: {e}")
        update_excel_processing_queue("remove_from_queue", session_id, 1)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")detection.
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Query
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
import tempfile

from excel_resume_parser.main import ExcelResumeParserManager
from core.custom_logger import CustomLogger
from core.llm_config import LLMConfigManager, LLMProvider
from core.config import AppConfig

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("excel_resume_parser_api")

# Initialize LLM config manager
llm_manager = LLMConfigManager()

# Create router
router = APIRouter()

# Global tracking for processing
EXCEL_PROCESSING_QUEUE = {
    "current_queue_size": 0,
    "total_processed_today": 0,
    "active_sessions": {},
}


def update_excel_processing_queue(action: str, session_id: str = None, count: int = 0):
    """Update the global Excel processing queue statistics."""
    global EXCEL_PROCESSING_QUEUE

    if action == "add_to_queue":
        EXCEL_PROCESSING_QUEUE["current_queue_size"] += count
        if session_id:
            EXCEL_PROCESSING_QUEUE["active_sessions"][session_id] = {
                "start_time": time.time(),
                "count": count,
                "status": "processing",
            }
    elif action == "remove_from_queue":
        EXCEL_PROCESSING_QUEUE["current_queue_size"] = max(
            0, EXCEL_PROCESSING_QUEUE["current_queue_size"] - count
        )
        EXCEL_PROCESSING_QUEUE["total_processed_today"] += count
        if session_id and session_id in EXCEL_PROCESSING_QUEUE["active_sessions"]:
            del EXCEL_PROCESSING_QUEUE["active_sessions"][session_id]
    elif action == "complete_session":
        if session_id and session_id in EXCEL_PROCESSING_QUEUE["active_sessions"]:
            EXCEL_PROCESSING_QUEUE["active_sessions"][session_id][
                "status"
            ] = "completed"


@router.post("/excel-resume-parser/upload")
async def upload_excel_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    username: str = Form(...),
    sheet_name: Optional[str] = Form(None),
):
    """
    Upload and process an Excel file containing resume data.

    Args:
        file: Excel file to upload (.xlsx, .xls, .xlsm)
        user_id: Base user ID for generated resumes
        username: Base username for generated resumes
        sheet_name: Specific sheet to process (optional, uses first sheet if not provided)

    Returns:
        Processing results including parsed resumes and database operations

    Note:
        llm_provider is auto-detected from system configuration
        save_temp_file is automatically set to False for security
    """
    session_id = f"excel_{user_id}_{int(time.time())}"
    temp_file_path = None

    try:
        logger.info(f"Starting Excel resume processing for session {session_id}")

        # Auto-detect LLM provider from system configuration
        llm_provider = AppConfig.LLM_PROVIDER
        logger.info(f"Auto-detected LLM provider: {llm_provider}")

        # Set save_temp_file to False for security (files are processed in memory)
        save_temp_file = False
        logger.info("save_temp_file automatically set to False for security")

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

        # Initialize Excel Resume Parser Manager with auto-detected LLM provider
        parser_manager = ExcelResumeParserManager(llm_provider=llm_provider)

        # Add to processing queue
        update_excel_processing_queue("add_to_queue", session_id, 1)

        # Process the Excel file
        logger.info(f"Processing Excel file: {file.filename}")
        results = parser_manager.process_excel_file_from_bytes(
            file_bytes=file_content,
            filename=file.filename,
            base_user_id=user_id,
            base_username=username,
            sheet_name=sheet_name,
            save_temp_file=save_temp_file,
        )

        # Update processing queue
        update_excel_processing_queue("complete_session", session_id)
        update_excel_processing_queue("remove_from_queue", session_id, 1)

        # Add session info to results
        results["session_id"] = session_id
        results["file_info"] = {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(file_content),
            "temp_file_saved": False,  # Always False for security
            "temp_file_path": None,
            "llm_provider": llm_provider,  # Include auto-detected LLM provider
        }

        logger.info(f"Excel processing completed for session {session_id}")
        return results

    except HTTPException:
        # Re-raise HTTP exceptions
        update_excel_processing_queue("remove_from_queue", session_id, 1)
        raise
    except Exception as e:
        logger.error(f"Error processing Excel file for session {session_id}: {e}")
        update_excel_processing_queue("remove_from_queue", session_id, 1)

        # Cleanup temporary file on general exception
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
                logger.info(
                    f"Cleaned up temporary file after exception: {temp_file_path}"
                )
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to cleanup temp file after exception: {cleanup_error}"
                )

        raise HTTPException(
            status_code=500, detail=f"Error processing Excel file: {str(e)}"
        )


@router.post("/excel-resume-parser/analyze")
async def analyze_excel_file(
    file: UploadFile = File(...),
):
    """
    Analyze an Excel file to get structure information without processing resumes.

    Args:
        file: Excel file to analyze

    Returns:
        Excel file structure information (sheets, columns, sample data)
    """
    try:
        logger.info(f"Analyzing Excel file: {file.filename}")

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

        # Initialize Excel Resume Parser Manager
        parser_manager = ExcelResumeParserManager()

        # Get Excel file information
        excel_info = parser_manager.get_excel_info(
            file_bytes=file_content, filename=file.filename
        )

        # Add file metadata
        excel_info["file_info"] = {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(file_content),
        }

        logger.info(f"Excel analysis completed for: {file.filename}")
        return excel_info

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error analyzing Excel file {file.filename}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error analyzing Excel file: {str(e)}"
        )


@router.get("/excel-resume-parser/queue-status")
async def get_queue_status():
    """
    Get current processing queue status for Excel resume parser.

    Returns:
        Current queue statistics and active sessions
    """
    try:
        return {
            "status": "success",
            "queue_info": EXCEL_PROCESSING_QUEUE,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting queue status: {str(e)}"
        )


@router.get("/excel-resume-parser/statistics")
async def get_processing_statistics():
    """
    Get Excel resume parser statistics and configuration.

    Returns:
        Processing statistics and system configuration
    """
    try:
        # Initialize parser manager to get statistics
        parser_manager = ExcelResumeParserManager()
        stats = parser_manager.get_processing_statistics()

        # Add queue information
        stats["queue_statistics"] = EXCEL_PROCESSING_QUEUE
        stats["timestamp"] = datetime.now().isoformat()

        return stats

    except Exception as e:
        logger.error(f"Error getting processing statistics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting processing statistics: {str(e)}"
        )


@router.post("/excel-resume-parser/cleanup-temp")
async def cleanup_temp_files(
    age_limit_minutes: int = Query(
        60, description="Age limit in minutes for file cleanup"
    )
):
    """
    Cleanup temporary files created during Excel processing.

    Args:
        age_limit_minutes: Age limit in minutes for file cleanup

    Returns:
        Cleanup operation results
    """
    try:
        logger.info(
            f"Starting temp file cleanup with age limit: {age_limit_minutes} minutes"
        )

        # Initialize parser manager
        parser_manager = ExcelResumeParserManager()

        # Perform cleanup
        parser_manager.cleanup_temp_files(age_limit_minutes)

        return {
            "status": "success",
            "message": f"Temp file cleanup completed (age limit: {age_limit_minutes} minutes)",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error during temp file cleanup: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error during temp file cleanup: {str(e)}"
        )


@router.get("/excel-resume-parser/supported-formats")
async def get_supported_formats():
    """
    Get information about supported Excel formats and requirements.

    Returns:
        Supported formats and processing requirements
    """
    return {
        "status": "success",
        "supported_formats": [
            {
                "extension": ".xlsx",
                "description": "Excel 2007+ format (recommended)",
                "max_size_mb": 50,
            },
            {
                "extension": ".xls",
                "description": "Excel 97-2003 format",
                "max_size_mb": 50,
            },
            {
                "extension": ".xlsm",
                "description": "Excel macro-enabled format",
                "max_size_mb": 50,
            },
        ],
        "processing_features": [
            "Automatic duplicate header detection and removal",
            "Column name standardization",
            "Row-wise resume data extraction",
            "Integration with existing resume parser",
            "Duplicate detection across resumes",
            "Batch processing support",
        ],
        "requirements": {
            "columns": "No specific column names required - flexible mapping",
            "data_format": "Each row should represent one resume/candidate",
            "sheet_selection": "Specific sheet can be selected or first sheet used by default",
        },
        "timestamp": datetime.now().isoformat(),
    }
