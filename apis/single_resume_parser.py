from fastapi import APIRouter, HTTPException
import os
from fastapi import File, UploadFile
from pathlib import Path
from datetime import datetime, timedelta
from GroqcloudLLM.text_extraction import extract_and_clean_text
from GroqcloudLLM.main import ResumeParser
from Expericecal.total_exp import format_experience, calculator
from core.custom_logger import CustomLogger
from typing import List, Dict, Any
import json
import re


def clean_and_extract_skills(skills_input: List[str]) -> List[str]:
    """
    Extract and clean individual skills from various input formats

    Handles scenarios like:
    - "python,machine learning,"
    - "technical skills:python,sql,opencv,azure cloud"
    - "Python | React | AWS"
    - "Skills: Java, Spring Boot, Microservices"
    - Mixed formats in single list
    """

    if not skills_input:
        return []

    all_skills = []

    for skill_item in skills_input:
        if not skill_item or not isinstance(skill_item, str):
            continue

        # Convert to lowercase for processing
        skill_text = skill_item.strip()

        # Remove common prefixes
        prefixes_to_remove = [
            r"^technical\s*skills?\s*:?\s*",
            r"^skills?\s*:?\s*",
            r"^technologies?\s*:?\s*",
            r"^expertise\s*:?\s*",
            r"^programming\s*:?\s*",
            r"^tools?\s*:?\s*",
        ]

        for prefix in prefixes_to_remove:
            skill_text = re.sub(prefix, "", skill_text, flags=re.IGNORECASE)

        # Split by various delimiters
        delimiters = [",", "|", ";", "/", "\\n", "\\r", "\n", "\r"]

        # Create regex pattern for splitting
        delimiter_pattern = "|".join(map(re.escape, delimiters))
        skills_parts = re.split(delimiter_pattern, skill_text)

        # Process each skill part
        for skill in skills_parts:
            cleaned_skill = clean_individual_skill(skill)
            if cleaned_skill:
                all_skills.append(cleaned_skill)

    # Remove duplicates while preserving order
    unique_skills = []
    seen = set()
    for skill in all_skills:
        skill_lower = skill.lower()
        if skill_lower not in seen:
            seen.add(skill_lower)
            unique_skills.append(skill)

    return unique_skills


def clean_individual_skill(skill: str) -> str:
    """Clean and normalize individual skill"""
    if not skill:
        return ""

    # Remove extra whitespace and common unwanted characters
    skill = skill.strip(" \t\n\r,;|/")

    # Remove numbers at the beginning (like "1. Python")
    skill = re.sub(r"^\d+\.?\s*", "", skill)

    # Remove brackets and parentheses content
    skill = re.sub(r"\([^)]*\)", "", skill)
    skill = re.sub(r"\[[^\]]*\]", "", skill)

    # Remove extra spaces
    skill = re.sub(r"\s+", " ", skill).strip()

    # Skip if too short or contains only special characters
    if len(skill) < 2 or re.match(r"^[^a-zA-Z0-9]*$", skill):
        return ""

    return skill


def process_user_json(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process the entire user JSON and clean skills"""

    # Create a copy to avoid modifying original
    processed_data = json_data.copy()

    # Process main skills
    if "skills" in processed_data:
        processed_data["skills"] = clean_and_extract_skills(processed_data["skills"])

    # Process may_also_known_skills
    if "may_also_known_skills" in processed_data:
        processed_data["may_also_known_skills"] = clean_and_extract_skills(
            processed_data["may_also_known_skills"]
        )

    return processed_data


# Initialize the GroqcloudLLM parser (single resume parsing only)
parser = ResumeParser()

# Create a router instance
router = APIRouter()

BASE_FOLDER = "dummy_data_save"
TEMP_FOLDER = os.path.join(BASE_FOLDER, "temp_text_extract")
TEMP_DIR = Path(os.path.join(BASE_FOLDER, "temp_files"))

# Ensure the directories exist
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

logger_manager = CustomLogger()
logging = logger_manager.get_logger("single_resume_parser")


def cleanup_temp_directory(age_limit_minutes: int = 60):
    """
    Cleanup temporary directory by deleting files older than the specified age limit.
    """
    now = datetime.now()
    for file_path in TEMP_DIR.iterdir():
        if file_path.is_file():
            file_age = now - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age > timedelta(minutes=age_limit_minutes):
                try:
                    file_path.unlink()
                    logging.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logging.error(f"Failed to delete file {file_path}: {e}")


@router.post(
    "/resume-parser",
    tags=[
        "Single Resume Parser",
    ],
)
async def upload_resume(file: UploadFile = File(...)):
    """
    Single Resume Parser - Extract and parse text from uploaded file using GroqcloudLLM.

    This endpoint:
    - Uses GroqcloudLLM module for parsing
    - Does NOT generate embeddings
    - Does NOT save to database
    - Returns only parsed resume data
    """
    file_location = None
    try:
        # Clean up old temp files
        cleanup_temp_directory()

        # Define the path to save the uploaded file
        file_location = os.path.join(TEMP_FOLDER, file.filename)

        # Save the uploaded file
        with open(file_location, "wb") as temp_file:
            temp_file.write(await file.read())

        # Extract text
        _, file_extension = os.path.splitext(file.filename)
        if file_extension.lower() not in [".txt", ".pdf", ".docx"]:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Only .txt, .pdf, and .docx are supported.",
            )

        # Extract and save the total resume text
        total_resume_text = extract_and_clean_text(file_location)
        logging.info(
            f"Extracted text from {file.filename}, length: {len(total_resume_text)}"
        )

        # Process the resume using GroqcloudLLM
        resume_parser = parser.process_resume(total_resume_text)

        if not resume_parser:
            raise HTTPException(status_code=400, detail="No resume data found.")

        # Initialize total_experience if not present
        if "total_experience" not in resume_parser:
            resume_parser["total_experience"] = 0

        # Calculate experience if needed
        # res = calculator.calculate_experience(resume_parser)
        # resume_parser["total_experience"] = format_experience(res[0], res[1])

        # Process and clean the JSON data
        resume_parser = process_user_json(resume_parser)

        logging.info(f"Successfully parsed resume: {file.filename}")

        return {
            "message": "Resume parsed successfully using GroqcloudLLM",
            "filename": file.filename,
            "total_resume_text": total_resume_text,
            "resume_parser": resume_parser,
            "parser_used": "GroqcloudLLM",
            "embeddings_generated": False,
            "saved_to_database": False,
        }

    except HTTPException as http_ex:
        logging.info(f"HTTPException in single resume parser: {http_ex.detail}")
        raise http_ex

    except Exception as e:
        import traceback

        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e) if str(e) else "Unknown error occurred",
            "traceback": traceback.format_exc(),
        }
        logging.error(f"Error in single resume parser: {error_details}")

        detail_message = f"Error processing file: {error_details['error_type']}: {error_details['error_message']}"
        if not error_details["error_message"]:
            detail_message = f"Error processing file: {error_details['error_type']} - Check logs for details"

        raise HTTPException(status_code=500, detail=detail_message)

    finally:
        # Always cleanup the temp file
        if file_location and os.path.exists(file_location):
            try:
                os.remove(file_location)
                logging.debug(f"Cleaned up temp file: {file_location}")
            except Exception as cleanup_error:
                logging.warning(
                    f"Failed to cleanup temp file {file_location}: {cleanup_error}"
                )


@router.get("/resume-parser-info", tags=["Single Resume Parser"])
async def get_single_resume_parser_info():
    """
    Get information about the single resume parser endpoint.
    """
    return {
        "endpoint": "/resume-parser",
        "method": "POST",
        "description": "Single Resume Parser using GroqcloudLLM",
        "features": {
            "parser_module": "GroqcloudLLM",
            "embeddings_generation": False,
            "database_saving": False,
            "use_case": "Quick resume parsing for testing and single file processing",
        },
        "supported_formats": [".txt", ".pdf", ".docx"],
        "max_files": 1,
        "response_includes": [
            "parsed_resume_data",
            "extracted_text",
            "processing_metadata",
        ],
        "response_excludes": ["embeddings", "database_id", "vector_data"],
    }
