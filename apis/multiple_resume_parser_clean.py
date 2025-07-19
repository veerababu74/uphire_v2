from fastapi import APIRouter, HTTPException, File, UploadFile
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the multiple resume parser module
from multipleresumepraser.main import ResumeParser
from GroqcloudLLM.text_extraction import extract_and_clean_text, clean_text
from mangodatabase.operations import ResumeOperations, SkillsTitlesOperations
from mangodatabase.client import get_collection, get_skills_titles_collection
from embeddings.vectorizer import AddUserDataVectorizer
from schemas.add_user_schemas import ResumeData
from core.custom_logger import CustomLogger
from core.llm_config import LLMConfigManager, LLMProvider

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("multiple_resume_parser_api")

# Initialize database connections and operations
collection = get_collection()
skills_titles_collection = get_skills_titles_collection()
skills_ops = SkillsTitlesOperations(skills_titles_collection)
add_user_vectorizer = AddUserDataVectorizer()
resume_ops = ResumeOperations(collection, add_user_vectorizer)

# Initialize LLM config manager
llm_manager = LLMConfigManager()

# Create router
router = APIRouter()

# Directory configuration
BASE_FOLDER = "dummy_data_save"
TEMP_FOLDER = os.path.join(BASE_FOLDER, "temp_text_extract")
TEMP_DIR = Path(os.path.join(BASE_FOLDER, "temp_files"))

# Ensure directories exist
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_temp_directory(age_limit_minutes: int = 60):
    """Cleanup temporary directory by deleting old files."""
    now = datetime.now()
    for file_path in TEMP_DIR.iterdir():
        if file_path.is_file():
            file_age = now - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age > timedelta(minutes=age_limit_minutes):
                try:
                    file_path.unlink()
                    logger.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete file {file_path}: {e}")


def generate_embeddings_for_resume(parsed_data: dict) -> dict:
    """
    Generate embeddings for all relevant resume fields.

    Args:
        parsed_data (dict): Parsed resume data

    Returns:
        dict: Resume data with generated embeddings
    """
    try:
        logger.info("Generating embeddings for resume data")

        # Generate embeddings for text fields
        embeddings = {}

        # Skills embeddings
        skills = parsed_data.get("skills", [])
        if skills and isinstance(skills, list):
            skills_text = " ".join([str(skill) for skill in skills if skill])
            if skills_text.strip():
                embeddings["skills_embedding"] = add_user_vectorizer.generate_embedding(
                    skills_text
                )

        # Experience embeddings
        experience = parsed_data.get("experience", [])
        if experience and isinstance(experience, list):
            experience_texts = []
            for exp in experience:
                if isinstance(exp, dict):
                    title = exp.get("title", "")
                    company = exp.get("company", "")
                    if title or company:
                        experience_texts.append(f"{title} at {company}")

            if experience_texts:
                experience_text = " ".join(experience_texts)
                embeddings["experience_embedding"] = (
                    add_user_vectorizer.generate_embedding(experience_text)
                )

        # Education embeddings
        education = parsed_data.get("academic_details", [])
        if education and isinstance(education, list):
            education_texts = []
            for edu in education:
                if isinstance(edu, dict):
                    edu_text = (
                        f"{edu.get('education', '')} from {edu.get('college', '')}"
                    )
                    education_texts.append(edu_text)
            if education_texts:
                education_text = " ".join(education_texts)
                embeddings["education_embedding"] = (
                    add_user_vectorizer.generate_embedding(education_text)
                )

        # Combined profile embedding
        profile_parts = []
        if skills:
            profile_parts.append(" ".join([str(skill) for skill in skills if skill]))

        contact_details = parsed_data.get("contact_details", {})
        if isinstance(contact_details, dict):
            total_exp = parsed_data.get("total_experience", "")
            if total_exp:
                profile_parts.append(f"Experience: {total_exp}")
            current_city = contact_details.get("current_city", "")
            if current_city:
                profile_parts.append(f"Location: {current_city}")

        if profile_parts:
            profile_text = " ".join(profile_parts)
            embeddings["profile_embedding"] = add_user_vectorizer.generate_embedding(
                profile_text
            )

        # Add embeddings to the parsed data
        if embeddings:
            parsed_data["embeddings"] = embeddings
            parsed_data["embeddings_generated_at"] = datetime.now().isoformat()
            logger.info(f"Generated {len(embeddings)} embedding types for resume")

        return parsed_data

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        # Return original data without embeddings if generation fails
        return parsed_data


def process_multiple_resumes_with_embeddings(
    files_data: List[Dict[str, Any]], llm_provider: str = None, max_concurrent: int = 3
) -> List[Dict[str, Any]]:
    """
    Process multiple resume files using the multiple resume parser and generate embeddings.

    Args:
        files_data: List of file data dictionaries
        llm_provider: Optional LLM provider to use
        max_concurrent: Maximum concurrent processing tasks

    Returns:
        List[Dict[str, Any]]: Processed resume data with embeddings
    """
    results = []

    def process_single_file(file_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            filename = file_data["filename"]
            content = file_data["content"]

            logger.info(f"Processing resume: {filename}")

            # Cleanup temp directory
            cleanup_temp_directory()

            # Save file temporarily
            temp_file_path = TEMP_DIR / filename
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(content)

            # Extract text from file
            total_resume_text = extract_and_clean_text(str(temp_file_path))

            if not total_resume_text.strip():
                return {
                    "filename": filename,
                    "status": "error",
                    "error": "Could not extract text from file",
                    "parsed_data": None,
                }

            # Clean the extracted text
            total_resume_text = clean_text(total_resume_text)

            # Initialize the multiple resume parser
            try:
                if llm_provider:
                    parser = ResumeParser(llm_provider=llm_provider)
                else:
                    parser = ResumeParser()

                logger.info(
                    f"Initialized parser with provider: {parser.provider.value}"
                )

            except Exception as e:
                logger.error(f"Failed to initialize parser: {e}")
                parser = ResumeParser()  # Use default

            # Process the resume
            parsed_data = parser.process_resume(total_resume_text)

            # Check if parsing was successful
            if "error" in parsed_data:
                logger.warning(f"Resume parsing had issues: {parsed_data.get('error')}")

            # Generate embeddings for the parsed data
            parsed_data_with_embeddings = generate_embeddings_for_resume(parsed_data)

            # Add metadata
            parsed_data_with_embeddings.update(
                {
                    "filename": filename,
                    "processing_timestamp": datetime.now().isoformat(),
                    "llm_provider_used": parser.provider.value,
                    "text_length": len(total_resume_text),
                }
            )

            # Clean up temp file
            try:
                temp_file_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete temp file {temp_file_path}: {e}")

            logger.info(f"Successfully processed {filename}")

            # Create response data without embeddings for user
            response_data = parsed_data_with_embeddings.copy()
            if "embeddings" in response_data:
                del response_data["embeddings"]

            return {
                "filename": filename,
                "status": "success",
                "parsed_data": response_data,
                "llm_provider": parser.provider.value,
                "embeddings_generated": "embeddings" in parsed_data_with_embeddings,
            }

        except Exception as e:
            logger.error(f"Error processing resume {filename}: {str(e)}")
            return {
                "filename": filename,
                "status": "error",
                "error": str(e),
                "parsed_data": None,
            }

    # Process files concurrently
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, file_data): file_data["filename"]
            for file_data in files_data
        }

        # Collect results as they complete
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed processing: {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                results.append(
                    {
                        "filename": filename,
                        "status": "error",
                        "error": str(e),
                        "parsed_data": None,
                    }
                )

    return results


@router.post("/resume-parser-multiple")
async def parse_multiple_resumes(files: List[UploadFile] = File(...)):
    """
    Parse multiple resumes and automatically save them to the database.

    This endpoint:
    - Accepts 1 or more resume files
    - Uses multipleresumepraser module for parsing
    - Generates embeddings automatically
    - Saves parsed data to database automatically
    - Returns summary of processed files

    Args:
        files: List of resume files to process (1 or more required)
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        if len(files) > 20:  # Limit for safety
            raise HTTPException(
                status_code=400, detail="Too many files. Maximum 20 files allowed."
            )

        logger.info(f"Processing {len(files)} resumes")

        # Prepare file data
        files_data = []
        for file in files:
            content = await file.read()
            files_data.append({"filename": file.filename, "content": content})

        # Process files with default settings (no query parameters)
        processing_results = process_multiple_resumes_with_embeddings(
            files_data=files_data,
            llm_provider=None,  # Use default
            max_concurrent=3,  # Use default
        )

        # Automatically save all successfully parsed resumes to database
        successful_parses = [
            result["parsed_data"]
            for result in processing_results
            if result["status"] == "success" and result["parsed_data"]
        ]

        # For database saving, we need the full data with embeddings
        # We'll get it from the original processing results
        full_data_for_db = []
        for result in processing_results:
            if result["status"] == "success" and result["parsed_data"]:
                # Get the original parsed data with embeddings from the processing function
                filename = result["filename"]
                # Re-generate the full data with embeddings for database saving
                parsed_data = result["parsed_data"].copy()

                # Add back embeddings if they were generated
                if result.get("embeddings_generated"):
                    # We need to regenerate embeddings for database saving
                    # since we removed them from the response
                    parsed_data = generate_embeddings_for_resume(parsed_data)

                full_data_for_db.append(parsed_data)

        database_results = []
        if full_data_for_db:
            # Save to database
            for i, parsed_data in enumerate(full_data_for_db):
                try:
                    # Auto-generate required fields
                    if "user_id" not in parsed_data or not parsed_data["user_id"]:
                        parsed_data["user_id"] = (
                            f"auto_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                        )
                    if "username" not in parsed_data or not parsed_data["username"]:
                        parsed_data["username"] = parsed_data["user_id"]

                    # Ensure contact details structure
                    if "contact_details" not in parsed_data:
                        parsed_data["contact_details"] = {}

                    contact = parsed_data["contact_details"]
                    if "name" not in contact:
                        contact["name"] = parsed_data.get("name", f"User_{i}")
                    if "email" not in contact:
                        contact["email"] = f"user{i}@example.com"
                    if "phone" not in contact:
                        contact["phone"] = "+1-000-000-0000"
                    if "current_city" not in contact:
                        contact["current_city"] = "Unknown"
                    if "looking_for_jobs_in" not in contact:
                        contact["looking_for_jobs_in"] = []
                    if "pan_card" not in contact:
                        contact["pan_card"] = "UNKNOWN"

                    # Save to database
                    result = resume_ops.create_resume(parsed_data)

                    database_results.append(
                        {
                            "filename": parsed_data.get("filename"),
                            "database_id": str(result.get("id")),
                            "user_id": parsed_data["user_id"],
                            "status": "saved",
                            "embeddings_saved": "embeddings" in parsed_data,
                        }
                    )

                    logger.info(f"Saved resume to database with ID: {result.get('id')}")

                except Exception as e:
                    logger.error(f"Error saving resume to database: {e}")
                    database_results.append(
                        {
                            "filename": parsed_data.get("filename"),
                            "status": "save_failed",
                            "error": str(e),
                        }
                    )

        # Calculate final statistics
        total_files = len(processing_results)
        successful_parses_count = len(
            [r for r in processing_results if r["status"] == "success"]
        )
        failed_parses = total_files - successful_parses_count
        saved_to_database = len([r for r in database_results if r["status"] == "saved"])
        embeddings_generated = len(
            [
                r
                for r in processing_results
                if r["status"] == "success" and "embeddings" in r.get("parsed_data", {})
            ]
        )

        return {
            "message": f"Processed {total_files} resumes and saved {saved_to_database} to database",
            "summary": {
                "total_files_uploaded": total_files,
                "successful_parses": successful_parses_count,
                "failed_parses": failed_parses,
                "saved_to_database": saved_to_database,
                "embeddings_generated": embeddings_generated,
            },
            "processing_details": processing_results,
            "database_results": database_results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in parse_multiple_resumes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/info")
async def get_multiple_resume_parser_info():
    """
    Get information about the multiple resume parser endpoint.
    """
    return {
        "endpoint": {
            "url": "/parse-multiple-resumes",
            "method": "POST",
            "description": "Upload 1 or more resume files for automatic parsing and database storage",
            "max_files": 20,
        },
        "features": {
            "parser_module": "multipleresumepraser",
            "embeddings_generation": True,
            "automatic_database_saving": True,
            "concurrent_processing": True,
            "no_query_parameters_required": True,
        },
        "supported_formats": [".txt", ".pdf", ".docx"],
        "workflow": [
            "1. Upload resume files",
            "2. Extract and clean text",
            "3. Parse resume data using LLM",
            "4. Generate embeddings",
            "5. Automatically save to database",
            "6. Return processing summary",
        ],
    }
