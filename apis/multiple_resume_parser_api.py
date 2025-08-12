from fastapi import APIRouter, HTTPException, File, UploadFile, Form
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the multiple resume parser module
from GroqcloudLLM.main import ResumeParser
from GroqcloudLLM.text_extraction import extract_and_clean_text, clean_text
from mangodatabase.operations import ResumeOperations, SkillsTitlesOperations
from mangodatabase.client import (
    get_collection,
    get_skills_titles_collection,
    get_resume_extracted_text_collection,
)
from mangodatabase.duplicate_detection import DuplicateDetectionOperations
from embeddings.vectorizer import AddUserDataVectorizer
from schemas.add_user_schemas import ResumeData
from schemas.multiple_resume_schemas import MultipleResumeUploadRequest
from core.custom_logger import CustomLogger
from core.llm_config import LLMConfigManager, LLMProvider

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("multiple_resume_parser_api")

# Initialize database connections and operations
collection = get_collection()
skills_titles_collection = get_skills_titles_collection()
extracted_text_collection = get_resume_extracted_text_collection()
skills_ops = SkillsTitlesOperations(skills_titles_collection)
add_user_vectorizer = AddUserDataVectorizer()
resume_ops = ResumeOperations(collection, add_user_vectorizer)
duplicate_ops = DuplicateDetectionOperations(extracted_text_collection)

# Initialize LLM config manager
llm_manager = LLMConfigManager()

# Create router
router = APIRouter()

# Global queue tracking for concurrent processing
PROCESSING_QUEUE = {
    "current_queue_size": 0,
    "total_processed_today": 0,
    "active_sessions": {},
}

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


def update_processing_queue(action: str, session_id: str = None, count: int = 0):
    """Update the global processing queue statistics."""
    global PROCESSING_QUEUE

    if action == "add_to_queue":
        PROCESSING_QUEUE["current_queue_size"] += count
        if session_id:
            PROCESSING_QUEUE["active_sessions"][session_id] = {
                "start_time": time.time(),
                "count": count,
                "status": "processing",
            }
    elif action == "remove_from_queue":
        PROCESSING_QUEUE["current_queue_size"] = max(
            0, PROCESSING_QUEUE["current_queue_size"] - count
        )
        PROCESSING_QUEUE["total_processed_today"] += count
        if session_id and session_id in PROCESSING_QUEUE["active_sessions"]:
            del PROCESSING_QUEUE["active_sessions"][session_id]
    elif action == "complete_session":
        if session_id and session_id in PROCESSING_QUEUE["active_sessions"]:
            PROCESSING_QUEUE["active_sessions"][session_id]["status"] = "completed"


def get_queue_status():
    """Get current queue status information."""
    active_sessions = len(
        [
            s
            for s in PROCESSING_QUEUE["active_sessions"].values()
            if s["status"] == "processing"
        ]
    )
    return {
        "current_queue_size": PROCESSING_QUEUE["current_queue_size"],
        "active_processing_sessions": active_sessions,
        "total_processed_today": PROCESSING_QUEUE["total_processed_today"],
        "queue_status": (
            "busy" if PROCESSING_QUEUE["current_queue_size"] > 0 else "available"
        ),
    }


def calculate_processing_time(start_time: float, end_time: float) -> dict:
    """Calculate processing time statistics."""
    total_seconds = end_time - start_time
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    return {
        "total_seconds": round(total_seconds, 2),
        "formatted_time": f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s",
        "start_time": datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
    }


def normalize_text_data(text: str) -> str:
    """
    Normalize text data by converting to lowercase and removing extra spaces.
    """
    if not text or not isinstance(text, str):
        return text
    return text.strip().lower()


def normalize_text_list(text_list: List[str]) -> List[str]:
    """
    Normalize a list of text strings by converting to lowercase and removing extra spaces.
    """
    if not text_list or not isinstance(text_list, list):
        return text_list
    return [normalize_text_data(text) for text in text_list if text]


def filter_empty_strings_from_list(text_list: List[str]) -> List[str]:
    """
    Filter out empty strings and whitespace-only strings from a list.
    """
    if not text_list or not isinstance(text_list, list):
        return []
    return [
        text.strip()
        for text in text_list
        if text and isinstance(text, str) and text.strip()
    ]


def clean_string_field(value) -> str:
    """
    Clean a string field by returning empty string if None or whitespace-only.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else ""
    return str(value) if value else ""


def clean_resume_data(
    resume_dict: dict, user_id: str = None, username: str = None
) -> dict:
    """
    Clean and normalize resume data following the same pattern as add_userdata.py
    """
    # Ensure contact_details exists and has required fields
    if "contact_details" not in resume_dict:
        resume_dict["contact_details"] = {}

    contact = resume_dict["contact_details"]

    # Clean and ensure required contact fields
    contact["name"] = clean_string_field(contact.get("name")) or "Unknown"
    contact["email"] = clean_string_field(contact.get("email")) or "unknown@example.com"
    contact["phone"] = clean_string_field(contact.get("phone")) or "+1-000-000-0000"
    contact["current_city"] = (
        clean_string_field(contact.get("current_city")) or "Unknown"
    )
    contact["pan_card"] = clean_string_field(contact.get("pan_card")) or "UNKNOWN"

    # Optional contact fields
    contact["alternative_phone"] = clean_string_field(contact.get("alternative_phone"))
    contact["gender"] = clean_string_field(contact.get("gender"))
    contact["naukri_profile"] = clean_string_field(contact.get("naukri_profile"))
    contact["linkedin_profile"] = clean_string_field(contact.get("linkedin_profile"))
    contact["portfolio_link"] = clean_string_field(contact.get("portfolio_link"))
    contact["aadhar_card"] = clean_string_field(contact.get("aadhar_card"))

    # Clean looking_for_jobs_in list
    if "looking_for_jobs_in" in contact:
        contact["looking_for_jobs_in"] = filter_empty_strings_from_list(
            contact["looking_for_jobs_in"]
        )
    else:
        contact["looking_for_jobs_in"] = []

    # Ensure required main fields - use provided user_id and username if available
    if user_id:
        resume_dict["user_id"] = user_id
    else:
        resume_dict["user_id"] = (
            clean_string_field(resume_dict.get("user_id"))
            or f"auto_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    if username:
        resume_dict["username"] = username
    else:
        resume_dict["username"] = (
            clean_string_field(resume_dict.get("username")) or resume_dict["user_id"]
        )

    # Optional main fields
    resume_dict["notice_period"] = clean_string_field(resume_dict.get("notice_period"))
    resume_dict["currency"] = clean_string_field(resume_dict.get("currency"))
    resume_dict["pay_duration"] = clean_string_field(resume_dict.get("pay_duration"))
    resume_dict["source"] = clean_string_field(resume_dict.get("source"))
    resume_dict["last_working_day"] = clean_string_field(
        resume_dict.get("last_working_day")
    )
    resume_dict["comment"] = clean_string_field(resume_dict.get("comment"))
    resume_dict["exit_reason"] = clean_string_field(resume_dict.get("exit_reason"))

    # Clean skills lists
    if "skills" in resume_dict:
        resume_dict["skills"] = filter_empty_strings_from_list(resume_dict["skills"])
    else:
        resume_dict["skills"] = []

    if "may_also_known_skills" in resume_dict:
        resume_dict["may_also_known_skills"] = filter_empty_strings_from_list(
            resume_dict["may_also_known_skills"]
        )
    else:
        resume_dict["may_also_known_skills"] = []

    if "labels" in resume_dict and resume_dict["labels"]:
        resume_dict["labels"] = filter_empty_strings_from_list(resume_dict["labels"])
    else:
        resume_dict["labels"] = []

    # Clean experience data
    if "experience" in resume_dict and resume_dict["experience"]:
        for exp in resume_dict["experience"]:
            exp["company"] = clean_string_field(exp.get("company")) or "Unknown Company"
            exp["title"] = clean_string_field(exp.get("title")) or "Unknown Position"
            exp["from_date"] = clean_string_field(exp.get("from_date")) or "Unknown"
            exp["to"] = clean_string_field(exp.get("to"))
            if "until" in exp:
                exp["until"] = clean_string_field(exp.get("until"))
            # Normalize job titles
            if exp["title"]:
                exp["title"] = normalize_text_data(exp["title"])
    else:
        resume_dict["experience"] = []

    # Clean academic details
    if "academic_details" in resume_dict and resume_dict["academic_details"]:
        for edu in resume_dict["academic_details"]:
            edu["education"] = clean_string_field(edu.get("education")) or "Unknown"
            edu["college"] = clean_string_field(edu.get("college")) or "Unknown"
    else:
        resume_dict["academic_details"] = []

    # Convert contact details URLs to strings (only if not empty/None) - like add_userdata.py
    contact = resume_dict["contact_details"]
    if contact.get("naukri_profile"):
        contact["naukri_profile"] = str(contact["naukri_profile"])
    if contact.get("linkedin_profile"):
        contact["linkedin_profile"] = str(contact["linkedin_profile"])
    if contact.get("portfolio_link"):
        contact["portfolio_link"] = str(contact["portfolio_link"])

    # Normalize skills
    if resume_dict["skills"]:
        resume_dict["skills"] = normalize_text_list(resume_dict["skills"])
    if resume_dict["may_also_known_skills"]:
        resume_dict["may_also_known_skills"] = normalize_text_list(
            resume_dict["may_also_known_skills"]
        )

    return resume_dict


def generate_embeddings_for_resume(
    parsed_data: dict, user_id: str = None, username: str = None
) -> dict:
    """
    Generate embeddings for all relevant resume fields using the proper vectorizer.
    """
    try:
        logger.info("Generating embeddings for resume data")

        # Clean and normalize the data first
        cleaned_data = clean_resume_data(parsed_data, user_id, username)

        # Add created_at timestamp (using UTC timezone like add_userdata.py)
        cleaned_data["created_at"] = datetime.now(timezone.utc)

        # Generate combined_resume text like add_userdata.py does
        contact_details = cleaned_data.get("contact_details", {})

        combined_resume = f"""
RESUME

PERSONAL INFORMATION
-------------------
Name: {contact_details.get('name', 'N/A')}
Contact Details:
  Email: {contact_details.get('email', 'N/A')}
  Phone: {contact_details.get('phone', 'N/A')}
  Alternative Phone: {contact_details.get('alternative_phone', 'N/A')}
  Current City: {contact_details.get('current_city', 'N/A')}
  Looking for jobs in: {', '.join(contact_details.get('looking_for_jobs_in', [])) if contact_details.get('looking_for_jobs_in') else 'N/A'}
  Gender: {contact_details.get('gender', 'N/A')}
  Age: {contact_details.get('age', 'N/A')}
  PAN Card: {contact_details.get('pan_card', 'N/A')}
  Aadhar Card: {contact_details.get('aadhar_card', 'N/A')}

PROFESSIONAL SUMMARY
-------------------
Total Experience: {cleaned_data.get('total_experience', 'N/A')} years
Notice Period: {cleaned_data.get('notice_period', 'N/A')} days
Current Salary: {cleaned_data.get('currency', 'N/A')} {cleaned_data.get('current_salary', 'N/A')} ({cleaned_data.get('pay_duration', 'N/A')})
Expected Salary: {cleaned_data.get('currency', 'N/A')} {cleaned_data.get('expected_salary', 'N/A')} ({cleaned_data.get('pay_duration', 'N/A')})
Hike Expected: {cleaned_data.get('hike', 'N/A')}%
Last Working Day: {cleaned_data.get('last_working_day', 'N/A')}
Exit Reason: {cleaned_data.get('exit_reason', 'N/A')}

SKILLS
------
Primary Skills: {', '.join(cleaned_data.get('skills', [])) if cleaned_data.get('skills') else 'N/A'}
Additional Skills: {', '.join(cleaned_data.get('may_also_known_skills', [])) if cleaned_data.get('may_also_known_skills') else 'N/A'}
Labels: {', '.join(cleaned_data.get('labels', [])) if cleaned_data.get('labels') else 'N/A'}

PROFESSIONAL EXPERIENCE
----------------------
{chr(10).join([f'''
Company: {exp.get('company', 'N/A')}
Title: {exp.get('title', 'N/A')}
Duration: {exp.get('from_date', 'N/A')} to {exp.get('until', 'Present') if exp.get('until') else 'Present'}
''' for exp in cleaned_data.get('experience', [])]) if cleaned_data.get('experience') else 'N/A'}

EDUCATION
---------
{chr(10).join([f'''
Degree: {edu.get('education', 'N/A')}
College: {edu.get('college', 'N/A')}
Pass Year: {edu.get('pass_year', 'N/A')}
''' for edu in cleaned_data.get('academic_details', [])]) if cleaned_data.get('academic_details') else 'N/A'}

ADDITIONAL INFORMATION
---------------------
Tier 1 MBA: {'Yes' if cleaned_data.get('is_tier1_mba') else 'No'}
Tier 1 Engineering: {'Yes' if cleaned_data.get('is_tier1_engineering') else 'No'}
Comments: {cleaned_data.get('comment', 'N/A')}

PROFESSIONAL LINKS
-----------------
Naukri Profile: {contact_details.get('naukri_profile', 'N/A')}
LinkedIn Profile: {contact_details.get('linkedin_profile', 'N/A')}
Portfolio: {contact_details.get('portfolio_link', 'N/A')}
"""

        # Add the combined resume to the dictionary
        cleaned_data["combined_resume"] = combined_resume

        # Add flattened fields for compatibility with search APIs
        # The AddUserDataVectorizer expects some fields at the root level
        contact_details = cleaned_data.get("contact_details", {})
        cleaned_data["name"] = contact_details.get("name", "Unknown")
        cleaned_data["email"] = contact_details.get("email", "unknown@example.com")
        cleaned_data["phone"] = contact_details.get("phone", "+1-000-000-0000")

        # Map total_experience to total_exp for search compatibility
        if "total_experience" in cleaned_data:
            cleaned_data["total_exp"] = cleaned_data["total_experience"]

        # Use the AddUserDataVectorizer's generate_resume_embeddings method
        resume_with_embeddings = add_user_vectorizer.generate_resume_embeddings(
            cleaned_data
        )

        # Add metadata about embedding generation
        resume_with_embeddings["embeddings_generated_at"] = datetime.now().isoformat()

        # Count the number of vector fields generated
        vector_fields = [
            key for key in resume_with_embeddings.keys() if key.endswith("_vector")
        ]
        logger.info(
            f"Generated {len(vector_fields)} vector fields for resume: {vector_fields}"
        )

        return resume_with_embeddings

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        # Return cleaned data without embeddings if generation fails
        return clean_resume_data(parsed_data, user_id, username)


def process_multiple_resumes_with_embeddings(
    files_data: List[Dict[str, Any]],
    user_id: str,
    username: str,
    llm_provider: str = None,
    max_concurrent: int = 10,
) -> List[Dict[str, Any]]:
    """
    Process multiple resume files using the multiple resume parser and generate embeddings.

    Args:
        files_data: List of file data dictionaries
        user_id: The user ID to assign to all resumes
        username: The username to assign to all resumes
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

            # **NEW: Check for duplicate content before parsing**
            is_duplicate, similar_documents = duplicate_ops.check_duplicate_content(
                user_id, total_resume_text
            )

            if is_duplicate:
                logger.info(
                    f"Duplicate content detected for {filename}. Skipping parsing."
                )

                # Clean up temp file
                try:
                    temp_file_path.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete temp file {temp_file_path}: {e}")

                return {
                    "filename": filename,
                    "status": "duplicate",
                    "error": "Duplicate content detected",
                    "error_type": "duplicate_content",
                    "similar_documents": similar_documents,
                    "message": f"This resume content is {similar_documents[0]['similarity_score']:.1%} similar to existing content",
                    "parsed_data": None,
                }

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

            # Process the resume using enhanced LLM (which now includes intelligent content validation)
            parsed_data = parser.process_resume(total_resume_text)

            # Check if parsing was successful or if there's an error
            if "error" in parsed_data:
                error_type = parsed_data.get("error_type", "parsing_error")

                if error_type == "invalid_content":
                    # Handle non-resume content detected by LLM's intelligent validation
                    logger.warning(
                        f"LLM detected invalid content in {filename}: {parsed_data.get('error')}"
                    )
                    return {
                        "filename": filename,
                        "status": "error",
                        "error_type": "invalid_content",
                        "error": parsed_data.get("error"),
                        "suggestion": parsed_data.get(
                            "suggestion", "Please upload a valid resume document."
                        ),
                        "parsed_data": None,
                    }
                else:
                    # Handle other parsing errors
                    logger.warning(
                        f"Resume parsing had issues for {filename}: {parsed_data.get('error')}"
                    )
                    return {
                        "filename": filename,
                        "status": "error",
                        "error_type": "parsing_error",
                        "error": parsed_data.get("error"),
                        "parsed_data": None,
                    }

            # Handle experience dates - convert 'to' field to 'until' if present
            if "experience" in parsed_data and parsed_data["experience"]:
                for exp in parsed_data["experience"]:
                    if "to" in exp and exp["to"] is not None:
                        exp["until"] = exp["to"]
                        del exp["to"]

            # Generate embeddings for the parsed data (this also cleans the data)
            parsed_data_with_embeddings = generate_embeddings_for_resume(
                parsed_data, user_id, username
            )

            # Add metadata
            parsed_data_with_embeddings.update(
                {
                    "filename": filename,
                    "processing_timestamp": datetime.now().isoformat(),
                    "llm_provider_used": parser.provider.value,
                    "text_length": len(total_resume_text),
                }
            )

            # **NEW: Save extracted text for future duplicate detection**
            save_result = duplicate_ops.save_extracted_text(
                user_id, username, filename, total_resume_text
            )

            if not save_result["success"]:
                logger.warning(
                    f"Failed to save extracted text for {filename}: {save_result.get('error')}"
                )

            # Clean up temp file
            try:
                temp_file_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete temp file {temp_file_path}: {e}")

            logger.info(f"Successfully processed {filename}")

            # Create response data without vector embeddings for user
            response_data = parsed_data_with_embeddings.copy()
            # Remove vector fields from response (they're large and not needed for user display)
            vector_fields = [
                key for key in response_data.keys() if key.endswith("_vector")
            ]
            for field in vector_fields:
                if field in response_data:
                    del response_data[field]

            return {
                "filename": filename,
                "status": "success",
                "parsed_data": response_data,
                "full_data_with_vectors": parsed_data_with_embeddings,  # Keep full data for DB (internal use only)
                "llm_provider": parser.provider.value,
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
async def parse_multiple_resumes(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    username: str = Form(...),
    max_concurrent: Optional[int] = Form(10),
):
    """
    Parse multiple resumes and automatically save them to the database.

    This endpoint:
    - Accepts 1 or more resume files along with user_id and username
    - Uses multipleresumepraser module for parsing
    - Cleans and normalizes parsed data
    - Assigns the provided user_id and username to all resumes
    - Saves parsed data to database automatically
    - Updates skills and titles collections
    - Returns detailed processing summary with statistics

    Args:
        files: List of resume files to process (1 or more required)
        user_id: The user ID to assign to all uploaded resumes
        username: The username to assign to all uploaded resumes
        max_concurrent: Maximum concurrent processing threads (default: 10)
    """
    # Generate unique session ID for tracking
    session_id = f"{user_id}_{int(time.time())}"
    start_time = time.time()

    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        if len(files) > 100:  # Limit for safety - increased for bulk processing
            raise HTTPException(
                status_code=400, detail="Too many files. Maximum 100 files allowed."
            )

        # Add to processing queue
        update_processing_queue("add_to_queue", session_id, len(files))

        logger.info(
            f"Processing {len(files)} resumes for user: {username} (ID: {user_id}) - Session: {session_id}"
        )

        # Get initial queue status
        initial_queue_status = get_queue_status()

        # Prepare file data
        files_data = []
        for file in files:
            content = await file.read()
            files_data.append({"filename": file.filename, "content": content})

        # Process files with optimized settings for bulk processing
        max_concurrent_threads = min(
            max_concurrent or 10, max(3, len(files_data) // 10)
        )  # Use provided max_concurrent or dynamic thread count

        processing_results = process_multiple_resumes_with_embeddings(
            files_data=files_data,
            user_id=user_id,
            username=username,
            llm_provider=None,  # Use default
            max_concurrent=max_concurrent_threads,  # Dynamic based on file count
        )

        # Automatically save all successfully parsed resumes to database
        successful_parses = [
            result["parsed_data"]
            for result in processing_results
            if result["status"] == "success" and result["parsed_data"]
        ]

        # Get full data with vectors for database saving
        full_data_for_db = [
            result["full_data_with_vectors"]
            for result in processing_results
            if result["status"] == "success" and result.get("full_data_with_vectors")
        ]

        database_results = []
        if full_data_for_db:
            # Save to database in batches for better performance
            batch_size = 20
            for i in range(0, len(full_data_for_db), batch_size):
                batch = full_data_for_db[i : i + batch_size]
                logger.info(
                    f"Saving batch {i//batch_size + 1} of {len(batch)} resumes to database"
                )

                for j, parsed_data in enumerate(batch):
                    try:
                        # Direct MongoDB insertion like add_userdata.py
                        result = collection.insert_one(parsed_data)

                        # Extract skills and experience titles for skills_titles collection
                        skills = []
                        if "skills" in parsed_data and parsed_data["skills"]:
                            skills.extend(parsed_data["skills"])
                        if (
                            "may_also_known_skills" in parsed_data
                            and parsed_data["may_also_known_skills"]
                        ):
                            skills.extend(parsed_data["may_also_known_skills"])

                        experience_titles = []
                        if "experience" in parsed_data and parsed_data["experience"]:
                            for experience in parsed_data["experience"]:
                                if "title" in experience and experience["title"]:
                                    experience_titles.append(experience["title"])

                        # Add skills and titles to skills_titles collection
                        normalized_skills = (
                            normalize_text_list(skills) if skills else []
                        )
                        normalized_experience_titles = (
                            normalize_text_list(experience_titles)
                            if experience_titles
                            else []
                        )

                        try:
                            if normalized_skills:
                                skills_ops.add_multiple_skills(normalized_skills)
                            if normalized_experience_titles:
                                skills_ops.add_multiple_titles(
                                    normalized_experience_titles
                                )
                            logger.info(f"Added skills: {normalized_skills}")
                            logger.info(
                                f"Added experience titles: {normalized_experience_titles}"
                            )
                        except Exception as skills_error:
                            logger.warning(
                                f"Skills/titles insertion error: {str(skills_error)}"
                            )
                            # Don't fail the whole operation for skills insertion errors

                        # Check if vector embeddings are present
                        vector_fields = [
                            key for key in parsed_data.keys() if key.endswith("_vector")
                        ]

                        database_results.append(
                            {
                                "filename": parsed_data.get("filename"),
                                "database_id": str(result.inserted_id),
                                "user_id": parsed_data["user_id"],
                                "status": "saved",
                                "skills_added": len(normalized_skills),
                                "titles_added": len(normalized_experience_titles),
                            }
                        )

                        logger.info(
                            f"Saved resume to database with ID: {result.inserted_id}"
                        )

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
        invalid_content_count = len(
            [r for r in processing_results if r.get("error_type") == "invalid_content"]
        )
        parsing_error_count = len(
            [r for r in processing_results if r.get("error_type") == "parsing_error"]
        )
        duplicate_content_count = len(
            [r for r in processing_results if r.get("status") == "duplicate"]
        )
        failed_parses = total_files - successful_parses_count - duplicate_content_count
        saved_to_database = len([r for r in database_results if r["status"] == "saved"])
        total_skills_added = sum(
            [
                r.get("skills_added", 0)
                for r in database_results
                if r["status"] == "saved"
            ]
        )
        total_titles_added = sum(
            [
                r.get("titles_added", 0)
                for r in database_results
                if r["status"] == "saved"
            ]
        )

        # Calculate processing time and update queue
        end_time = time.time()
        processing_time_stats = calculate_processing_time(start_time, end_time)
        update_processing_queue("remove_from_queue", session_id, total_files)
        update_processing_queue("complete_session", session_id)
        final_queue_status = get_queue_status()

        # Clean processing results for user response (remove internal fields)
        cleaned_processing_results = []
        for result in processing_results:
            cleaned_result = result.copy()
            # Remove internal fields that shouldn't be exposed to users
            if "full_data_with_vectors" in cleaned_result:
                del cleaned_result["full_data_with_vectors"]
            cleaned_processing_results.append(cleaned_result)

        # Create lists of resume names by status
        successfully_parsed_resumes = [
            result["filename"]
            for result in processing_results
            if result["status"] == "success"
        ]

        failed_to_parse_resumes = [
            {
                "filename": result["filename"],
                "error": result.get("error", "Unknown error"),
            }
            for result in processing_results
            if result["status"] == "error"
        ]

        duplicate_content_resumes = [
            {
                "filename": result["filename"],
                "error": result.get("error", "Duplicate content detected"),
                "similar_documents": result.get("similar_documents", []),
                "message": result.get("message", ""),
            }
            for result in processing_results
            if result["status"] == "duplicate"
        ]

        successfully_saved_resumes = [
            result["filename"]
            for result in database_results
            if result["status"] == "saved"
        ]

        failed_to_save_resumes = [
            {
                "filename": result["filename"],
                "error": result.get("error", "Unknown database error"),
            }
            for result in database_results
            if result["status"] == "save_failed"
        ]

        # Enhanced response with detailed statistics
        return {
            "message": f"✅ Successfully processed {successful_parses_count}/{total_files} resumes for {username} in {processing_time_stats['formatted_time']}",
            "resume_processing_summary": {
                "successfully_parsed_resumes": successfully_parsed_resumes,
                "failed_to_parse_resumes": failed_to_parse_resumes,
                "duplicate_content_resumes": duplicate_content_resumes,
                "successfully_saved_to_database": successfully_saved_resumes,
                "failed_to_save_to_database": failed_to_save_resumes,
            },
            "processing_statistics": {
                "session_id": session_id,
                "user_id": user_id,
                "username": username,
                "total_files_uploaded": total_files,
                "successfully_parsed": successful_parses_count,
                "failed_to_parse": failed_parses,
                "invalid_content_detected": invalid_content_count,
                "parsing_errors": parsing_error_count,
                "duplicate_content_detected": duplicate_content_count,
                "successfully_saved_to_database": saved_to_database,
                "parsing_success_rate": (
                    f"{(successful_parses_count/total_files)*100:.1f}%"
                    if total_files > 0
                    else "0%"
                ),
                "content_validation_rate": (
                    f"{((total_files - invalid_content_count)/total_files)*100:.1f}%"
                    if total_files > 0
                    else "0%"
                ),
                "database_save_success_rate": (
                    f"{(saved_to_database/successful_parses_count)*100:.1f}%"
                    if successful_parses_count > 0
                    else "0%"
                ),
                "skills_added_to_collection": total_skills_added,
                "titles_added_to_collection": total_titles_added,
            },
            "processing_time": processing_time_stats,
            "queue_information": {
                "initial_queue_status": initial_queue_status,
                "final_queue_status": final_queue_status,
                "processing_settings": {
                    "concurrent_threads_used": max_concurrent_threads,
                    "batch_processing": True,
                    "database_batch_size": 20,
                },
            },
            # "detailed_results": {
            #     "processing_details": cleaned_processing_results,
            #     "database_results": database_results,
            # },
        }

    except HTTPException:
        # Update queue on HTTP exceptions
        end_time = time.time()
        update_processing_queue(
            "remove_from_queue", session_id, len(files) if "files" in locals() else 0
        )
        raise
    except Exception as e:
        # Update queue and calculate time even on errors
        end_time = time.time()
        processing_time_stats = calculate_processing_time(start_time, end_time)
        update_processing_queue(
            "remove_from_queue", session_id, len(files) if "files" in locals() else 0
        )

        logger.error(
            f"Error in parse_multiple_resumes (Session: {session_id}): {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Internal server error: {str(e)}",
                "session_id": session_id,
                "processing_time": processing_time_stats,
                "queue_status": get_queue_status(),
            },
        )


@router.post("/resume-parser-bulk")
async def parse_bulk_resumes(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    username: str = Form(...),
    max_concurrent: Optional[int] = Form(15),
    batch_size: Optional[int] = Form(25),
):
    """
    Parse a large batch of resumes (up to 100) with optimized performance settings.

    This endpoint is specifically designed for bulk processing with:
    - Higher concurrency limits
    - Batch database operations
    - Progress tracking
    - Memory optimization
    - Detailed processing statistics

    Args:
        files: List of resume files (up to 100)
        user_id: The user ID to assign to all uploaded resumes
        username: The username to assign to all uploaded resumes
        max_concurrent: Maximum concurrent processing threads (default: 15)
        batch_size: Database save batch size (default: 25)
    """
    # Generate unique session ID for tracking
    session_id = f"bulk_{user_id}_{int(time.time())}"
    start_time = time.time()

    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        if len(files) > 100:
            raise HTTPException(
                status_code=400,
                detail="Too many files. Maximum 100 files allowed for bulk processing.",
            )

        # Validate parameters
        max_concurrent = min(max_concurrent or 15, 20)  # Cap at 20 for stability
        batch_size = min(batch_size or 25, 50)  # Cap at 50 for memory

        # Add to processing queue
        update_processing_queue("add_to_queue", session_id, len(files))
        initial_queue_status = get_queue_status()

        logger.info(
            f"Bulk processing {len(files)} resumes for user: {username} (ID: {user_id}) with {max_concurrent} threads, batch size {batch_size} - Session: {session_id}"
        )

        # Prepare file data
        files_data = []
        for file in files:
            content = await file.read()
            files_data.append({"filename": file.filename, "content": content})

        # Process with bulk-optimized settings
        processing_results = process_multiple_resumes_with_embeddings(
            files_data=files_data,
            user_id=user_id,
            username=username,
            llm_provider=None,
            max_concurrent=max_concurrent,
        )

        # Bulk database operations
        successful_results = [
            result
            for result in processing_results
            if result["status"] == "success" and result["parsed_data"]
        ]

        database_results = []
        total_saved = 0

        # Process in batches for better memory management
        for batch_num, i in enumerate(range(0, len(successful_results), batch_size)):
            batch = successful_results[i : i + batch_size]
            logger.info(
                f"Processing database batch {batch_num + 1}/{(len(successful_results) + batch_size - 1) // batch_size}"
            )

            for j, result in enumerate(batch):
                try:
                    # Use the full data with vectors for database saving
                    parsed_data = result.get(
                        "full_data_with_vectors", result["parsed_data"]
                    ).copy()

                    # Direct MongoDB insertion like add_userdata.py
                    db_result = collection.insert_one(parsed_data)

                    # Extract skills and experience titles for skills_titles collection
                    skills = []
                    if "skills" in parsed_data and parsed_data["skills"]:
                        skills.extend(parsed_data["skills"])
                    if (
                        "may_also_known_skills" in parsed_data
                        and parsed_data["may_also_known_skills"]
                    ):
                        skills.extend(parsed_data["may_also_known_skills"])

                    experience_titles = []
                    if "experience" in parsed_data and parsed_data["experience"]:
                        for experience in parsed_data["experience"]:
                            if "title" in experience and experience["title"]:
                                experience_titles.append(experience["title"])

                    # Add skills and titles to skills_titles collection
                    normalized_skills = normalize_text_list(skills) if skills else []
                    normalized_experience_titles = (
                        normalize_text_list(experience_titles)
                        if experience_titles
                        else []
                    )

                    try:
                        if normalized_skills:
                            skills_ops.add_multiple_skills(normalized_skills)
                        if normalized_experience_titles:
                            skills_ops.add_multiple_titles(normalized_experience_titles)
                        logger.info(f"Added skills: {normalized_skills}")
                        logger.info(
                            f"Added experience titles: {normalized_experience_titles}"
                        )
                    except Exception as skills_error:
                        logger.warning(
                            f"Skills/titles insertion error: {str(skills_error)}"
                        )
                        # Don't fail the whole operation for skills insertion errors

                    # Check if vector embeddings are present
                    vector_fields = [
                        key for key in parsed_data.keys() if key.endswith("_vector")
                    ]

                    database_results.append(
                        {
                            "filename": result["filename"],
                            "database_id": str(db_result.inserted_id),
                            "user_id": parsed_data["user_id"],
                            "status": "saved",
                            "batch_number": batch_num + 1,
                            "skills_added": len(normalized_skills),
                            "titles_added": len(normalized_experience_titles),
                        }
                    )

                    total_saved += 1

                except Exception as e:
                    logger.error(
                        f"Error saving resume {result['filename']} to database: {e}"
                    )
                    database_results.append(
                        {
                            "filename": result["filename"],
                            "status": "save_failed",
                            "error": str(e),
                            "batch_number": batch_num + 1,
                        }
                    )

        # Final statistics
        total_files = len(files)
        successful_parses = len(successful_results)
        duplicate_content_count = len(
            [r for r in processing_results if r.get("status") == "duplicate"]
        )
        invalid_content_count = len(
            [r for r in processing_results if r.get("error_type") == "invalid_content"]
        )
        parsing_error_count = len(
            [r for r in processing_results if r.get("error_type") == "parsing_error"]
        )
        failed_parses = total_files - successful_parses - duplicate_content_count
        total_skills_added = sum(
            [
                r.get("skills_added", 0)
                for r in database_results
                if r["status"] == "saved"
            ]
        )
        total_titles_added = sum(
            [
                r.get("titles_added", 0)
                for r in database_results
                if r["status"] == "saved"
            ]
        )

        # Calculate processing time and update queue
        end_time = time.time()
        processing_time_stats = calculate_processing_time(start_time, end_time)
        update_processing_queue("remove_from_queue", session_id, total_files)
        update_processing_queue("complete_session", session_id)
        final_queue_status = get_queue_status()

        # Clean processing results for user response (remove internal fields)
        cleaned_processing_results = []
        for result in processing_results:
            cleaned_result = result.copy()
            # Remove internal fields that shouldn't be exposed to users
            if "full_data_with_vectors" in cleaned_result:
                del cleaned_result["full_data_with_vectors"]
            cleaned_processing_results.append(cleaned_result)

        # Create lists of resume names by status
        successfully_parsed_resumes = [
            result["filename"]
            for result in processing_results
            if result["status"] == "success"
        ]

        failed_to_parse_resumes = [
            {
                "filename": result["filename"],
                "error": result.get("error", "Unknown error"),
            }
            for result in processing_results
            if result["status"] == "error"
        ]

        duplicate_content_resumes = [
            {
                "filename": result["filename"],
                "error": result.get("error", "Duplicate content detected"),
                "similar_documents": result.get("similar_documents", []),
                "message": result.get("message", ""),
            }
            for result in processing_results
            if result["status"] == "duplicate"
        ]

        successfully_saved_resumes = [
            result["filename"]
            for result in database_results
            if result["status"] == "saved"
        ]

        failed_to_save_resumes = [
            {
                "filename": result["filename"],
                "error": result.get("error", "Unknown database error"),
            }
            for result in database_results
            if result["status"] == "save_failed"
        ]

        # Enhanced bulk response with detailed statistics
        return {
            "message": f"🚀 Bulk processed {successful_parses}/{total_files} resumes for {username} in {processing_time_stats['formatted_time']} - {total_saved} saved to database",
            "resume_processing_summary": {
                "successfully_parsed_resumes": successfully_parsed_resumes,
                "failed_to_parse_resumes": failed_to_parse_resumes,
                "duplicate_content_resumes": duplicate_content_resumes,
                "successfully_saved_to_database": successfully_saved_resumes,
                "failed_to_save_to_database": failed_to_save_resumes,
            },
            "bulk_processing_statistics": {
                "session_id": session_id,
                "user_id": user_id,
                "username": username,
                "total_files_uploaded": total_files,
                "successfully_parsed": successful_parses,
                "failed_to_parse": failed_parses,
                "invalid_content_detected": invalid_content_count,
                "parsing_errors": parsing_error_count,
                "duplicate_content_detected": duplicate_content_count,
                "successfully_saved_to_database": total_saved,
                "parsing_success_rate": (
                    f"{(successful_parses/total_files)*100:.1f}%"
                    if total_files > 0
                    else "0%"
                ),
                "database_save_success_rate": (
                    f"{(total_saved/successful_parses)*100:.1f}%"
                    if successful_parses > 0
                    else "0%"
                ),
                "skills_added_to_collection": total_skills_added,
                "titles_added_to_collection": total_titles_added,
            },
            "processing_time": processing_time_stats,
            "queue_information": {
                "initial_queue_status": initial_queue_status,
                "final_queue_status": final_queue_status,
                "processing_settings": {
                    "max_concurrent_threads": max_concurrent,
                    "database_batch_size": batch_size,
                    "total_batches": (len(successful_results) + batch_size - 1)
                    // batch_size,
                    "performance_optimized": True,
                },
            },
            # "detailed_results": {
            #     "processing_details": cleaned_processing_results,
            #     "database_results": database_results,
            # },
        }

    except HTTPException:
        # Update queue on HTTP exceptions
        end_time = time.time()
        update_processing_queue(
            "remove_from_queue", session_id, len(files) if "files" in locals() else 0
        )
        raise
    except Exception as e:
        # Update queue and calculate time even on errors
        end_time = time.time()
        processing_time_stats = calculate_processing_time(start_time, end_time)
        update_processing_queue(
            "remove_from_queue", session_id, len(files) if "files" in locals() else 0
        )

        logger.error(f"Error in parse_bulk_resumes (Session: {session_id}): {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Bulk processing error: {str(e)}",
                "session_id": session_id,
                "processing_time": processing_time_stats,
                "queue_status": get_queue_status(),
            },
        )


@router.get("/queue-status")
async def get_processing_queue_status():
    """
    Get current processing queue status and statistics.

    Returns information about:
    - Current queue size
    - Active processing sessions
    - Total processed today
    - Queue availability status
    """
    queue_status = get_queue_status()
    active_sessions_details = []

    # Get details of active sessions
    for session_id, session_info in PROCESSING_QUEUE["active_sessions"].items():
        if session_info["status"] == "processing":
            current_time = time.time()
            elapsed_time = current_time - session_info["start_time"]
            active_sessions_details.append(
                {
                    "session_id": session_id,
                    "resume_count": session_info["count"],
                    "elapsed_time_seconds": round(elapsed_time, 2),
                    "elapsed_time_formatted": (
                        f"{int(elapsed_time//60)}m {int(elapsed_time%60)}s"
                        if elapsed_time > 60
                        else f"{int(elapsed_time)}s"
                    ),
                    "status": session_info["status"],
                }
            )

    return {
        "queue_status": queue_status,
        "active_sessions": active_sessions_details,
        "queue_health": {
            "is_available": queue_status["current_queue_size"]
            < 50,  # Threshold for availability
            "load_level": (
                "low"
                if queue_status["current_queue_size"] < 10
                else "medium" if queue_status["current_queue_size"] < 30 else "high"
            ),
            "estimated_wait_time": (
                "immediate"
                if queue_status["current_queue_size"] == 0
                else f"~{queue_status['current_queue_size'] * 2} minutes"
            ),
        },
        "statistics": {
            "total_active_sessions": len(active_sessions_details),
            "total_resumes_in_queue": queue_status["current_queue_size"],
            "total_processed_today": queue_status["total_processed_today"],
            "server_status": "healthy",
        },
    }


@router.get("/info")
async def get_multiple_resume_parser_info():
    """
    Get information about the multiple resume parser endpoints.
    """
    return {
        "endpoints": {
            "standard_processing": {
                "url": "/resume-parser-multiple",
                "method": "POST",
                "description": "Upload 1-100 resume files for automatic parsing and database storage",
                "required_form_fields": ["files", "user_id", "username"],
                "optional_form_fields": ["max_concurrent"],
                "max_files": 100,
                "concurrent_threads": "Dynamic (3-10 based on file count) or configurable",
                "use_case": "General purpose, moderate volume processing",
            },
            "bulk_processing": {
                "url": "/resume-parser-bulk",
                "method": "POST",
                "description": "Optimized bulk processing for large volumes (up to 100 resumes)",
                "required_form_fields": ["files", "user_id", "username"],
                "optional_form_fields": ["max_concurrent", "batch_size"],
                "max_files": 100,
                "concurrent_threads": "Configurable (up to 20, default 15)",
                "batch_size": "Configurable (up to 50, default 25)",
                "use_case": "High volume processing with performance optimization",
            },
        },
        "request_format": {
            "content_type": "multipart/form-data",
            "required_fields": {
                "files": "List of resume files (PDF, DOCX, TXT)",
                "user_id": "String - User ID to assign to all resumes",
                "username": "String - Username to assign to all resumes",
            },
            "optional_fields": {
                "max_concurrent": "Integer - Maximum concurrent threads",
                "batch_size": "Integer - Database batch size (bulk endpoint only)",
            },
        },
        "features": {
            "parser_module": "multipleresumepraser",
            "user_assignment": "All resumes assigned to provided user_id and username",
            "automatic_database_saving": True,
            "concurrent_processing": True,
            "batch_database_operations": True,
            "memory_optimization": True,
            "progress_tracking": True,
            "skills_management": True,
            "queue_management": True,
            "processing_statistics": True,
            "detailed_timing": True,
            "success_rate_tracking": True,
            "content_validation": True,
            "enhanced_error_handling": True,
            "resume_content_detection": True,
        },
        "supported_formats": [".txt", ".pdf", ".docx"],
        "performance_specs": {
            "max_resumes_per_request": 100,
            "recommended_bulk_size": "50-100 resumes",
            "concurrent_threads": "Up to 20 threads",
            "database_batch_size": "Up to 50 records per batch",
            "estimated_processing_time": "2-5 minutes for 100 resumes (depends on file size and LLM provider)",
        },
        "workflow": [
            "1. Upload resume files (up to 100) along with user_id and username",
            "2. Add files to processing queue and generate session ID",
            "3. Extract and clean text from each file",
            "4. Validate if content appears to be resume-related",
            "5. Parse resume data using enhanced LLM prompts (concurrent processing)",
            "6. Clean and normalize parsed data",
            "7. Assign provided user_id and username to all resumes",
            "8. Save to database in batches",
            "9. Update skills and titles collections",
            "10. Calculate processing statistics and success rates",
            "11. Update queue status and return comprehensive summary with detailed error analysis",
        ],
        "new_features": {
            "enhanced_system_prompts": {
                "description": "Improved LLM prompts with detailed extraction guidelines",
                "benefits": "Better data quality, more comprehensive extraction, consistent formatting",
                "components": [
                    "Expert role definition",
                    "Detailed instructions",
                    "Data quality requirements",
                    "Output specifications",
                ],
            },
            "content_validation": {
                "description": "AI-powered resume content detection before processing",
                "validation_criteria": [
                    "Resume-specific keywords",
                    "Contact information patterns",
                    "Professional sections",
                    "Employment terms",
                ],
                "scoring_system": "Multi-factor scoring with threshold-based validation",
                "error_handling": "Specific error messages and suggestions for invalid content",
            },
            "enhanced_error_handling": {
                "invalid_content_detection": "Identifies non-resume content with helpful suggestions",
                "detailed_error_types": [
                    "invalid_content",
                    "parsing_error",
                    "unknown_error",
                ],
                "user_feedback": "Specific suggestions for each error type",
                "error_statistics": "Tracking of content validation rates and error distributions",
            },
            "processing_statistics": {
                "success_rates": "Parsing and database save success percentages",
                "content_validation_rates": "Percentage of files containing valid resume content",
                "timing_details": "Start time, end time, and formatted duration",
                "queue_tracking": "Initial and final queue status",
            },
            "queue_management": {
                "queue_status_endpoint": "/queue-status",
                "session_tracking": "Unique session IDs for each processing request",
                "concurrent_monitoring": "Track active processing sessions",
                "load_balancing": "Queue health and estimated wait times",
            },
            "enhanced_responses": {
                "detailed_statistics": "Comprehensive processing metrics including validation rates",
                "failure_analysis": "Individual file processing status with error types",
                "performance_metrics": "Thread usage and batch processing stats",
                "content_quality_metrics": "Resume validation success rates",
            },
        },
    }
