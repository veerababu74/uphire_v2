import re
import os
import numpy as np
from fastapi import APIRouter, Body, HTTPException, status, UploadFile, File
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from mangodatabase.client import get_collection, get_users_collection
from mangodatabase.user_operations import (
    UserOperations,
    get_effective_user_id_for_search,
)
from core.helpers import format_resume
from embeddings.vectorizer import Vectorizer
from schemas.vector_search_scehma import VectorSearchQuery
from GroqcloudLLM.text_extraction import extract_and_clean_text
from pathlib import Path

from datetime import datetime, timedelta

BASE_FOLDER = "dummy_data_save"
TEMP_FOLDER = os.path.join(BASE_FOLDER, "temp_text_extract")
TEMP_DIR = Path(os.path.join(BASE_FOLDER, "temp_files"))

# Ensure the directories exist
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
from core.custom_logger import CustomLogger

logger_manager = CustomLogger()
logging = logger_manager.get_logger("vector_search_api")


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


"""
Vector Search API Documentation:

Field Mapping Options:
- skills: Searches through candidate's technical and soft skills
- experience: Searches through work experience descriptions
- education: Searches through educational background
- projects: Searches through project descriptions
- full_text: Searches across the entire resume

Expected Input Format:
{
    "query": "string",      # Search query text
    "field": "string",      # One of the field mapping options above
    "num_results": int,     # Number of results to return (default: 10)
    "min_score": float      # Minimum similarity score threshold (0.0 to 1.0)
}
"""

# Get MongoDB collection
resumes_collection = get_collection()
users_collection = get_users_collection()
user_ops = UserOperations(users_collection)
enhanced_search_router = APIRouter(prefix="/aiv1", tags=["enhanced ai vector search"])
vectorizer = Vectorizer()


class SearchError(Exception):
    """Custom exception for search-related errors"""

    pass


def calculate_priority_score(
    candidate: Dict[str, Any], query: str
) -> Tuple[float, str]:
    """
    Calculate priority-based relevance score for vector search results:
    1st Priority: Designation/Role (40% weight)
    2nd Priority: Location (30% weight)
    3rd Priority: Skills, Experience, and Salary (30% combined weight)
    """
    query_lower = query.lower()
    score_components = []
    reasons = []

    # 1st Priority: Designation/Role matching (weight: 0.4)
    designation_score = calculate_designation_score(candidate, query_lower)
    if designation_score > 0:
        score_components.append(designation_score * 0.4)
        reasons.append("Designation match")

    # 2nd Priority: Location matching (weight: 0.3)
    location_score = calculate_location_score(candidate, query_lower)
    if location_score > 0:
        score_components.append(location_score * 0.3)
        reasons.append("Location match")

    # 3rd Priority: Skills matching (weight: 0.15)
    skills_score = calculate_skills_score(candidate, query_lower)
    if skills_score > 0:
        score_components.append(skills_score * 0.15)
        reasons.append("Skills match")

    # 3rd Priority: Experience matching (weight: 0.1)
    experience_score = calculate_experience_score(candidate, query_lower)
    if experience_score > 0:
        score_components.append(experience_score * 0.1)
        reasons.append("Experience match")

    # 3rd Priority: Salary matching (weight: 0.05)
    salary_score = calculate_salary_score(candidate, query_lower)
    if salary_score > 0:
        score_components.append(salary_score * 0.05)
        reasons.append("Salary match")

    final_score = sum(score_components) if score_components else 0.0
    reason = "; ".join(reasons) if reasons else "Basic vector similarity"

    return final_score, reason


def calculate_designation_score(candidate: Dict[str, Any], query_lower: str) -> float:
    """Calculate designation/role matching score"""
    score = 0.0

    # Check experience roles
    experience = candidate.get("experience", [])
    for exp in experience[:3]:  # Check top 3 recent roles
        role = exp.get("role", "").lower()
        if any(word in role for word in query_lower.split() if len(word) > 2):
            score = max(score, 1.0 if exp == experience[0] else 0.8)

    # Check labels
    labels = candidate.get("labels", [])
    for label in labels:
        if any(word in label.lower() for word in query_lower.split() if len(word) > 2):
            score = max(score, 0.7)

    return score


def calculate_location_score(candidate: Dict[str, Any], query_lower: str) -> float:
    """Calculate location matching score"""
    score = 0.0

    # Common Indian cities
    cities = [
        "mumbai",
        "delhi",
        "bangalore",
        "chennai",
        "kolkata",
        "pune",
        "hyderabad",
        "ahmedabad",
        "surat",
        "jaipur",
        "noida",
        "gurgaon",
    ]

    query_cities = [city for city in cities if city in query_lower]

    if query_cities:
        current_city = (
            candidate.get("contact_details", {}).get("current_city", "").lower()
        )
        looking_for_jobs_in = candidate.get("contact_details", {}).get(
            "looking_for_jobs_in", []
        )

        for query_city in query_cities:
            if query_city in current_city:
                score = max(score, 1.0)
            for job_city in looking_for_jobs_in:
                if query_city in job_city.lower():
                    score = max(score, 0.8)

    return score


def calculate_skills_score(candidate: Dict[str, Any], query_lower: str) -> float:
    """Calculate skills matching score"""
    score = 0.0

    skills = candidate.get("skills", []) + candidate.get("may_also_known_skills", [])
    query_words = [word for word in query_lower.split() if len(word) > 2]

    for skill in skills:
        skill_lower = skill.lower()
        for word in query_words:
            if word in skill_lower:
                score = max(score, 1.0)
                break

    return min(score, 1.0)


def calculate_experience_score(candidate: Dict[str, Any], query_lower: str) -> float:
    """Calculate experience matching score"""
    score = 0.0

    # Extract experience numbers from query
    exp_match = re.search(r"(\d+)\s*(?:year|yr|yrs|years|exp|experience)", query_lower)
    if exp_match:
        target_exp = float(exp_match.group(1))
        total_exp_str = candidate.get("total_experience", "0")

        try:
            candidate_exp = float(
                re.search(r"(\d+(?:\.\d+)?)", str(total_exp_str)).group(1)
            )

            if abs(candidate_exp - target_exp) <= 1:  # Within 1 year
                score = 1.0
            elif abs(candidate_exp - target_exp) <= 2:  # Within 2 years
                score = 0.8
            elif candidate_exp >= target_exp * 0.8:  # At least 80% of required
                score = 0.6
        except (ValueError, AttributeError):
            pass

    return score


def calculate_salary_score(candidate: Dict[str, Any], query_lower: str) -> float:
    """Calculate salary matching score"""
    score = 0.0

    # Extract salary from query (in lakhs)
    salary_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac|l)", query_lower)
    if salary_match:
        target_salary = float(salary_match.group(1))
        expected_salary = candidate.get("expected_salary", 0)

        try:
            if expected_salary > 1000000:  # Convert from rupees to lakhs
                expected_salary = expected_salary / 100000

            if expected_salary <= target_salary:
                score = 1.0
            elif expected_salary <= target_salary * 1.2:  # Within 20% over budget
                score = 0.8
        except (ValueError, TypeError):
            pass

    return score


@enhanced_search_router.post(
    "/search",
    response_model=List[Dict[str, Any]],
    summary="AI-Powered Resume Search for Specific User",
    description="""
    Perform semantic search across resume database using AI embeddings for a specific user.
    
    **Input Fields:**
    - user_id: User ID (MANDATORY) - only resumes for this user will be searched
    - query: Search text (e.g., "Python developer with 5 years experience in machine learning")
    - field: Search scope (default: "full_text")
    - num_results: Number of results to return (default: 10)
    - min_score: Minimum similarity threshold (default: 0.2)
    - relevant_score: Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned (default: 40.0)
    
    **Example Input:**
    ```json
    {
        "user_id": "64f123abc456def789012345",
        "query": "experienced machine learning engineer with python",
        "field": "full_text",
        "num_results": 10,
        "min_score": 0.2,
        "relevant_score": 40.0
    }
    ```
    
    **Output Fields:**
    - user_id: Unique identifier for the user (matches the input user_id)
    - username: Username of the candidate
    - name: Candidate's full name
    - contact_details: Email, phone, location etc.
    - education: List of educational qualifications
    - experience: List of work experiences
    - projects: List of projects
    - total_experience: Years of experience
    - skills: List of technical and soft skills
    - certifications: List of certifications
    - relevance_score: Match score (0-100)
    
    **Search Fields:**
    - full_text: Search entire resume (default)
    - skills: Search only skills section
    - experience: Search work experience
    - education: Search educational background
    - projects: Search project descriptions
    
    **Note:** This endpoint will only return resumes belonging to the specified user_id.
    """,
    responses={
        200: {
            "description": "Successful search results",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "user_id": "64f123abc456def789012345",
                            "username": "john.doe",
                            "name": "John Doe",
                            "contact_details": {
                                "email": "john@example.com",
                                "phone": "+1234567890",
                                "location": "New York, USA",
                            },
                            "education": [
                                {
                                    "degree": "Master of Science in Computer Science",
                                    "institution": "Stanford University",
                                    "year": "2020",
                                }
                            ],
                            "experience": [
                                {
                                    "title": "Senior Machine Learning Engineer",
                                    "company": "Tech Corp",
                                    "duration": "2020-Present",
                                    "description": "Led ML projects using Python and TensorFlow",
                                }
                            ],
                            "projects": [
                                {
                                    "name": "AI Chatbot",
                                    "description": "Developed NLP-based customer service bot",
                                }
                            ],
                            "total_experience": 5.5,
                            "skills": ["Python", "Machine Learning", "TensorFlow"],
                            "certifications": ["AWS Machine Learning Specialty"],
                            "relevance_score": 95.5,
                        }
                    ]
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "User ID is mandatory and cannot be empty"}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"detail": "An unexpected error occurred"}
                }
            },
        },
    },
)
async def vector_search(search_query: VectorSearchQuery):
    try:
        # Input validation
        if not search_query.query.strip():
            raise SearchError("Search query cannot be empty")

        if not search_query.user_id.strip():
            raise SearchError("User ID is mandatory and cannot be empty")

        if search_query.num_results < 1:
            raise SearchError("Number of results must be greater than 0")

        # Get effective user ID for search (handles admin access)
        effective_user_id = await get_effective_user_id_for_search(
            user_ops, search_query.user_id
        )

        # Generate embedding for search query
        try:
            query_embedding = vectorizer.generate_embedding(search_query.query)
        except Exception as e:
            raise SearchError(f"Failed to generate embedding: {str(e)}")

        vector_field_mapping = {
            "skills": "skills_vector",
            "experience": "experience_text_vector",
            "education": "education_text_vector",  # Fixed: was academic_details_vector
            "projects": "projects_text_vector",  # Fixed: should search projects specifically
            "full_text": "combined_resume_vector",
        }

        vector_field = vector_field_mapping.get(search_query.field)
        if not vector_field:
            raise SearchError(
                f"Invalid field name. Choose from: {', '.join(vector_field_mapping.keys())}"
            )

        # Enhanced search pipeline with scoring
        match_filter = {}
        if effective_user_id:  # If None, admin searches all documents
            match_filter["user_id"] = effective_user_id

        pipeline = [
            {
                "$search": {
                    "index": "vector_search_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": vector_field,
                        "k": search_query.num_results
                        * 5,  # Get more results to filter later
                    },
                }
            },
            {"$set": {"score": {"$meta": "searchScore"}}},
            {
                "$match": {
                    **{"score": {"$gte": search_query.min_score}},
                    **match_filter,  # Add user_id filter only if not admin
                }
            },
            {
                "$project": {
                    "user_id": 1,
                    "username": 1,
                    "name": 1,
                    "contact_details": 1,
                    "education": 1,
                    "experience": 1,
                    "projects": 1,
                    "total_experience": 1,
                    "skills": 1,
                    "certifications": 1,
                    "score": 1,
                }
            },
            {"$sort": {"score": -1}},
        ]

        try:
            results = list(resumes_collection.aggregate(pipeline))
        except Exception as e:
            raise SearchError(f"Database query failed: {str(e)}")

        if not results:
            return []

        # Limit results to requested number
        limited_results = results[: search_query.num_results]

        formatted_results = [format_resume(result) for result in limited_results]

        # Apply priority-based scoring and add relevance score to results
        for result in formatted_results:
            # Calculate priority score based on search query
            priority_score, match_reason = calculate_priority_score(
                result, search_query.query
            )

            # Get base vector score
            base_score = result.get("score", 0)

            # Combine priority score with vector similarity score
            if priority_score > 0:
                # Priority score takes precedence (70%) with vector score as secondary (30%)
                final_score = (priority_score * 0.7) + (base_score * 0.3)
            else:
                # Fall back to vector score only
                final_score = base_score

            # Set relevance score and additional metrics
            result["relevance_score"] = round(final_score * 100, 2)
            result["match_reason"] = match_reason
            result["priority_score"] = round(priority_score * 100, 2)
            result["vector_score"] = round(base_score * 100, 2)

        # Sort results by priority-based relevance score (highest first)
        formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Filter results based on relevant_score threshold (using the schema's relevant_score parameter)
        if search_query.relevant_score > 0:
            formatted_results = [
                result
                for result in formatted_results
                if result.get("relevance_score", 0) >= search_query.relevant_score
            ]

        return formatted_results

    except SearchError as se:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(se))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@enhanced_search_router.post(
    "/search-by-jd",
    response_model=List[Dict[str, Any]],
    summary="AI-Powered Resume Search Based on Job Description File for Specific User",
    description="""
    Upload a job description file (.txt, .pdf, or .docx) and find matching resumes for a specific user.
    
    The system will extract and clean the text from the file and use AI to find semantically relevant candidates 
    belonging to the specified user_id.
    
    **Parameters:**
    - user_id: User ID (MANDATORY) - only resumes for this user will be searched
    - file: Job description file (.txt, .pdf, or .docx)
    - field: Search scope (default: "full_text")
    - num_results: Number of results to return (default: 10)
    - min_score: Minimum similarity threshold (default: 0.0)
    - relevant_score: Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned (default: 40.0)
    """,
)
async def search_by_jd(
    user_id: str,  # Made mandatory
    file: UploadFile = File(...),
    field: str = "full_text",
    num_results: int = 10,
    min_score: float = 0.0,
    relevant_score: float = 40.0,
):
    try:
        # Input validation
        if not user_id.strip():
            raise SearchError("User ID is mandatory and cannot be empty")

        # Get effective user ID for search (handles admin access)
        effective_user_id = await get_effective_user_id_for_search(user_ops, user_id)

        # Step 1: Save uploaded file to temp directory
        file_location = os.path.join(TEMP_FOLDER, file.filename)

        with open(file_location, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        # Step 2: Extract and clean text from file
        try:
            _, file_extension = os.path.splitext(file.filename)
            if file_extension.lower() not in [".txt", ".pdf", ".docx"]:
                raise SearchError(
                    "Unsupported file type. Only .txt, .pdf, and .docx are supported."
                )

            jd_text = extract_and_clean_text(file_location)
            if not jd_text.strip():
                raise SearchError("Extracted job description is empty.")
        finally:
            # Clean up the temporary file
            try:
                os.remove(file_location)
                logging.info(f"Deleted temporary file: {file_location}")
            except Exception as e:
                logging.error(f"Failed to delete temporary file {file_location}: {e}")

        # Step 3: Generate embedding from cleaned JD text
        try:
            query_embedding = vectorizer.generate_embedding(jd_text)
        except Exception as e:
            raise SearchError(f"Failed to generate embedding from JD: {str(e)}")

        # Step 4: Map field to vector path
        vector_field_mapping = {
            "skills": "skills_vector",
            "experience": "experience_text_vector",
            "education": "academic_details_vector",
            "projects": "experience_text_vector",  # Projects search through experience for better relevance
            "full_text": "combined_resume_vector",
        }

        vector_field = vector_field_mapping.get(field)
        if not vector_field:
            raise SearchError(
                f"Invalid field name. Choose from: {', '.join(vector_field_mapping.keys())}"
            )

        # Step 5: Run vector search pipeline with admin access control
        match_filter = {}
        if effective_user_id:  # If None, admin searches all documents
            match_filter["user_id"] = effective_user_id

        pipeline = [
            {
                "$search": {
                    "index": "vector_search_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": vector_field,
                        "k": num_results * 5,  # Get more results to filter later
                    },
                }
            },
            {"$set": {"score": {"$meta": "searchScore"}}},
            {
                "$match": {
                    **{"score": {"$gte": min_score}},
                    **match_filter,  # Add user_id filter only if not admin
                }
            },
            {
                "$project": {
                    "user_id": 1,
                    "username": 1,
                    "name": 1,
                    "contact_details": 1,
                    "education": 1,
                    "experience": 1,
                    "projects": 1,
                    "total_experience": 1,
                    "skills": 1,
                    "certifications": 1,
                    "score": 1,
                }
            },
            {"$sort": {"score": -1}},
        ]

        try:
            results = list(resumes_collection.aggregate(pipeline))
        except Exception as e:
            raise SearchError(f"Database query failed: {str(e)}")

        if not results:
            return []

        # Limit results to requested number
        limited_results = results[:num_results]

        formatted_results = [format_resume(result) for result in limited_results]

        # Apply priority-based scoring and add relevance score to results
        for result in formatted_results:
            # Calculate priority score based on JD content
            priority_score, match_reason = calculate_priority_score(result, jd_text)

            # Get base vector score
            base_score = result.get("score", 0)

            # Combine priority score with vector similarity score
            if priority_score > 0:
                # Priority score takes precedence (70%) with vector score as secondary (30%)
                final_score = (priority_score * 0.7) + (base_score * 0.3)
            else:
                # Fall back to vector score only
                final_score = base_score

            # Set relevance score and additional metrics
            result["relevance_score"] = round(final_score * 100, 2)
            result["match_reason"] = match_reason
            result["priority_score"] = round(priority_score * 100, 2)
            result["vector_score"] = round(base_score * 100, 2)

        # Sort results by priority-based relevance score (highest first)
        formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Filter results based on relevant_score threshold
        if relevant_score > 0:
            formatted_results = [
                result
                for result in formatted_results
                if result.get("relevance_score", 0) >= relevant_score
            ]

        return formatted_results

    except SearchError as se:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(se))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
