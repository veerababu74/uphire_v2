from fastapi import APIRouter, HTTPException, File, UploadFile, status
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
import os
import re
from Retrivers.retriver import MangoRetriever, LangChainRetriever

from GroqcloudLLM.text_extraction import extract_and_clean_text
from core.config import AppConfig
from core.custom_logger import CustomLogger
from datetime import datetime, timedelta
from pathlib import Path
from recent_search_uts.recent_ai_search import save_ai_search_to_recent
from schemas.user_schemas import UserSearchRequest
from mangodatabase.client import get_users_collection
from mangodatabase.user_operations import UserOperations

# Initialize logger
logger_instance = CustomLogger()
logger = logger_instance.get_logger("retriever_api")

# Initialize router
router = APIRouter(
    prefix="/search",
    tags=[
        "enhanced ai vector search - retriever",
    ],
    responses={404: {"description": "Not found"}},
)

# Initialize retrievers with lazy initialization to avoid startup blocking
mango_retriever = MangoRetriever(lazy_init=True)
langchain_retriever = LangChainRetriever(lazy_init=True)

# Initialize user operations
users_collection = get_users_collection()
user_ops = UserOperations(users_collection)

# Setup temp directory
TEMP_FOLDER = "temp_uploads"
TEMP_DIR = Path(TEMP_FOLDER)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


logger_manager = CustomLogger()
logging = logger_manager.get_logger("vector_search_api")


def cleanup_temp_directory(age_limit_minutes: int = 60):
    """Cleanup temporary directory by deleting files older than the specified age limit."""
    now = datetime.now()


def calculate_priority_score(
    candidate: Dict[str, Any], query: str
) -> Tuple[float, str]:
    """
    Calculate priority-based relevance score for retriever search results:
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
    reason = "; ".join(reasons) if reasons else "Basic retriever similarity"

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


def apply_priority_scoring_to_results(
    results: Dict, query: str, relevant_score_threshold: Optional[float] = None
) -> Dict:
    """Apply priority-based scoring to search results and filter based on threshold"""
    if "results" not in results:
        return results

    enhanced_results = []
    for result in results["results"]:
        # Calculate priority score
        priority_score, match_reason = calculate_priority_score(result, query)

        # Get base retriever score
        base_score = result.get(
            "relevance_score",
            result.get("match_score", result.get("score", 0)),
        )

        # Normalize base score to 0-1 range if it's 0-100
        if base_score > 1.0:
            base_score = base_score / 100

        # Combine priority score with retriever score
        if priority_score > 0:
            # Priority score takes precedence (70%) with retriever score as secondary (30%)
            final_score = (priority_score * 0.7) + (base_score * 0.3)
        else:
            # Fall back to retriever score only
            final_score = base_score

        # Update result with enhanced scores
        result["relevance_score"] = final_score * 100  # Convert back to 0-100 range
        result["match_reason"] = match_reason
        result["priority_score"] = priority_score * 100
        result["retriever_score"] = base_score * 100

        enhanced_results.append(result)

    # Sort by priority-based relevance score
    enhanced_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    results["results"] = enhanced_results

    # Filter based on relevant_score threshold
    if relevant_score_threshold is not None and relevant_score_threshold > 0:
        filtered_results = [
            result
            for result in enhanced_results
            if result.get("relevance_score", 0) >= relevant_score_threshold
        ]
        results["results"] = filtered_results
        results["total_count"] = len(filtered_results)
    else:
        results["total_count"] = len(enhanced_results)

    return results

    for file_path in TEMP_DIR.iterdir():
        if file_path.is_file():
            file_age = now - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age > timedelta(minutes=age_limit_minutes):
                try:
                    file_path.unlink()
                    logging.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logging.error(f"Failed to delete file {file_path}: {e}")


class SearchRequest(BaseModel):
    user_id: str = Field(
        ...,
        description="User ID - if exists in users collection can search all documents, otherwise only their own",
    )
    query: str
    limit: Optional[int] = 5
    relevant_score: Optional[float] = Field(
        default=40.0,
        ge=0.0,
        le=100.0,
        description="Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned",
    )


class SearchResponse(BaseModel):
    results: List[Dict]
    total_count: int
    query: str


class SearchError(Exception):
    """Custom exception for search-related errors"""

    pass


def get_effective_user_id_for_search(requesting_user_id: str) -> Optional[str]:
    """
    Determine the effective user_id for search based on user existence in collection.

    Args:
        requesting_user_id: The user_id making the request

    Returns:
        - None if user exists in users collection (search all documents)
        - requesting_user_id if user does not exist in users collection (search only their documents)
    """
    try:
        user_exists = user_ops.user_exists(requesting_user_id)
        if user_exists:
            logger.info(
                f"User {requesting_user_id} exists in collection - searching all documents"
            )
            return None  # User exists in collection - can search all documents
        else:
            logger.info(
                f"User {requesting_user_id} not in collection - searching only their documents"
            )
            return requesting_user_id  # User not in collection - can only search their own documents
    except Exception as e:
        logger.warning(
            f"Error checking user existence for user {requesting_user_id}: {e}"
        )
        # On error, default to restricting to user's own documents for security
        return requesting_user_id


@router.post("/mango", response_model=SearchResponse)
async def search_mango(request: SearchRequest):
    """
    Search using the Mango retriever

    - Users in users collection can search all documents across all users
    - Users not in users collection can only search their own documents
    """
    try:
        # Determine effective user_id for search based on user existence in collection
        effective_user_id = get_effective_user_id_for_search(request.user_id)

        results = mango_retriever.search_and_rank(
            request.query, request.limit, effective_user_id
        )
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])

        # Apply priority-based scoring and filtering
        results = apply_priority_scoring_to_results(
            results, request.query, request.relevant_score
        )

        # Save the search to recent searches
        if request.user_id:
            await save_ai_search_to_recent(request.user_id, request.query)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/langchain", response_model=SearchResponse)
async def search_langchain(request: SearchRequest):
    """
    Search using the LangChain retriever

    - Users in users collection can search all documents across all users
    - Users not in users collection can only search their own documents
    """
    try:
        # Determine effective user_id for search based on user existence in collection
        effective_user_id = get_effective_user_id_for_search(request.user_id)

        results = langchain_retriever.search_and_rank(
            request.query, request.limit, effective_user_id
        )
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])

        # Apply priority-based scoring and filtering
        results = apply_priority_scoring_to_results(
            results, request.query, request.relevant_score
        )

        # Save the search to recent searches
        if request.user_id:
            await save_ai_search_to_recent(request.user_id, request.query)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/mango/search-by-jd",
    response_model=SearchResponse,
    summary="AI-Powered Resume Search Based on Job Description File",
    description="""
    Upload a job description file (.txt, .pdf, or .docx) and find matching resumes.
    
    The system will extract and clean the text from the file and use AI to find semantically relevant candidates.
    
    **Access Control:**
    - Users in users collection can search all documents across all users
    - Users not in users collection can only search their own documents
    
    **Parameters:**
    - user_id: User ID (MANDATORY) - determines search scope based on user existence in collection
    - file: Job description file (.txt, .pdf, or .docx)
    - limit: Maximum number of results to return (default: 10)
    - relevant_score: Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned (default: 40.0)
    """,
)
async def search_by_jd_mango(
    user_id: str,
    file: UploadFile = File(...),
    limit: int = 10,
    relevant_score: float = 40.0,
):
    try:
        # Determine effective user_id for search based on user existence in collection
        effective_user_id = get_effective_user_id_for_search(user_id)

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

        # Step 3: Generate embedding from cleaned JD text and search
        try:
            results = mango_retriever.search_and_rank(jd_text, limit, effective_user_id)

            # Filter results based on relevant_score threshold
            if relevant_score > 0:
                if "results" in results:
                    filtered_results = []
                    for result in results["results"]:
                        # Check various possible score fields
                        score = result.get(
                            "relevance_score",
                            result.get("match_score", result.get("score", 0)),
                        )
                        # Normalize score to 0-100 range if it's between 0-1
                        if score <= 1.0:
                            normalized_score = score * 100
                        else:
                            normalized_score = score

                        # Update the result with normalized score
                        result["relevance_score"] = normalized_score

                        if normalized_score >= relevant_score:
                            filtered_results.append(result)
                    results["results"] = filtered_results
                    results["total_count"] = len(filtered_results)

            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    except SearchError as se:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(se))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.post(
    "/langchain/search-by-jd",
    response_model=SearchResponse,
    summary="AI-Powered Resume Search Based on Job Description File",
    description="""
    Upload a job description file (.txt, .pdf, or .docx) and find matching resumes.
    
    The system will extract and clean the text from the file and use AI to find semantically relevant candidates.
    
    **Access Control:**
    - Users in users collection can search all documents across all users
    - Users not in users collection can only search their own documents
    
    **Parameters:**
    - user_id: User ID (MANDATORY) - determines search scope based on user existence in collection
    - file: Job description file (.txt, .pdf, or .docx)
    - limit: Maximum number of results to return (default: 10)
    - relevant_score: Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned (default: 40.0)
    """,
)
async def search_by_jd_langchain(
    user_id: str,
    file: UploadFile = File(...),
    limit: int = 10,
    relevant_score: float = 40.0,
):
    try:
        # Determine effective user_id for search based on user existence in collection
        effective_user_id = get_effective_user_id_for_search(user_id)

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

        # Step 3: Generate embedding from cleaned JD text and search
        try:
            results = langchain_retriever.search_and_rank(
                jd_text, limit, effective_user_id
            )

            # Filter results based on relevant_score threshold
            if relevant_score > 0:
                if "results" in results:
                    filtered_results = []
                    for result in results["results"]:
                        # Check various possible score fields
                        score = result.get(
                            "relevance_score",
                            result.get("match_score", result.get("score", 0)),
                        )
                        # Normalize score to 0-100 range if it's between 0-1
                        if score <= 1.0:
                            normalized_score = score * 100
                        else:
                            normalized_score = score

                        # Update the result with normalized score
                        result["relevance_score"] = normalized_score

                        if normalized_score >= relevant_score:
                            filtered_results.append(result)
                    results["results"] = filtered_results
                    results["total_count"] = len(filtered_results)

            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    except SearchError as se:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(se))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
