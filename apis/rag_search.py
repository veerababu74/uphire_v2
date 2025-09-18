from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi import status
from typing import List, Dict, Any, Optional
from Rag.runner import initialize_rag_app, ask_resume_question_enhanced
from core.custom_logger import CustomLogger
from pydantic import BaseModel, Field
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
from GroqcloudLLM.text_extraction import extract_and_clean_text
from core.custom_logger import CustomLogger
from recent_search_uts.recent_ai_search import save_ai_search_to_recent
from mangodatabase.client import get_users_collection
from mangodatabase.user_operations import UserOperations
from mangodatabase.client import get_users_collection
from mangodatabase.user_operations import UserOperations


def safe_float(value, default=0.0):
    """Safely convert value to float, handling None values"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_str(value, default=""):
    """Safely convert value to string, handling None values"""
    if value is None:
        return default
    return str(value)


def safe_list(value, default=None):
    """Safely convert value to list, handling None values"""
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, list):
        return value
    return default


def safe_object_id(value, default=""):
    """Safely convert ObjectId or any object to string, handling None values"""
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


# Initialize logger
logger = CustomLogger().get_logger("rag_search")

# Define base folders
BASE_FOLDER = "dummy_data_save"
TEMP_FOLDER = os.path.join(BASE_FOLDER, "temp_text_extract")
TEMP_DIR = Path(os.path.join(BASE_FOLDER, "temp_files"))

# Ensure the directories exist
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging


class SearchError(Exception):
    """Custom exception for search-related errors"""

    pass


# Pydantic models for request bodies
class VectorSimilaritySearchRequest(BaseModel):
    user_id: str = Field(
        ...,
        description="User ID - if exists in users collection can search all documents, otherwise only their own",
    )
    query: str = Field(..., description="Search query text")
    limit: int = Field(
        default=50, description="Maximum number of results to return", ge=1, le=100
    )
    relevant_score: Optional[float] = Field(
        default=40.0,
        ge=0.0,
        le=100.0,
        description="Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned",
    )
    use_enhanced_search: Optional[bool] = Field(
        default=True,
        description="Use enhanced search engine with better query understanding and relevance scoring (optional, defaults to True)",
    )


class LLMContextSearchRequest(BaseModel):
    user_id: str = Field(
        ...,
        description="User ID - if exists in users collection can search all documents, otherwise only their own",
    )
    query: str = Field(..., description="Search query text")
    context_size: int = Field(
        default=5, description="Number of documents to analyze", ge=1, le=20
    )
    relevant_score: Optional[float] = Field(
        default=40.0,
        ge=0.0,
        le=100.0,
        description="Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned",
    )
    use_enhanced_search: Optional[bool] = Field(
        default=True,
        description="Use enhanced search engine with better query understanding and context analysis (optional, defaults to True)",
    )


# Pydantic models for response bodies
class ContactDetails(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    alternative_phone: str = ""
    current_city: str = ""
    looking_for_jobs_in: List[str] = []
    gender: str = ""
    age: int = 0
    naukri_profile: str = ""
    linkedin_profile: str = ""
    portfolio_link: str = ""
    pan_card: str = ""
    aadhar_card: str = ""


class VectorSearchResult(BaseModel):
    model_config = {"populate_by_name": True}

    id: str = Field(default="", serialization_alias="_id")
    user_id: str = ""
    username: str = ""
    contact_details: ContactDetails
    total_experience: str = ""
    notice_period: str = ""
    currency: str = ""
    pay_duration: str = ""
    current_salary: float = 0.0
    hike: float = 0.0
    expected_salary: float = 0.0
    skills: List[str] = []
    may_also_known_skills: List[str] = []
    labels: List[str] = []
    experience: List[Dict[str, Any]] = []
    academic_details: List[Dict[str, Any]] = []
    source: str = ""
    last_working_day: str = ""
    is_tier1_mba: bool = False
    is_tier1_engineering: bool = False
    comment: str = ""
    exit_reason: str = ""
    similarity_score: float
    vector_score: Optional[float] = None  # Original vector similarity score
    match_reason: Optional[str] = ""  # Enhanced match explanation


class LLMSearchResult(BaseModel):
    model_config = {"populate_by_name": True}

    id: str = Field(default="", serialization_alias="_id")
    user_id: str = ""
    username: str = ""
    contact_details: ContactDetails
    total_experience: str = ""
    notice_period: str = ""
    currency: str = ""
    pay_duration: str = ""
    current_salary: float = 0.0
    hike: float = 0.0
    expected_salary: float = 0.0
    skills: List[str] = []
    may_also_known_skills: List[str] = []
    labels: List[str] = []
    experience: List[Dict[str, Any]] = []
    academic_details: List[Dict[str, Any]] = []
    source: str = ""
    last_working_day: str = ""
    is_tier1_mba: bool = False
    is_tier1_engineering: bool = False
    comment: str = ""
    exit_reason: str = ""
    relevance_score: float
    llm_score: Optional[float] = None  # Original LLM relevance score
    match_reason: str = ""


class SearchStatistics(BaseModel):
    retrieved: int = 0
    analyzed: Optional[int] = None
    query: str = ""


class VectorSimilaritySearchResponse(BaseModel):
    results: List[VectorSearchResult]
    total_found: int
    statistics: SearchStatistics


class LLMContextSearchResponse(BaseModel):
    results: List[LLMSearchResult]
    total_analyzed: int
    statistics: SearchStatistics


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


# Create router instance
router = APIRouter(
    prefix="/rag",
    tags=[
        "ai rag search",
    ],
    responses={404: {"description": "Not found"}},
)

# Initialize user operations for admin checking
users_collection = get_users_collection()
user_ops = UserOperations(users_collection)


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


@router.post(
    "/vector-similarity-search",
    response_model=VectorSimilaritySearchResponse,
    summary="Vector Similarity Search",
    description="""
    Perform vector similarity search on resumes using the RAG system.
    
    **Parameters:**
    - user_id: The user ID who performed the search (mandatory)
    - query: The search query text
    - limit: Maximum number of results to return (default: 50)
    - relevant_score: Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned (default: 40.0)
    - use_enhanced_search: Use enhanced search with better query understanding and relevance scoring (optional, default: True)
    
    **Enhanced Search Features (enabled by default):**
    - Intelligent query parsing (extracts role, experience, salary, skills, domain)
    - Multi-factor relevance scoring with weighted criteria
    - Better filtering and ranking based on parsed requirements
    - Detailed match explanations for each candidate
    
    **Returns:**
    Dictionary containing:
    - total_found: Total number of matches found
    - results: List of matching resumes with similarity scores (filtered by user_id)
    """,
    responses={
        200: {
            "description": "Successful search results",
            "content": {
                "application/json": {
                    "example": {
                        "total_found": 10,
                        "results": [
                            {
                                "_id": "687bc9ec05f065045059f618",
                                "user_id": "user123",
                                "username": "john_doe",
                                "contact_details": {
                                    "name": "John Doe",
                                    "current_city": "Mumbai",
                                },
                                "skills": ["Python", "React", "AWS"],
                                "total_experience": "5 years",
                                "similarity_score": 0.85,
                            }
                        ],
                    }
                }
            },
        },
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def vector_similarity_search(request: VectorSimilaritySearchRequest):
    """
    Perform vector similarity search on resumes.
    """
    try:
        # Initialize RAG application
        rag_app = initialize_rag_app()

        # Determine effective user_id for search based on user existence in collection
        effective_user_id = get_effective_user_id_for_search(request.user_id)

        # Perform vector similarity search with user_id filter
        result = rag_app.vector_similarity_search(
            request.query,
            request.limit,
            user_id=effective_user_id,
            use_enhanced=request.use_enhanced_search,
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Format results to match the expected VectorSimilaritySearchResponse structure
        formatted_results = []
        for candidate in result.get("results", []):
            # Handle _id field more explicitly
            candidate_id = candidate.get("_id")
            if candidate_id is None:
                candidate_id = candidate.get("id", "")  # Try alternative key
            candidate_id = str(candidate_id) if candidate_id is not None else ""

            formatted_candidate = {
                "_id": candidate_id,
                "user_id": candidate.get("user_id", ""),
                "username": candidate.get("username", ""),
                "contact_details": {
                    "name": candidate.get("contact_details", {}).get("name", ""),
                    "email": candidate.get("contact_details", {}).get("email", ""),
                    "phone": candidate.get("contact_details", {}).get("phone", ""),
                    "alternative_phone": candidate.get("contact_details", {}).get(
                        "alternative_phone", ""
                    ),
                    "current_city": candidate.get("contact_details", {}).get(
                        "current_city", ""
                    ),
                    "looking_for_jobs_in": safe_list(
                        candidate.get("contact_details", {}).get(
                            "looking_for_jobs_in", []
                        )
                    ),
                    "gender": candidate.get("contact_details", {}).get("gender", ""),
                    "age": candidate.get("contact_details", {}).get("age", 0),
                    "naukri_profile": candidate.get("contact_details", {}).get(
                        "naukri_profile", ""
                    ),
                    "linkedin_profile": candidate.get("contact_details", {}).get(
                        "linkedin_profile", ""
                    ),
                    "portfolio_link": candidate.get("contact_details", {}).get(
                        "portfolio_link", ""
                    ),
                    "pan_card": candidate.get("contact_details", {}).get(
                        "pan_card", ""
                    ),
                    "aadhar_card": candidate.get("contact_details", {}).get(
                        "aadhar_card", ""
                    ),
                },
                "total_experience": safe_str(candidate.get("total_experience", "0.0")),
                "notice_period": candidate.get("notice_period", ""),
                "currency": candidate.get("currency", ""),
                "pay_duration": candidate.get("pay_duration", ""),
                "current_salary": safe_float(candidate.get("current_salary", 0)),
                "hike": safe_float(candidate.get("hike", 0)),
                "expected_salary": safe_float(candidate.get("expected_salary", 0)),
                "skills": safe_list(candidate.get("skills", [])),
                "may_also_known_skills": safe_list(
                    candidate.get("may_also_known_skills", [])
                ),
                "labels": safe_list(candidate.get("labels", [])),
                "experience": safe_list(candidate.get("experience", [])),
                "academic_details": safe_list(candidate.get("academic_details", [])),
                "source": candidate.get("source", ""),
                "last_working_day": candidate.get("last_working_day", ""),
                "is_tier1_mba": bool(candidate.get("is_tier1_mba", False)),
                "is_tier1_engineering": bool(
                    candidate.get("is_tier1_engineering", False)
                ),
                "comment": candidate.get("comment", ""),
                "exit_reason": candidate.get("exit_reason", ""),
                "similarity_score": safe_float(candidate.get("similarity_score", 0.0)),
                "vector_score": safe_float(
                    candidate.get("vector_score")
                ),  # Enhanced: Include original vector score
                "match_reason": candidate.get(
                    "match_reason", ""
                ),  # Enhanced: Include match explanation
            }

            # Normalize similarity_score to 0-100 range if it's in 0-1 range
            similarity_score = formatted_candidate["similarity_score"]
            if similarity_score <= 1.0:
                formatted_candidate["similarity_score"] = round(
                    similarity_score * 100, 2
                )

            formatted_results.append(formatted_candidate)

        # Filter results based on relevant_score threshold
        if request.relevant_score is not None and request.relevant_score > 0:
            formatted_results = [
                result
                for result in formatted_results
                if result.get("similarity_score", 0) >= request.relevant_score
            ]

        # Format response according to VectorSimilaritySearchResponse model
        formatted_response = {
            "results": formatted_results,
            "total_found": len(formatted_results),  # Update total_found after filtering
            "statistics": {
                "retrieved": result.get("statistics", {}).get("retrieved", 0),
                "query": result.get("statistics", {}).get("query", request.query),
            },
        }

        # Save the search to recent searches
        await save_ai_search_to_recent(request.user_id, request.query)

        logger.info(
            f"Vector similarity search completed successfully. Found {len(formatted_results)} results"
        )
        return formatted_response

    except Exception as e:
        logger.error(f"Vector similarity search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post(
    "/llm-context-search",
    response_model=LLMContextSearchResponse,
    summary="LLM Context Search",
    description="""
    Perform LLM-powered context search on resumes using the RAG system.
    
    **Parameters:**
    - user_id: The user ID who performed the search (mandatory)
    - query: The search query text
    - context_size: Number of documents to analyze (default: 5)
    - relevant_score: Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned (default: 40.0)
    - use_enhanced_search: Use enhanced search with better query understanding and context analysis (optional, default: True)
    
    **Enhanced Search Features (enabled by default):**
    - Intelligent query parsing and requirement extraction
    - Context-aware LLM prompts with structured search requirements
    - Multi-factor scoring combining LLM analysis with domain-specific criteria
    - Enhanced candidate filtering and ranking
    - Detailed match explanations and reasoning
    
    **Returns:**
    Dictionary containing:
    - total_found: Total number of matches found
    - total_analyzed: Number of documents analyzed
    - statistics: Search statistics
    - results: List of matching resumes with relevance scores and match reasons (filtered by user_id)
    """,
    responses={
        200: {
            "description": "Successful search results",
            "content": {
                "application/json": {
                    "example": {
                        "total_found": 10,
                        "total_analyzed": 5,
                        "statistics": {
                            "avg_relevance": 0.85,
                            "match_distribution": {"high": 3, "medium": 2, "low": 0},
                        },
                        "results": [
                            {
                                "_id": "687bc9ec05f065045059f618",
                                "user_id": "user123",
                                "username": "john_doe",
                                "contact_details": {
                                    "name": "John Doe",
                                    "current_city": "Mumbai",
                                },
                                "skills": ["Python", "React", "AWS"],
                                "total_experience": "5 years",
                                "relevance_score": 0.92,
                                "match_reason": "Strong match in Python and AWS skills",
                            }
                        ],
                    }
                }
            },
        },
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def llm_context_search(request: LLMContextSearchRequest):
    """
    Perform LLM-powered context search on resumes.
    """
    try:
        # Initialize RAG application
        rag_app = initialize_rag_app()

        # Determine effective user_id for search based on user existence in collection
        effective_user_id = get_effective_user_id_for_search(request.user_id)

        # Perform LLM context search with user_id filter
        result = rag_app.llm_context_search(
            request.query,
            request.context_size,
            user_id=effective_user_id,
            use_enhanced=request.use_enhanced_search,
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Normalize scores to 0-100 range if they're in 0-1 range and fix _id field
        if "results" in result:
            for res in result["results"]:
                # Handle _id field properly - ensure it's always a string
                candidate_id = res.get("_id")
                if candidate_id is None:
                    candidate_id = res.get("id", "")  # Try alternative key
                res["_id"] = str(candidate_id) if candidate_id is not None else ""

                # Normalize relevance_score
                if "relevance_score" in res:
                    relevance_score = res["relevance_score"]
                    if relevance_score <= 1.0:
                        res["relevance_score"] = round(relevance_score * 100, 2)

                # Normalize match_score
                if "match_score" in res:
                    match_score = res["match_score"]
                    if match_score <= 1.0:
                        res["match_score"] = round(match_score * 100, 2)

        # Filter results based on relevant_score threshold
        if (
            request.relevant_score is not None
            and request.relevant_score > 0
            and "results" in result
        ):
            result["results"] = [
                res
                for res in result["results"]
                if res.get("relevance_score", 0) >= request.relevant_score
                or res.get("match_score", 0) >= request.relevant_score
            ]
            # Update total_found after filtering
            result["total_found"] = len(result["results"])

        # save the search to recent searches
        await save_ai_search_to_recent(request.user_id, request.query)

        return result

    except Exception as e:
        logger.error(f"LLM context search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post(
    "/llm-context-search/by-jd",
    response_model=LLMContextSearchResponse,
    summary="AI-Powered Resume Search Based on Job Description File",
    description="""
    Upload a job description file (.txt, .pdf, or .docx) and find matching resumes for a specific user.
    
    **Parameters:**
    - user_id: The user ID who performed the search (mandatory)
    - file: Job description file to upload
    - limit: Maximum number of results to return (default: 10)
    
    The system will extract and clean the text from the file and use AI to find semantically relevant candidates for the specified user only.
    """,
)
async def llm_search_by_jd(
    user_id: str = Query(
        ..., description="User ID who performed the search (mandatory)"
    ),
    file: UploadFile = File(...),
    limit: int = 10,
):
    try:
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

            # Extract text and clean it
            jd_text = extract_and_clean_text(file_location)

            # Additional cleaning to remove code-like content
            if jd_text.startswith("#") or "import " in jd_text or "def " in jd_text:
                # This appears to be code, not a job description
                raise SearchError(
                    "Invalid job description format. The file appears to contain code instead of a job description."
                )

            if not jd_text.strip():
                raise SearchError("Extracted job description is empty.")

            # Log the cleaned text for debugging
            logger.info(
                f"Cleaned job description text: {jd_text[:200]}..."
            )  # Log first 200 chars

        finally:
            # Clean up the temporary file
            try:
                os.remove(file_location)
                logging.info(f"Deleted temporary file: {file_location}")
            except Exception as e:
                logging.error(f"Failed to delete temporary file {file_location}: {e}")

        try:
            # Initialize RAG application
            logger.info(
                f"Initializing RAG application for JD search with limit: {limit}"
            )
            rag_app = initialize_rag_app()

            # Determine effective user_id for search based on user existence in collection
            effective_user_id = get_effective_user_id_for_search(user_id)

            # Perform LLM context search with user_id filter
            logger.info(
                f"Performing LLM context search with text length: {len(jd_text)} for user: {user_id} (effective_user_id: {effective_user_id})"
            )
            result = rag_app.llm_context_search(
                jd_text, context_size=limit, user_id=effective_user_id
            )

            # Log the result for debugging
            logger.info(f"LLM context search completed. Result type: {type(result)}")
            if isinstance(result, dict):
                logger.info(
                    f"Result keys: {list(result.keys()) if result else 'Empty dict'}"
                )

            # Check for errors in result
            if "error" in result:
                logger.error(f"RAG application returned error: {result['error']}")
                raise HTTPException(status_code=500, detail=result["error"])

            # Validate result structure
            if not isinstance(result, dict):
                logger.error(f"Invalid result type: {type(result)}")
                raise HTTPException(
                    status_code=500, detail="Invalid response format from RAG system"
                )

            # Format results to match the expected structure
            formatted_results = []
            for candidate in result.get("results", []):
                formatted_candidate = {
                    "_id": safe_object_id(candidate.get("_id", "")),
                    "user_id": candidate.get("user_id", ""),
                    "username": candidate.get("username", ""),
                    "contact_details": {
                        "name": candidate.get("contact_details", {}).get("name", ""),
                        "email": candidate.get("contact_details", {}).get("email", ""),
                        "phone": candidate.get("contact_details", {}).get("phone", ""),
                        "alternative_phone": candidate.get("contact_details", {}).get(
                            "alternative_phone", ""
                        ),
                        "current_city": candidate.get("contact_details", {}).get(
                            "current_city", ""
                        ),
                        "looking_for_jobs_in": safe_list(
                            candidate.get("contact_details", {}).get(
                                "looking_for_jobs_in", []
                            )
                        ),
                        "gender": candidate.get("contact_details", {}).get(
                            "gender", ""
                        ),
                        "age": candidate.get("contact_details", {}).get("age", 0),
                        "naukri_profile": candidate.get("contact_details", {}).get(
                            "naukri_profile", ""
                        ),
                        "linkedin_profile": candidate.get("contact_details", {}).get(
                            "linkedin_profile", ""
                        ),
                        "portfolio_link": candidate.get("contact_details", {}).get(
                            "portfolio_link", ""
                        ),
                        "pan_card": candidate.get("contact_details", {}).get(
                            "pan_card", ""
                        ),
                        "aadhar_card": candidate.get("contact_details", {}).get(
                            "aadhar_card", ""
                        ),
                    },
                    "total_experience": safe_str(
                        candidate.get("total_experience", "0.0")
                    ),
                    "notice_period": candidate.get("notice_period", ""),
                    "currency": candidate.get("currency", ""),
                    "pay_duration": candidate.get("pay_duration", ""),
                    "current_salary": safe_float(candidate.get("current_salary", 0)),
                    "hike": safe_float(candidate.get("hike", 0)),
                    "expected_salary": safe_float(candidate.get("expected_salary", 0)),
                    "skills": safe_list(candidate.get("skills", [])),
                    "may_also_known_skills": safe_list(
                        candidate.get("may_also_known_skills", [])
                    ),
                    "labels": safe_list(candidate.get("labels", [])),
                    "experience": safe_list(candidate.get("experience", [])),
                    "academic_details": safe_list(
                        candidate.get("academic_details", [])
                    ),
                    "source": candidate.get("source", ""),
                    "last_working_day": candidate.get("last_working_day", ""),
                    "is_tier1_mba": bool(candidate.get("is_tier1_mba", False)),
                    "is_tier1_engineering": bool(
                        candidate.get("is_tier1_engineering", False)
                    ),
                    "comment": candidate.get("comment", ""),
                    "exit_reason": candidate.get("exit_reason", ""),
                    "relevance_score": safe_float(
                        candidate.get("relevance_score", 0.0)
                    ),
                    "match_reason": candidate.get("match_reason", ""),
                }

                # Normalize relevance_score to 0-100 range if it's in 0-1 range
                relevance_score = formatted_candidate["relevance_score"]
                if relevance_score <= 1.0:
                    formatted_candidate["relevance_score"] = round(
                        relevance_score * 100, 2
                    )

                formatted_results.append(formatted_candidate)

            # Format response according to LLMContextSearchResponse model
            formatted_response = {
                "results": formatted_results,
                "total_analyzed": result.get("total_analyzed", 0),
                "statistics": {
                    "retrieved": result.get("statistics", {}).get("retrieved", 0),
                    "analyzed": result.get("statistics", {}).get("analyzed", 0),
                    "query": (
                        jd_text[:200] + "..." if len(jd_text) > 200 else jd_text
                    ),  # Truncate long queries
                },
            }

            logger.info("LLM context search by JD completed successfully")
            return formatted_response

        except Exception as e:
            logger.error(f"Error during RAG search: {str(e)}")
            raise HTTPException(status_code=500, detail=f"RAG search failed: {str(e)}")

    except SearchError as se:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(se))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.post(
    "/vector-similarity-search/by-jd",
    response_model=VectorSimilaritySearchResponse,
    summary="AI-Powered Resume Search Based on Job Description File",
    description="""
    Upload a job description file (.txt, .pdf, or .docx) and find matching resumes for a specific user.
    
    **Parameters:**
    - user_id: The user ID who performed the search (mandatory)
    - file: Job description file to upload
    - limit: Maximum number of results to return (default: 10)
    
    The system will extract and clean the text from the file and use AI to find semantically relevant candidates for the specified user only.
    """,
)
async def vector_search_by_jd(
    user_id: str = Query(
        ..., description="User ID who performed the search (mandatory)"
    ),
    file: UploadFile = File(...),
    limit: int = 10,
):
    try:
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
                logging.error(
                    f"Failed to delete temporary file {file_location}: {e}"
                )  # Step 3: Generate embedding from cleaned JD text
        try:
            # Initialize RAG application
            rag_app = initialize_rag_app()

            # Perform vector similarity search with user_id filter
            result = rag_app.vector_similarity_search(jd_text, limit, user_id=user_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Format results to match the expected VectorSimilaritySearchResponse structure
        formatted_results = []
        for candidate in result.get("results", []):
            formatted_candidate = {
                "_id": safe_object_id(candidate.get("_id", "")),
                "user_id": candidate.get("user_id", ""),
                "username": candidate.get("username", ""),
                "contact_details": {
                    "name": candidate.get("contact_details", {}).get("name", ""),
                    "email": candidate.get("contact_details", {}).get("email", ""),
                    "phone": candidate.get("contact_details", {}).get("phone", ""),
                    "alternative_phone": candidate.get("contact_details", {}).get(
                        "alternative_phone", ""
                    ),
                    "current_city": candidate.get("contact_details", {}).get(
                        "current_city", ""
                    ),
                    "looking_for_jobs_in": safe_list(
                        candidate.get("contact_details", {}).get(
                            "looking_for_jobs_in", []
                        )
                    ),
                    "gender": candidate.get("contact_details", {}).get("gender", ""),
                    "age": candidate.get("contact_details", {}).get("age", 0),
                    "naukri_profile": candidate.get("contact_details", {}).get(
                        "naukri_profile", ""
                    ),
                    "linkedin_profile": candidate.get("contact_details", {}).get(
                        "linkedin_profile", ""
                    ),
                    "portfolio_link": candidate.get("contact_details", {}).get(
                        "portfolio_link", ""
                    ),
                    "pan_card": candidate.get("contact_details", {}).get(
                        "pan_card", ""
                    ),
                    "aadhar_card": candidate.get("contact_details", {}).get(
                        "aadhar_card", ""
                    ),
                },
                "total_experience": safe_str(candidate.get("total_experience", "0.0")),
                "notice_period": candidate.get("notice_period", ""),
                "currency": candidate.get("currency", ""),
                "pay_duration": candidate.get("pay_duration", ""),
                "current_salary": safe_float(candidate.get("current_salary", 0)),
                "hike": safe_float(candidate.get("hike", 0)),
                "expected_salary": safe_float(candidate.get("expected_salary", 0)),
                "skills": safe_list(candidate.get("skills", [])),
                "may_also_known_skills": safe_list(
                    candidate.get("may_also_known_skills", [])
                ),
                "labels": safe_list(candidate.get("labels", [])),
                "experience": safe_list(candidate.get("experience", [])),
                "academic_details": safe_list(candidate.get("academic_details", [])),
                "source": candidate.get("source", ""),
                "last_working_day": candidate.get("last_working_day", ""),
                "is_tier1_mba": bool(candidate.get("is_tier1_mba", False)),
                "is_tier1_engineering": bool(
                    candidate.get("is_tier1_engineering", False)
                ),
                "comment": candidate.get("comment", ""),
                "exit_reason": candidate.get("exit_reason", ""),
                "similarity_score": safe_float(candidate.get("similarity_score", 0.0)),
            }

            # Normalize similarity_score to 0-100 range if it's in 0-1 range
            similarity_score = formatted_candidate["similarity_score"]
            if similarity_score <= 1.0:
                formatted_candidate["similarity_score"] = round(
                    similarity_score * 100, 2
                )

            formatted_results.append(formatted_candidate)

        # Format response according to VectorSimilaritySearchResponse model
        formatted_response = {
            "results": formatted_results,
            "total_found": result.get("total_found", 0),
            "statistics": {
                "retrieved": result.get("statistics", {}).get("retrieved", 0),
                "query": (
                    jd_text[:200] + "..." if len(jd_text) > 200 else jd_text
                ),  # Truncate long queries
            },
        }

        logger.info("Vector similarity search by JD completed successfully")
        return formatted_response

    except SearchError as se:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(se))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
