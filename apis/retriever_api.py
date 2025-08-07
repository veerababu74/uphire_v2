from fastapi import APIRouter, HTTPException, File, UploadFile, status
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import os
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

        # Filter results based on relevant_score threshold
        if request.relevant_score is not None and request.relevant_score > 0:
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

                    if normalized_score >= request.relevant_score:
                        filtered_results.append(result)
                results["results"] = filtered_results
                results["total_count"] = len(filtered_results)

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

        # Filter results based on relevant_score threshold
        if request.relevant_score is not None and request.relevant_score > 0:
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

                    if normalized_score >= request.relevant_score:
                        filtered_results.append(result)
                results["results"] = filtered_results
                results["total_count"] = len(filtered_results)

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
