from fastapi import APIRouter, HTTPException, Query, File, UploadFile
from typing import List, Dict, Any, Optional
from Rag.runner import initialize_rag_app, ask_resume_question_enhanced
from core.custom_logger import CustomLogger
from mangodatabase.client import get_users_collection
from mangodatabase.user_operations import (
    UserOperations,
    get_effective_user_id_for_search,
)
import os
from pathlib import Path

# Initialize logger
logger = CustomLogger().get_logger("enhanced_search")

# Initialize user operations
users_collection = get_users_collection()
user_ops = UserOperations(users_collection)

# Create router instance
router = APIRouter(
    prefix="/enhanced-search",
    tags=["Enhanced Search"],
    responses={404: {"description": "Not found"}},
)

# Configuration for performance presets
PERFORMANCE_PRESETS = {
    "fast": {"mongodb_limit": 20, "llm_limit": 3},
    "balanced": {"mongodb_limit": 50, "llm_limit": 10},
    "comprehensive": {"mongodb_limit": 100, "llm_limit": 20},
    "exhaustive": {"mongodb_limit": 200, "llm_limit": 30},
}


@router.post(
    "/smart-search",
    response_model=Dict[str, Any],
    summary="Smart Search with RAG and Vector Similarity",
    description="""
    Perform a smart search that combines RAG and vector similarity capabilities.
    
    **Access Control:**
    - If user_id exists in users collection → User can search ALL documents
    - If user_id does NOT exist in users collection → User can only search their own documents
    
    **Parameters:**
    - user_id: User ID (required) - determines search scope based on user existence in collection
    - query: The search query text
    - preset: Performance preset ("fast", "balanced", "comprehensive", "exhaustive")
    - min_score: Minimum similarity score (0.0 to 1.0)
    - relevant_score: Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned (default: 40.0)
    
    **Returns:**
    Dictionary containing:
    - total_found: Total number of matches found
    - statistics: Search statistics
    - results: List of matching resumes with relevance scores
    """,
)
async def smart_search(
    user_id: str = Query(
        ..., description="User ID (required for user-specific search)"
    ),
    query: str = Query(..., description="Search query text"),
    preset: str = Query(
        default="balanced",
        description="Performance preset",
        enum=list(PERFORMANCE_PRESETS.keys()),
    ),
    min_score: float = Query(
        default=0.0,
        description="Minimum similarity score",
        ge=0.0,
        le=1.0,
    ),
    relevant_score: float = Query(
        default=40.0,
        description="Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned",
        ge=0.0,
        le=100.0,
    ),
):
    """
    Perform smart search combining RAG and vector similarity.
    """
    try:
        # Input validation
        if not user_id.strip():
            raise HTTPException(
                status_code=400, detail="User ID is mandatory and cannot be empty"
            )

        # Get effective user ID for search (handles admin access)
        effective_user_id = await get_effective_user_id_for_search(user_ops, user_id)

        # Initialize RAG application
        rag_app = initialize_rag_app()

        # Get preset configuration
        config = PERFORMANCE_PRESETS[preset]

        # Perform enhanced RAG search
        rag_result = ask_resume_question_enhanced(
            query,
            mongodb_limit=config["mongodb_limit"],
            llm_limit=config["llm_limit"],
        )

        if "error" in rag_result:
            raise HTTPException(status_code=500, detail=rag_result["error"])

        # Filter results by minimum score
        if "scored_documents" in rag_result:
            rag_result["scored_documents"] = [
                doc
                for doc in rag_result["scored_documents"]
                if doc.get("score", 0) >= min_score
            ]

        # Filter results by relevant_score threshold
        if relevant_score > 0 and "scored_documents" in rag_result:
            filtered_docs = []
            for doc in rag_result["scored_documents"]:
                # Check various possible score fields
                score = doc.get(
                    "relevance_score", doc.get("match_score", doc.get("score", 0))
                )
                # Normalize score to 0-100 range if it's between 0-1
                if score <= 1.0:
                    normalized_score = score * 100
                else:
                    normalized_score = score

                # Update the document with normalized score
                doc["relevance_score"] = normalized_score

                if normalized_score >= relevant_score:
                    filtered_docs.append(doc)
            rag_result["scored_documents"] = filtered_docs

        return rag_result

    except Exception as e:
        logger.error(f"Smart search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post(
    "/context-search",
    response_model=Dict[str, Any],
    summary="Context-Aware Resume Search",
    description="""
    Perform a context-aware search that uses LLM to understand and match resume content.
    
    **Access Control:**
    - If user_id exists in users collection → User can search ALL documents
    - If user_id does NOT exist in users collection → User can only search their own documents
    
    **Parameters:**
    - user_id: User ID (required) - determines search scope based on user existence in collection
    - query: The search query text
    - context_size: Number of documents to analyze (1-20)
    - min_relevance: Minimum relevance score (0.0 to 1.0)
    - relevant_score: Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned (default: 40.0)
    
    **Returns:**
    Dictionary containing:
    - total_found: Total number of matches found
    - total_analyzed: Number of documents analyzed
    - statistics: Search statistics
    - results: List of matching resumes with relevance scores
    """,
)
async def context_search(
    user_id: str = Query(
        ..., description="User ID (required for user-specific search)"
    ),
    query: str = Query(..., description="Search query text"),
    context_size: int = Query(
        default=5,
        description="Number of documents to analyze",
        ge=1,
        le=20,
    ),
    min_relevance: float = Query(
        default=0.0,
        description="Minimum relevance score",
        ge=0.0,
        le=1.0,
    ),
    relevant_score: float = Query(
        default=40.0,
        description="Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned",
        ge=0.0,
        le=100.0,
    ),
):
    """
    Perform context-aware search using LLM.
    """
    try:
        # Input validation
        if not user_id.strip():
            raise HTTPException(
                status_code=400, detail="User ID is mandatory and cannot be empty"
            )

        # Get effective user ID for search (handles admin access)
        effective_user_id = await get_effective_user_id_for_search(user_ops, user_id)

        # Initialize RAG application
        rag_app = initialize_rag_app()

        # Perform LLM context search
        result = rag_app.llm_context_search(query, context_size)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Filter results by minimum relevance
        if "results" in result:
            result["results"] = [
                res
                for res in result["results"]
                if res.get("relevance_score", 0) >= min_relevance
            ]

        # Filter results by relevant_score threshold
        if relevant_score > 0 and "results" in result:
            filtered_results = []
            for res in result["results"]:
                # Check various possible score fields
                score = res.get(
                    "relevance_score", res.get("match_score", res.get("score", 0))
                )
                # Normalize score to 0-100 range if it's between 0-1
                if score <= 1.0:
                    normalized_score = score * 100
                else:
                    normalized_score = score

                # Update the result with normalized score
                res["relevance_score"] = normalized_score

                if normalized_score >= relevant_score:
                    filtered_results.append(res)
            result["results"] = filtered_results

        return result

    except Exception as e:
        logger.error(f"Context search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post(
    "/jd-based-search",
    response_model=Dict[str, Any],
    summary="Job Description Based Resume Search",
    description="""
    Upload a job description file and find matching resumes using advanced search capabilities.
    
    **Access Control:**
    - If user_id exists in users collection → User can search ALL documents
    - If user_id does NOT exist in users collection → User can only search their own documents
    
    **Parameters:**
    - user_id: User ID (required) - determines search scope based on user existence in collection
    - file: Job description file (.txt, .pdf, or .docx)
    - preset: Performance preset ("fast", "balanced", "comprehensive", "exhaustive")
    - min_score: Minimum similarity score (0.0 to 1.0)
    - relevant_score: Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned (default: 40.0)
    
    **Returns:**
    Dictionary containing:
    - total_found: Total number of matches found
    - statistics: Search statistics
    - results: List of matching resumes with relevance scores
    """,
)
async def jd_based_search(
    user_id: str = Query(
        ..., description="User ID (required for user-specific search)"
    ),
    file: UploadFile = File(...),
    preset: str = Query(
        default="balanced",
        description="Performance preset",
        enum=list(PERFORMANCE_PRESETS.keys()),
    ),
    min_score: float = Query(
        default=0.0,
        description="Minimum similarity score",
        ge=0.0,
        le=1.0,
    ),
    relevant_score: float = Query(
        default=40.0,
        description="Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned",
        ge=0.0,
        le=100.0,
    ),
):
    """
    Perform search based on job description file.
    """
    try:
        # Input validation
        if not user_id.strip():
            raise HTTPException(
                status_code=400, detail="User ID is mandatory and cannot be empty"
            )

        # Get effective user ID for search (handles admin access)
        effective_user_id = await get_effective_user_id_for_search(user_ops, user_id)

        # Create temp directory if it doesn't exist
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        # Save uploaded file
        file_location = temp_dir / file.filename
        with open(file_location, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        try:
            # Extract text from file
            _, file_extension = os.path.splitext(file.filename)
            if file_extension.lower() not in [".txt", ".pdf", ".docx"]:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file type. Only .txt, .pdf, and .docx are supported.",
                )

            # TODO: Implement text extraction from different file types
            # For now, just read text files
            if file_extension.lower() == ".txt":
                with open(file_location, "r", encoding="utf-8") as f:
                    jd_text = f.read()
            else:
                raise HTTPException(
                    status_code=400,
                    detail="PDF and DOCX support coming soon. Please use TXT files for now.",
                )

            if not jd_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Extracted job description is empty.",
                )

            # Get preset configuration
            config = PERFORMANCE_PRESETS[preset]

            # Initialize RAG application
            rag_app = initialize_rag_app()

            # Perform enhanced RAG search with JD text
            result = ask_resume_question_enhanced(
                jd_text,
                mongodb_limit=config["mongodb_limit"],
                llm_limit=config["llm_limit"],
            )

            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])

            # Filter results by minimum score
            if "scored_documents" in result:
                result["scored_documents"] = [
                    doc
                    for doc in result["scored_documents"]
                    if doc.get("score", 0) >= min_score
                ]

            # Filter results by relevant_score threshold
            if relevant_score > 0 and "scored_documents" in result:
                result["scored_documents"] = [
                    doc
                    for doc in result["scored_documents"]
                    if doc.get("relevance_score", 0) >= relevant_score
                    or doc.get("match_score", 0) >= relevant_score
                ]

            return result

        finally:
            # Clean up the temporary file
            try:
                os.remove(file_location)
                logger.info(f"Deleted temporary file: {file_location}")
            except Exception as e:
                logger.error(f"Failed to delete temporary file {file_location}: {e}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"JD-based search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
