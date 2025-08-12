"""
API endpoints for duplicate detection management.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
from mangodatabase.client import get_resume_extracted_text_collection
from mangodatabase.duplicate_detection import DuplicateDetectionOperations
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("duplicate_detection_api")

# Initialize duplicate detection operations
extracted_text_collection = get_resume_extracted_text_collection()
duplicate_ops = DuplicateDetectionOperations(extracted_text_collection)

# Create router
router = APIRouter()


@router.get("/duplicate-detection/statistics/{user_id}")
async def get_duplicate_statistics(user_id: str):
    """
    Get duplicate detection statistics for a specific user.

    Args:
        user_id: The user ID to get statistics for

    Returns:
        Dictionary with duplicate detection statistics
    """
    try:
        stats = duplicate_ops.get_duplicate_statistics(user_id)
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error(f"Error getting duplicate statistics for user {user_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get duplicate statistics: {str(e)}"
        )


@router.get("/duplicate-detection/extracted-texts/{user_id}")
async def get_user_extracted_texts(
    user_id: str,
    limit: Optional[int] = Query(50, description="Maximum number of texts to return"),
):
    """
    Get all extracted texts for a specific user.

    Args:
        user_id: The user ID
        limit: Maximum number of documents to return (default: 50)

    Returns:
        List of extracted text documents
    """
    try:
        texts = duplicate_ops.get_user_extracted_texts(user_id, limit)
        return {"status": "success", "data": texts, "count": len(texts)}
    except Exception as e:
        logger.error(f"Error getting extracted texts for user {user_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get extracted texts: {str(e)}"
        )


@router.delete("/duplicate-detection/extracted-text/{document_id}")
async def delete_extracted_text(document_id: str):
    """
    Delete an extracted text document.

    Args:
        document_id: The document ID to delete

    Returns:
        Dictionary with deletion result
    """
    try:
        result = duplicate_ops.delete_extracted_text(document_id)

        if result["success"]:
            return {"status": "success", "message": result["message"]}
        else:
            raise HTTPException(
                status_code=404 if "not found" in result["message"].lower() else 500,
                detail=result["message"],
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting extracted text {document_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete extracted text: {str(e)}"
        )


@router.post("/duplicate-detection/check-similarity")
async def check_text_similarity(user_id: str, text: str):
    """
    Check if provided text is similar to existing content for the user.

    Args:
        user_id: The user ID to check against
        text: The text to check for similarity

    Returns:
        Dictionary with similarity check results
    """
    try:
        is_duplicate, similar_documents = duplicate_ops.check_duplicate_content(
            user_id, text
        )

        return {
            "status": "success",
            "is_duplicate": is_duplicate,
            "similarity_threshold": duplicate_ops.similarity_threshold,
            "similar_documents": similar_documents,
            "message": (
                f"Found {len(similar_documents)} similar documents"
                if is_duplicate
                else "No similar content found"
            ),
        }

    except Exception as e:
        logger.error(f"Error checking text similarity for user {user_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to check text similarity: {str(e)}"
        )


@router.get("/duplicate-detection/info")
async def get_duplicate_detection_info():
    """
    Get information about the duplicate detection system.

    Returns:
        Dictionary with system information
    """
    try:
        return {
            "status": "success",
            "system_info": {
                "similarity_threshold": duplicate_ops.similarity_threshold,
                "description": "Duplicate content detection system for resume processing",
                "features": [
                    "Text normalization for better comparison",
                    "70% similarity threshold for duplicate detection",
                    "Automatic storage of extracted text",
                    "User-specific duplicate detection",
                    "Detailed similarity reporting",
                ],
                "endpoints": {
                    "statistics": "/duplicate-detection/statistics/{user_id}",
                    "extracted_texts": "/duplicate-detection/extracted-texts/{user_id}",
                    "delete_text": "/duplicate-detection/extracted-text/{document_id}",
                    "check_similarity": "/duplicate-detection/check-similarity",
                    "system_info": "/duplicate-detection/info",
                },
            },
        }
    except Exception as e:
        logger.error(f"Error getting duplicate detection info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get system info: {str(e)}"
        )
