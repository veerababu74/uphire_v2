from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
from pymongo.collection import Collection
from mangodatabase.operations import ResumeOperations
from embeddings.vectorizer import AddUserDataVectorizer
from core.database import get_database
from mangodatabase.client import get_collection, get_users_collection
from mangodatabase.user_operations import UserOperations

router = APIRouter(prefix="/resumes", tags=["resumes crud"])

# Global vectorizer instance
_vectorizer = None

# Initialize user operations for access control
users_collection = get_users_collection()
user_ops = UserOperations(users_collection)


def get_effective_user_id_for_resume_operations(
    requesting_user_id: str,
) -> Optional[str]:
    """
    Determine the effective user_id for resume operations based on user existence in collection.

    Args:
        requesting_user_id: The user_id making the request

    Returns:
        - None if user exists in users collection (can access all resumes)
        - requesting_user_id if user does not exist in users collection (can only access their own resumes)
    """
    try:
        user_exists = user_ops.user_exists(requesting_user_id)
        if user_exists:
            print(
                f"User {requesting_user_id} exists in collection - can access all resumes"
            )
            return None  # User exists in collection - can access all resumes
        else:
            print(
                f"User {requesting_user_id} not in collection - can only access their own resumes"
            )
            return requesting_user_id  # User not in collection - can only access their own resumes
    except Exception as e:
        print(f"Error checking user existence for user {requesting_user_id}: {e}")
        # On error, default to restricting to user's own resumes for security
        return requesting_user_id


def validate_user_access_to_resume(
    requesting_user_id: str, resume: Dict[str, Any]
) -> bool:
    """
    Validate if the requesting user has access to the given resume.

    Args:
        requesting_user_id: The user_id making the request
        resume: The resume document from database

    Returns:
        True if user has access, False otherwise
    """
    try:
        # Check if user exists in users collection
        user_exists = user_ops.user_exists(requesting_user_id)

        if user_exists:
            # User exists in collection - can access all resumes
            return True
        else:
            # User not in collection - can only access their own resumes
            resume_user_id = resume.get("user_id")
            return resume_user_id == requesting_user_id
    except Exception as e:
        print(f"Error validating user access: {e}")
        # On error, default to denying access for security
        return False


def get_vectorizer() -> AddUserDataVectorizer:
    """Get or create the AddUserDataVectorizer instance"""
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = AddUserDataVectorizer()
    return _vectorizer


def get_resume_operations(
    db: Collection = Depends(get_database),
) -> ResumeOperations:
    vectorizer = get_vectorizer()
    return ResumeOperations(db, vectorizer)


@router.post("/", response_model=Dict[str, str])
async def create_resume(
    resume_data: Dict[str, Any],
    requesting_user_id: str = Query(
        ..., description="User ID of the user making the request"
    ),
    operations: ResumeOperations = Depends(get_resume_operations),
):
    """Create a new resume with vector embeddings and user access control"""
    try:
        if not resume_data:
            raise HTTPException(status_code=400, detail="Resume data cannot be empty")

        # Validate required fields
        if "user_id" not in resume_data:
            raise HTTPException(
                status_code=400, detail="user_id is required in resume data"
            )

        # Check user permissions
        effective_user_id = get_effective_user_id_for_resume_operations(
            requesting_user_id
        )

        if effective_user_id is not None:
            # User can only create resumes for themselves
            if resume_data["user_id"] != requesting_user_id:
                raise HTTPException(
                    status_code=403,
                    detail="You can only create resumes for your own user_id",
                )

        print(
            f"User {requesting_user_id} creating resume for user_id: {resume_data['user_id']}"
        )

        result = operations.create_resume(resume_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.put("/{resume_id}", response_model=Dict[str, str])
async def update_resume(
    resume_id: str,
    resume_data: Dict[str, Any],
    requesting_user_id: str = Query(
        ..., description="User ID of the user making the request"
    ),
    operations: ResumeOperations = Depends(get_resume_operations),
):
    """Update an existing resume with vector embeddings and user access control"""
    try:
        if not resume_id.strip():
            raise HTTPException(status_code=400, detail="Resume ID cannot be empty")

        if not resume_data:
            raise HTTPException(status_code=400, detail="Resume data cannot be empty")

        # Validate required fields for AddUserData format
        required_fields = ["user_id", "username"]
        missing_fields = [
            field for field in required_fields if field not in resume_data
        ]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {', '.join(missing_fields)}",
            )

        # Check if resume exists and validate user access
        try:
            existing_resume = operations.get_resume(resume_id)
        except HTTPException as e:
            if e.status_code == 404:
                raise HTTPException(status_code=404, detail="Resume not found")
            raise

        # Validate user access to this resume
        if not validate_user_access_to_resume(requesting_user_id, existing_resume):
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to update this resume",
            )

        # Check user permissions for the new user_id in resume_data
        effective_user_id = get_effective_user_id_for_resume_operations(
            requesting_user_id
        )

        if effective_user_id is not None:
            # User can only update resumes to their own user_id
            if resume_data["user_id"] != requesting_user_id:
                raise HTTPException(
                    status_code=403,
                    detail="You can only update resumes with your own user_id",
                )

        # Log the update operation
        print(
            f"User {requesting_user_id} updating resume {resume_id} for user_id: {resume_data['user_id']}"
        )

        result = operations.update_resume(resume_id, resume_data)

        # Log successful update
        print(f"Resume {resume_id} updated successfully. All vectors regenerated.")

        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{resume_id}", response_model=Dict[str, Any])
async def get_resume(
    resume_id: str,
    requesting_user_id: str = Query(
        ..., description="User ID of the user making the request"
    ),
    operations: ResumeOperations = Depends(get_resume_operations),
):
    """Get a resume by ID with user access control"""
    try:
        if not resume_id.strip():
            raise HTTPException(status_code=400, detail="Resume ID cannot be empty")

        # Get the resume first
        result = operations.get_resume(resume_id)

        # Validate user access to this resume
        if not validate_user_access_to_resume(requesting_user_id, result):
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to access this resume",
            )

        print(f"User {requesting_user_id} accessed resume {resume_id}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/{resume_id}", response_model=Dict[str, str])
async def delete_resume(
    resume_id: str,
    requesting_user_id: str = Query(
        ..., description="User ID of the user making the request"
    ),
    operations: ResumeOperations = Depends(get_resume_operations),
):
    """Delete a resume by ID with user access control"""
    try:
        if not resume_id.strip():
            raise HTTPException(status_code=400, detail="Resume ID cannot be empty")

        # Check if resume exists and validate user access
        try:
            existing_resume = operations.get_resume(resume_id)
        except HTTPException as e:
            if e.status_code == 404:
                raise HTTPException(status_code=404, detail="Resume not found")
            raise

        # Validate user access to this resume
        if not validate_user_access_to_resume(requesting_user_id, existing_resume):
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to delete this resume",
            )

        print(f"User {requesting_user_id} deleting resume {resume_id}")

        result = operations.delete_resume(resume_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/", response_model=List[Dict[str, Any]])
async def list_resumes(
    requesting_user_id: str = Query(
        ..., description="User ID of the user making the request"
    ),
    skip: int = 0,
    limit: int = 10,
    operations: ResumeOperations = Depends(get_resume_operations),
):
    """List resumes with pagination and user access control"""
    try:
        if skip < 0:
            raise HTTPException(
                status_code=400, detail="Skip parameter cannot be negative"
            )

        if limit <= 0 or limit > 100:
            raise HTTPException(
                status_code=400, detail="Limit must be between 1 and 100"
            )

        # Determine user access level
        effective_user_id = get_effective_user_id_for_resume_operations(
            requesting_user_id
        )

        if effective_user_id is None:
            # User exists in collection - can list all resumes
            print(f"User {requesting_user_id} listing all resumes")
            result = operations.list_resumes(skip, limit)
        else:
            # User not in collection - can only list their own resumes
            print(f"User {requesting_user_id} listing only their own resumes")
            result = operations.list_resumes_by_user(requesting_user_id, skip, limit)

        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/update-embeddings", response_model=Dict[str, str])
async def update_all_embeddings(
    requesting_user_id: str = Query(
        ..., description="User ID of the user making the request"
    ),
    operations: ResumeOperations = Depends(get_resume_operations),
):
    """Update vector embeddings for all accessible resumes based on user permissions"""
    try:
        # Check user permissions
        effective_user_id = get_effective_user_id_for_resume_operations(
            requesting_user_id
        )

        if effective_user_id is None:
            # User exists in collection - can update all resumes
            print(f"User {requesting_user_id} updating embeddings for all resumes")
            result = operations.update_all_vector_embeddings()
        else:
            # User not in collection - can only update their own resumes
            print(
                f"User {requesting_user_id} updating embeddings for their own resumes only"
            )
            result = operations.update_vector_embeddings_by_user(requesting_user_id)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.patch("/{resume_id}/regenerate-vectors", response_model=Dict[str, str])
async def regenerate_resume_vectors(
    resume_id: str,
    requesting_user_id: str = Query(
        ..., description="User ID of the user making the request"
    ),
    operations: ResumeOperations = Depends(get_resume_operations),
):
    """Regenerate vector embeddings for a specific resume with user access control"""
    try:
        if not resume_id.strip():
            raise HTTPException(status_code=400, detail="Resume ID cannot be empty")

        # Check if resume exists and validate user access
        try:
            existing_resume = operations.get_resume(resume_id)
        except HTTPException as e:
            if e.status_code == 404:
                raise HTTPException(status_code=404, detail="Resume not found")
            raise

        # Validate user access to this resume
        if not validate_user_access_to_resume(requesting_user_id, existing_resume):
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to regenerate vectors for this resume",
            )

        print(f"User {requesting_user_id} regenerating vectors for resume {resume_id}")

        # Regenerate vectors using existing data
        result = operations.update_resume(resume_id, existing_resume)

        return {
            "message": f"Vector embeddings regenerated successfully for resume {resume_id}"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.patch("/{resume_id}", response_model=Dict[str, str])
async def partial_update_resume(
    resume_id: str,
    resume_data: Dict[str, Any],
    requesting_user_id: str = Query(
        ..., description="User ID of the user making the request"
    ),
    regenerate_vectors: bool = True,
    operations: ResumeOperations = Depends(get_resume_operations),
):
    """Partially update resume fields with optional vector regeneration and user access control"""
    try:
        if not resume_id.strip():
            raise HTTPException(status_code=400, detail="Resume ID cannot be empty")

        if not resume_data:
            raise HTTPException(status_code=400, detail="Resume data cannot be empty")

        # Check if resume exists and validate user access
        try:
            existing_resume = operations.get_resume(resume_id)
        except HTTPException as e:
            if e.status_code == 404:
                raise HTTPException(status_code=404, detail="Resume not found")
            raise

        # Validate user access to this resume
        if not validate_user_access_to_resume(requesting_user_id, existing_resume):
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to update this resume",
            )

        # If user_id is being updated, validate permissions
        if "user_id" in resume_data:
            effective_user_id = get_effective_user_id_for_resume_operations(
                requesting_user_id
            )

            if effective_user_id is not None:
                # User can only update resumes to their own user_id
                if resume_data["user_id"] != requesting_user_id:
                    raise HTTPException(
                        status_code=403,
                        detail="You can only update resumes with your own user_id",
                    )

        print(f"User {requesting_user_id} partially updating resume {resume_id}")

        # Use the new method for partial updates
        result = operations.update_resume_fields_only(
            resume_id, resume_data, regenerate_vectors
        )

        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
