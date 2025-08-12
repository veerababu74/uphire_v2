# apis/user_management.py
from fastapi import APIRouter, HTTPException, status, Query
from typing import Dict, List, Optional
from schemas.user_schemas import UserCreate, UserUpdate, UserResponse, UserListResponse
from mangodatabase.client import get_users_collection
from mangodatabase.user_operations import UserOperations
from core.custom_logger import CustomLogger

# Initialize logger
logger_instance = CustomLogger()
logger = logger_instance.get_logger("user_management_api")

# Initialize router
router = APIRouter(
    prefix="/users",
    tags=["User Management"],
    responses={404: {"description": "Not found"}},
)

# Initialize user operations
users_collection = get_users_collection()
user_ops = UserOperations(users_collection)


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user_data: UserCreate):
    """
    Create a new user

    - **user_id**: Unique user identifier
    - **email**: User email address (must be unique)
    - **name**: User full name (optional)
    - **is_admin**: Whether user has admin access to all documents (default: False)
    """
    try:
        logger.info(f"Creating user with user_id: {user_data.user_id}")
        result = user_ops.create_user(user_data)
        logger.info(f"User created successfully: {result.user_id}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """
    Get user by user_id
    """
    try:
        logger.info(f"Retrieving user with user_id: {user_id}")
        result = user_ops.get_user_by_id(user_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with user_id '{user_id}' not found",
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, user_data: UserUpdate):
    """
    Update user by user_id

    - **email**: User email address (optional)
    - **name**: User full name (optional)
    - **is_admin**: Whether user has admin access to all documents (optional)
    """
    try:
        logger.info(f"Updating user with user_id: {user_id}")
        result = user_ops.update_user(user_id, user_data)
        logger.info(f"User updated successfully: {user_id}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str):
    """
    Delete user by user_id
    """
    try:
        logger.info(f"Deleting user with user_id: {user_id}")
        user_ops.delete_user(user_id)
        logger.info(f"User deleted successfully: {user_id}")
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.get("/", response_model=Dict)
async def list_users(
    skip: int = Query(0, ge=0, description="Number of users to skip"),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of users to return"
    ),
):
    """
    List all users with pagination

    - **skip**: Number of users to skip (for pagination)
    - **limit**: Maximum number of users to return (1-1000)
    """
    try:
        logger.info(f"Listing users with skip={skip}, limit={limit}")
        result = user_ops.list_users(skip=skip, limit=limit)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error listing users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.get("/admin/list", response_model=List[UserResponse])
async def list_admin_users():
    """
    Get all admin users
    """
    try:
        logger.info("Retrieving all admin users")
        result = user_ops.get_admin_users()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving admin users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.get("/{user_id}/admin-status", response_model=Dict[str, bool])
async def check_admin_status(user_id: str):
    """
    Check if user has admin access
    """
    try:
        logger.info(f"Checking admin status for user_id: {user_id}")
        is_admin = user_ops.is_admin_user(user_id)
        return {"user_id": user_id, "is_admin": is_admin}
    except Exception as e:
        logger.error(f"Unexpected error checking admin status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )
