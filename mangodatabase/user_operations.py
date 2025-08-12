# mangodatabase/user_operations.py
from pymongo.collection import Collection
from bson import ObjectId
from typing import Dict, Any, Optional, List
from datetime import datetime
from pymongo.errors import DuplicateKeyError
from fastapi import HTTPException
from schemas.user_schemas import UserCreate, UserUpdate, UserResponse
from core.custom_logger import CustomLogger

# Initialize logger
logger = CustomLogger().get_logger("user_operations")


class UserOperations:
    """Database operations for user management"""

    def __init__(self, collection: Collection):
        self.collection = collection

    def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user"""
        try:
            # Check if user_id already exists
            existing_user = self.collection.find_one({"user_id": user_data.user_id})
            if existing_user:
                raise HTTPException(
                    status_code=400,
                    detail=f"User with user_id '{user_data.user_id}' already exists",
                )

            # Check if email already exists
            existing_email = self.collection.find_one({"email": user_data.email})
            if existing_email:
                raise HTTPException(
                    status_code=400,
                    detail=f"User with email '{user_data.email}' already exists",
                )

            # Create user document
            user_doc = {
                "user_id": user_data.user_id,
                "email": user_data.email,
                "name": user_data.name,
                "is_admin": user_data.is_admin,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            result = self.collection.insert_one(user_doc)

            # Return the created user
            created_user = self.collection.find_one({"_id": result.inserted_id})
            return self._format_user_response(created_user)

        except DuplicateKeyError:
            raise HTTPException(
                status_code=400, detail="User with this user_id or email already exists"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error creating user: {str(e)}"
            )

    def get_user_by_id(self, user_id: str) -> Optional[UserResponse]:
        """Get user by user_id"""
        try:
            user = self.collection.find_one({"user_id": user_id})
            if user:
                return self._format_user_response(user)
            return None
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error retrieving user: {str(e)}"
            )

    def get_user_by_mongo_id(self, mongo_id: str) -> Optional[UserResponse]:
        """Get user by MongoDB ObjectId"""
        try:
            user = self.collection.find_one({"_id": ObjectId(mongo_id)})
            if user:
                return self._format_user_response(user)
            return None
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error retrieving user: {str(e)}"
            )

    def update_user(
        self, user_id: str, user_data: UserUpdate
    ) -> Optional[UserResponse]:
        """Update user by user_id"""
        try:
            # Build update document
            update_doc = {"updated_at": datetime.utcnow()}

            # Only update fields that are provided
            if user_data.email is not None:
                # Check if email already exists for another user
                existing_email = self.collection.find_one(
                    {"email": user_data.email, "user_id": {"$ne": user_id}}
                )
                if existing_email:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Email '{user_data.email}' is already used by another user",
                    )
                update_doc["email"] = user_data.email

            if user_data.name is not None:
                update_doc["name"] = user_data.name

            if user_data.is_admin is not None:
                update_doc["is_admin"] = user_data.is_admin

            # Update the user
            result = self.collection.update_one(
                {"user_id": user_id}, {"$set": update_doc}
            )

            if result.matched_count == 0:
                raise HTTPException(
                    status_code=404, detail=f"User with user_id '{user_id}' not found"
                )

            # Return updated user
            updated_user = self.collection.find_one({"user_id": user_id})
            return self._format_user_response(updated_user)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error updating user: {str(e)}"
            )

    def delete_user(self, user_id: str) -> bool:
        """Delete user by user_id"""
        try:
            result = self.collection.delete_one({"user_id": user_id})
            if result.deleted_count == 0:
                raise HTTPException(
                    status_code=404, detail=f"User with user_id '{user_id}' not found"
                )
            return True
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error deleting user: {str(e)}"
            )

    def list_users(self, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """List all users with pagination"""
        try:
            # Get total count
            total_count = self.collection.count_documents({})

            # Get users with pagination
            cursor = (
                self.collection.find({}).sort("created_at", -1).skip(skip).limit(limit)
            )
            users = [self._format_user_response(user) for user in cursor]

            return {
                "users": users,
                "total_count": total_count,
                "skip": skip,
                "limit": limit,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error listing users: {str(e)}"
            )

    def is_admin_user(self, user_id: str) -> bool:
        """Check if user is admin"""
        try:
            user = self.collection.find_one({"user_id": user_id, "is_admin": True})
            return user is not None
        except Exception:
            return False

    def user_exists(self, user_id: str) -> bool:
        """Check if user exists in the users collection"""
        try:
            user = self.collection.find_one({"user_id": user_id})
            return user is not None
        except Exception:
            return False

    def get_admin_users(self) -> List[UserResponse]:
        """Get all admin users"""
        try:
            cursor = self.collection.find({"is_admin": True}).sort("created_at", -1)
            return [self._format_user_response(user) for user in cursor]
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error retrieving admin users: {str(e)}"
            )

    def _format_user_response(self, user_doc: Dict) -> UserResponse:
        """Format user document to response model"""
        return UserResponse(
            id=str(user_doc["_id"]),
            user_id=user_doc["user_id"],
            email=user_doc["email"],
            name=user_doc.get("name"),
            is_admin=user_doc.get("is_admin", False),
            created_at=user_doc["created_at"],
            updated_at=user_doc["updated_at"],
        )


async def get_effective_user_id_for_search(
    user_ops: "UserOperations", user_id: str
) -> Optional[str]:
    """
    Get the effective user ID for search operations.

    Logic:
    - If user_id exists in users collection → Return None (show ALL data)
    - If user_id does NOT exist in users collection → Return user_id (show only that user's data)

    Args:
        user_ops: UserOperations instance
        user_id: The user ID to check

    Returns:
        Optional[str]: None if user exists in collection (show all data),
                      user_id if user not in collection (show only user's data)
    """
    try:
        # Check if user exists in the users collection
        user_exists = user_ops.user_exists(user_id)

        if user_exists:
            # User exists in collection → can access ALL data
            return None
        else:
            # User does NOT exist in collection → can only access their own data
            return user_id

    except Exception as e:
        # If there's an error checking user existence, default to restricting to user's own data
        logger.error(f"Error checking user existence for user {user_id}: {str(e)}")
        return user_id
