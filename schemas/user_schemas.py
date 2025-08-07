# schemas/user_schemas.py
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime
from bson import ObjectId


class UserCreate(BaseModel):
    """Schema for creating a new user"""

    user_id: str = Field(..., description="Unique user identifier")
    email: EmailStr = Field(..., description="User email address")
    name: Optional[str] = Field(None, description="User full name")
    is_admin: bool = Field(
        default=False, description="Whether user has admin access to all documents"
    )

    class Config:
        json_encoders = {ObjectId: str}


class UserUpdate(BaseModel):
    """Schema for updating an existing user"""

    email: Optional[EmailStr] = Field(None, description="User email address")
    name: Optional[str] = Field(None, description="User full name")
    is_admin: Optional[bool] = Field(
        None, description="Whether user has admin access to all documents"
    )

    class Config:
        json_encoders = {ObjectId: str}


class UserResponse(BaseModel):
    """Schema for user response"""

    id: str = Field(..., description="MongoDB document ID")
    user_id: str = Field(..., description="Unique user identifier")
    email: str = Field(..., description="User email address")
    name: Optional[str] = Field(None, description="User full name")
    is_admin: bool = Field(
        ..., description="Whether user has admin access to all documents"
    )
    created_at: datetime = Field(..., description="User creation timestamp")
    updated_at: datetime = Field(..., description="User last update timestamp")

    class Config:
        json_encoders = {ObjectId: str, datetime: lambda v: v.isoformat()}


class UserListResponse(BaseModel):
    """Schema for user list response"""

    users: list[UserResponse]
    total_count: int


class UserSearchRequest(BaseModel):
    """Enhanced search request that considers admin users"""

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
