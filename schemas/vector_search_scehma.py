# resume_api/models/search.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union


class VectorSearchQuery(BaseModel):
    user_id: str = Field(
        ...,
        description="User ID - if exists in users collection can search all documents, otherwise only their own",
    )
    query: str = Field(..., description="Search query for semantic search")
    field: Literal["full_text"] = Field(
        default="full_text",
        description="Field to search in (fixed to full_text for total resume search)",
    )
    num_results: Literal[10] = Field(
        default=10, description="Fixed number of results to return"
    )
    min_score: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Minimum similarity score threshold"
    )
    relevant_score: float = Field(
        default=40.0,
        ge=0.0,
        le=100.0,
        description="Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned",
    )

    class Config:
        validate_assignment = True
