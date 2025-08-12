"""
Pydantic schemas for AI Candidate Ranking API

Author: Uphire Team
Version: 1.0.0
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class SkillsMatchDetail(BaseModel):
    """Detailed skills matching information"""

    matched_skills: List[str] = Field(
        default=[], description="Skills that matched the job description"
    )
    missing_skills: List[str] = Field(
        default=[], description="Skills required but missing from candidate"
    )
    additional_skills: List[str] = Field(
        default=[], description="Candidate skills not in job requirements"
    )
    skills_match_percentage: float = Field(
        description="Percentage of required skills matched"
    )


class ExperienceRelevance(BaseModel):
    """Experience relevance details"""

    relevant_experience_years: float = Field(description="Years of relevant experience")
    total_experience_years: float = Field(description="Total years of experience")
    experience_match_percentage: float = Field(
        description="Relevance percentage of experience"
    )
    relevant_roles: List[str] = Field(
        default=[], description="Job titles that match requirements"
    )


class CandidateRanking(BaseModel):
    """Individual candidate ranking result"""

    _id: str = Field(description="MongoDB document ID")
    candidate_id: str = Field(description="Candidate's database ID")
    user_id: str = Field(description="Candidate's user ID")
    username: str = Field(description="Candidate's username")
    name: str = Field(description="Candidate's full name")

    # Core ranking metrics
    overall_match_score: float = Field(description="Overall match score (0-100)")
    skills_match: SkillsMatchDetail = Field(description="Detailed skills matching")
    experience_relevance: ExperienceRelevance = Field(
        description="Experience relevance details"
    )

    # Additional metrics
    education_relevance_score: float = Field(
        description="Education relevance score (0-100)"
    )
    location_compatibility_score: float = Field(
        description="Location compatibility score (0-100)"
    )
    salary_expectation_alignment: float = Field(
        description="Salary alignment score (0-100)"
    )

    # Status and metadata
    status: str = Field(description="Candidate status (Accepted/Rejected)")
    ranking_reason: str = Field(description="AI explanation for the ranking")
    is_auto_rejected: bool = Field(description="Whether candidate was auto-rejected")

    # Contact and basic info
    contact_details: Dict[str, Any] = Field(description="Candidate contact information")
    total_experience: str = Field(description="Total experience string")
    current_salary: float = Field(default=0.0, description="Current salary")
    expected_salary: float = Field(default=0.0, description="Expected salary")

    # Timestamp
    ranked_at: datetime = Field(
        default_factory=datetime.now, description="Ranking timestamp"
    )


class RankingResponse(BaseModel):
    """Response model for candidate ranking"""

    job_description_summary: str = Field(description="Summary of the job description")
    total_candidates_analyzed: int = Field(
        description="Total number of candidates analyzed"
    )
    accepted_candidates: int = Field(description="Number of candidates accepted")
    rejected_candidates: int = Field(description="Number of candidates rejected")
    candidates: List[CandidateRanking] = Field(description="Ranked list of candidates")
    ranking_criteria: Dict[str, Any] = Field(description="Criteria used for ranking")
    processed_at: datetime = Field(
        default_factory=datetime.now, description="Processing timestamp"
    )


class JobDescriptionRequest(BaseModel):
    """Request model for job description text input"""

    job_description: str = Field(..., min_length=50, description="Job description text")
    user_id: str = Field(..., description="User ID performing the ranking")
    max_candidates: int = Field(
        default=50, ge=1, le=100, description="Maximum candidates to analyze"
    )
    include_rejected: bool = Field(
        default=False, description="Include rejected candidates in response"
    )


class RankingStatistics(BaseModel):
    """Statistics about AI ranking performance"""

    database_overview: Dict[str, Any] = Field(
        description="Database overview statistics"
    )
    skills_analysis: Dict[str, Any] = Field(description="Skills distribution analysis")
    experience_distribution: Dict[str, int] = Field(
        description="Experience ranges distribution"
    )
    ranking_configuration: Dict[str, Any] = Field(
        description="Current ranking configuration"
    )
    recommendations: List[str] = Field(
        description="Recommendations for better matching"
    )
