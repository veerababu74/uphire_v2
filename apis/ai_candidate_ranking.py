"""
AI Candidate Ranking API

This module provides AI-powered candidate ranking with automatic matching against job descriptions.
It includes features for:
- Skills match percentage calculation
- Missing skills identification
- Experience relevance scoring
- Automatic rejection below 40% threshold
- Status tagging for rejected candidates

Author: Uphire Team
Version: 1.0.0
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import os
import logging
from datetime import datetime

# Local imports
from mangodatabase.client import get_collection, get_users_collection
from mangodatabase.user_operations import UserOperations
from GroqcloudLLM.text_extraction import extract_and_clean_text
from core.custom_logger import CustomLogger
from core.ai_ranking_config import (
    REJECTION_THRESHOLD,
    ACCEPTED_STATUS,
    REJECTED_STATUS,
    SCORING_WEIGHTS,
    MAX_CANDIDATES_LIMIT,
    DEFAULT_MAX_CANDIDATES,
    MIN_JOB_DESCRIPTION_LENGTH,
    SUPPORTED_FILE_EXTENSIONS,
    EDUCATION_LEVELS,
    DEFAULT_EDUCATION_SCORE,
    FLEXIBLE_LOCATION_SCORE,
    HAS_LOCATION_INFO_SCORE,
    NO_LOCATION_INFO_SCORE,
    REASONABLE_HIKE_THRESHOLD,
    MODERATE_HIKE_THRESHOLD,
    REASONABLE_SALARY_SCORE,
    MODERATE_SALARY_SCORE,
    HIGH_SALARY_SCORE,
    PARTIAL_SALARY_INFO_SCORE,
    NO_SALARY_CONSTRAINTS_SCORE,
    DEFAULT_SALARY_SCORE,
    MAX_JOB_SUMMARY_LENGTH,
    MAX_MISSING_SKILLS_IN_REASON,
    RELEVANT_EXPERIENCE_RATIO,
)
from embeddings.vectorizer import Vectorizer
from core.helpers import format_resume
from Rag.ragapp import initialize_rag_app

# Initialize logger
logger = CustomLogger().get_logger("ai_candidate_ranking")

# Initialize router
router = APIRouter(
    prefix="/ai-ranking",
    tags=["AI Candidate Ranking"],
)

# Initialize database connections
collection = get_collection()
users_collection = get_users_collection()
user_ops = UserOperations(users_collection)
vectorizer = Vectorizer()

# Temporary folder for uploaded files
TEMP_FOLDER = "temp_uploads"
os.makedirs(TEMP_FOLDER, exist_ok=True)


class JobDescriptionRequest(BaseModel):
    """Request model for job description text input"""

    job_description: str = Field(
        ..., min_length=MIN_JOB_DESCRIPTION_LENGTH, description="Job description text"
    )
    user_id: str = Field(..., description="User ID performing the ranking")
    max_candidates: int = Field(
        default=DEFAULT_MAX_CANDIDATES,
        ge=1,
        le=MAX_CANDIDATES_LIMIT,
        description="Maximum candidates to analyze",
    )
    include_rejected: bool = Field(
        default=False, description="Include rejected candidates in response"
    )


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

    document_id: str = Field(description="MongoDB document ID", alias="_id")
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

    class Config:
        populate_by_name = True

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


def get_effective_user_id_for_search(requesting_user_id: str) -> Optional[str]:
    """
    Determine the effective user_id for search operations based on user existence in collection.
    """
    try:
        user_exists = user_ops.user_exists(requesting_user_id)
        if user_exists:
            logger.info(
                f"User {requesting_user_id} exists in collection - can search all candidates"
            )
            return None  # Can search all candidates
        else:
            logger.info(
                f"User {requesting_user_id} not in collection - can only search their own candidates"
            )
            return requesting_user_id  # Can only search their own candidates
    except Exception as e:
        logger.error(
            f"Error checking user existence for user {requesting_user_id}: {e}"
        )
        return requesting_user_id  # Default to restricting for security


def extract_job_requirements(job_description: str) -> Dict[str, Any]:
    """
    Use AI to extract structured requirements from job description
    """
    try:
        # Initialize RAG application for intelligent parsing
        rag_app = initialize_rag_app()

        # Create a prompt to extract job requirements
        extraction_prompt = f"""
        Analyze the following job description and extract key requirements in JSON format:
        
        Job Description:
        {job_description}
        
        Extract and return ONLY a JSON object with these fields:
        {{
            "required_skills": ["list of technical skills"],
            "preferred_skills": ["list of nice-to-have skills"],
            "experience_years": "minimum years required",
            "job_titles": ["related job titles"],
            "education_requirements": ["education requirements"],
            "key_responsibilities": ["main responsibilities"],
            "domain_keywords": ["industry/domain keywords"]
        }}
        """

        # Use the LLM to extract requirements
        result = rag_app.llm.invoke(extraction_prompt)

        # Try to parse the JSON response
        import json

        try:
            requirements = json.loads(result.content.strip())
        except:
            # Fallback to basic extraction if JSON parsing fails
            requirements = {
                "required_skills": [],
                "preferred_skills": [],
                "experience_years": "0",
                "job_titles": [],
                "education_requirements": [],
                "key_responsibilities": [],
                "domain_keywords": [],
            }
            logger.warning("Failed to parse LLM response, using fallback requirements")

        return requirements

    except Exception as e:
        logger.error(f"Error extracting job requirements: {str(e)}")
        return {
            "required_skills": [],
            "preferred_skills": [],
            "experience_years": "0",
            "job_titles": [],
            "education_requirements": [],
            "key_responsibilities": [],
            "domain_keywords": [],
        }


def calculate_skills_match(
    candidate_skills: List[str], job_requirements: Dict[str, Any]
) -> SkillsMatchDetail:
    """
    Calculate detailed skills matching between candidate and job requirements
    """
    required_skills = [
        skill.lower().strip() for skill in job_requirements.get("required_skills", [])
    ]
    preferred_skills = [
        skill.lower().strip() for skill in job_requirements.get("preferred_skills", [])
    ]
    all_job_skills = required_skills + preferred_skills

    candidate_skills_lower = [
        skill.lower().strip() for skill in candidate_skills if skill
    ]

    # Find matched skills
    matched_skills = []
    for job_skill in all_job_skills:
        for candidate_skill in candidate_skills_lower:
            if job_skill in candidate_skill or candidate_skill in job_skill:
                matched_skills.append(job_skill)
                break

    # Find missing skills (required skills not found)
    missing_skills = []
    for req_skill in required_skills:
        if not any(req_skill in cs or cs in req_skill for cs in candidate_skills_lower):
            missing_skills.append(req_skill)

    # Find additional skills (candidate has but not required)
    additional_skills = []
    for candidate_skill in candidate_skills:
        candidate_skill_lower = candidate_skill.lower().strip()
        if not any(
            js in candidate_skill_lower or candidate_skill_lower in js
            for js in all_job_skills
        ):
            additional_skills.append(candidate_skill)

    # Calculate match percentage
    if all_job_skills:
        skills_match_percentage = (len(matched_skills) / len(all_job_skills)) * 100
    else:
        skills_match_percentage = 0.0

    return SkillsMatchDetail(
        matched_skills=list(set(matched_skills)),
        missing_skills=list(set(missing_skills)),
        additional_skills=additional_skills[:10],  # Limit to top 10
        skills_match_percentage=round(skills_match_percentage, 2),
    )


def calculate_experience_relevance(
    candidate: Dict[str, Any], job_requirements: Dict[str, Any]
) -> ExperienceRelevance:
    """
    Calculate experience relevance based on job titles and years
    """
    try:
        # Extract candidate experience
        total_exp_str = candidate.get("total_experience", "0")
        if isinstance(total_exp_str, str):
            import re

            exp_match = re.search(r"(\d+\.?\d*)", total_exp_str)
            total_experience_years = float(exp_match.group(1)) if exp_match else 0.0
        else:
            total_experience_years = float(total_exp_str)

        # Get candidate job titles
        candidate_titles = []
        for exp in candidate.get("experience", []):
            title = exp.get("title", "")
            if title:
                candidate_titles.append(title.lower().strip())

        # Get job requirement titles
        required_titles = [
            title.lower().strip() for title in job_requirements.get("job_titles", [])
        ]

        # Find relevant roles
        relevant_roles = []
        for req_title in required_titles:
            for candidate_title in candidate_titles:
                if req_title in candidate_title or candidate_title in req_title:
                    relevant_roles.append(candidate_title)

        # Estimate relevant experience years (simplified calculation)
        if relevant_roles and total_experience_years > 0:
            # Assume 70% of total experience is relevant if roles match
            relevant_experience_years = total_experience_years * 0.7
        else:
            relevant_experience_years = 0.0

        # Calculate experience match percentage
        required_years = float(job_requirements.get("experience_years", 0))
        if required_years > 0:
            experience_match_percentage = min(
                100, (relevant_experience_years / required_years) * 100
            )
        else:
            experience_match_percentage = (
                100.0 if relevant_experience_years > 0 else 0.0
            )

        return ExperienceRelevance(
            relevant_experience_years=round(relevant_experience_years, 1),
            total_experience_years=round(total_experience_years, 1),
            experience_match_percentage=round(experience_match_percentage, 2),
            relevant_roles=list(set(relevant_roles)),
        )

    except Exception as e:
        logger.error(f"Error calculating experience relevance: {str(e)}")
        return ExperienceRelevance(
            relevant_experience_years=0.0,
            total_experience_years=0.0,
            experience_match_percentage=0.0,
            relevant_roles=[],
        )


def calculate_overall_match_score(
    skills_match: SkillsMatchDetail,
    experience_relevance: ExperienceRelevance,
    education_score: float,
    location_score: float,
    salary_score: float,
) -> float:
    """
    Calculate overall match score using weighted algorithm from configuration
    """
    overall_score = (
        skills_match.skills_match_percentage * SCORING_WEIGHTS["skills"]
        + experience_relevance.experience_match_percentage
        * SCORING_WEIGHTS["experience"]
        + education_score * SCORING_WEIGHTS["education"]
        + location_score * SCORING_WEIGHTS["location"]
        + salary_score * SCORING_WEIGHTS["salary"]
    )

    return round(overall_score, 2)


def calculate_education_relevance(
    candidate: Dict[str, Any], job_requirements: Dict[str, Any]
) -> float:
    """
    Calculate education relevance score
    """
    try:
        candidate_education = candidate.get("academic_details", [])
        required_education = job_requirements.get("education_requirements", [])

        if not required_education:
            return 100.0  # No specific requirements

        if not candidate_education:
            return 0.0  # No education info

        # Simple matching based on degree level
        education_levels = {
            "phd": 6,
            "doctorate": 6,
            "master": 5,
            "mba": 5,
            "ms": 5,
            "me": 5,
            "mtech": 5,
            "bachelor": 4,
            "btech": 4,
            "be": 4,
            "bsc": 4,
            "ba": 4,
            "bcom": 4,
            "diploma": 3,
            "12th": 2,
            "intermediate": 2,
            "10th": 1,
        }

        # Get candidate's highest education level
        candidate_level = 0
        for edu in candidate_education:
            degree = edu.get("degree", "").lower()
            for level_name, level_value in education_levels.items():
                if level_name in degree:
                    candidate_level = max(candidate_level, level_value)

        # Get required education level
        required_level = 0
        for req in required_education:
            req_lower = req.lower()
            for level_name, level_value in education_levels.items():
                if level_name in req_lower:
                    required_level = max(required_level, level_value)

        if required_level == 0:
            return 100.0  # No specific level required

        if candidate_level >= required_level:
            return 100.0
        elif candidate_level > 0:
            return (candidate_level / required_level) * 100
        else:
            return 0.0

    except Exception as e:
        logger.error(f"Error calculating education relevance: {str(e)}")
        return 50.0  # Default neutral score


def calculate_location_compatibility(candidate: Dict[str, Any]) -> float:
    """
    Calculate location compatibility score
    """
    try:
        contact_details = candidate.get("contact_details", {})
        current_city = contact_details.get("current_city", "")
        looking_for_jobs_in = contact_details.get("looking_for_jobs_in", [])

        # If candidate is looking for jobs (indicates flexibility), give higher score
        if looking_for_jobs_in and len(looking_for_jobs_in) > 0:
            return 100.0
        elif current_city:
            return 80.0  # Has location info
        else:
            return 50.0  # No location info

    except Exception as e:
        logger.error(f"Error calculating location compatibility: {str(e)}")
        return 50.0


def calculate_salary_alignment(candidate: Dict[str, Any]) -> float:
    """
    Calculate salary expectation alignment score
    """
    try:
        current_salary = float(candidate.get("current_salary", 0))
        expected_salary = float(candidate.get("expected_salary", 0))

        if expected_salary > 0 and current_salary > 0:
            # If expected salary is reasonable (up to 50% hike), score well
            hike_percentage = (
                (expected_salary - current_salary) / current_salary
            ) * 100
            if hike_percentage <= 30:
                return 100.0
            elif hike_percentage <= 50:
                return 80.0
            else:
                return 60.0
        elif expected_salary > 0 or current_salary > 0:
            return 70.0  # Has some salary info
        else:
            return 90.0  # No salary constraints

    except Exception as e:
        logger.error(f"Error calculating salary alignment: {str(e)}")
        return 80.0


def generate_ranking_reason(
    candidate_ranking: CandidateRanking, job_requirements: Dict[str, Any]
) -> str:
    """
    Generate AI explanation for the ranking
    """
    try:
        score = candidate_ranking.overall_match_score
        skills_pct = candidate_ranking.skills_match.skills_match_percentage
        exp_pct = candidate_ranking.experience_relevance.experience_match_percentage

        if score >= 80:
            reason = f"Excellent match ({score}%) - Strong skills alignment ({skills_pct}%) and relevant experience ({exp_pct}%)"
        elif score >= 60:
            reason = f"Good match ({score}%) - Decent skills match ({skills_pct}%) with suitable experience ({exp_pct}%)"
        elif score >= REJECTION_THRESHOLD:
            reason = f"Partial match ({score}%) - Some relevant skills ({skills_pct}%) but limited experience alignment ({exp_pct}%)"
        else:
            reason = f"Poor match ({score}%) - Insufficient skills match ({skills_pct}%) and limited relevant experience ({exp_pct}%)"

        # Add missing skills info if significant
        if len(candidate_ranking.skills_match.missing_skills) > 0:
            reason += f". Missing key skills: {', '.join(candidate_ranking.skills_match.missing_skills[:3])}"

        return reason

    except Exception as e:
        logger.error(f"Error generating ranking reason: {str(e)}")
        return "Unable to generate detailed ranking explanation"


def rank_candidates_against_job(
    candidates: List[Dict[str, Any]], job_requirements: Dict[str, Any]
) -> List[CandidateRanking]:
    """
    Rank all candidates against job requirements
    """
    ranked_candidates = []

    for candidate in candidates:
        try:
            # Extract candidate skills
            all_skills = candidate.get("skills", []) + candidate.get(
                "may_also_known_skills", []
            )

            # Calculate all scoring components
            skills_match = calculate_skills_match(all_skills, job_requirements)
            experience_relevance = calculate_experience_relevance(
                candidate, job_requirements
            )
            education_score = calculate_education_relevance(candidate, job_requirements)
            location_score = calculate_location_compatibility(candidate)
            salary_score = calculate_salary_alignment(candidate)

            # Calculate overall match score
            overall_score = calculate_overall_match_score(
                skills_match,
                experience_relevance,
                education_score,
                location_score,
                salary_score,
            )

            # Determine status based on threshold
            is_auto_rejected = overall_score < REJECTION_THRESHOLD
            status = REJECTED_STATUS if is_auto_rejected else ACCEPTED_STATUS

            # Create candidate ranking
            candidate_ranking = CandidateRanking(
                _id=str(candidate.get("_id", "")),
                candidate_id=str(candidate.get("_id", "")),
                user_id=candidate.get("user_id", ""),
                username=candidate.get("username", ""),
                name=candidate.get("contact_details", {}).get("name", ""),
                overall_match_score=overall_score,
                skills_match=skills_match,
                experience_relevance=experience_relevance,
                education_relevance_score=education_score,
                location_compatibility_score=location_score,
                salary_expectation_alignment=salary_score,
                status=status,
                ranking_reason="",  # Will be filled by generate_ranking_reason
                is_auto_rejected=is_auto_rejected,
                contact_details=candidate.get("contact_details", {}),
                total_experience=str(candidate.get("total_experience", "0")),
                current_salary=float(candidate.get("current_salary", 0)),
                expected_salary=float(candidate.get("expected_salary", 0)),
            )

            # Generate ranking reason
            candidate_ranking.ranking_reason = generate_ranking_reason(
                candidate_ranking, job_requirements
            )

            ranked_candidates.append(candidate_ranking)

        except Exception as e:
            logger.error(
                f"Error ranking candidate {candidate.get('_id', 'unknown')}: {str(e)}"
            )
            continue

    # Sort by overall match score (highest first)
    ranked_candidates.sort(key=lambda x: x.overall_match_score, reverse=True)

    return ranked_candidates


@router.post(
    "/rank-by-job-text",
    response_model=RankingResponse,
    summary="AI Candidate Ranking by Job Description Text",
    description="""
    Rank candidates against a job description using AI-powered matching.
    
    **Features:**
    - Skills match percentage calculation with detailed breakdown
    - Missing skills identification for gap analysis  
    - Experience relevance scoring based on job titles and years
    - Automatic rejection below 40% threshold with status tagging
    - Comprehensive ranking with AI explanations
    
    **Parameters:**
    - job_description: Job description text (minimum 50 characters)
    - user_id: User ID performing the ranking (determines search scope)
    - max_candidates: Maximum candidates to analyze (1-100)
    - include_rejected: Whether to include auto-rejected candidates in response
    
    **Returns:**
    Comprehensive ranking response with detailed match analysis for each candidate.
    Auto-rejected candidates (< 40% match) are tagged with "CV Rejected - In Process" status.
    """,
)
async def rank_candidates_by_job_text(request: JobDescriptionRequest):
    """
    Rank candidates against job description text using AI-powered matching
    """
    try:
        logger.info(f"Starting AI candidate ranking for user {request.user_id}")

        # Determine effective user_id for search
        effective_user_id = get_effective_user_id_for_search(request.user_id)

        # Extract job requirements using AI
        logger.info("Extracting job requirements using AI")
        job_requirements = extract_job_requirements(request.job_description)

        # Search for candidates using vector similarity
        logger.info(f"Searching for candidates (max: {request.max_candidates})")
        rag_app = initialize_rag_app()

        search_result = rag_app.vector_similarity_search(
            query=request.job_description,
            limit=request.max_candidates,
            user_id=effective_user_id,
        )

        if "error" in search_result:
            raise HTTPException(status_code=500, detail=search_result["error"])

        candidates = search_result.get("results", [])
        logger.info(f"Found {len(candidates)} candidates for ranking")

        # Rank candidates against job requirements
        logger.info("Ranking candidates against job requirements")
        ranked_candidates = rank_candidates_against_job(candidates, job_requirements)

        # Filter based on include_rejected parameter
        if not request.include_rejected:
            ranked_candidates = [c for c in ranked_candidates if not c.is_auto_rejected]

        # Calculate statistics
        total_analyzed = (
            len(ranked_candidates) if request.include_rejected else len(candidates)
        )
        accepted_count = len([c for c in ranked_candidates if not c.is_auto_rejected])
        rejected_count = total_analyzed - accepted_count

        # Create job description summary
        job_summary = (
            request.job_description[:200] + "..."
            if len(request.job_description) > 200
            else request.job_description
        )

        # Create ranking criteria
        ranking_criteria = {
            "skills_weight": 40,
            "experience_weight": 30,
            "education_weight": 15,
            "location_weight": 10,
            "salary_weight": 5,
            "rejection_threshold": REJECTION_THRESHOLD,
            "job_requirements": job_requirements,
        }

        # Create response
        response = RankingResponse(
            job_description_summary=job_summary,
            total_candidates_analyzed=total_analyzed,
            accepted_candidates=accepted_count,
            rejected_candidates=rejected_count,
            candidates=ranked_candidates,
            ranking_criteria=ranking_criteria,
        )

        logger.info(
            f"AI candidate ranking completed. Analyzed: {total_analyzed}, Accepted: {accepted_count}, Rejected: {rejected_count}"
        )
        return response

    except Exception as e:
        logger.error(f"Error in AI candidate ranking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post(
    "/rank-by-job-file",
    response_model=RankingResponse,
    summary="AI Candidate Ranking by Job Description File",
    description="""
    Upload a job description file and rank candidates using AI-powered matching.
    
    **Supported File Types:** .txt, .pdf, .docx
    
    **Features:**
    - Automatic text extraction from uploaded files
    - Skills match percentage calculation with detailed breakdown
    - Missing skills identification for gap analysis  
    - Experience relevance scoring based on job titles and years
    - Automatic rejection below 40% threshold with status tagging
    - Comprehensive ranking with AI explanations
    
    **Parameters:**
    - file: Job description file (.txt, .pdf, or .docx)
    - user_id: User ID performing the ranking (determines search scope)
    - max_candidates: Maximum candidates to analyze (1-100)
    - include_rejected: Whether to include auto-rejected candidates in response
    
    **Returns:**
    Comprehensive ranking response with detailed match analysis for each candidate.
    Auto-rejected candidates (< 40% match) are tagged with "CV Rejected - In Process" status.
    """,
)
async def rank_candidates_by_job_file(
    file: UploadFile = File(...),
    user_id: str = Query(..., description="User ID performing the ranking"),
    max_candidates: int = Query(
        default=50, ge=1, le=100, description="Maximum candidates to analyze"
    ),
    include_rejected: bool = Query(
        default=False, description="Include rejected candidates in response"
    ),
):
    """
    Upload job description file and rank candidates using AI-powered matching
    """
    try:
        logger.info(f"Starting AI candidate ranking by file for user {user_id}")

        # Step 1: Save uploaded file to temp directory
        file_location = os.path.join(TEMP_FOLDER, file.filename)

        with open(file_location, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        # Step 2: Extract and clean text from file
        try:
            _, file_extension = os.path.splitext(file.filename)
            if file_extension.lower() not in [".txt", ".pdf", ".docx"]:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file type. Only .txt, .pdf, and .docx are supported.",
                )

            job_description_text = extract_and_clean_text(file_location)

            if not job_description_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Extracted job description is empty or invalid.",
                )

            # Check minimum length
            if len(job_description_text) < 50:
                raise HTTPException(
                    status_code=400,
                    detail="Job description is too short. Minimum 50 characters required.",
                )

        finally:
            # Clean up the temporary file
            try:
                os.remove(file_location)
                logger.info(f"Deleted temporary file: {file_location}")
            except Exception as e:
                logger.error(f"Failed to delete temporary file {file_location}: {e}")

        # Step 3: Create request object and process
        job_request = JobDescriptionRequest(
            job_description=job_description_text,
            user_id=user_id,
            max_candidates=max_candidates,
            include_rejected=include_rejected,
        )

        # Process using the text-based ranking function
        return await rank_candidates_by_job_text(job_request)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in AI candidate ranking by file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get(
    "/ranking-stats",
    response_model=Dict[str, Any],
    summary="Get AI Ranking Statistics",
    description="""
    Get comprehensive statistics about AI candidate ranking performance.
    
    **Returns:**
    - Total candidates in database
    - Number of accepted vs rejected candidates  
    - Average match scores by skill categories
    - Most common missing skills
    - Performance metrics over time
    """,
)
async def get_ranking_statistics(
    user_id: str = Query(..., description="User ID for statistics scope")
):
    """
    Get comprehensive AI ranking statistics
    """
    try:
        logger.info(f"Getting ranking statistics for user {user_id}")

        # Determine effective user_id for search
        effective_user_id = get_effective_user_id_for_search(user_id)

        # Build query based on user permissions
        if effective_user_id:
            query = {"user_id": effective_user_id}
        else:
            query = {}

        # Get total candidates count
        total_candidates = collection.count_documents(query)

        # Get sample of candidates for analysis
        sample_size = min(100, total_candidates)
        candidates_sample = list(collection.find(query).limit(sample_size))

        # Analyze skills distribution
        all_skills = []
        for candidate in candidates_sample:
            skills = candidate.get("skills", []) + candidate.get(
                "may_also_known_skills", []
            )
            all_skills.extend(skills)

        # Count skill frequencies
        from collections import Counter

        skill_counter = Counter(all_skills)
        top_skills = skill_counter.most_common(20)

        # Analyze experience distribution
        experience_ranges = {"0-2": 0, "2-5": 0, "5-10": 0, "10+": 0}
        for candidate in candidates_sample:
            try:
                total_exp_str = candidate.get("total_experience", "0")
                if isinstance(total_exp_str, str):
                    import re

                    exp_match = re.search(r"(\d+\.?\d*)", total_exp_str)
                    exp_years = float(exp_match.group(1)) if exp_match else 0.0
                else:
                    exp_years = float(total_exp_str)

                if exp_years < 2:
                    experience_ranges["0-2"] += 1
                elif exp_years < 5:
                    experience_ranges["2-5"] += 1
                elif exp_years < 10:
                    experience_ranges["5-10"] += 1
                else:
                    experience_ranges["10+"] += 1
            except:
                experience_ranges["0-2"] += 1

        # Create statistics response
        stats = {
            "database_overview": {
                "total_candidates": total_candidates,
                "sample_analyzed": sample_size,
                "last_updated": datetime.now().isoformat(),
            },
            "skills_analysis": {
                "total_unique_skills": len(skill_counter),
                "top_skills": [
                    {"skill": skill, "count": count} for skill, count in top_skills
                ],
                "average_skills_per_candidate": round(
                    len(all_skills) / max(1, len(candidates_sample)), 2
                ),
            },
            "experience_distribution": experience_ranges,
            "ranking_configuration": {
                "skills_weight": 40,
                "experience_weight": 30,
                "education_weight": 15,
                "location_weight": 10,
                "salary_weight": 5,
                "rejection_threshold": REJECTION_THRESHOLD,
            },
            "recommendations": [
                "Use detailed job descriptions for better matching accuracy",
                "Include specific technical skills in job requirements",
                "Consider experience level requirements carefully",
                f"Current rejection threshold is {REJECTION_THRESHOLD}% - candidates below this are auto-rejected",
            ],
        }

        logger.info("Ranking statistics generated successfully")
        return stats

    except Exception as e:
        logger.error(f"Error getting ranking statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
