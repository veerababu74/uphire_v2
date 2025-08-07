import re
from fastapi import APIRouter, Body, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from mangodatabase.client import (
    get_collection,
    get_manual_recent_search_collection,
    get_users_collection,
)
from mangodatabase.user_operations import UserOperations
from core.helpers import format_resume
from datetime import datetime, timezone
import uuid

resumes_collection = get_collection()

# Initialize user operations for admin checking
users_collection = get_users_collection()
user_ops = UserOperations(users_collection)


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
            print(
                f"User {requesting_user_id} exists in collection - searching all documents"
            )
            return None  # User exists in collection - can search all documents
        else:
            print(
                f"User {requesting_user_id} not in collection - searching only their documents"
            )
            return requesting_user_id  # User not in collection - can only search their own documents
    except Exception as e:
        print(f"Error checking user existence for user {requesting_user_id}: {e}")
        # On error, default to restricting to user's own documents for security
        return requesting_user_id


# Define request and response models for better documentation
class ManualSearchRequest(BaseModel):
    userid: str = Field(
        description="User ID - if exists in users collection can search all documents, otherwise only their own",
        example="user123",
    )
    experience_titles: Optional[List[str]] = Field(
        default=None,
        description="List of job titles to search for",
        example=["Software Engineer", "Python Developer"],
    )
    skills: Optional[List[str]] = Field(
        default=None,
        description="Technical skills to match",
        example=["Python", "React", "AWS"],
    )
    min_education: Optional[List[str]] = Field(
        default=None,
        description="List of minimum education qualifications",
        example=["BTech", "BSc"],
    )
    min_experience: Optional[str] = Field(
        default=None,
        description="Minimum required experience",
        example="2 years 6 months",
    )
    max_experience: Optional[str] = Field(
        default=None,
        description="Maximum required experience",
        example="5 years",
    )
    locations: Optional[List[str]] = Field(
        default=None,
        description="Preferred locations",
        example=["Mumbai", "Pune", "Bangalore"],
    )
    min_salary: Optional[float] = Field(
        default=None,
        description="Minimum expected salary",
        example=500000.0,
    )
    max_salary: Optional[float] = Field(
        default=None,
        description="Maximum expected salary",
        example=1500000.0,
    )
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum number of results (optional)",
        example=10,
    )
    relevant_score: Optional[float] = Field(
        default=40.0,
        ge=0.0,
        le=100.0,
        description="Minimum relevance score threshold (0-100). Only results with match_score >= this value will be returned",
        example=40.0,
    )


router = APIRouter(
    prefix="/manualsearch",
    tags=[
        "Manual Resume Search",
    ],
)


@router.post(
    "/",
    response_model=List[Dict[str, Any]],
    summary="Advanced Resume Search",
    description="""
    Search for resumes using multiple criteria with intelligent ranking.

    **Access Control:**
    - Users in users collection can search all documents across all users
    - Users not in users collection can only search their own documents
    
    **User ID is Mandatory:**
    - userid: Required field - determines search scope based on user existence in collection
    
    **All Other Search Criteria are Optional:**
    - Experience Titles (Optional): Job titles to match (e.g., ["Software Engineer", "Developer"])
    - Skills (Optional): Technical skills to match (includes both 'skills' and 'may_also_known_skills')
    - Minimum Education (Optional): Education level (e.g., "BTech", "MCA", "MBA")
    - Minimum Experience (Optional): Required experience (e.g., "2 years 6 months")
    - Maximum Experience (Optional): Maximum experience limit (e.g., "5 years")
    - Locations (Optional): Preferred cities (matches both current_city and looking_for_jobs_in)
    - Min Salary (Optional): Minimum expected salary filter
    - Max Salary (Optional): Maximum expected salary filter (budget limit)
    - Relevant Score (Optional): Minimum relevance score threshold (default: 40.0, range: 0-100)
    
    **Relevance Score Filtering:**
    - Only candidates with match_score >= relevant_score will be returned
    - Default threshold is 40% to ensure reasonable quality matches
    - Set to 0 to return all matches regardless of score
    - Higher values (60-80+) return only highly relevant candidates
    
    **Salary Filtering:**
    - Uses expected_salary if available, falls back to current_salary
    - Candidates without salary information are excluded when salary filters are applied
    - Salary values should be in the same currency as stored in database
    
    **Data Structure:**
    - Contact Details: name, email, phone, current_city, looking_for_jobs_in, pan_card, etc.
    - Academic Details: education, college, pass_year
    - Experience: company, title, from_date, to (optional)
    - Skills: skills array and may_also_known_skills array
    - Financial: current_salary, expected_salary, currency, pay_duration
    
    **Education Levels Hierarchy:**
    1. 10th/SSC
    2. 12th/HSC/Intermediate
    3. Diploma/ITI
    4. Bachelor's (BTech/BE/BSc/BA/BCom)
    5. Master's (MTech/ME/MSc/MA/MCom/MBA)    6. PhD/Doctorate
    
    **Enhanced Scoring System (0-100 Scale):**
    - Experience Titles: Up to 25 points (normalized with diminishing returns)
    - Skills Match: Up to 25 points (normalized with diminishing returns) 
    - Education Level: Up to 15 points (normalized based on education level hierarchy)
    - Experience Range: Up to 10 points (10 for perfect match, 5 for close match)
    - Location Match: Up to 15 points (normalized based on current city vs job preferences)
    - Salary Range: Up to 10 points (full points for within range)
    - Field Diversity Bonus: Up to 10 additional points for matching multiple field types
    
    **Score Calculation:**
    - Base Score: Sum of normalized category scores (0-100)
    - Field Diversity Bonus: 2 points per different field type matched (max 10)
    - Final Score: Base Score + Field Diversity Bonus (capped at 100)
    
    **Category Score Details:**
    - Experience Titles: 8 points for first match + 2 points for each additional (max 25)
    - Skills: 6 points for first match + 1.5 points for each additional (max 25)
    - Education: 2.5 points per education level value (max 15)
    - Location: 3 points for current city match, 2 points for job preference (max 15)
    - All scores are normalized to ensure total never exceeds 100
    
    **Priority Ranking:**
    1. Primary: Number of different field types matched (experience, skills, education, location, etc.)
    2. Secondary: Total match score (base score + field match bonus)
    3. Tertiary: Total individual matches across all fields
    
    **Enhanced Search Features:**
    - Comprehensive keyword matching: finds ALL candidates matching ANY of the provided keywords
    - Detailed match tracking: shows exactly which keywords matched for each candidate
    - Field-based prioritization: candidates matching more field types appear first
    - Enhanced match details: includes matched_experience_titles, matched_skills, matched_education, matched_locations
    
    **Returns:**
    Sorted list of matching resumes with comprehensive match details and scores (best matches first).
    If no results are found, returns an informative object with search summary and suggestions.
    """,
    responses={
        200: {
            "description": "Successful search results or informative no-results message",
            "content": {
                "application/json": {
                    "examples": {
                        "successful_search": {
                            "summary": "Successful search with results",
                            "value": [
                                {
                                    "user_id": "user123",
                                    "username": "johndoe",
                                    "contact_details": {
                                        "name": "John Doe",
                                        "email": "john@example.com",
                                        "phone": "+91-1234567890",
                                        "current_city": "Mumbai",
                                        "looking_for_jobs_in": ["Mumbai", "Pune"],
                                        "pan_card": "ABCDE1234F",
                                        "gender": "Male",
                                        "age": 28,
                                    },
                                    "academic_details": [
                                        {
                                            "education": "BTech in Computer Science",
                                            "college": "Mumbai University",
                                            "pass_year": 2020,
                                        }
                                    ],
                                    "experience": [
                                        {
                                            "company": "Tech Corp",
                                            "title": "Senior Software Engineer",
                                            "from_date": "2021-01",
                                            "to": "2023-12",
                                        }
                                    ],
                                    "skills": ["Python", "React", "AWS"],
                                    "may_also_known_skills": ["Docker", "Kubernetes"],
                                    "total_experience": "2 years 6 months",
                                    "expected_salary": 1500000.0,
                                    "current_salary": 1200000.0,
                                    "notice_period": "30 days",
                                    "match_score": 85.5,
                                    "base_score": 75.5,
                                    "field_diversity_bonus": 10,
                                    "category_scores": {
                                        "experience_titles": 16.0,
                                        "skills": 21.5,
                                        "education": 10.0,
                                        "experience_range": 10.0,
                                        "location": 12.0,
                                        "salary": 10.0,
                                    },
                                    "match_details": {
                                        "experience_title_matches": 1,
                                        "skills_matches": 3,
                                        "education_matches": 1,
                                        "location_matches": 2,
                                        "experience_range_match": True,
                                        "salary_range_match": True,
                                        "matched_experience_titles": [
                                            "Software Engineer"
                                        ],
                                        "matched_skills": ["Python", "React", "AWS"],
                                        "matched_education": ["BTech"],
                                        "matched_locations": [
                                            "Mumbai (current)",
                                            "Mumbai (preference)",
                                        ],
                                        "fields_matched": 5,
                                    },
                                    "total_individual_matches": 9,
                                }
                            ],
                        },
                        "no_results_found": {
                            "summary": "No results found with helpful suggestions",
                            "value": [
                                {
                                    "message": "No matching resumes found",
                                    "search_summary": {
                                        "user_id": "user123",
                                        "total_candidates_searched": 0,
                                        "search_criteria_used": {
                                            "experience_titles": [
                                                "Software Engineer",
                                                "Python Developer",
                                            ],
                                            "skills": ["Python", "React", "AWS"],
                                            "min_education": ["BTech", "BSc"],
                                            "min_experience": "2 years 6 months",
                                            "max_experience": "5 years",
                                            "locations": [
                                                "Mumbai",
                                                "Pune",
                                                "Bangalore",
                                            ],
                                            "min_salary": 500000,
                                            "max_salary": 1500000,
                                            "relevant_score": 40.0,
                                        },
                                        "suggestions": [
                                            "Try using broader or alternative job titles",
                                            "Consider removing some specific skills or using more general skill terms",
                                            "Adjust the experience range requirements",
                                            "Expand the location search to include nearby cities",
                                            "Adjust the salary range to be more flexible",
                                            "Consider accepting lower education qualifications",
                                        ],
                                    },
                                    "results": [],
                                }
                            ],
                        },
                    }
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "No search criteria provided"}
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"detail": "Manual search failed: Database error"}
                }
            },
        },
    },
)
async def manual_resume_search(search_params: ManualSearchRequest):
    try:
        # Check if at least one search criteria is provided
        has_criteria = any(
            [
                search_params.experience_titles,
                search_params.skills,
                search_params.min_education,
                search_params.min_experience,
                search_params.max_experience,
                search_params.locations,
                search_params.min_salary is not None,
                search_params.max_salary is not None,
            ]
        )

        # Save search query to recent searches (userid is now mandatory)
        if has_criteria:
            try:
                recent_collection = get_manual_recent_search_collection()

                # Generate unique search ID
                search_id = str(uuid.uuid4())

                # Create search document
                search_document = {
                    "search_id": search_id,
                    "user_id": search_params.userid,
                    "search_criteria": {
                        "experience_titles": search_params.experience_titles,
                        "skills": search_params.skills,
                        "min_education": search_params.min_education,
                        "min_experience": search_params.min_experience,
                        "max_experience": search_params.max_experience,
                        "locations": search_params.locations,
                        "min_salary": search_params.min_salary,
                        "max_salary": search_params.max_salary,
                        "relevant_score": search_params.relevant_score,
                    },
                    "timestamp": datetime.now(timezone.utc),
                    "search_type": "manual",
                    "is_recent": True,
                }

                # Check if user has too many recent searches (limit to 100)
                user_recent_count = recent_collection.count_documents(
                    {"user_id": search_params.userid}
                )
                if user_recent_count >= 100:
                    # Remove oldest search for this user
                    oldest_search = recent_collection.find_one(
                        {"user_id": search_params.userid}, sort=[("timestamp", 1)]
                    )
                    if oldest_search:
                        recent_collection.delete_one({"_id": oldest_search["_id"]})

                # Insert new recent search
                recent_collection.insert_one(search_document)

            except Exception as save_error:
                # Log the error but don't fail the search
                print(f"Warning: Failed to save recent search: {str(save_error)}")

        # Determine effective user_id for search based on user existence in collection
        effective_user_id = get_effective_user_id_for_search(search_params.userid)

        # Base query to filter by user_id (admin users search all, regular users search only their own)
        base_query = {}
        if effective_user_id is not None:
            base_query["user_id"] = effective_user_id

        if not has_criteria:
            # If no criteria provided, return all resumes for this user (or all if admin) sorted by most recent
            results = list(resumes_collection.find(base_query))
        else:
            # Enhanced query building to capture ALL matching candidates for this user
            # Use comprehensive OR logic to find any resume matching any criteria
            or_conditions = []

            # Add experience title conditions - search for ALL titles
            if search_params.experience_titles:
                for title in search_params.experience_titles:
                    title_pattern = re.compile(f".*{re.escape(title)}.*", re.IGNORECASE)
                    or_conditions.append(
                        {"experience.title": {"$regex": title_pattern}}
                    )

            # Add skills conditions - search for ALL skills in both fields
            if search_params.skills:
                for skill in search_params.skills:
                    skill_pattern = re.compile(f".*{re.escape(skill)}.*", re.IGNORECASE)
                    or_conditions.extend(
                        [
                            {"skills": {"$regex": skill_pattern}},
                            {"may_also_known_skills": {"$regex": skill_pattern}},
                        ]
                    )

            # Add education conditions - search for ALL education levels
            if search_params.min_education:
                for edu in search_params.min_education:
                    education_pattern = re.compile(
                        f".*{re.escape(edu)}.*", re.IGNORECASE
                    )
                    or_conditions.append(
                        {"academic_details.education": {"$regex": education_pattern}}
                    )

            # Add location conditions - search for ALL locations
            if search_params.locations:
                for location in search_params.locations:
                    location_pattern = re.compile(
                        f".*{re.escape(location)}.*", re.IGNORECASE
                    )
                    or_conditions.extend(
                        [
                            {
                                "contact_details.current_city": {
                                    "$regex": location_pattern
                                }
                            },
                            {
                                "contact_details.looking_for_jobs_in": {
                                    "$regex": location_pattern
                                }
                            },
                        ]
                    )

            # Build final query combining user_id filter with search criteria
            if or_conditions:
                final_query = {"$and": [base_query, {"$or": or_conditions}]}
            else:
                # If only experience/salary filters, filter by user_id only
                final_query = base_query

            results = list(
                resumes_collection.find(final_query)
            )  # Parse min and max experience strings to months
        min_experience_months = 0
        max_experience_months = float("inf")

        if search_params.min_experience:
            min_experience_months = parse_experience_to_months(
                search_params.min_experience
            )

        if search_params.max_experience:
            max_experience_months = parse_experience_to_months(
                search_params.max_experience
            )

        # Calculate scores and apply additional filters
        scored_results = []

        for resume in results:
            # Define maximum possible scores for normalization
            max_scores = {
                "experience_titles": 25,  # Max 25 points for experience titles
                "skills": 25,  # Max 25 points for skills
                "education": 15,  # Max 15 points for education
                "experience_range": 10,  # Max 10 points for experience range
                "location": 15,  # Max 15 points for location
                "salary": 10,  # Max 10 points for salary
            }

            # Track raw scores for each category
            category_scores = {
                "experience_titles": 0,
                "skills": 0,
                "education": 0,
                "experience_range": 0,
                "location": 0,
                "salary": 0,
            }

            match_details = {
                "experience_title_matches": 0,
                "skills_matches": 0,
                "education_matches": 0,
                "location_matches": 0,
                "experience_range_match": False,
                "salary_range_match": False,
                "matched_experience_titles": [],
                "matched_skills": [],
                "matched_education": [],
                "matched_locations": [],
                "fields_matched": 0,  # Track number of different fields that have matches
            }
            should_include = True

            # Calculate experience title match score - Enhanced to find ALL matching titles
            if search_params.experience_titles:
                field_has_match = False
                title_score = 0
                for exp in resume.get("experience", []):
                    exp_title = exp.get("title", "").lower()
                    for title in search_params.experience_titles:
                        if title.lower() in exp_title:
                            match_details["experience_title_matches"] += 1
                            match_details["matched_experience_titles"].append(title)
                            title_score += 1
                            field_has_match = True

                if field_has_match:
                    # Normalize to max 25 points, with diminishing returns for multiple matches
                    category_scores["experience_titles"] = min(
                        25, title_score * 8 + (title_score - 1) * 2
                    )
                    match_details["fields_matched"] += 1

            # Calculate skills match score - Enhanced to find ALL matching skills
            if search_params.skills:
                field_has_match = False
                skills_score = 0
                resume_skills = [skill.lower() for skill in resume.get("skills", [])]
                may_also_known_skills = [
                    skill.lower() for skill in resume.get("may_also_known_skills", [])
                ]
                all_resume_skills = resume_skills + may_also_known_skills

                for skill in search_params.skills:
                    if skill.lower() in all_resume_skills:
                        match_details["skills_matches"] += 1
                        match_details["matched_skills"].append(skill)
                        skills_score += 1
                        field_has_match = True

                if field_has_match:
                    # Normalize to max 25 points, with diminishing returns for multiple matches
                    category_scores["skills"] = min(
                        25, skills_score * 6 + (skills_score - 1) * 1.5
                    )
                    match_details["fields_matched"] += 1

            # Education level scoring - Enhanced to find ALL matching education levels
            if search_params.min_education:
                field_has_match = False
                education_score = 0
                education_levels = {
                    "10th": 1,
                    "ssc": 1,
                    "12th": 2,
                    "hsc": 2,
                    "inter": 2,
                    "intermediate": 2,
                    "diploma": 3,
                    "iti": 3,
                    "associate": 3,
                    "bachelor": 4,
                    "degree": 4,
                    "graduate": 4,
                    "btech": 4,
                    "be": 4,
                    "bsc": 4,
                    "ba": 4,
                    "bcom": 4,
                    "master": 5,
                    "post graduate": 5,
                    "mtech": 5,
                    "me": 5,
                    "msc": 5,
                    "ma": 5,
                    "mcom": 5,
                    "mba": 5,
                    "phd": 6,
                    "doctorate": 6,
                }

                for edu in resume.get("academic_details", []):
                    education = edu.get("education", "").lower()
                    for search_edu in search_params.min_education:
                        if search_edu.lower() in education:
                            match_details["education_matches"] += 1
                            match_details["matched_education"].append(search_edu)
                            # Get education level for scoring
                            for level_name, level_value in education_levels.items():
                                if level_name in education:
                                    education_score += level_value
                                    field_has_match = True
                                    break

                if field_has_match:
                    # Normalize to max 15 points
                    category_scores["education"] = min(15, education_score * 2.5)
                    match_details["fields_matched"] += 1

            # Experience duration filtering and scoring
            if min_experience_months > 0 or max_experience_months < float("inf"):
                total_exp = resume.get("total_experience", "")
                resume_exp_months = 0

                if total_exp and total_exp != "N/A":
                    resume_exp_months = parse_experience_to_months(str(total_exp))
                else:
                    for exp in resume.get("experience", []):
                        exp_months = calculate_experience_from_dates(
                            exp.get("from_date", ""), exp.get("to", "")
                        )
                        resume_exp_months += exp_months

                if (
                    resume_exp_months >= min_experience_months
                    and resume_exp_months <= max_experience_months
                ):
                    match_details["experience_range_match"] = True
                    category_scores["experience_range"] = (
                        10  # Full points for perfect match
                    )
                    match_details["fields_matched"] += 1
                elif (
                    resume_exp_months >= min_experience_months * 0.8
                ):  # 80% of min experience
                    category_scores["experience_range"] = (
                        5  # Partial points for close match
                    )

            # Location filtering and scoring - Enhanced to track all matching locations
            if search_params.locations:
                field_has_match = False
                location_score = 0
                current_city = resume.get("contact_details", {}).get("current_city", "")
                looking_for_jobs_in = resume.get("contact_details", {}).get(
                    "looking_for_jobs_in", []
                )

                for location in search_params.locations:
                    # Check current city
                    if current_city and current_city != "N/A":
                        if search_city_in_location(current_city, location):
                            match_details["location_matches"] += 1
                            match_details["matched_locations"].append(
                                f"{location} (current)"
                            )
                            location_score += 3  # Higher weight for current city
                            field_has_match = True

                    # Check looking for jobs in
                    if looking_for_jobs_in:
                        for job_location in looking_for_jobs_in:
                            if job_location and job_location != "N/A":
                                if search_city_in_location(job_location, location):
                                    match_details["location_matches"] += 1
                                    match_details["matched_locations"].append(
                                        f"{location} (preference)"
                                    )
                                    location_score += 2  # Good weight for preference
                                    field_has_match = True

                if field_has_match:
                    # Normalize to max 15 points
                    category_scores["location"] = min(15, location_score * 2)
                    match_details["fields_matched"] += 1

            # Salary filtering and scoring - Enhanced with better tracking
            if (
                search_params.min_salary is not None
                or search_params.max_salary is not None
            ):
                expected_salary = resume.get("expected_salary", 0)
                current_salary = resume.get("current_salary", 0)
                candidate_salary = (
                    expected_salary
                    if expected_salary and expected_salary > 0
                    else current_salary
                )

                if candidate_salary and candidate_salary > 0:
                    salary_in_range = True

                    if (
                        search_params.min_salary is not None
                        and candidate_salary < search_params.min_salary
                    ):
                        salary_in_range = False

                    if (
                        search_params.max_salary is not None
                        and candidate_salary > search_params.max_salary
                    ):
                        salary_in_range = False

                    if salary_in_range:
                        match_details["salary_range_match"] = True
                        category_scores["salary"] = 10  # Full points for salary match
                        match_details["fields_matched"] += 1

            # Include resume with enhanced match details
            formatted_resume = format_resume(resume)
            if "total_resume_text" in formatted_resume:
                del formatted_resume["total_resume_text"]

            # Calculate final normalized score (0-100)
            total_possible_score = sum(max_scores.values())  # 100 points total
            actual_score = sum(category_scores.values())

            # Normalize to 0-100 scale
            normalized_score = (actual_score / total_possible_score) * 100

            # Apply field diversity bonus (up to 10 additional points for matching multiple fields)
            field_diversity_bonus = min(10, match_details["fields_matched"] * 2)
            final_score = min(100, normalized_score + field_diversity_bonus)

            formatted_resume["match_score"] = round(final_score, 2)
            formatted_resume["base_score"] = round(normalized_score, 2)
            formatted_resume["field_diversity_bonus"] = field_diversity_bonus
            formatted_resume["category_scores"] = category_scores
            formatted_resume["match_details"] = match_details
            formatted_resume["total_individual_matches"] = sum(
                [
                    match_details["experience_title_matches"],
                    match_details["skills_matches"],
                    match_details["education_matches"],
                    match_details["location_matches"],
                    1 if match_details["experience_range_match"] else 0,
                    1 if match_details["salary_range_match"] else 0,
                ]
            )

            scored_results.append(formatted_resume)

        # Filter results based on relevant_score threshold
        if (
            search_params.relevant_score is not None
            and search_params.relevant_score > 0
        ):
            scored_results = [
                result
                for result in scored_results
                if result.get("match_score", 0) >= search_params.relevant_score
            ]

        # Enhanced sorting: prioritize by fields matched, then by total score, then by individual matches
        sorted_results = sorted(
            scored_results,
            key=lambda x: (
                x.get("match_details", {}).get(
                    "fields_matched", 0
                ),  # Primary: number of different fields matched
                x.get("match_score", 0),  # Secondary: total match score
                x.get(
                    "total_individual_matches", 0
                ),  # Tertiary: total individual matches
            ),
            reverse=True,
        )

        # Return results based on limit (if provided)
        if search_params.limit and search_params.limit > 0:
            final_results = sorted_results[: search_params.limit]
        else:
            final_results = sorted_results

        # If no results found, return informative message instead of empty array
        if not final_results:
            no_results_info = {
                "message": "No matching resumes found",
                "search_summary": {
                    "user_id": search_params.userid,
                    "total_candidates_searched": (
                        len(results) if "results" in locals() else 0
                    ),
                    "search_criteria_used": {},
                    "suggestions": [],
                },
                "results": [],
            }

            # Add search criteria that were used
            if search_params.experience_titles:
                no_results_info["search_summary"]["search_criteria_used"][
                    "experience_titles"
                ] = search_params.experience_titles
            if search_params.skills:
                no_results_info["search_summary"]["search_criteria_used"][
                    "skills"
                ] = search_params.skills
            if search_params.min_education:
                no_results_info["search_summary"]["search_criteria_used"][
                    "min_education"
                ] = search_params.min_education
            if search_params.min_experience:
                no_results_info["search_summary"]["search_criteria_used"][
                    "min_experience"
                ] = search_params.min_experience
            if search_params.max_experience:
                no_results_info["search_summary"]["search_criteria_used"][
                    "max_experience"
                ] = search_params.max_experience
            if search_params.locations:
                no_results_info["search_summary"]["search_criteria_used"][
                    "locations"
                ] = search_params.locations
            if search_params.min_salary is not None:
                no_results_info["search_summary"]["search_criteria_used"][
                    "min_salary"
                ] = search_params.min_salary
            if search_params.max_salary is not None:
                no_results_info["search_summary"]["search_criteria_used"][
                    "max_salary"
                ] = search_params.max_salary
            if search_params.relevant_score is not None:
                no_results_info["search_summary"]["search_criteria_used"][
                    "relevant_score"
                ] = search_params.relevant_score

            # Add helpful suggestions
            suggestions = []
            if search_params.experience_titles:
                suggestions.append("Try using broader or alternative job titles")
            if search_params.skills:
                suggestions.append(
                    "Consider removing some specific skills or using more general skill terms"
                )
            if search_params.min_experience or search_params.max_experience:
                suggestions.append("Adjust the experience range requirements")
            if search_params.locations:
                suggestions.append(
                    "Expand the location search to include nearby cities"
                )
            if search_params.min_salary or search_params.max_salary:
                suggestions.append("Adjust the salary range to be more flexible")
            if search_params.min_education:
                suggestions.append("Consider accepting lower education qualifications")
            if search_params.relevant_score and search_params.relevant_score > 0:
                suggestions.append(
                    "Try lowering the relevance score threshold to see more candidates"
                )

            if not suggestions:
                suggestions.append(
                    "Try adding some search criteria to find matching candidates"
                )

            no_results_info["search_summary"]["suggestions"] = suggestions

            return [no_results_info]

        return final_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manual search failed: {str(e)}")


def search_city_in_location(location: str, search_city: str) -> bool:
    """
    Check if a city name exists in the location string.

    Args:
        location: The location string (current_city or job location)
        search_city: The city name to search for

    Returns:
        True if the city is found in the location, False otherwise
    """
    if not location or not search_city or location == "N/A":
        return False

    location_lower = location.lower()
    search_city_lower = search_city.lower()
    return search_city_lower in location_lower


def parse_experience_to_months(experience_str: str) -> int:
    """
    Parse experience string in format 'X years Y months' or similar to total months.
    """
    if not experience_str:
        return 0

    years = 0
    months = 0

    year_match = re.search(
        r"(\d+)\s*(?:year|years|yr|yrs)", experience_str, re.IGNORECASE
    )
    if year_match:
        years = int(year_match.group(1))

    month_match = re.search(
        r"(\d+)\s*(?:month|months|mo|mos)", experience_str, re.IGNORECASE
    )
    if month_match:
        months = int(month_match.group(1))

    if not year_match and not month_match:
        num_match = re.search(r"(\d+)", experience_str)
        if num_match:
            years = int(num_match.group(1))

    return (years * 12) + months


def calculate_experience_from_dates(from_date: str, to_date: str = None) -> int:
    """
    Calculate experience in months from date range.

    Args:
        from_date: Start date in 'YYYY-MM' format
        to_date: End date in 'YYYY-MM' format (None for current job)

    Returns:
        Experience duration in months
    """
    if not from_date:
        return 0

    try:
        from_parts = from_date.split("-")
        if len(from_parts) != 2:
            return 0

        from_year = int(from_parts[0])
        from_month = int(from_parts[1])

        if to_date and to_date.strip():
            to_parts = to_date.split("-")
            if len(to_parts) != 2:
                return 0
            to_year = int(to_parts[0])
            to_month = int(to_parts[1])
        else:
            from datetime import datetime

            current_date = datetime.now()
            to_year = current_date.year
            to_month = current_date.month

        total_months = (to_year - from_year) * 12 + (to_month - from_month)
        return max(1, total_months)

    except (ValueError, IndexError):
        return 0
