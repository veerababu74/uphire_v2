# resume_api/api/skills_search.py
from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Any
from mangodatabase.client import get_skills_titles_collection
import json
import re
import os

router = APIRouter(
    prefix="/skills",
    tags=["Skills Autocomplete"],
)

# Initialize database collections
skills_titles_collection = get_skills_titles_collection()  # Skills collection


# Load skills from JSON file
def load_skills_from_json():
    """Load all skills from the JSON file and remove duplicates"""
    try:
        json_path = os.path.join("data", "all_skills.json")
        with open(json_path, "r", encoding="utf-8") as f:
            skills_data = json.load(f)
            # Remove duplicates while preserving case, using lowercase for comparison
            skills_list = [
                skill.strip() for skill in skills_data["skills"] if skill.strip()
            ]
            seen_lower = set()
            unique_skills = []
            for skill in skills_list:
                skill_lower = skill.lower()
                if skill_lower not in seen_lower:
                    seen_lower.add(skill_lower)
                    unique_skills.append(skill)
            return unique_skills
    except Exception as e:
        print(f"Error loading skills from JSON: {e}")
        return []


# Load skills from database
def load_skills_from_db():
    """Load all skills from the skills_titles collection where type='skill' and remove duplicates"""
    try:
        skills_cursor = skills_titles_collection.find({"type": "skill"})
        skills_list = [
            doc["value"]
            for doc in skills_cursor
            if "value" in doc and doc["value"].strip()
        ]
        # Remove duplicates while preserving case, using lowercase for comparison
        seen_lower = set()
        unique_skills = []
        for skill in skills_list:
            skill_lower = skill.lower()
            if skill_lower not in seen_lower:
                seen_lower.add(skill_lower)
                unique_skills.append(skill)
        return unique_skills
    except Exception as e:
        print(f"Error loading skills from database: {e}")
        return []


# Load skills from both sources
json_skills = load_skills_from_json()
db_skills = load_skills_from_db()


# Combine and deduplicate skills from both sources
def combine_skills(json_skills_list, db_skills_list):
    """Combine skills from both sources and remove duplicates case-insensitively"""
    all_skills = json_skills_list + db_skills_list
    seen_lower = set()
    unique_combined = []
    for skill in all_skills:
        skill_lower = skill.lower()
        if skill_lower not in seen_lower:
            seen_lower.add(skill_lower)
            unique_combined.append(skill)
    return unique_combined


all_combined_skills = combine_skills(json_skills, db_skills)

# Debug: Print the number of skills loaded
print(f"Loaded {len(json_skills)} skills from JSON file")
print(f"Loaded {len(db_skills)} skills from database")
print(f"Total combined unique skills: {len(all_combined_skills)}")


@router.get(
    "/autocomplete/json/",
    response_model=List[str],
    summary="Autocomplete Skills from JSON File Only",
    description="""
    Get autocomplete suggestions for skills based on input prefix from JSON file only.
    
    **Parameters:**
    - q: Search prefix for skill name (e.g., "pyth" for "Python")
    - limit: Maximum number of suggestions to return
    
    **Returns:**
    List of matching skills sorted alphabetically from the JSON file
    """,
)
async def autocomplete_skills_json_only(
    q: str = Query(
        ..., description="Skill name prefix to search for", min_length=1, example="pyth"
    ),
    limit: int = Query(
        default=10,
        description="Maximum number of suggestions to return",
        ge=1,
        le=50,
        example=10,
    ),
):
    """
    Get autocomplete suggestions for skills from JSON file only.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    try:
        q_lower = q.lower()
        filtered_skills = [
            skill for skill in json_skills if skill.lower().startswith(q_lower)
        ]
        # Remove duplicates from filtered results
        seen_lower = set()
        unique_filtered = []
        for skill in sorted(filtered_skills):
            skill_lower = skill.lower()
            if skill_lower not in seen_lower:
                seen_lower.add(skill_lower)
                unique_filtered.append(skill)
        return unique_filtered[:limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Autocomplete failed: {str(e)}")


@router.get(
    "/autocomplete/database/",
    response_model=List[str],
    summary="Autocomplete Skills from Database Only",
    description="""
    Get autocomplete suggestions for skills based on input prefix from database only.
    
    **Parameters:**
    - q: Search prefix for skill name (e.g., "pyth" for "Python")
    - limit: Maximum number of suggestions to return
    
    **Returns:**
    List of matching skills sorted alphabetically from the database
    """,
)
async def autocomplete_skills_database_only(
    q: str = Query(
        ..., description="Skill name prefix to search for", min_length=1, example="pyth"
    ),
    limit: int = Query(
        default=10,
        description="Maximum number of suggestions to return",
        ge=1,
        le=50,
        example=10,
    ),
):
    """
    Get autocomplete suggestions for skills from database only.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    try:
        # Direct database search for better performance
        regex_pattern = f"^{re.escape(q)}"
        db_results = skills_titles_collection.find(
            {"type": "skill", "value": {"$regex": regex_pattern, "$options": "i"}}
        ).limit(limit)

        db_skills_found = [doc["value"] for doc in db_results if "value" in doc]

        if db_skills_found:
            # Remove duplicates from database results
            seen_lower = set()
            unique_db_skills = []
            for skill in sorted(db_skills_found):
                skill_lower = skill.lower()
                if skill_lower not in seen_lower:
                    seen_lower.add(skill_lower)
                    unique_db_skills.append(skill)
            return unique_db_skills[:limit]

        # Fallback to in-memory search if database search doesn't yield results
        q_lower = q.lower()
        filtered_skills = [
            skill for skill in db_skills if skill.lower().startswith(q_lower)
        ]
        # Remove duplicates from filtered results
        seen_lower = set()
        unique_filtered = []
        for skill in sorted(filtered_skills):
            skill_lower = skill.lower()
            if skill_lower not in seen_lower:
                seen_lower.add(skill_lower)
                unique_filtered.append(skill)
        return unique_filtered[:limit]

    except Exception as e:
        # Fallback to in-memory search on database error
        q_lower = q.lower()
        filtered_skills = [
            skill for skill in db_skills if skill.lower().startswith(q_lower)
        ]
        # Remove duplicates from fallback results
        seen_lower = set()
        unique_filtered = []
        for skill in sorted(filtered_skills):
            skill_lower = skill.lower()
            if skill_lower not in seen_lower:
                seen_lower.add(skill_lower)
                unique_filtered.append(skill)
        return unique_filtered[:limit]


@router.get(
    "/autocomplete/combined/",
    response_model=List[str],
    summary="Autocomplete Skills from Both JSON and Database",
    description="""
    Get autocomplete suggestions for skills based on input prefix from both JSON file and database.
    
    **Parameters:**
    - q: Search prefix for skill name (e.g., "pyth" for "Python")
    - limit: Maximum number of suggestions to return
    
    **Returns:**
    List of matching skills sorted alphabetically from both sources (deduplicated)
    """,
)
async def autocomplete_skills_combined(
    q: str = Query(
        ..., description="Skill name prefix to search for", min_length=1, example="pyth"
    ),
    limit: int = Query(
        default=10,
        description="Maximum number of suggestions to return",
        ge=1,
        le=50,
        example=10,
    ),
):
    """
    Get autocomplete suggestions for skills from both JSON file and database.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    try:
        q_lower = q.lower()
        filtered_skills = [
            skill for skill in all_combined_skills if skill.lower().startswith(q_lower)
        ]
        # Remove duplicates from filtered results (though all_combined_skills should already be unique)
        seen_lower = set()
        unique_filtered = []
        for skill in sorted(filtered_skills):
            skill_lower = skill.lower()
            if skill_lower not in seen_lower:
                seen_lower.add(skill_lower)
                unique_filtered.append(skill)
        return unique_filtered[:limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Autocomplete failed: {str(e)}")


@router.get(
    "/all/json/",
    response_model=List[str],
    summary="Get All Skills from JSON File",
    description="""
    Get all skills from the JSON file.
    
    **Returns:**
    List of all skills from the JSON file sorted alphabetically
    """,
)
async def get_all_skills_json():
    """
    Get all skills from JSON file.
    """
    try:
        return sorted(json_skills)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get skills from JSON: {str(e)}"
        )


@router.get(
    "/all/database/",
    response_model=List[str],
    summary="Get All Skills from Database",
    description="""
    Get all skills from the database.
    
    **Returns:**
    List of all skills from the database sorted alphabetically
    """,
)
async def get_all_skills_database():
    """
    Get all skills from database.
    """
    try:
        return sorted(db_skills)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get skills from database: {str(e)}"
        )


@router.get(
    "/all/combined/",
    response_model=List[str],
    summary="Get All Skills from Both JSON and Database",
    description="""
    Get all skills from both JSON file and database (deduplicated).
    
    **Returns:**
    List of all unique skills from both sources sorted alphabetically
    """,
)
async def get_all_skills_combined():
    """
    Get all skills from both JSON file and database.
    """
    try:
        return sorted(all_combined_skills)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get combined skills: {str(e)}"
        )


@router.get(
    "/count/",
    response_model=Dict[str, int],
    summary="Get Skills Count from All Sources",
    description="""
    Get the count of skills from all sources.
    
    **Returns:**
    Dictionary with counts from JSON, database, and combined total
    """,
)
async def get_skills_count():
    """
    Get count of skills from all sources.
    """
    try:
        return {
            "json_skills_count": len(json_skills),
            "database_skills_count": len(db_skills),
            "combined_unique_count": len(all_combined_skills),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get skills count: {str(e)}"
        )


@router.get(
    "/refresh/",
    response_model=Dict[str, Any],
    summary="Refresh Skills Cache",
    description="""
    Refresh the in-memory skills cache by reloading from both JSON file and database.
    
    **Returns:**
    Updated counts and status message
    """,
)
async def refresh_skills_cache():
    """
    Refresh the in-memory skills cache from both sources.
    """
    try:
        global json_skills, db_skills, all_combined_skills

        json_skills = load_skills_from_json()
        db_skills = load_skills_from_db()
        all_combined_skills = combine_skills(json_skills, db_skills)

        return {
            "json_skills_count": len(json_skills),
            "database_skills_count": len(db_skills),
            "combined_unique_count": len(all_combined_skills),
            "message": "Skills cache refreshed successfully from both sources",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to refresh skills cache: {str(e)}"
        )
