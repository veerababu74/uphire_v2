from fastapi import APIRouter, Query, Body, Depends, HTTPException
from mangodatabase.client import get_collection
from embeddings.vectorizer import Vectorizer
from core.helpers import format_resume
from typing import List, Dict, Any
import pymongo
import re

router = APIRouter(
    prefix="/autocomplete",
    tags=["Job Search Autocomplete"],
)

collection = get_collection()
vectorizer = Vectorizer()


def process_results(results, key):
    """
    Process the results by converting to lowercase, stripping whitespace,
    and removing duplicates while maintaining order.
    Also filters out empty or blank strings.
    """
    processed = []
    seen = set()
    for result in results:
        value = result.get(key)
        if not isinstance(value, str):
            continue  # Skip invalid or None values
        stripped = value.strip()
        if not stripped:  # Skip empty or whitespace-only strings
            continue
        lower_value = stripped.lower()
        if lower_value not in seen:
            seen.add(lower_value)
            processed.append(value)  # Preserve original casing
    return processed


def extract_skills(raw_data: str) -> List[str]:
    if not raw_data or not isinstance(raw_data, str):
        return []

    cleaned = re.sub(r".*?:", "", raw_data)
    parenthetical_content = re.findall(r"\((.*?)\)", cleaned)
    cleaned = re.sub(r"\(.*?\)", ",", cleaned)
    skills = re.split(r"[,/&]|\band\b", cleaned)
    for content in parenthetical_content:
        skills.extend(re.split(r"[,\s]+", content))

    processed_skills = []
    for skill in skills:
        skill = re.sub(r"[^\w\s-]", "", skill).strip().lower()
        if skill and skill not in {"others", "and", "in", "of"}:
            processed_skills.append(skill)
    return processed_skills


@router.get(
    "/job_titles/",
    response_model=List[str],
    summary="Autocomplete Job Titles",
    description="""
    Get autocomplete suggestions for job titles based on input prefix.
    Uses both exact matching and semantic search for better results.
    **Parameters:**
    - prefix: Text to search for in job titles (e.g., "software eng")
    - limit: Maximum number of suggestions to return
    **Returns:**
    List of matching job titles sorted by relevance
    **examples Usage:**
    - prefix="soft" might return ["Software Engineer", "Software Developer", "Software Architect"]
    - prefix="data" might return ["Data Scientist", "Data Engineer", "Data Analyst"]
    """,
    responses={
        200: {
            "description": "Successful job title suggestions",
            "content": {
                "application/json": {
                    "example": [
                        "software engineer",
                        "software developer",
                        "senior software engineer",
                        "software architect",
                        "software team lead",
                    ]
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "Search prefix cannot be empty"}
                }
            },
        },
    },
)
async def autocomplete_titles(
    prefix: str = Query(
        ...,
        description="Job title prefix to search for",
        min_length=2,
        example="software eng",
    ),
    limit: int = Query(
        default=10, description="Maximum number of suggestions", ge=1, le=50, example=5
    ),
):
    try:
        pipeline = [
            {"$unwind": "$experience"},
            {
                "$match": {
                    "experience.title": {"$regex": f".*{prefix}.*", "$options": "i"}
                }
            },
            {"$group": {"_id": "$experience.title"}},
            {"$limit": limit},
            {"$project": {"title": "$_id", "_id": 0}},
        ]
        results = list(collection.aggregate(pipeline))
        titles = process_results(results, "title")

        if len(titles) < limit:
            query_embedding = vectorizer.generate_embedding(prefix)
            semantic_pipeline = [
                {
                    "$search": {
                        "index": "vector_search_index",
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": "experience_text_vector",
                            "k": limit,
                        },
                    }
                },
                {"$unwind": "$experience"},
                {"$project": {"title": "$experience.title"}},
                {"$limit": limit},
            ]
            semantic_results = list(collection.aggregate(semantic_pipeline))
            semantic_titles = process_results(semantic_results, "title")
            titles.extend(
                [
                    title
                    for title in semantic_titles
                    if title.lower().strip()
                    not in map(str.lower, map(str.strip, titles))
                ]
            )

        return titles[:limit]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Title autocomplete failed: {str(e)}"
        )


@router.get(
    "/job_skillsv1/",
    response_model=List[str],
    summary="Autocomplete Technical Skills",
    description="""
    Get autocomplete suggestions for technical skills based on input prefix.
    Uses both exact matching and semantic search for better results.
    **Parameters:**
    - prefix: Text to search for in skills (e.g., "py" for Python)
    - limit: Maximum number of suggestions to return
    **Returns:**
    List of matching skills sorted by relevance
    **examples Usage:**
    - prefix="py" might return ["python", "pytorch", "pyqt"]
    - prefix="java" might return ["javascript", "java", "java spring"]
    """,
    responses={
        200: {
            "description": "Successful skill suggestions",
            "content": {
                "application/json": {
                    "example": [
                        "python",
                        "pytorch",
                        "python django",
                        "python flask",
                        "python scripting",
                    ]
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "Search prefix cannot be empty"}
                }
            },
        },
    },
)
async def autocomplete_skills(
    prefix: str = Query(
        ..., description="Skill prefix to search for", min_length=2, example="py"
    ),
    limit: int = Query(
        default=10, description="Maximum number of suggestions", ge=1, le=50, example=5
    ),
):
    try:
        # First, let's try a broader search to see if we have any skills data
        pipeline = [
            {"$unwind": "$skills"},
            {"$match": {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}}},
            {"$group": {"_id": "$skills"}},
            {"$limit": limit * 5},  # Increase limit to account for splitting
            {"$project": {"raw_skill": "$_id", "_id": 0}},
        ]
        raw_results = list(collection.aggregate(pipeline))

        # If no exact matches, try a different field or broader search
        if not raw_results:
            # Try searching in different skill fields or use a more generic approach
            alternative_pipeline = [
                {
                    "$match": {
                        "$or": [
                            {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}},
                            {
                                "technical_skills": {
                                    "$regex": f".*{prefix}.*",
                                    "$options": "i",
                                }
                            },
                            {
                                "experience.skills": {
                                    "$regex": f".*{prefix}.*",
                                    "$options": "i",
                                }
                            },
                        ]
                    }
                },
                {"$limit": limit * 2},
            ]
            alternative_results = list(collection.aggregate(alternative_pipeline))

            # Extract skills from various fields
            all_skills = []
            for doc in alternative_results:
                if doc.get("skills"):
                    if isinstance(doc["skills"], list):
                        all_skills.extend(doc["skills"])
                    else:
                        all_skills.append(doc["skills"])
                if doc.get("technical_skills"):
                    if isinstance(doc["technical_skills"], list):
                        all_skills.extend(doc["technical_skills"])
                    else:
                        all_skills.append(doc["technical_skills"])

            # Filter skills that match the prefix
            matching_skills = [
                skill.strip()
                for skill in all_skills
                if isinstance(skill, str) and prefix.lower() in skill.lower()
            ]

            # Remove duplicates while preserving order
            unique_skills = []
            seen = set()
            for skill in matching_skills:
                skill_lower = skill.lower()
                if skill_lower not in seen:
                    seen.add(skill_lower)
                    unique_skills.append(skill)

            return unique_skills[:limit]

        raw_skills = [
            result["raw_skill"] for result in raw_results if result.get("raw_skill")
        ]

        extracted_skills = [
            skill for raw_skill in raw_skills for skill in extract_skills(raw_skill)
        ]

        unique_skills = list({skill for skill in extracted_skills})
        sorted_skills = sorted(
            unique_skills, key=lambda s: (not s.startswith(prefix.lower()), s)
        )

        return sorted_skills[:limit]
    except Exception as e:
        # Return a more detailed error for debugging
        raise HTTPException(
            status_code=500, detail=f"Skills autocomplete failed: {str(e)}"
        )


# New combined route
@router.get(
    "/jobs_and_skills/",
    response_model=Dict[str, List[str]],
    summary="Autocomplete Job Titles and Skills",
    description="""
    Get autocomplete suggestions for both job titles and technical skills based on input prefix.
    Combines results from exact matching and semantic search.
    **Parameters:**
    - prefix: Text to search for in both job titles and skills (e.g., "py" for Python-related jobs and skills)
    - limit: Maximum number of suggestions for each category
    **Returns:**
    Object containing two arrays: one for matching job titles and one for matching skills
    **examples Response:**
    {
      "titles": ["Python Developer", "Senior Python Engineer", "Python Team Lead"],
      "skills": ["Python", "Python Django", "Python Flask"]
    }
    """,
    responses={
        200: {
            "description": "Successful job title and skill suggestions",
            "content": {
                "application/json": {
                    "example": {
                        "titles": [
                            "Python Developer",
                            "Senior Python Engineer",
                            "Python Team Lead",
                        ],
                        "skills": [
                            "Python",
                            "Python Django",
                            "Python Flask",
                        ],
                    }
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "Search prefix cannot be empty"}
                }
            },
        },
    },
)
async def autocomplete_jobs_and_skills(
    prefix: str = Query(
        ...,
        description="Prefix to search in titles and skills",
        min_length=2,
        example="py",
    ),
    limit: int = Query(
        default=5, description="Maximum number of suggestions per category", ge=1, le=50
    ),
):
    try:
        # --- Fetch and process job titles ---
        title_pipeline = [
            {"$unwind": "$experience"},
            {
                "$match": {
                    "experience.title": {"$regex": f".*{prefix}.*", "$options": "i"}
                }
            },
            {"$group": {"_id": "$experience.title"}},
            {"$limit": limit},
            {"$project": {"title": "$_id", "_id": 0}},
        ]
        title_results = list(collection.aggregate(title_pipeline))
        titles = process_results(title_results, "title")

        if len(titles) < limit:
            query_embedding = vectorizer.generate_embedding(prefix)
            title_semantic_pipeline = [
                {
                    "$search": {
                        "index": "vector_search_index",
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": "experience_text_vector",
                            "k": limit,
                        },
                    }
                },
                {"$unwind": "$experience"},
                {"$project": {"title": "$experience.title"}},
                {"$limit": limit},
            ]
            semantic_title_results = list(collection.aggregate(title_semantic_pipeline))
            semantic_titles = process_results(semantic_title_results, "title")
            titles.extend(
                [
                    title
                    for title in semantic_titles
                    if title.lower().strip()
                    not in map(str.lower, map(str.strip, titles))
                ]
            )

        # --- Fetch and process skills ---
        skill_pipeline = [
            {"$unwind": "$skills"},
            {"$match": {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}}},
            {"$group": {"_id": "$skills"}},
            {"$limit": limit * 5},
            {"$project": {"raw_skill": "$_id", "_id": 0}},
        ]
        raw_skill_results = list(collection.aggregate(skill_pipeline))
        raw_skills = [
            result["raw_skill"]
            for result in raw_skill_results
            if result.get("raw_skill")
        ]
        extracted_skills = [
            skill for raw_skill in raw_skills for skill in extract_skills(raw_skill)
        ]
        unique_skills = list(set(extracted_skills))
        sorted_skills = sorted(
            unique_skills, key=lambda s: (not s.startswith(prefix.lower()), s)
        )

        return {
            "titles": titles[:limit],
            "skills": sorted_skills[:limit],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Combined autocomplete failed: {str(e)}"
        )


# @router.get(
#     "/debug/sample_data",
#     summary="Debug: Sample Data Structure",
#     description="Returns a sample document to understand the data structure",
# )
# async def get_sample_data():
#     try:
#         # Get a sample document to understand the structure
#         sample = collection.find_one({}, {"_id": 0})
#         if sample:
#             # Show keys and sample values
#             structure = {}
#             for key, value in sample.items():
#                 if isinstance(value, list) and value:
#                     structure[key] = (
#                         f"Array with {len(value)} items, sample: {value[0] if value else 'empty'}"
#                     )
#                 elif isinstance(value, dict):
#                     structure[key] = f"Object with keys: {list(value.keys())}"
#                 else:
#                     structure[key] = (
#                         f"Type: {type(value).__name__}, value: {str(value)[:100]}"
#                     )
#             return {"structure": structure, "sample_keys": list(sample.keys())}
#         else:
#             return {"message": "No documents found in collection"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")


# @router.get(
#     "/debug/skills_fields",
#     summary="Debug: Available Skills Fields",
#     description="Check what skill-related fields exist in the database",
# )
# async def check_skills_fields():
#     try:
#         # Check for various skill-related fields
#         pipeline = [
#             {"$limit": 10},
#             {
#                 "$project": {
#                     "has_skills": {"$ifNull": ["$skills", "NOT_FOUND"]},
#                     "has_technical_skills": {
#                         "$ifNull": ["$technical_skills", "NOT_FOUND"]
#                     },
#                     "has_experience_skills": {
#                         "$ifNull": ["$experience.skills", "NOT_FOUND"]
#                     },
#                     "skills_type": {"$type": "$skills"},
#                     "skills_sample": {"$slice": ["$skills", 3]},
#                 }
#             },
#         ]
#         results = list(collection.aggregate(pipeline))
#         return {"sample_documents": results, "total_checked": len(results)}
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Skills fields check failed: {str(e)}"
#         )


"""


from fastapi import APIRouter, Query, Body, Depends, HTTPException
from mangodatabase.client import get_collection
from embeddings.vectorizer import Vectorizer
from core.helpers import format_resume
from typing import List, Dict, Any
import pymongo
import re
import difflib
import time
from functools import lru_cache

router = APIRouter(
    prefix="/autocomplete",
    tags=["Job Search Autocomplete"],
)

collection = get_collection()
vectorizer = Vectorizer()

# Simple in-memory cache for frequently accessed results
_autocomplete_cache = {}
_cache_expiry = {}
CACHE_DURATION = 300  # 5 minutes cache


def process_results(results, key):

    processed = []
    seen = set()
    for result in results:
        value = result.get(key)
        if not isinstance(value, str):
            continue  # Skip invalid or None values
        stripped = value.strip()
        if not stripped:  # Skip empty or whitespace-only strings
            continue
        lower_value = stripped.lower()
        if lower_value not in seen:
            seen.add(lower_value)
            processed.append(value)  # Preserve original casing
    return processed


def extract_skills(raw_data: str) -> List[str]:
    if not raw_data or not isinstance(raw_data, str):
        return []

    # Handle different skill formats more carefully
    skills = []

    # Split by common delimiters while preserving compound skills
    delimiters = r"[,;|/&]|\sand\s|\sor\s"
    raw_skills = re.split(delimiters, raw_data, flags=re.IGNORECASE)

    for skill in raw_skills:
        # Clean the skill but preserve important characters
        cleaned = re.sub(
            r"^[:\-\s]+|[:\-\s]+$", "", skill
        )  # Remove leading/trailing separators
        cleaned = re.sub(r"\s+", " ", cleaned)  # Normalize whitespace
        cleaned = cleaned.strip()

        # Filter out non-skills and very short/long entries
        if (
            cleaned
            and len(cleaned) >= 2
            and len(cleaned) <= 50
            and not re.match(r"^\d+$", cleaned)  # Not just numbers
            and cleaned.lower()
            not in {
                "others",
                "and",
                "in",
                "of",
                "the",
                "with",
                "using",
                "etc",
                "various",
            }
        ):
            skills.append(cleaned)

    return skills


def calculate_relevance_score(text: str, prefix: str) -> float:
    # "Calculate relevance score for autocomplete results with fuzzy matching"
    if not text or not prefix:
        return 0.0

    text_lower = text.lower()
    prefix_lower = prefix.lower()

    # Exact match gets highest score
    if text_lower == prefix_lower:
        return 1.0

    # Starts with prefix gets high score
    if text_lower.startswith(prefix_lower):
        return 0.9 - (len(text) - len(prefix)) * 0.01

    # Contains prefix gets medium score
    if prefix_lower in text_lower:
        # Position matters - earlier is better
        position = text_lower.find(prefix_lower)
        position_penalty = position * 0.01
        return 0.7 - position_penalty

    # Word boundary match
    if re.search(r"\b" + re.escape(prefix_lower), text_lower):
        return 0.6

    # Fuzzy matching for typos and partial matches
    similarity = difflib.SequenceMatcher(None, prefix_lower, text_lower).ratio()
    if similarity > 0.7:  # 70% similarity threshold
        return 0.5 * similarity

    # Check if any word in text starts with prefix
    words = text_lower.split()
    for word in words:
        if word.startswith(prefix_lower):
            return 0.4

    # Check character-level similarity for abbreviations
    if len(prefix) >= 2:
        # Check if prefix characters appear in order
        prefix_chars = list(prefix_lower)
        text_chars = list(text_lower.replace(" ", ""))

        i = 0
        for char in text_chars:
            if i < len(prefix_chars) and char == prefix_chars[i]:
                i += 1

        if i == len(prefix_chars):  # All prefix characters found in order
            return 0.3

    return 0.0


def fuzzy_search_enhancement(
    candidates: List[str], prefix: str, limit: int
) -> List[str]:
    #"Enhance search results with fuzzy matching"
    if not candidates:
        return []

    # Score all candidates
    scored_candidates = []
    for candidate in candidates:
        score = calculate_relevance_score(candidate, prefix)
        if score > 0:
            scored_candidates.append((candidate, score))

    # Sort by score and return top results
    scored_candidates.sort(key=lambda x: (-x[1], len(x[0]), x[0].lower()))
    return [candidate for candidate, _ in scored_candidates[:limit]]


def get_cache_key(prefix: str, limit: int, search_type: str) -> str:
    # Generate cache key for results
    return f"{search_type}:{prefix.lower()}:{limit}"


def is_cache_valid(key: str) -> bool:
    # Check if cache entry is still valid
    return key in _cache_expiry and time.time() < _cache_expiry[key]


def set_cache(key: str, value: Any):
    #Set cache value with expiry
    _autocomplete_cache[key] = value
    _cache_expiry[key] = time.time() + CACHE_DURATION


def get_enhanced_skills(prefix: str, limit: int) -> List[str]:
    # Enhanced skill search across multiple fields with caching
    cache_key = get_cache_key(prefix, limit, "skills")

    # Check cache first
    if is_cache_valid(cache_key):
        return _autocomplete_cache[cache_key]

    # Optimized search with better indexing strategy
    multi_field_pipeline = [
        {
            "$match": {
                "$or": [
                    {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}},
                    {"technical_skills": {"$regex": f".*{prefix}.*", "$options": "i"}},
                    {"experience.skills": {"$regex": f".*{prefix}.*", "$options": "i"}},
                ]
            }
        },
        {"$limit": limit * 8},  # Optimized limit
        {
            "$project": {
                "skills": 1,
                "technical_skills": 1,
                "experience.skills": 1,
                "_id": 0,  # Exclude _id for better performance
            }
        },
    ]

    results = list(collection.aggregate(multi_field_pipeline))
    skill_candidates = set()

    # More efficient skill extraction
    for doc in results:
        # Process all skill fields uniformly
        for field_name in ["skills", "technical_skills"]:
            field_data = doc.get(field_name)
            if field_data:
                if isinstance(field_data, list):
                    for item in field_data:
                        if isinstance(item, str) and prefix.lower() in item.lower():
                            skill_candidates.update(extract_skills(item))
                elif (
                    isinstance(field_data, str) and prefix.lower() in field_data.lower()
                ):
                    skill_candidates.update(extract_skills(field_data))

        # Handle experience skills efficiently
        experience_data = doc.get("experience")
        if isinstance(experience_data, list):
            for exp in experience_data[:3]:  # Limit experience entries for performance
                exp_skills = exp.get("skills") if isinstance(exp, dict) else None
                if exp_skills:
                    if isinstance(exp_skills, list):
                        for skill in exp_skills:
                            if (
                                isinstance(skill, str)
                                and prefix.lower() in skill.lower()
                            ):
                                skill_candidates.update(extract_skills(skill))
                    elif (
                        isinstance(exp_skills, str)
                        and prefix.lower() in exp_skills.lower()
                    ):
                        skill_candidates.update(extract_skills(exp_skills))

    # Filter candidates more efficiently
    valid_skills = []
    for skill in skill_candidates:
        if len(skill) >= 2 and len(skill) <= 50 and prefix.lower() in skill.lower():
            valid_skills.append(skill)

    # Score and sort
    scored_skills = [
        (skill, calculate_relevance_score(skill, prefix)) for skill in valid_skills
    ]
    scored_skills = [(skill, score) for skill, score in scored_skills if score > 0]
    scored_skills.sort(key=lambda x: (-x[1], len(x[0]), x[0].lower()))

    result = [skill for skill, score in scored_skills[:limit]]

    # Cache the result
    set_cache(cache_key, result)

    return result


@router.get(
    "/job_titles/",
    response_model=List[str],
    summary="Autocomplete Job Titles",

    responses={
        200: {
            "description": "Successful job title suggestions",
            "content": {
                "application/json": {
                    "example": [
                        "software engineer",
                        "software developer",
                        "senior software engineer",
                        "software architect",
                        "software team lead",
                    ]
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "Search prefix cannot be empty"}
                }
            },
        },
    },
)
async def autocomplete_titles(
    prefix: str = Query(
        ...,
        description="Job title prefix to search for",
        min_length=2,
        example="software eng",
    ),
    limit: int = Query(
        default=10, description="Maximum number of suggestions", ge=1, le=50, example=5
    ),
):
    try:
        pipeline = [
            {"$unwind": "$experience"},
            {
                "$match": {
                    "experience.title": {"$regex": f".*{prefix}.*", "$options": "i"}
                }
            },
            {"$group": {"_id": "$experience.title"}},
            {"$limit": limit},
            {"$project": {"title": "$_id", "_id": 0}},
        ]
        results = list(collection.aggregate(pipeline))
        titles = process_results(results, "title")

        if len(titles) < limit:
            query_embedding = vectorizer.generate_embedding(prefix)
            semantic_pipeline = [
                {
                    "$search": {
                        "index": "vector_search_index",
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": "experience_text_vector",
                            "k": limit,
                        },
                    }
                },
                {"$unwind": "$experience"},
                {"$project": {"title": "$experience.title"}},
                {"$limit": limit},
            ]
            semantic_results = list(collection.aggregate(semantic_pipeline))
            semantic_titles = process_results(semantic_results, "title")
            titles.extend(
                [
                    title
                    for title in semantic_titles
                    if title.lower().strip()
                    not in map(str.lower, map(str.strip, titles))
                ]
            )

        return titles[:limit]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Title autocomplete failed: {str(e)}"
        )


@router.get(
    "/job_skillsv1/",
    response_model=List[str],
    summary="Autocomplete Technical Skills",
   
    responses={
        200: {
            "description": "Successful skill suggestions",
            "content": {
                "application/json": {
                    "example": [
                        "python",
                        "pytorch",
                        "python django",
                        "python flask",
                        "python scripting",
                    ]
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "Search prefix cannot be empty"}
                }
            },
        },
    },
)
async def autocomplete_skills(
    prefix: str = Query(
        ..., description="Skill prefix to search for", min_length=2, example="py"
    ),
    limit: int = Query(
        default=10, description="Maximum number of suggestions", ge=1, le=50, example=5
    ),
):
    try:
        # First, let's try a broader search to see if we have any skills data
        pipeline = [
            {"$unwind": "$skills"},
            {"$match": {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}}},
            {"$group": {"_id": "$skills"}},
            {"$limit": limit * 5},  # Increase limit to account for splitting
            {"$project": {"raw_skill": "$_id", "_id": 0}},
        ]
        raw_results = list(collection.aggregate(pipeline))

        # If no exact matches, try a different field or broader search
        if not raw_results:
            # Try searching in different skill fields or use a more generic approach
            alternative_pipeline = [
                {
                    "$match": {
                        "$or": [
                            {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}},
                            {
                                "technical_skills": {
                                    "$regex": f".*{prefix}.*",
                                    "$options": "i",
                                }
                            },
                            {
                                "experience.skills": {
                                    "$regex": f".*{prefix}.*",
                                    "$options": "i",
                                }
                            },
                        ]
                    }
                },
                {"$limit": limit * 2},
            ]
            alternative_results = list(collection.aggregate(alternative_pipeline))

            # Extract skills from various fields
            all_skills = []
            for doc in alternative_results:
                if doc.get("skills"):
                    if isinstance(doc["skills"], list):
                        all_skills.extend(doc["skills"])
                    else:
                        all_skills.append(doc["skills"])
                if doc.get("technical_skills"):
                    if isinstance(doc["technical_skills"], list):
                        all_skills.extend(doc["technical_skills"])
                    else:
                        all_skills.append(doc["technical_skills"])

            # Filter skills that match the prefix
            matching_skills = [
                skill.strip()
                for skill in all_skills
                if isinstance(skill, str) and prefix.lower() in skill.lower()
            ]

            # Remove duplicates while preserving order
            unique_skills = []
            seen = set()
            for skill in matching_skills:
                skill_lower = skill.lower()
                if skill_lower not in seen:
                    seen.add(skill_lower)
                    unique_skills.append(skill)

            return unique_skills[:limit]

        raw_skills = [
            result["raw_skill"] for result in raw_results if result.get("raw_skill")
        ]

        extracted_skills = [
            skill for raw_skill in raw_skills for skill in extract_skills(raw_skill)
        ]

        unique_skills = list({skill for skill in extracted_skills})
        sorted_skills = sorted(
            unique_skills, key=lambda s: (not s.startswith(prefix.lower()), s)
        )

        return sorted_skills[:limit]
    except Exception as e:
        # Return a more detailed error for debugging
        raise HTTPException(
            status_code=500, detail=f"Skills autocomplete failed: {str(e)}"
        )


# New combined route
@router.get(
    "/jobs_and_skills/",
    response_model=Dict[str, List[str]],
    summary="Autocomplete Job Titles and Skills",
   
    responses={
        200: {
            "description": "Successful job title and skill suggestions",
            "content": {
                "application/json": {
                    "example": {
                        "titles": [
                            "Python Developer",
                            "Senior Python Engineer",
                            "Python Team Lead",
                        ],
                        "skills": [
                            "Python",
                            "Python Django",
                            "Python Flask",
                        ],
                    }
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "Search prefix cannot be empty"}
                }
            },
        },
    },
)
async def autocomplete_jobs_and_skills(
    prefix: str = Query(
        ...,
        description="Prefix to search in titles and skills",
        min_length=2,
        max_length=100,
        example="py",
    ),
    limit: int = Query(
        default=5, description="Maximum number of suggestions per category", ge=1, le=50
    ),
):
    try:
        # Input validation and sanitization
        prefix = prefix.strip()
        if not prefix or len(prefix) < 2:
            raise HTTPException(
                status_code=400,
                detail="Search prefix must be at least 2 characters long",
            )

        # Sanitize input to prevent regex injection
        prefix = re.sub(r"[^\w\s\-\+\#\.]", "", prefix)
        if not prefix:
            raise HTTPException(
                status_code=400, detail="Invalid search prefix after sanitization"
            )

        # Check cache for complete response
        cache_key = get_cache_key(prefix, limit, "combined")
        if is_cache_valid(cache_key):
            return _autocomplete_cache[cache_key]

        # --- Enhanced Job Title Search with Performance Optimization ---
        title_candidates = []

        # Optimized title search pipeline
        title_pipeline = [
            {"$unwind": "$experience"},
            {
                "$match": {
                    "$and": [
                        {"experience.title": {"$exists": True, "$ne": "", "$ne": None}},
                        {
                            "experience.title": {
                                "$regex": f".*{prefix}.*",
                                "$options": "i",
                            }
                        },
                    ]
                }
            },
            {"$group": {"_id": "$experience.title", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit * 2},  # Reduced for better performance
            {"$project": {"title": "$_id", "frequency": "$count", "_id": 0}},
        ]

        try:
            title_results = list(
                collection.aggregate(title_pipeline, maxTimeMS=5000)
            )  # 5 second timeout

            # Process titles with enhanced scoring
            for result in title_results:
                title = result.get("title", "").strip()
                if title and len(title) >= 2:
                    relevance_score = calculate_relevance_score(title, prefix)
                    frequency_bonus = min(result.get("frequency", 1) / 100, 0.1)
                    final_score = relevance_score + frequency_bonus
                    if final_score > 0.1:  # Filter low-relevance results
                        title_candidates.append((title, final_score))

        except Exception as title_error:
            print(f"Title search error: {title_error}")
            # Fallback to simpler search
            simple_title_pipeline = [
                {"$unwind": "$experience"},
                {
                    "$match": {
                        "experience.title": {"$regex": f"^{prefix}.*", "$options": "i"}
                    }
                },
                {"$group": {"_id": "$experience.title"}},
                {"$limit": limit},
                {"$project": {"title": "$_id", "_id": 0}},
            ]
            fallback_results = list(collection.aggregate(simple_title_pipeline))
            for result in fallback_results:
                title = result.get("title", "").strip()
                if title:
                    title_candidates.append((title, 0.8))

        # Enhanced semantic search for titles (only if needed and vectorizer available)
        if len(title_candidates) < limit and hasattr(vectorizer, "generate_embedding"):
            try:
                query_embedding = vectorizer.generate_embedding(prefix + " job title")
                semantic_title_pipeline = [
                    {
                        "$search": {
                            "index": "vector_search_index",
                            "knnBeta": {
                                "vector": query_embedding,
                                "path": "experience_text_vector",
                                "k": limit,
                            },
                        }
                    },
                    {"$unwind": "$experience"},
                    {"$match": {"experience.title": {"$exists": True, "$ne": ""}}},
                    {
                        "$project": {
                            "title": "$experience.title",
                            "score": {"$meta": "searchScore"},
                        }
                    },
                    {"$limit": limit},
                ]

                semantic_results = list(
                    collection.aggregate(semantic_title_pipeline, maxTimeMS=3000)
                )
                existing_titles = {title.lower() for title, _ in title_candidates}

                for result in semantic_results:
                    title = result.get("title", "").strip()
                    if (
                        title
                        and title.lower() not in existing_titles
                        and len(title) >= 2
                    ):
                        semantic_score = 0.3 + min(result.get("score", 0) / 10, 0.2)
                        title_candidates.append((title, semantic_score))

            except Exception as semantic_error:
                print(f"Title semantic search failed: {semantic_error}")

        # Sort and finalize titles
        title_candidates.sort(key=lambda x: (-x[1], len(x[0]), x[0].lower()))
        final_titles = [title for title, _ in title_candidates[:limit]]

        # --- Enhanced Skills Search ---
        enhanced_skills = get_enhanced_skills(prefix, limit)

        # Fallback semantic search for skills if needed
        if len(enhanced_skills) < limit // 2 and hasattr(
            vectorizer, "generate_embedding"
        ):
            try:
                query_embedding = vectorizer.generate_embedding(
                    f"{prefix} programming technology skill"
                )
                skill_semantic_pipeline = [
                    {
                        "$search": {
                            "index": "vector_search_index",
                            "knnBeta": {
                                "vector": query_embedding,
                                "path": "experience_text_vector",
                                "k": limit * 2,
                            },
                        }
                    },
                    {"$project": {"skills": 1, "technical_skills": 1}},
                    {"$limit": limit * 2},
                ]

                semantic_skill_results = list(
                    collection.aggregate(skill_semantic_pipeline, maxTimeMS=3000)
                )
                existing_skills_lower = {skill.lower() for skill in enhanced_skills}

                for doc in semantic_skill_results:
                    if len(enhanced_skills) >= limit:
                        break

                    for field in ["skills", "technical_skills"]:
                        field_data = doc.get(field)
                        if not field_data:
                            continue

                        # Process field data efficiently
                        items_to_process = []
                        if isinstance(field_data, list):
                            items_to_process = [
                                item for item in field_data if isinstance(item, str)
                            ][:3]
                        elif isinstance(field_data, str):
                            items_to_process = [field_data]

                        for item in items_to_process:
                            if prefix.lower() in item.lower():
                                extracted = extract_skills(item)
                                for skill in extracted:
                                    if (
                                        skill.lower() not in existing_skills_lower
                                        and prefix.lower() in skill.lower()
                                        and len(skill) >= 2
                                    ):
                                        enhanced_skills.append(skill)
                                        existing_skills_lower.add(skill.lower())
                                        if len(enhanced_skills) >= limit:
                                            break

            except Exception as skill_semantic_error:
                print(f"Skills semantic search failed: {skill_semantic_error}")

        # Prepare final response
        response = {
            "titles": final_titles,
            "skills": enhanced_skills[:limit],
        }

        # Cache the response
        set_cache(cache_key, response)

        return response

    except Exception as e:
        # Return partial results instead of complete failure
        print(f"Autocomplete error: {str(e)}")
        return {
            "titles": [],
            "skills": [],
            "error": f"Search temporarily unavailable: {str(e)[:100]}",
        }


@router.get(
    "/clear_cache/",
    summary="Clear Autocomplete Cache",
    description="Clear the internal cache for autocomplete results. Use this if you need fresh results.",
)
async def clear_autocomplete_cache():
   
    try:
        global _autocomplete_cache, _cache_expiry
        cleared_entries = len(_autocomplete_cache)
        _autocomplete_cache.clear()
        _cache_expiry.clear()
        return {
            "message": f"Cache cleared successfully. Removed {cleared_entries} entries.",
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


@router.get(
    "/cache_stats/",
    summary="Cache Statistics",
    description="Get statistics about the autocomplete cache usage.",
)
async def get_cache_stats():
   
    try:
        current_time = time.time()
        valid_entries = sum(
            1 for expiry_time in _cache_expiry.values() if current_time < expiry_time
        )
        expired_entries = len(_cache_expiry) - valid_entries

        return {
            "total_entries": len(_autocomplete_cache),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "cache_duration_seconds": CACHE_DURATION,
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache stats failed: {str(e)}")


# @router.get(
#     "/debug/sample_data",
#     summary="Debug: Sample Data Structure",
#     description="Returns a sample document to understand the data structure",
# )
# async def get_sample_data():
#     try:
#         # Get a sample document to understand the structure
#         sample = collection.find_one({}, {"_id": 0})
#         if sample:
#             # Show keys and sample values
#             structure = {}
#             for key, value in sample.items():
#                 if isinstance(value, list) and value:
#                     structure[key] = (
#                         f"Array with {len(value)} items, sample: {value[0] if value else 'empty'}"
#                     )
#                 elif isinstance(value, dict):
#                     structure[key] = f"Object with keys: {list(value.keys())}"
#                 else:
#                     structure[key] = (
#                         f"Type: {type(value).__name__}, value: {str(value)[:100]}"
#                     )
#             return {"structure": structure, "sample_keys": list(sample.keys())}
#         else:
#             return {"message": "No documents found in collection"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")


# @router.get(
#     "/debug/skills_fields",
#     summary="Debug: Available Skills Fields",
#     description="Check what skill-related fields exist in the database",
# )
# async def check_skills_fields():
#     try:
#         # Check for various skill-related fields
#         pipeline = [
#             {"$limit": 10},
#             {
#                 "$project": {
#                     "has_skills": {"$ifNull": ["$skills", "NOT_FOUND"]},
#                     "has_technical_skills": {
#                         "$ifNull": ["$technical_skills", "NOT_FOUND"]
#                     },
#                     "has_experience_skills": {
#                         "$ifNull": ["$experience.skills", "NOT_FOUND"]
#                     },
#                     "skills_type": {"$type": "$skills"},
#                     "skills_sample": {"$slice": ["$skills", 3]},
#                 }
#             },
#         ]
#         results = list(collection.aggregate(pipeline))
#         return {"sample_documents": results, "total_checked": len(results)}
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Skills fields check failed: {str(e)}"
#         )


Keyword arguments:
argument -- description
Return: return_description
"""
