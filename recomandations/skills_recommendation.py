"""
Skills Recommendation API

This module provides position-based skills recommendations by analyzing skills data
from the MongoDB database. It recommends skills based on job positions like
"Python Developer", "Backend Developer", etc.

Author: Uphire Team
Version: 1.0.0
"""

from fastapi import APIRouter, HTTPException, Query
from mangodatabase.client import get_collection, get_skills_titles_collection
from mangodatabase.operations import SkillsTitlesOperations
from typing import List, Dict, Any, Optional
import pymongo
import re
from collections import Counter
from core.custom_logger import CustomLogger

# Initialize logger
logger = CustomLogger().get_logger("skills_recommendation")

# Initialize router
router = APIRouter(
    prefix="/recommendations",
    tags=["Skills Recommendations"],
)

# Initialize database connections
collection = get_collection()
skills_titles_collection = get_skills_titles_collection()
skills_operations = SkillsTitlesOperations(skills_titles_collection)


class SkillsRecommendationEngine:
    """Engine for generating position-based skills recommendations"""

    def __init__(self):
        self.resume_collection = collection
        self.skills_collection = skills_titles_collection

        # Define position keywords mapping for better matching
        self.position_keywords = {
            "python": ["python", "django", "flask", "fastapi", "pandas", "numpy"],
            "java": ["java", "spring", "hibernate", "maven", "gradle"],
            "javascript": ["javascript", "node", "react", "angular", "vue"],
            "frontend": [
                "html",
                "css",
                "javascript",
                "react",
                "angular",
                "vue",
                "sass",
                "less",
            ],
            "backend": [
                "python",
                "java",
                "node",
                "express",
                "django",
                "spring",
                "rest",
                "api",
            ],
            "fullstack": [
                "javascript",
                "python",
                "react",
                "node",
                "django",
                "flask",
                "mongodb",
                "postgresql",
            ],
            "data": [
                "python",
                "sql",
                "pandas",
                "numpy",
                "machine learning",
                "tensorflow",
                "pytorch",
            ],
            "devops": [
                "docker",
                "kubernetes",
                "aws",
                "jenkins",
                "terraform",
                "ansible",
            ],
            "mobile": ["android", "ios", "react native", "flutter", "swift", "kotlin"],
            "database": [
                "sql",
                "mysql",
                "postgresql",
                "mongodb",
                "redis",
                "elasticsearch",
            ],
            "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform"],
            "ai": [
                "machine learning",
                "deep learning",
                "tensorflow",
                "pytorch",
                "nlp",
                "computer vision",
            ],
            "ml": [
                "machine learning",
                "deep learning",
                "tensorflow",
                "pytorch",
                "scikit-learn",
                "pandas",
            ],
            "web": [
                "html",
                "css",
                "javascript",
                "react",
                "angular",
                "php",
                "wordpress",
            ],
            "testing": ["selenium", "pytest", "junit", "cypress", "jest", "automation"],
            "security": [
                "cybersecurity",
                "penetration testing",
                "firewall",
                "encryption",
                "authentication",
            ],
        }

    def normalize_position(self, position: str) -> List[str]:
        """
        Normalize position input to extract relevant keywords
        """
        position_lower = position.lower().strip()
        keywords = []

        # Extract direct matches
        for key, values in self.position_keywords.items():
            if key in position_lower:
                keywords.extend(values)

        # Extract technology names directly mentioned
        for tech in [
            "python",
            "java",
            "javascript",
            "react",
            "angular",
            "vue",
            "django",
            "flask",
            "spring",
        ]:
            if tech in position_lower:
                keywords.append(tech)

        # Extract role-based keywords
        if any(
            word in position_lower for word in ["developer", "engineer", "programmer"]
        ):
            if "frontend" in position_lower or "front-end" in position_lower:
                keywords.extend(self.position_keywords["frontend"])
            elif "backend" in position_lower or "back-end" in position_lower:
                keywords.extend(self.position_keywords["backend"])
            elif "fullstack" in position_lower or "full-stack" in position_lower:
                keywords.extend(self.position_keywords["fullstack"])
            elif "mobile" in position_lower:
                keywords.extend(self.position_keywords["mobile"])

        # Remove duplicates while preserving order
        return list(dict.fromkeys(keywords))

    def get_skills_from_resumes(
        self, position_keywords: List[str], limit: int = 100
    ) -> List[Dict]:
        """
        Get skills from resumes that match position keywords
        """
        try:
            # Create regex patterns for position keywords
            patterns = [
                re.compile(keyword, re.IGNORECASE) for keyword in position_keywords
            ]

            # Build MongoDB query to find resumes with matching skills or experience titles
            query = {
                "$or": [
                    {"skills": {"$in": [{"$regex": pattern} for pattern in patterns]}},
                    {
                        "may_also_known_skills": {
                            "$in": [{"$regex": pattern} for pattern in patterns]
                        }
                    },
                    {
                        "experience.title": {
                            "$in": [{"$regex": pattern} for pattern in patterns]
                        }
                    },
                    {
                        "experience.description": {
                            "$in": [{"$regex": pattern} for pattern in patterns]
                        }
                    },
                ]
            }

            # Find matching resumes
            resumes = list(self.resume_collection.find(query).limit(limit))

            return resumes

        except Exception as e:
            logger.error(f"Error querying resumes: {str(e)}")
            return []

    def extract_skills_from_resumes(self, resumes: List[Dict]) -> List[str]:
        """
        Extract and aggregate skills from resume documents
        """
        skills_counter = Counter()

        for resume in resumes:
            # Extract from skills field
            if "skills" in resume and isinstance(resume["skills"], list):
                for skill in resume["skills"]:
                    if isinstance(skill, str) and skill.strip():
                        skills_counter[skill.strip().lower()] += 1

            # Extract from may_also_known_skills field
            if "may_also_known_skills" in resume and isinstance(
                resume["may_also_known_skills"], list
            ):
                for skill in resume["may_also_known_skills"]:
                    if isinstance(skill, str) and skill.strip():
                        skills_counter[skill.strip().lower()] += 1

            # Extract from experience descriptions (technology mentions)
            if "experience" in resume and isinstance(resume["experience"], list):
                for exp in resume["experience"]:
                    if isinstance(exp, dict):
                        # Extract from job titles
                        if "title" in exp and isinstance(exp["title"], str):
                            title_words = re.findall(r"\b\w+\b", exp["title"].lower())
                            for word in title_words:
                                if len(word) > 2:  # Avoid very short words
                                    skills_counter[
                                        word
                                    ] += 0.5  # Lower weight for title words

                        # Extract from descriptions
                        if "description" in exp and isinstance(exp["description"], str):
                            # Look for technology patterns in descriptions
                            tech_patterns = [
                                r"\b(python|java|javascript|react|angular|vue|django|flask|spring|node|express)\b",
                                r"\b(html|css|sql|mongodb|postgresql|mysql|redis|aws|azure|docker|kubernetes)\b",
                                r"\b(machine learning|deep learning|tensorflow|pytorch|scikit-learn|pandas|numpy)\b",
                            ]

                            for pattern in tech_patterns:
                                matches = re.findall(
                                    pattern, exp["description"].lower()
                                )
                                for match in matches:
                                    skills_counter[
                                        match
                                    ] += 0.3  # Lower weight for description mentions

        return skills_counter

    def get_skills_from_titles_collection(
        self, position_keywords: List[str]
    ) -> List[str]:
        """
        Get relevant skills from the skills_titles collection
        """
        try:
            # Create regex patterns
            patterns = [
                re.compile(keyword, re.IGNORECASE) for keyword in position_keywords
            ]

            # Query skills collection
            query = {
                "type": "skill",
                "value": {"$in": [{"$regex": pattern} for pattern in patterns]},
            }

            skills = list(self.skills_collection.find(query))
            return [skill["value"] for skill in skills]

        except Exception as e:
            logger.error(f"Error querying skills collection: {str(e)}")
            return []

    def recommend_skills(self, position: str, limit: int = 20) -> Dict[str, Any]:
        """
        Generate skills recommendations for a given position
        """
        try:
            # Normalize position to get relevant keywords
            position_keywords = self.normalize_position(position)

            if not position_keywords:
                # If no specific keywords found, use the position as-is
                position_keywords = [position.lower()]

            logger.info(f"Searching for skills related to position: {position}")
            logger.info(f"Using keywords: {position_keywords}")

            # Get skills from resumes
            matching_resumes = self.get_skills_from_resumes(
                position_keywords, limit=200
            )
            skills_from_resumes = self.extract_skills_from_resumes(matching_resumes)

            # Get skills from skills collection
            skills_from_collection = self.get_skills_from_titles_collection(
                position_keywords
            )

            # Combine and rank skills
            all_skills = Counter()

            # Add skills from resumes with their frequency
            for skill, count in skills_from_resumes.items():
                all_skills[skill] += count

            # Add skills from collection (give them a base score)
            for skill in skills_from_collection:
                all_skills[skill.lower()] += 2.0  # Base score for collection skills

            # Filter and clean skills
            cleaned_skills = {}
            for skill, score in all_skills.items():
                # Skip very short skills or common words
                if len(skill) > 2 and skill not in [
                    "and",
                    "the",
                    "for",
                    "with",
                    "in",
                    "on",
                    "at",
                    "to",
                    "of",
                ]:
                    cleaned_skills[skill] = score

            # Sort by score and get top skills
            recommended_skills = sorted(
                cleaned_skills.items(), key=lambda x: x[1], reverse=True
            )[:limit]

            # Prepare response
            skills_list = []
            for skill, score in recommended_skills:
                skills_list.append(
                    {
                        "skill": skill.title(),
                        "relevance_score": round(score, 2),
                        "frequency": int(score),
                    }
                )

            return {
                "position": position,
                "total_skills_found": len(skills_list),
                "resumes_analyzed": len(matching_resumes),
                "recommended_skills": skills_list,
                "search_keywords": position_keywords,
            }

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error generating recommendations: {str(e)}"
            )


# Initialize recommendation engine
recommendation_engine = SkillsRecommendationEngine()


@router.get("/skills/{position}")
async def get_skills_recommendations(
    position: str,
    limit: int = Query(20, ge=1, le=100, description="Number of skills to recommend"),
) -> Dict[str, Any]:
    """
    Get skills recommendations based on job position

    Args:
        position: Job position (e.g., "Python Developer", "Backend Developer")
        limit: Maximum number of skills to return (1-100)

    Returns:
        Dictionary containing recommended skills with relevance scores

    Examples:
        - /recommendations/skills/Python Developer
        - /recommendations/skills/Backend Developer
        - /recommendations/skills/Full Stack Developer
        - /recommendations/skills/Data Scientist
    """
    try:
        if not position or not position.strip():
            raise HTTPException(status_code=400, detail="Position cannot be empty")

        logger.info(f"Generating skills recommendations for position: {position}")

        recommendations = recommendation_engine.recommend_skills(position, limit)

        logger.info(
            f"Generated {len(recommendations['recommended_skills'])} recommendations for {position}"
        )

        return {
            "success": True,
            "data": recommendations,
            "message": f"Successfully generated skills recommendations for {position}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in skills recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/skills/search/{keyword}")
async def search_skills_by_keyword(
    keyword: str,
    limit: int = Query(10, ge=1, le=50, description="Number of skills to return"),
) -> Dict[str, Any]:
    """
    Search for skills containing a specific keyword

    Args:
        keyword: Keyword to search for in skills
        limit: Maximum number of skills to return

    Returns:
        List of skills matching the keyword
    """
    try:
        if not keyword or not keyword.strip():
            raise HTTPException(status_code=400, detail="Keyword cannot be empty")

        # Search in skills collection
        pattern = re.compile(keyword, re.IGNORECASE)
        skills_query = {"type": "skill", "value": {"$regex": pattern}}

        skills = list(skills_titles_collection.find(skills_query).limit(limit))

        # Also search in resumes
        resume_skills = set()
        resume_query = {
            "$or": [
                {"skills": {"$regex": pattern}},
                {"may_also_known_skills": {"$regex": pattern}},
            ]
        }

        resumes = collection.find(resume_query).limit(50)
        for resume in resumes:
            for skill_field in ["skills", "may_also_known_skills"]:
                if skill_field in resume and isinstance(resume[skill_field], list):
                    for skill in resume[skill_field]:
                        if isinstance(skill, str) and keyword.lower() in skill.lower():
                            resume_skills.add(skill.strip())

        # Combine results
        all_skills = set()
        for skill in skills:
            all_skills.add(skill["value"])

        all_skills.update(resume_skills)

        # Convert to list and sort
        result_skills = sorted(list(all_skills))[:limit]

        return {
            "success": True,
            "keyword": keyword,
            "total_found": len(result_skills),
            "skills": result_skills,
        }

    except Exception as e:
        logger.error(f"Error searching skills by keyword: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching skills: {str(e)}")


@router.get("/positions/popular")
async def get_popular_positions(
    limit: int = Query(20, ge=1, le=100, description="Number of positions to return")
) -> Dict[str, Any]:
    """
    Get popular job positions based on resume data

    Args:
        limit: Maximum number of positions to return

    Returns:
        List of popular job positions with frequency counts
    """
    try:
        # Aggregate job titles from experience data
        pipeline = [
            {"$unwind": "$experience"},
            {
                "$group": {
                    "_id": {"$toLower": "$experience.title"},
                    "count": {"$sum": 1},
                    "original_title": {"$first": "$experience.title"},
                }
            },
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]

        results = list(collection.aggregate(pipeline))

        positions = []
        for result in results:
            if result["_id"] and result["_id"].strip():
                positions.append(
                    {"position": result["original_title"], "frequency": result["count"]}
                )

        return {
            "success": True,
            "total_positions": len(positions),
            "popular_positions": positions,
        }

    except Exception as e:
        logger.error(f"Error getting popular positions: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching popular positions: {str(e)}"
        )


@router.get("/skills/trending")
async def get_trending_skills(
    limit: int = Query(30, ge=1, le=100, description="Number of skills to return")
) -> Dict[str, Any]:
    """
    Get trending/popular skills based on frequency in resumes

    Args:
        limit: Maximum number of skills to return

    Returns:
        List of trending skills with frequency counts
    """
    try:
        # Get all skills from resumes
        skills_counter = Counter()

        # Query resumes in batches to avoid memory issues
        batch_size = 1000
        skip = 0

        while True:
            batch = list(
                collection.find({}, {"skills": 1, "may_also_known_skills": 1})
                .skip(skip)
                .limit(batch_size)
            )
            if not batch:
                break

            for resume in batch:
                # Count skills
                for skill_field in ["skills", "may_also_known_skills"]:
                    if skill_field in resume and isinstance(resume[skill_field], list):
                        for skill in resume[skill_field]:
                            if isinstance(skill, str) and skill.strip():
                                skills_counter[skill.strip().lower()] += 1

            skip += batch_size

            # Limit iterations to prevent infinite loop
            if skip > 10000:  # Process max 10k resumes
                break

        # Get top skills
        trending_skills = []
        for skill, count in skills_counter.most_common(limit):
            if len(skill) > 2:  # Filter out very short skills
                trending_skills.append({"skill": skill.title(), "frequency": count})

        return {
            "success": True,
            "total_skills": len(trending_skills),
            "trending_skills": trending_skills,
        }

    except Exception as e:
        logger.error(f"Error getting trending skills: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching trending skills: {str(e)}"
        )
