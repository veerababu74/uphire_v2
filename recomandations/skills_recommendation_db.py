"""
Database-Focused Skills Recommendation API

This module provides position-based skills recommendations using ONLY skills that exist
in the MongoDB skills_titles collection. It does not extract or analyze skills from resume text.

Author: Uphire Team
Version: 2.0.0 (Database-focused)
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
logger = CustomLogger().get_logger("skills_recommendation_db")

# Initialize router
router = APIRouter(
    prefix="/recommendations",
    tags=["Skills Recommendations (Database Only)"],
)

# Initialize database connections
collection = get_collection()
skills_titles_collection = get_skills_titles_collection()
skills_operations = SkillsTitlesOperations(skills_titles_collection)


class DatabaseSkillsRecommendationEngine:
    """Engine for generating position-based skills recommendations from database skills ONLY"""

    def __init__(self):
        self.resume_collection = collection
        self.skills_collection = skills_titles_collection

        # Define position keywords mapping for better matching
        self.position_keywords = {
            "python": [
                "python",
                "django",
                "flask",
                "fastapi",
                "pandas",
                "numpy",
                "pytest",
                "sqlalchemy",
            ],
            "java": [
                "java",
                "spring",
                "hibernate",
                "maven",
                "gradle",
                "junit",
                "tomcat",
            ],
            "javascript": [
                "javascript",
                "node",
                "react",
                "angular",
                "vue",
                "express",
                "typescript",
            ],
            "frontend": [
                "html",
                "css",
                "javascript",
                "react",
                "angular",
                "vue",
                "sass",
                "less",
                "bootstrap",
                "webpack",
                "jquery",
                "typescript",
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
                "microservices",
                "php",
                "laravel",
                "ruby",
                "rails",
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
                "express",
                "angular",
                "vue",
            ],
            "data": [
                "python",
                "sql",
                "pandas",
                "numpy",
                "machine learning",
                "tensorflow",
                "pytorch",
                "tableau",
                "powerbi",
                "spark",
                "hadoop",
            ],
            "devops": [
                "docker",
                "kubernetes",
                "aws",
                "jenkins",
                "terraform",
                "ansible",
                "ci/cd",
                "gitlab",
                "azure",
                "monitoring",
            ],
            "mobile": [
                "android",
                "ios",
                "react native",
                "flutter",
                "swift",
                "kotlin",
                "xamarin",
                "ionic",
            ],
            "database": [
                "sql",
                "mysql",
                "postgresql",
                "mongodb",
                "redis",
                "elasticsearch",
                "nosql",
                "oracle",
                "cassandra",
            ],
            "cloud": [
                "aws",
                "azure",
                "gcp",
                "docker",
                "kubernetes",
                "terraform",
                "cloudformation",
                "serverless",
            ],
            "ai": [
                "machine learning",
                "deep learning",
                "tensorflow",
                "pytorch",
                "nlp",
                "computer vision",
                "neural networks",
            ],
            "ml": [
                "machine learning",
                "deep learning",
                "tensorflow",
                "pytorch",
                "scikit-learn",
                "pandas",
                "numpy",
                "data science",
            ],
            "web": [
                "html",
                "css",
                "javascript",
                "react",
                "angular",
                "php",
                "wordpress",
                "bootstrap",
                "jquery",
            ],
            "testing": [
                "selenium",
                "pytest",
                "junit",
                "cypress",
                "jest",
                "automation",
                "testing",
                "qa",
            ],
            "security": [
                "cybersecurity",
                "penetration testing",
                "firewall",
                "encryption",
                "authentication",
                "security",
            ],
        }

    def normalize_position(self, position: str) -> List[str]:
        """
        Normalize position input to extract relevant keywords
        """
        position_lower = position.lower().strip()
        keywords = []

        # Extract direct matches from position keywords
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

    def get_all_database_skills(self) -> List[str]:
        """
        Get all skills from the skills_titles collection
        """
        try:
            skills_cursor = self.skills_collection.find({"type": "skill"})
            skills = [skill["value"] for skill in skills_cursor]
            logger.info(f"Retrieved {len(skills)} skills from database")
            return skills
        except Exception as e:
            logger.error(f"Error retrieving skills from database: {str(e)}")
            return []

    def match_skills_to_position(
        self, position_keywords: List[str], all_skills: List[str]
    ) -> Dict[str, float]:
        """
        Match database skills to position keywords and calculate relevance scores
        """
        matched_skills = {}

        # Convert position keywords to lowercase for matching
        position_keywords_lower = [keyword.lower() for keyword in position_keywords]

        for skill in all_skills:
            skill_lower = skill.lower().strip()
            relevance_score = 0.0

            # Exact match gets highest score
            if skill_lower in position_keywords_lower:
                relevance_score = 10.0
            else:
                # Check for partial matches
                for keyword in position_keywords_lower:
                    keyword = keyword.strip()
                    # If keyword is contained in skill or vice versa
                    if keyword in skill_lower:
                        relevance_score += 8.0
                    elif skill_lower in keyword:
                        relevance_score += 6.0
                    # Check if skill contains any part of the keyword
                    elif any(
                        part in skill_lower for part in keyword.split() if len(part) > 2
                    ):
                        relevance_score += 4.0
                    # Check if keyword contains any part of the skill
                    elif any(
                        part in keyword for part in skill_lower.split() if len(part) > 2
                    ):
                        relevance_score += 3.0

            # Only include skills with some relevance
            if relevance_score > 0:
                matched_skills[skill] = relevance_score

        return matched_skills

    def get_skill_popularity_from_resumes(self, skills: List[str]) -> Dict[str, int]:
        """
        Get popularity of skills in resumes to boost popular skills
        """
        skill_popularity = {}

        try:
            # Query resumes that contain any of these skills
            query = {
                "$or": [
                    {"skills": {"$in": skills}},
                    {"may_also_known_skills": {"$in": skills}},
                ]
            }

            resumes = self.resume_collection.find(
                query, {"skills": 1, "may_also_known_skills": 1}
            ).limit(
                1000
            )  # Limit for performance

            # Count exact skill matches
            for resume in resumes:
                for field in ["skills", "may_also_known_skills"]:
                    if field in resume and isinstance(resume[field], list):
                        for resume_skill in resume[field]:
                            if isinstance(resume_skill, str):
                                resume_skill_clean = resume_skill.lower().strip()
                                # Find exact matches
                                for target_skill in skills:
                                    if (
                                        target_skill.lower().strip()
                                        == resume_skill_clean
                                    ):
                                        skill_popularity[target_skill] = (
                                            skill_popularity.get(target_skill, 0) + 1
                                        )

            return skill_popularity

        except Exception as e:
            logger.error(f"Error getting skill popularity: {str(e)}")
            return {}

    def recommend_skills(self, position: str, limit: int = 20) -> Dict[str, Any]:
        """
        Generate skills recommendations for a given position using ONLY database skills
        """
        try:
            # Normalize position to get relevant keywords
            position_keywords = self.normalize_position(position)

            if not position_keywords:
                # If no specific keywords found, use the position as-is
                position_keywords = [position.lower()]

            logger.info(f"Searching for skills related to position: {position}")
            logger.info(f"Using keywords: {position_keywords}")

            # Get all skills from database
            all_skills = self.get_all_database_skills()

            if not all_skills:
                return {
                    "position": position,
                    "total_skills_found": 0,
                    "database_skills_count": 0,
                    "recommended_skills": [],
                    "search_keywords": position_keywords,
                    "message": "No skills found in database skills_titles collection",
                    "source": "database_only",
                }

            # Match skills to position keywords
            matched_skills = self.match_skills_to_position(
                position_keywords, all_skills
            )

            if not matched_skills:
                return {
                    "position": position,
                    "total_skills_found": 0,
                    "database_skills_count": len(all_skills),
                    "recommended_skills": [],
                    "search_keywords": position_keywords,
                    "message": f"No matching skills found for position '{position}' in database skills_titles collection",
                    "source": "database_only",
                }

            # Get skill popularity from resumes to boost popular skills
            skill_popularity = self.get_skill_popularity_from_resumes(
                list(matched_skills.keys())
            )

            # Calculate final scores combining relevance and popularity
            final_scores = {}
            for skill, relevance_score in matched_skills.items():
                popularity_boost = (
                    skill_popularity.get(skill, 0) * 0.5
                )  # Weight popularity lower than relevance
                final_score = relevance_score + popularity_boost
                final_scores[skill] = final_score

            # Sort by score and get top skills
            recommended_skills = sorted(
                final_scores.items(), key=lambda x: x[1], reverse=True
            )[:limit]

            # Prepare response
            skills_list = []
            for skill, score in recommended_skills:
                skills_list.append(
                    {
                        "skill": skill,  # Keep original casing from database
                        "relevance_score": round(score, 2),
                        "frequency_in_resumes": skill_popularity.get(skill, 0),
                        "source": "skills_titles_collection",
                    }
                )

            return {
                "position": position,
                "total_skills_found": len(skills_list),
                "database_skills_count": len(all_skills),
                "recommended_skills": skills_list,
                "search_keywords": position_keywords,
                "message": f"Found {len(skills_list)} relevant skills from database skills_titles collection",
                "source": "database_only",
            }

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error generating recommendations: {str(e)}"
            )


# Initialize recommendation engine
recommendation_engine = DatabaseSkillsRecommendationEngine()


@router.get("/skills/{position}")
async def get_skills_recommendations(
    position: str,
    limit: int = Query(20, ge=1, le=100, description="Number of skills to recommend"),
) -> Dict[str, Any]:
    """
    Get skills recommendations based on job position (DATABASE SKILLS ONLY)

    This endpoint only returns skills that exist in the skills_titles collection.
    It does not analyze or extract skills from resume text.

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

        logger.info(
            f"Generating database skills recommendations for position: {position}"
        )

        recommendations = recommendation_engine.recommend_skills(position, limit)

        logger.info(
            f"Generated {len(recommendations['recommended_skills'])} recommendations for {position} from database"
        )

        return {
            "success": True,
            "data": recommendations,
            "message": f"Successfully generated skills recommendations for {position} from database",
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
    Search for skills containing a specific keyword (DATABASE SKILLS ONLY)

    Args:
        keyword: Keyword to search for in skills
        limit: Maximum number of skills to return

    Returns:
        List of skills matching the keyword from skills_titles collection
    """
    try:
        if not keyword or not keyword.strip():
            raise HTTPException(status_code=400, detail="Keyword cannot be empty")

        # Search in skills collection only
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        skills_query = {"type": "skill", "value": {"$regex": pattern}}

        skills = list(skills_titles_collection.find(skills_query).limit(limit))

        result_skills = [skill["value"] for skill in skills]

        return {
            "success": True,
            "keyword": keyword,
            "total_found": len(result_skills),
            "skills": result_skills,
            "source": "skills_titles_collection",
        }

    except Exception as e:
        logger.error(f"Error searching skills by keyword: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching skills: {str(e)}")


@router.get("/skills/all")
async def get_all_database_skills(
    limit: int = Query(100, ge=1, le=1000, description="Number of skills to return")
) -> Dict[str, Any]:
    """
    Get all skills from the database skills_titles collection

    Args:
        limit: Maximum number of skills to return

    Returns:
        List of all skills in the database
    """
    try:
        skills_cursor = skills_titles_collection.find({"type": "skill"}).limit(limit)
        skills = [skill["value"] for skill in skills_cursor]

        return {
            "success": True,
            "total_skills": len(skills),
            "skills": skills,
            "source": "skills_titles_collection",
        }

    except Exception as e:
        logger.error(f"Error getting all skills: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching skills: {str(e)}")


@router.get("/positions/popular")
async def get_popular_positions(
    limit: int = Query(20, ge=1, le=100, description="Number of positions to return")
) -> Dict[str, Any]:
    """
    Get popular job positions from the database titles

    Args:
        limit: Maximum number of positions to return

    Returns:
        List of popular job positions from skills_titles collection
    """
    try:
        # Get titles from skills_titles collection
        titles_cursor = skills_titles_collection.find({"type": "title"}).limit(limit)
        titles = [title["value"] for title in titles_cursor]

        # Format as positions with dummy frequency (since we can't calculate real frequency)
        positions = []
        for i, title in enumerate(titles):
            positions.append(
                {
                    "position": title,
                    "frequency": len(titles) - i,  # Simple ranking based on order
                }
            )

        return {
            "success": True,
            "total_positions": len(positions),
            "popular_positions": positions,
            "source": "skills_titles_collection",
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
    Get trending skills from the database (DATABASE SKILLS ONLY)

    This endpoint returns skills from the skills_titles collection,
    ordered by their frequency in actual resumes.

    Args:
        limit: Maximum number of skills to return

    Returns:
        List of trending skills with frequency counts from resumes
    """
    try:
        # Get all skills from database
        all_skills = recommendation_engine.get_all_database_skills()

        if not all_skills:
            return {
                "success": True,
                "total_skills": 0,
                "trending_skills": [],
                "source": "skills_titles_collection",
            }

        # Get popularity of these skills in resumes
        skill_popularity = recommendation_engine.get_skill_popularity_from_resumes(
            all_skills
        )

        # Sort by popularity
        trending_skills = []
        sorted_skills = sorted(
            skill_popularity.items(), key=lambda x: x[1], reverse=True
        )

        for skill, frequency in sorted_skills[:limit]:
            trending_skills.append(
                {
                    "skill": skill,
                    "frequency": frequency,
                    "source": "skills_titles_collection",
                }
            )

        # If we don't have enough skills with frequency, add remaining skills with 0 frequency
        if len(trending_skills) < limit:
            remaining_skills = [
                skill for skill in all_skills if skill not in skill_popularity
            ]
            for skill in remaining_skills[: limit - len(trending_skills)]:
                trending_skills.append(
                    {
                        "skill": skill,
                        "frequency": 0,
                        "source": "skills_titles_collection",
                    }
                )

        return {
            "success": True,
            "total_skills": len(trending_skills),
            "trending_skills": trending_skills,
            "source": "skills_titles_collection",
        }

    except Exception as e:
        logger.error(f"Error getting trending skills: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching trending skills: {str(e)}"
        )
