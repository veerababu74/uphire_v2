"""
Recommendations Module

This module provides various recommendation features including:
- Skills recommendations based on job positions (DATABASE SKILLS ONLY)
- Position-based skill suggestions from skills_titles collection
- Trending skills analysis from database

Author: Uphire Team
Version: 2.0.0 (Database-focused)
"""

from .skills_recommendation_db import router as skills_recommendation_router

__all__ = ["skills_recommendation_router"]
