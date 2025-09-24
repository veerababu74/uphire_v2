#!/usr/bin/env python3
"""
Test the manual search function directly without FastAPI server
"""

import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mangodatabase.client import get_collection, get_users_collection
from mangodatabase.user_operations import UserOperations
from apis.manual_search import ManualSearchRequest, get_effective_user_id_for_search
import json
import re
from core.helpers import format_resume


def test_manual_search_logic():
    """Test the manual search logic directly"""

    try:
        # Initialize collections
        resumes_collection = get_collection()
        users_collection = get_users_collection()
        user_ops = UserOperations(users_collection)

        # Create a test search request with the original failing criteria
        search_params = ManualSearchRequest(
            userid="66c8771a20bd68c725758679",
            experience_titles=["frontend developer"],
            skills=["cobol"],
            min_education=["10th Pass"],
            min_experience="6 Months",
            max_experience="1 Year",
            locations=["adwani"],
            min_salary=1.0,
            max_salary=2.0,
            relevant_score=40.0,
        )

        print(f"🔍 Testing manual search logic...")
        print(f"📋 Search parameters: {search_params}")

        # Check user access
        effective_user_id = get_effective_user_id_for_search(search_params.userid)
        print(f"👤 Effective user ID for search: {effective_user_id}")

        # Build base query
        base_query = {}
        if effective_user_id is not None:
            base_query["user_id"] = effective_user_id

        print(f"🔎 Base query: {base_query}")

        # Check if we have search criteria
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

        print(f"📊 Has search criteria: {has_criteria}")

        if not has_criteria:
            # Return all resumes for this user
            results = list(resumes_collection.find(base_query))
            print(f"📄 No criteria search - found {len(results)} results")
        else:
            # Build OR conditions
            or_conditions = []

            # Add skills conditions
            if search_params.skills:
                for skill in search_params.skills:
                    skill_pattern = re.compile(f".*{re.escape(skill)}.*", re.IGNORECASE)
                    or_conditions.extend(
                        [
                            {"skills": {"$regex": skill_pattern}},
                            {"may_also_known_skills": {"$regex": skill_pattern}},
                        ]
                    )

            print(f"🔍 OR conditions: {len(or_conditions)} conditions")

            # Build final query
            if or_conditions:
                final_query = {"$and": [base_query, {"$or": or_conditions}]}
            else:
                final_query = base_query

            print(f"🔎 Final query: {final_query}")

            # Execute query
            results = list(resumes_collection.find(final_query))
            print(f"📄 Found {len(results)} raw results")

            # Test the scoring logic on first result if available
            if results:
                resume = results[0]
                print(f"🎯 Testing scoring on first result:")
                print(f"   - User ID: {resume.get('user_id', 'N/A')}")
                print(
                    f"   - Name: {resume.get('contact_details', {}).get('name', 'N/A')}"
                )
                print(f"   - Skills: {resume.get('skills', [])}")
                print(f"   - May also know: {resume.get('may_also_known_skills', [])}")

                # Test skill matching
                if search_params.skills:
                    resume_skills = [
                        skill.lower() for skill in resume.get("skills", [])
                    ]
                    may_also_known_skills = [
                        skill.lower()
                        for skill in resume.get("may_also_known_skills", [])
                    ]
                    all_resume_skills = resume_skills + may_also_known_skills

                    skills_matches = 0
                    matched_skills = []

                    for skill in search_params.skills:
                        if skill.lower() in all_resume_skills:
                            skills_matches += 1
                            matched_skills.append(skill)

                    print(f"   - Skills matches: {skills_matches}")
                    print(f"   - Matched skills: {matched_skills}")

        print(f"✅ Manual search logic test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error in manual search logic: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_manual_search_logic()
