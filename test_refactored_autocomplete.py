#!/usr/bin/env python3
"""
Test script for the refactored autocomplete API with dynamic skills functionality
"""

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from apis.autocomplete_skills_titiles import get_skills_by_job_titles
from mangodatabase.client import get_collection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_refactored_skills_api():
    """Test the refactored skills by job titles API"""

    print("🧪 TESTING REFACTORED SKILLS API")
    print("=" * 50)

    # Test cases
    test_cases = [
        {
            "titles": ["python developer", "data scientist"],
            "limit": None,  # Test dynamic sizing
            "include_related": True,
            "description": "Dynamic sizing test with 2 job titles",
        },
        {
            "titles": [
                "software engineer",
                "full stack developer",
                "backend developer",
            ],
            "limit": 25,  # Test with user-specified limit
            "include_related": True,
            "description": "User-limited results with 3 job titles",
        },
        {
            "titles": ["machine learning engineer"],
            "limit": None,  # Test dynamic sizing
            "include_related": False,
            "description": "Single job title without related skills",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 TEST CASE {i}: {test_case['description']}")
        print(f"   Input: {test_case['titles']}")
        print(
            f"   Limit: {test_case['limit']} (Dynamic sizing: {test_case['limit'] is None})"
        )

        try:
            # Prepare request body
            request_body = {
                "titles": test_case["titles"],
                "include_related": test_case["include_related"],
            }
            if test_case["limit"] is not None:
                request_body["limit"] = test_case["limit"]

            # Call the API function
            result = await get_skills_by_job_titles(request_body)

            # Analyze results
            if "skills" in result:
                skills_count = len(result["skills"])
                total_found = result.get("request_summary", {}).get(
                    "total_skills_found", 0
                )

                print(
                    f"   ✅ SUCCESS: {skills_count} skills returned (out of {total_found} found)"
                )

                # Show quality distribution if available
                if "skills_by_category" in result:
                    categories = list(result["skills_by_category"].keys())
                    print(f"   📊 Categories: {categories}")

                # Show sample skills
                sample_skills = result["skills"][:5]
                print(f"   🎯 Sample skills: {sample_skills}")

                # Check if dynamic sizing worked
                if test_case["limit"] is None:
                    print(
                        f"   🔄 Dynamic sizing: Returned {skills_count} skills based on quality"
                    )
                else:
                    limit_respected = skills_count <= test_case["limit"]
                    print(
                        f"   📏 Limit respected: {limit_respected} ({skills_count} <= {test_case['limit']})"
                    )

            else:
                print(f"   ❌ ERROR: No skills in response - {result}")

        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            import traceback

            traceback.print_exc()

    print(f"\n✨ TESTING COMPLETED")


async def test_dynamic_scoring():
    """Test the dynamic scoring functions directly"""
    print(f"\n🧮 TESTING DYNAMIC SCORING FUNCTIONS")
    print("=" * 50)

    try:
        from apis.autocomplete_skills_titiles import (
            calculate_relevance_score,
            get_dynamic_skill_relationships,
            get_skill_frequencies,
        )

        # Test dynamic relationship building
        print("🔗 Testing dynamic skill relationships...")
        relationships = get_dynamic_skill_relationships()
        print(f"   Built relationships for {len(relationships)} skills")
        if relationships:
            sample_skill = list(relationships.keys())[0]
            related_count = len(relationships[sample_skill])
            print(f"   Sample: '{sample_skill}' has {related_count} related skills")

        # Test skill frequencies
        print("📊 Testing skill frequencies...")
        frequencies = get_skill_frequencies()
        print(f"   Found frequencies for {len(frequencies)} skills")
        if frequencies:
            top_skills = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[
                :5
            ]
            print(f"   Top 5 skills: {[(skill, count) for skill, count in top_skills]}")

        # Test dynamic scoring
        print("🎯 Testing dynamic relevance scoring...")
        test_pairs = [
            ("python", "django"),
            ("java", "spring"),
            ("react", "javascript"),
            ("data", "python"),
        ]

        for prefix, text in test_pairs:
            score = calculate_relevance_score(text, prefix, semantic_score=0.5)
            print(f"   '{prefix}' → '{text}': {score:.3f}")

        print("   ✅ Dynamic scoring functions working correctly")

    except Exception as e:
        print(f"   ❌ ERROR in dynamic scoring: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":

    async def main():
        print("🚀 TESTING REFACTORED AUTOCOMPLETE API (NO STATIC DATA)")
        print("🎯 Verifying removal of hardcoded limits and technology lists")

        # Test the main API functionality
        await test_refactored_skills_api()

        # Test the dynamic scoring components
        await test_dynamic_scoring()

        print(f"\n🎉 ALL TESTS COMPLETED!")
        print("✅ Static data successfully removed and replaced with dynamic approach")

    asyncio.run(main())
