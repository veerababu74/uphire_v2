"""
Test Script for Database-Focused Skills Recommendation API

This script demonstrates the usage of the new database-focused skills recommendation API.
This version only returns skills that exist in the skills_titles collection.

Usage:
    python test_skills_recommendation_db.py
"""

import requests
import json
from typing import Dict, Any

# Base URL for the API
BASE_URL = "http://localhost:8000"


def test_api_endpoint(endpoint: str, description: str) -> Dict[Any, Any]:
    """Test an API endpoint and display results"""
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"Endpoint: {endpoint}")
    print("=" * 70)

    try:
        response = requests.get(f"{BASE_URL}{endpoint}")

        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS!")

            # Pretty print with highlighting key information
            if "data" in data and "recommended_skills" in data["data"]:
                skills_data = data["data"]
                print(f"🎯 Position: {skills_data.get('position', 'N/A')}")
                print(f"📊 Skills found: {skills_data.get('total_skills_found', 0)}")
                print(
                    f"🗄️  Database skills: {skills_data.get('database_skills_count', 0)}"
                )
                print(f"🔍 Keywords used: {skills_data.get('search_keywords', [])}")
                print(f"📝 Source: {skills_data.get('source', 'unknown')}")
                print("\n📋 Recommended Skills:")

                for i, skill in enumerate(
                    skills_data.get("recommended_skills", [])[:10], 1
                ):
                    print(
                        f"  {i:2d}. {skill.get('skill', 'N/A')} "
                        f"(Score: {skill.get('relevance_score', 0)}, "
                        f"Resume Frequency: {skill.get('frequency_in_resumes', 0)})"
                    )
            else:
                print(json.dumps(data, indent=2))

            return data
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            return {}

    except requests.exceptions.ConnectionError:
        print(
            "❌ CONNECTION ERROR: Make sure the FastAPI server is running on port 8000"
        )
        return {}
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {str(e)}")
        return {}


def main():
    """Run all API tests"""
    print("🚀 Testing Database-Focused Skills Recommendation API")
    print("📋 This version only returns skills from the skills_titles collection")
    print("🌐 Make sure the FastAPI server is running on http://localhost:8000")

    # Test 1: Skills recommendations for Python Developer
    test_api_endpoint(
        "/recommendations/skills/Python Developer?limit=10",
        "Skills Recommendations for Python Developer (Database Only)",
    )

    # Test 2: Skills recommendations for Backend Developer
    test_api_endpoint(
        "/recommendations/skills/Backend Developer?limit=8",
        "Skills Recommendations for Backend Developer (Database Only)",
    )

    # Test 3: Skills recommendations for Full Stack Developer
    test_api_endpoint(
        "/recommendations/skills/Full Stack Developer?limit=12",
        "Skills Recommendations for Full Stack Developer (Database Only)",
    )

    # Test 4: Skills recommendations for Data Scientist
    test_api_endpoint(
        "/recommendations/skills/Data Scientist?limit=15",
        "Skills Recommendations for Data Scientist (Database Only)",
    )

    # Test 5: Skills recommendations for Frontend Developer
    test_api_endpoint(
        "/recommendations/skills/Frontend Developer?limit=10",
        "Skills Recommendations for Frontend Developer (Database Only)",
    )

    # Test 6: Search skills by keyword
    test_api_endpoint(
        "/recommendations/skills/search/python?limit=10",
        "Search Skills by Keyword: python (Database Only)",
    )

    # Test 7: Search skills by keyword
    test_api_endpoint(
        "/recommendations/skills/search/react?limit=8",
        "Search Skills by Keyword: react (Database Only)",
    )

    # Test 8: Get all database skills
    test_api_endpoint(
        "/recommendations/skills/all?limit=20", "Get All Database Skills (Sample)"
    )

    # Test 9: Get popular positions
    test_api_endpoint(
        "/recommendations/positions/popular?limit=15",
        "Popular Job Positions (Database Only)",
    )

    # Test 10: Get trending skills
    test_api_endpoint(
        "/recommendations/skills/trending?limit=20", "Trending Skills (Database Only)"
    )

    print(f"\n{'='*70}")
    print("🎉 All tests completed!")
    print("📊 Results Summary:")
    print("   - All skills come from the skills_titles collection")
    print("   - Relevance scores based on keyword matching")
    print("   - Frequency counts from actual resume data")
    print("   - No text analysis or extraction from resumes")
    print("=" * 70)


if __name__ == "__main__":
    main()
