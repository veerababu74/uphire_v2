"""
Test Script for Skills Recommendation API

This script demonstrates the usage of the new skills recommendation API.
Run this after starting the FastAPI server.

Usage:
    python test_skills_recommendation.py
"""

import requests
import json
from typing import Dict, Any

# Base URL for the API
BASE_URL = "http://localhost:8000"


def test_api_endpoint(endpoint: str, description: str) -> Dict[Any, Any]:
    """Test an API endpoint and display results"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Endpoint: {endpoint}")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}{endpoint}")

        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS!")
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
    print("🚀 Testing Skills Recommendation API")
    print("Make sure the FastAPI server is running on http://localhost:8000")

    # Test 1: Skills recommendations for Python Developer
    test_api_endpoint(
        "/recommendations/skills/Python Developer?limit=10",
        "Skills Recommendations for Python Developer",
    )

    # Test 2: Skills recommendations for Backend Developer
    test_api_endpoint(
        "/recommendations/skills/Backend Developer?limit=8",
        "Skills Recommendations for Backend Developer",
    )

    # Test 3: Skills recommendations for Full Stack Developer
    test_api_endpoint(
        "/recommendations/skills/Full Stack Developer?limit=12",
        "Skills Recommendations for Full Stack Developer",
    )

    # Test 4: Skills recommendations for Data Scientist
    test_api_endpoint(
        "/recommendations/skills/Data Scientist?limit=15",
        "Skills Recommendations for Data Scientist",
    )

    # Test 5: Search skills by keyword
    test_api_endpoint(
        "/recommendations/skills/search/python?limit=10",
        "Search Skills by Keyword: python",
    )

    # Test 6: Search skills by keyword
    test_api_endpoint(
        "/recommendations/skills/search/react?limit=8",
        "Search Skills by Keyword: react",
    )

    # Test 7: Get popular positions
    test_api_endpoint(
        "/recommendations/positions/popular?limit=15", "Popular Job Positions"
    )

    # Test 8: Get trending skills
    test_api_endpoint("/recommendations/skills/trending?limit=20", "Trending Skills")

    print(f"\n{'='*60}")
    print("🎉 All tests completed!")
    print("Check the results above to see the API responses.")
    print("=" * 60)


if __name__ == "__main__":
    main()
