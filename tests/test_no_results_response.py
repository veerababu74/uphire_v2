#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced no-results response from manual_search API
"""

import requests
import json


def test_no_results_response():
    """Test the API with criteria that likely won't match any results"""

    # API endpoint
    url = "http://127.0.0.1:8000/manualsearch/"

    # Test data that's very specific and unlikely to match
    test_data = {
        "userid": "nonexistent_user_test_123",
        "experience_titles": [
            "Very Specific Rare Job Title That Probably Doesn't Exist",
            "Another Ultra Specific Position",
        ],
        "skills": [
            "VeryRareSkillThatProbablyDoesntExist",
            "AnotherUltraSpecificTechnicalSkill",
        ],
        "min_education": ["PhD in Very Specific Field"],
        "min_experience": "15 years",
        "max_experience": "20 years",
        "locations": ["VerySpecificCityThatProbablyDoesntExist"],
        "min_salary": 10000000,  # Very high salary
        "max_salary": 15000000,
        "limit": 10,
    }

    headers = {"accept": "application/json", "Content-Type": "application/json"}

    try:
        print("Testing manual search API with criteria that won't match...")
        print(f"Request URL: {url}")
        print(f"Request Data: {json.dumps(test_data, indent=2)}")
        print("\n" + "=" * 50)

        response = requests.post(url, headers=headers, json=test_data)

        print(f"Response Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print("\nResponse Body:")

        if response.status_code == 200:
            response_data = response.json()
            print(json.dumps(response_data, indent=2))

            # Check if it's the enhanced no-results response
            if (
                len(response_data) == 1
                and isinstance(response_data[0], dict)
                and "message" in response_data[0]
            ):
                print("\n✅ SUCCESS: Enhanced no-results response working correctly!")
                print(f"Message: {response_data[0]['message']}")
                print(
                    f"Total candidates searched: {response_data[0]['search_summary']['total_candidates_searched']}"
                )
                print(
                    f"Suggestions provided: {len(response_data[0]['search_summary']['suggestions'])}"
                )
            else:
                print("\n❌ Unexpected response format")
        else:
            print(f"❌ API Error: {response.text}")

    except requests.exceptions.ConnectionError:
        print(
            "❌ Connection Error: Make sure the FastAPI server is running on http://127.0.0.1:8000"
        )
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    test_no_results_response()
