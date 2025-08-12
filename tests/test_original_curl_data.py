#!/usr/bin/env python3
"""
Test script using the original curl command data to test manual_search API
"""

import requests
import json


def test_original_curl_data():
    """Test the API with the original curl command data"""

    # API endpoint
    url = "http://127.0.0.1:8000/manualsearch/"

    # Original curl data
    test_data = {
        "userid": "user123",
        "experience_titles": ["Software Engineer", "Python Developer"],
        "skills": ["Python", "React", "AWS"],
        "min_education": ["BTech", "BSc"],
        "min_experience": "2 years 6 months",
        "max_experience": "5 years",
        "locations": ["Mumbai", "Pune", "Bangalore"],
        "min_salary": 500000,
        "max_salary": 1500000,
        "limit": 10,
    }

    headers = {"accept": "application/json", "Content-Type": "application/json"}

    try:
        print("Testing manual search API with original curl data...")
        print(f"Request URL: {url}")
        print(f"Request Data: {json.dumps(test_data, indent=2)}")
        print("\n" + "=" * 50)

        response = requests.post(url, headers=headers, json=test_data)

        print(f"Response Status Code: {response.status_code}")
        print("\nResponse Body:")

        if response.status_code == 200:
            response_data = response.json()
            print(json.dumps(response_data, indent=2))

            # Check response type
            if len(response_data) == 0:
                print("\n📝 OLD BEHAVIOR: Empty array returned")
            elif (
                len(response_data) == 1
                and isinstance(response_data[0], dict)
                and "message" in response_data[0]
            ):
                print("\n✅ NEW BEHAVIOR: Enhanced no-results response!")
                print(f"Message: {response_data[0]['message']}")
                print(
                    f"Total candidates searched: {response_data[0]['search_summary']['total_candidates_searched']}"
                )
                print("Suggestions:")
                for i, suggestion in enumerate(
                    response_data[0]["search_summary"]["suggestions"], 1
                ):
                    print(f"  {i}. {suggestion}")
            else:
                print(f"\n✅ RESULTS FOUND: {len(response_data)} candidates returned")
                for i, candidate in enumerate(response_data[:3], 1):  # Show first 3
                    if "contact_details" in candidate:
                        name = candidate["contact_details"].get("name", "N/A")
                        score = candidate.get("match_score", "N/A")
                        print(f"  {i}. {name} (Score: {score})")
                if len(response_data) > 3:
                    print(f"  ... and {len(response_data) - 3} more candidates")
        else:
            print(f"❌ API Error: {response.text}")

    except requests.exceptions.ConnectionError:
        print(
            "❌ Connection Error: Make sure the FastAPI server is running on http://127.0.0.1:8000"
        )
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    test_original_curl_data()
