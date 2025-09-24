#!/usr/bin/env python3
"""
Quick test of the manual search API
"""

import requests
import json


def test_api():
    url = "http://127.0.0.1:8000/manualsearch/"

    # Test 1: Original failing request
    print("=== Test 1: Original failing criteria ===")
    request_body = {
        "userid": "66c8771a20bd68c725758679",
        "experience_titles": ["frontend developer"],
        "skills": ["cobol"],
        "min_education": ["10th Pass"],
        "min_experience": "6 Months",
        "max_experience": "1 Year",
        "locations": ["adwani"],
        "min_salary": 1.0,
        "max_salary": 2.0,
        "relevant_score": 40.0,
    }

    try:
        response = requests.post(url, json=request_body, timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")

    print("\n" + "=" * 50 + "\n")

    # Test 2: Working request with python skill
    print("=== Test 2: Working criteria (python skill) ===")
    request_body = {
        "userid": "66c8771a20bd68c725758679",
        "skills": ["python"],
        "relevant_score": 0.0,
    }

    try:
        response = requests.post(url, json=request_body, timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Found {len(result)} results")
            if result and len(result) > 0:
                # Print first result summary
                first = result[0]
                print(
                    f"First result: {first.get('contact_details', {}).get('name', 'N/A')}"
                )
                print(f"Match score: {first.get('match_score', 'N/A')}")
                print(f"Skills: {first.get('skills', [])}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    test_api()
