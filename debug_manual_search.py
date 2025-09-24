#!/usr/bin/env python3
"""
Test script to debug the manual search issue
"""

import requests
import json
from typing import Dict, Any


def test_manual_search_with_proper_userid():
    """Test the manual search API with a proper user_id from the database"""

    # API endpoint - correct path
    url = "http://localhost:8000/manualsearch/"

    # Request body with the user_id that exists in the database
    request_body = {
        "userid": "66c8771a20bd68c725758679",  # This user_id exists in the database
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
        print("🔍 Testing Manual Search API...")
        print(f"📋 Request: {json.dumps(request_body, indent=2)}")

        # Make the API request
        response = requests.post(url, json=request_body, timeout=30)

        print(f"\n📊 Response Status: {response.status_code}")

        if response.status_code == 200:
            results = response.json()
            print(f"✅ API Request Successful!")
            print(f"📄 Response: {json.dumps(results, indent=2)}")

        else:
            print(f"❌ API Request Failed!")
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")


def test_simple_search():
    """Test with very simple criteria to see if we get any results"""

    url = "http://localhost:8000/manualsearch/"

    # Very simple request - just userid and one skill that exists in the database
    request_body = {
        "userid": "66c8771a20bd68c725758679",
        "skills": ["python"],  # This skill exists in the sample resume
        "relevant_score": 0.0,  # Set to 0 to get all matches
    }

    try:
        print("\n🔍 Testing Simple Search...")
        print(f"📋 Request: {json.dumps(request_body, indent=2)}")

        response = requests.post(url, json=request_body, timeout=30)

        print(f"\n📊 Response Status: {response.status_code}")

        if response.status_code == 200:
            results = response.json()
            print(f"✅ Simple Search Result:")
            print(f"📄 Found {len(results)} results")
            if results and len(results) > 0:
                print(f"First result keys: {list(results[0].keys())}")
            else:
                print("No results found")
        else:
            print(f"❌ Simple Search Failed!")
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")


def test_no_criteria_search():
    """Test with just userid to see if we get all resumes for that user"""

    url = "http://localhost:8000/manualsearch/"

    # Just userid - should return all resumes for this user
    request_body = {"userid": "66c8771a20bd68c725758679"}

    try:
        print("\n🔍 Testing No-Criteria Search (should return all user resumes)...")
        print(f"📋 Request: {json.dumps(request_body, indent=2)}")

        response = requests.post(url, json=request_body, timeout=30)

        print(f"\n📊 Response Status: {response.status_code}")

        if response.status_code == 200:
            results = response.json()
            print(f"✅ No-Criteria Search Result:")
            print(f"📄 Found {len(results)} results")
            if results and len(results) > 0:
                print(f"First result type: {type(results[0])}")
                if isinstance(results[0], dict):
                    print(f"First result keys: {list(results[0].keys())}")
                    if "contact_details" in results[0]:
                        print(
                            f"Name: {results[0]['contact_details'].get('name', 'N/A')}"
                        )
        else:
            print(f"❌ No-Criteria Search Failed!")
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")


if __name__ == "__main__":
    test_manual_search_with_proper_userid()
    test_simple_search()
    test_no_criteria_search()
