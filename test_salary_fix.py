#!/usr/bin/env python3
"""
Test script to verify the salary field validation fix
"""

import requests
import json


def test_manual_search_empty_salary():
    """Test the manual search API with empty salary strings"""

    # API endpoint
    url = "http://localhost:8000/manualsearch/"

    # Test data with empty salary strings (the problematic case)
    request_body = {
        "userid": "67b075f7fe29fc1b2d36e18b",
        "experience_titles": ["frontend developer", "software developer"],
        "locations": [],
        "max_experience": "",
        "max_salary": "",  # This was causing the error
        "min_education": ["10th Pass", "Graduate", "12th Pass"],
        "min_experience": "",
        "min_salary": "",  # This was causing the error
        "skills": ["html", "python"],
    }

    print("🧪 Testing Manual Search API with Empty Salary Fields")
    print("=" * 60)
    print("Request payload:")
    print(json.dumps(request_body, indent=2))
    print("-" * 60)

    try:
        response = requests.post(url, json=request_body, timeout=30)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            results = response.json()
            print("✅ SUCCESS: API request completed successfully!")
            print(
                f"📊 Results: {len(results) if isinstance(results, list) else 'Non-list response'}"
            )

            if isinstance(results, list) and results:
                print(f"First result sample:")
                print(
                    json.dumps(results[0], indent=2)[:500] + "..."
                    if len(str(results[0])) > 500
                    else json.dumps(results[0], indent=2)
                )
            elif isinstance(results, list):
                print("📝 Empty results list - no matching candidates found")
            else:
                print(f"📝 Response type: {type(results)}")

        elif response.status_code == 422:
            print("❌ VALIDATION ERROR (422):")
            try:
                error_detail = response.json()
                print(json.dumps(error_detail, indent=2))
            except:
                print(response.text)
        else:
            print(f"❌ ERROR ({response.status_code}):")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"❌ REQUEST EXCEPTION: {e}")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")


def test_manual_search_with_valid_salaries():
    """Test the manual search API with valid salary values"""

    url = "http://localhost:8000/manualsearch/"

    request_body = {
        "userid": "67b075f7fe29fc1b2d36e18b",
        "experience_titles": ["frontend developer", "software developer"],
        "locations": [],
        "max_experience": "",
        "max_salary": 1500000.0,  # Valid float
        "min_education": ["10th Pass", "Graduate", "12th Pass"],
        "min_experience": "",
        "min_salary": 500000.0,  # Valid float
        "skills": ["html", "python"],
    }

    print("\n🧪 Testing Manual Search API with Valid Salary Fields")
    print("=" * 60)
    print("Request payload:")
    print(json.dumps(request_body, indent=2))
    print("-" * 60)

    try:
        response = requests.post(url, json=request_body, timeout=30)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            results = response.json()
            print("✅ SUCCESS: API request completed successfully!")
            print(
                f"📊 Results: {len(results) if isinstance(results, list) else 'Non-list response'}"
            )
        elif response.status_code == 422:
            print("❌ VALIDATION ERROR (422):")
            try:
                error_detail = response.json()
                print(json.dumps(error_detail, indent=2))
            except:
                print(response.text)
        else:
            print(f"❌ ERROR ({response.status_code}):")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"❌ REQUEST EXCEPTION: {e}")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")


if __name__ == "__main__":
    test_manual_search_empty_salary()
    test_manual_search_with_valid_salaries()

    print("\n" + "=" * 60)
    print("🏁 Test completed!")
    print("=" * 60)
