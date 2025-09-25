#!/usr/bin/env python3
"""
Test script to validate the Pydantic model changes work correctly
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from apis.manual_search import ManualSearchRequest


def test_manual_search_request_validation():
    """Test the ManualSearchRequest model with various salary inputs"""

    print("🧪 Testing ManualSearchRequest Pydantic Model Validation")
    print("=" * 60)

    # Test 1: Empty strings for salary (the problematic case)
    print("\n1️⃣ Test: Empty strings for salary fields")
    try:
        data = {
            "userid": "test_user",
            "experience_titles": ["developer"],
            "min_salary": "",  # Empty string
            "max_salary": "",  # Empty string
        }
        model = ManualSearchRequest(**data)
        print(
            f"✅ SUCCESS: min_salary = {model.min_salary}, max_salary = {model.max_salary}"
        )
    except Exception as e:
        print(f"❌ FAILED: {e}")

    # Test 2: None values for salary
    print("\n2️⃣ Test: None values for salary fields")
    try:
        data = {
            "userid": "test_user",
            "experience_titles": ["developer"],
            "min_salary": None,
            "max_salary": None,
        }
        model = ManualSearchRequest(**data)
        print(
            f"✅ SUCCESS: min_salary = {model.min_salary}, max_salary = {model.max_salary}"
        )
    except Exception as e:
        print(f"❌ FAILED: {e}")

    # Test 3: Valid float values
    print("\n3️⃣ Test: Valid float values for salary fields")
    try:
        data = {
            "userid": "test_user",
            "experience_titles": ["developer"],
            "min_salary": 500000.0,
            "max_salary": 1500000.0,
        }
        model = ManualSearchRequest(**data)
        print(
            f"✅ SUCCESS: min_salary = {model.min_salary}, max_salary = {model.max_salary}"
        )
    except Exception as e:
        print(f"❌ FAILED: {e}")

    # Test 4: String numbers (should be converted)
    print("\n4️⃣ Test: String numbers for salary fields")
    try:
        data = {
            "userid": "test_user",
            "experience_titles": ["developer"],
            "min_salary": "500000",
            "max_salary": "1500000",
        }
        model = ManualSearchRequest(**data)
        print(
            f"✅ SUCCESS: min_salary = {model.min_salary}, max_salary = {model.max_salary}"
        )
    except Exception as e:
        print(f"❌ FAILED: {e}")

    # Test 5: Invalid string (should fail)
    print("\n5️⃣ Test: Invalid string for salary fields")
    try:
        data = {
            "userid": "test_user",
            "experience_titles": ["developer"],
            "min_salary": "invalid_number",
            "max_salary": "another_invalid",
        }
        model = ManualSearchRequest(**data)
        print(
            f"❌ UNEXPECTED SUCCESS: min_salary = {model.min_salary}, max_salary = {model.max_salary}"
        )
    except Exception as e:
        print(f"✅ EXPECTED FAILURE: {e}")

    # Test 6: Mixed valid and empty
    print("\n6️⃣ Test: Mixed valid and empty values")
    try:
        data = {
            "userid": "test_user",
            "experience_titles": ["developer"],
            "min_salary": 500000.0,
            "max_salary": "",  # Empty string
        }
        model = ManualSearchRequest(**data)
        print(
            f"✅ SUCCESS: min_salary = {model.min_salary}, max_salary = {model.max_salary}"
        )
    except Exception as e:
        print(f"❌ FAILED: {e}")


if __name__ == "__main__":
    test_manual_search_request_validation()

    print("\n" + "=" * 60)
    print("🏁 Validation tests completed!")
    print("=" * 60)
