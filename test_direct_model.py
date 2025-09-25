#!/usr/bin/env python3
"""
Direct test of the fixed ManualSearchRequest model
"""

import json
from typing import Dict, Any


# Simulate the exact request data that was failing
def test_original_failing_request():
    """Test with the exact data from the original error"""

    original_request_data = {
        "userid": "67b075f7fe29fc1b2d36e18b",
        "experience_titles": ["frontend developer", "software developer"],
        "locations": [],
        "max_experience": "",
        "max_salary": "",  # This was the problematic field
        "min_education": ["10th Pass", "Graduate", "12th Pass"],
        "min_experience": "",
        "min_salary": "",  # This was the problematic field
        "skills": ["html", "python"],
    }

    print("🧪 Testing Original Failing Request Data")
    print("=" * 50)
    print("Original request that caused 422 error:")
    print(json.dumps(original_request_data, indent=2))
    print("-" * 50)

    # Import the fixed model
    try:
        from apis.manual_search import ManualSearchRequest

        # Test the model creation
        model = ManualSearchRequest(**original_request_data)

        print("✅ SUCCESS: Model created successfully!")
        print(f"   - userid: {model.userid}")
        print(f"   - min_salary: {model.min_salary} (type: {type(model.min_salary)})")
        print(f"   - max_salary: {model.max_salary} (type: {type(model.max_salary)})")
        print(f"   - experience_titles: {model.experience_titles}")
        print(f"   - skills: {model.skills}")

        # Convert back to dict (as FastAPI would do)
        model_dict = model.dict()
        print("\n📋 Model dictionary representation:")
        print(json.dumps(model_dict, indent=2))

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_json_serialization():
    """Test JSON serialization of the fixed model"""

    print("\n🧪 Testing JSON Serialization")
    print("=" * 50)

    try:
        from apis.manual_search import ManualSearchRequest

        # Create model with empty salary strings
        data = {
            "userid": "test_user",
            "experience_titles": ["developer"],
            "min_salary": "",
            "max_salary": "",
        }

        model = ManualSearchRequest(**data)

        # Test JSON serialization
        json_str = model.json()
        print("✅ SUCCESS: JSON serialization worked!")
        print(f"JSON: {json_str}")

        # Test deserialization
        reconstructed = ManualSearchRequest.parse_raw(json_str)
        print("✅ SUCCESS: JSON deserialization worked!")
        print(f"Reconstructed min_salary: {reconstructed.min_salary}")
        print(f"Reconstructed max_salary: {reconstructed.max_salary}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


if __name__ == "__main__":
    success1 = test_original_failing_request()
    success2 = test_json_serialization()

    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED! The fix should resolve the 422 error.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 50)
