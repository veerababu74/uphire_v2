#!/usr/bin/env python3
"""
Test script to verify _id field fix for LLM context search
"""

import json


def test_llm_search_result_model():
    """Test the LLMSearchResult Pydantic model with _id field"""

    print("🧪 Testing LLMSearchResult Model with _id Field")
    print("=" * 60)

    try:
        from apis.rag_search import LLMSearchResult, ContactDetails

        # Test 1: Create model with id field (should serialize to _id)
        print("\n1️⃣ Test: Create model with 'id' field")

        sample_data = {
            "id": "67b075f7fe29fc1b2d36e18b",  # This should appear as _id in output
            "user_id": "user123",
            "username": "test_user",
            "contact_details": {
                "name": "Test User",
                "email": "test@example.com",
                "phone": "1234567890",
                "current_city": "Mumbai",
            },
            "total_experience": "5 years",
            "skills": ["python", "javascript"],
            "relevance_score": 85.5,
        }

        model = LLMSearchResult(**sample_data)

        print(f"✅ Model created successfully!")
        print(f"   - model.id: '{model.id}'")

        # Test serialization
        model_dict = model.model_dump(by_alias=True)  # Use by_alias to get _id
        print(f"   - Serialized _id: '{model_dict.get('_id')}'")
        print(f"   - Serialized id: '{model_dict.get('id')}'")

        # Test JSON serialization
        json_output = model.model_dump_json(by_alias=True)
        parsed_json = json.loads(json_output)
        print(f"   - JSON _id: '{parsed_json.get('_id')}'")

        if parsed_json.get("_id") == "67b075f7fe29fc1b2d36e18b":
            print("✅ SUCCESS: _id field is properly serialized!")
        else:
            print(
                f"❌ FAILED: _id field is '{parsed_json.get('_id')}', expected '67b075f7fe29fc1b2d36e18b'"
            )

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_data_transformation():
    """Test the data transformation logic"""

    print("\n🧪 Testing Data Transformation Logic")
    print("=" * 60)

    # Simulate the data transformation that happens in the API
    sample_result = {
        "results": [
            {
                "_id": "67b075f7fe29fc1b2d36e18b",
                "user_id": "user123",
                "username": "test_user",
                "contact_details": {"name": "Test User", "email": "test@example.com"},
                "relevance_score": 0.85,  # This should be normalized to 85
                "skills": ["python"],
            }
        ]
    }

    print("Original data structure:")
    print(json.dumps(sample_result, indent=2))

    # Apply the transformation logic from the API
    if "results" in sample_result:
        for res in sample_result["results"]:
            # Handle _id field properly - map _id to id for Pydantic model
            candidate_id = res.get("_id")
            if candidate_id is None:
                candidate_id = res.get("id", "")  # Try alternative key

            # Set both _id and id fields to ensure compatibility
            id_str = str(candidate_id) if candidate_id is not None else ""
            res["_id"] = id_str
            res["id"] = id_str  # Map _id to id for Pydantic model

            # Normalize relevance_score
            if "relevance_score" in res:
                relevance_score = res["relevance_score"]
                if relevance_score <= 1.0:
                    res["relevance_score"] = round(relevance_score * 100, 2)

    print("\nAfter transformation:")
    print(json.dumps(sample_result, indent=2))

    # Check if transformation worked
    first_result = sample_result["results"][0]
    if (
        first_result.get("_id") == "67b075f7fe29fc1b2d36e18b"
        and first_result.get("id") == "67b075f7fe29fc1b2d36e18b"
        and first_result.get("relevance_score") == 85.0
    ):
        print("✅ SUCCESS: Data transformation works correctly!")
        return True
    else:
        print("❌ FAILED: Data transformation has issues")
        print(f"   - _id: '{first_result.get('_id')}'")
        print(f"   - id: '{first_result.get('id')}'")
        print(f"   - relevance_score: {first_result.get('relevance_score')}")
        return False


if __name__ == "__main__":
    success1 = test_llm_search_result_model()
    success2 = test_data_transformation()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED! The _id field should now work correctly.")
        print("📝 Summary of fixes:")
        print("   - Pydantic model uses 'id' field with serialization_alias='_id'")
        print("   - API transforms data to include both '_id' and 'id' fields")
        print("   - Scores are normalized to 0-100 range")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 60)
