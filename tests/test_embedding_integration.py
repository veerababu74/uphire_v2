#!/usr/bin/env python3
"""
Test script to verify that the embedding fix resolves the integration issues.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_vectorizer_integration():
    """Test the vectorizer integration that was failing"""
    try:
        print("Testing AddUserDataVectorizer integration...")

        from embeddings.vectorizer import AddUserDataVectorizer

        # Create vectorizer instance
        vectorizer = AddUserDataVectorizer()

        # Test sample resume data
        sample_resume = {
            "user_id": "test_user",
            "username": "test_username",
            "contact_details": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "+1-555-0123",
                "current_city": "New York",
            },
            "total_experience": "5 years",
            "skills": ["Python", "Machine Learning", "Data Science"],
            "may_also_known_skills": ["TensorFlow", "PyTorch"],
            "experience": [
                {
                    "company": "Tech Corp",
                    "title": "Data Scientist",
                    "from_date": "2020-01",
                    "to": "2023-12",
                }
            ],
            "academic_details": [
                {
                    "education": "Masters in Computer Science",
                    "college": "MIT",
                    "pass_year": 2019,
                }
            ],
        }

        # Generate embeddings - this was failing before
        result = vectorizer.generate_resume_embeddings(sample_resume)

        # Check if embeddings were generated
        vector_fields = [key for key in result.keys() if key.endswith("_vector")]

        if vector_fields:
            print(f"✅ Successfully generated {len(vector_fields)} vector fields:")
            for field in vector_fields:
                vector_length = len(result[field]) if result[field] else 0
                print(f"  - {field}: {vector_length} dimensions")
            return True
        else:
            print("❌ No vector fields generated")
            return False

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_multiple_resume_api_simulation():
    """Simulate the multiple resume API embedding generation"""
    try:
        print("\nTesting multiple resume API simulation...")

        # Import the specific components that were failing
        from apis.multiple_resume_parser_api import generate_embeddings_for_resume

        # Sample parsed resume data (similar to what comes from LLM)
        sample_parsed_data = {
            "contact_details": {
                "name": "Jane Smith",
                "email": "jane.smith@example.com",
                "phone": "+1-555-0456",
                "current_city": "San Francisco",
                "looking_for_jobs_in": ["San Francisco", "Remote"],
                "pan_card": "ABCDE1234F",
            },
            "total_experience": "3 years",
            "skills": ["JavaScript", "React", "Node.js"],
            "may_also_known_skills": ["Vue.js", "Angular"],
            "experience": [
                {
                    "company": "Web Startup",
                    "title": "Frontend Developer",
                    "from_date": "2021-06",
                    "to": "2024-12",
                }
            ],
            "academic_details": [
                {
                    "education": "Bachelor of Computer Science",
                    "college": "Stanford University",
                    "pass_year": 2021,
                }
            ],
        }

        # Test the embedding generation function
        result = generate_embeddings_for_resume(
            sample_parsed_data, user_id="test_user_2", username="test_username_2"
        )

        # Check results
        vector_fields = [key for key in result.keys() if key.endswith("_vector")]

        if vector_fields:
            print(
                f"✅ API simulation successful - generated {len(vector_fields)} vector fields"
            )
            return True
        else:
            print("❌ API simulation failed - no vectors generated")
            return False

    except Exception as e:
        print(f"❌ API simulation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("🧪 Running integration tests for embedding fix...\n")

    success = True

    # Test 1: Basic vectorizer integration
    if not test_vectorizer_integration():
        success = False

    # Test 2: Multiple resume API simulation
    if not test_multiple_resume_api_simulation():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("🎉 All integration tests PASSED!")
        print("The embedding issues have been successfully resolved.")
        print("The application should now work without the meta tensor errors.")
    else:
        print("❌ Some integration tests FAILED!")
        print("Further investigation may be needed.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
