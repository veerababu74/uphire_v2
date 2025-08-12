"""
Test script for duplicate detection functionality.
This script tests the duplicate detection system without requiring actual file uploads.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mangodatabase.client import get_resume_extracted_text_collection
from mangodatabase.duplicate_detection import DuplicateDetectionOperations


def test_duplicate_detection():
    """Test the duplicate detection functionality."""
    print("🧪 Testing Duplicate Detection System...")

    # Initialize duplicate detection operations
    extracted_text_collection = get_resume_extracted_text_collection()
    duplicate_ops = DuplicateDetectionOperations(extracted_text_collection)

    # Test user ID
    test_user_id = "test_user_123"
    test_username = "John Doe"

    # Sample resume texts for testing
    sample_text_1 = """
    John Smith
    Software Engineer
    Email: john@email.com
    Phone: +1-234-567-8900
    
    Experience:
    - Software Developer at ABC Corp (2020-2023)
    - Junior Developer at XYZ Inc (2018-2020)
    
    Skills:
    Python, JavaScript, React, MongoDB, FastAPI
    
    Education:
    Bachelor of Computer Science
    University of Technology (2014-2018)
    """

    sample_text_2 = """
    John Smith
    Software Engineer
    Email: john@email.com
    Phone: +1-234-567-8900
    
    Experience:
    - Software Developer at ABC Corporation (2020-2023)
    - Junior Developer at XYZ Inc (2018-2020)
    
    Skills:
    Python, JavaScript, React, MongoDB, FastAPI, Docker
    
    Education:
    Bachelor of Computer Science
    University of Technology (2014-2018)
    """

    sample_text_3 = """
    Jane Wilson
    Data Scientist
    Email: jane@email.com
    Phone: +1-987-654-3210
    
    Experience:
    - Senior Data Scientist at DataCorp (2021-2023)
    - Data Analyst at Analytics Inc (2019-2021)
    
    Skills:
    Python, R, SQL, Pandas, Scikit-learn, TensorFlow
    
    Education:
    Master of Data Science
    Data University (2017-2019)
    """

    try:
        print(f"📊 Testing with user ID: {test_user_id}")

        # Test 1: Check for duplicates when no data exists
        print("\n🔍 Test 1: Checking for duplicates with no existing data...")
        is_duplicate, similar_docs = duplicate_ops.check_duplicate_content(
            test_user_id, sample_text_1
        )
        print(f"Is duplicate: {is_duplicate}")
        print(f"Similar documents found: {len(similar_docs)}")

        # Test 2: Save first text
        print("\n💾 Test 2: Saving first resume text...")
        save_result = duplicate_ops.save_extracted_text(
            test_user_id, test_username, "resume_1.pdf", sample_text_1
        )
        print(f"Save result: {save_result}")

        # Test 3: Check for duplicates with similar text (should detect duplicate)
        print("\n🔍 Test 3: Checking for duplicates with similar text...")
        is_duplicate, similar_docs = duplicate_ops.check_duplicate_content(
            test_user_id, sample_text_2
        )
        print(f"Is duplicate: {is_duplicate}")
        print(f"Similar documents found: {len(similar_docs)}")
        if similar_docs:
            for doc in similar_docs:
                print(
                    f"  - Filename: {doc['filename']}, Similarity: {doc['similarity_score']:.2%}"
                )

        # Test 4: Check for duplicates with completely different text (should not detect duplicate)
        print("\n🔍 Test 4: Checking for duplicates with different text...")
        is_duplicate, similar_docs = duplicate_ops.check_duplicate_content(
            test_user_id, sample_text_3
        )
        print(f"Is duplicate: {is_duplicate}")
        print(f"Similar documents found: {len(similar_docs)}")

        # Test 5: Save different text
        print("\n💾 Test 5: Saving different resume text...")
        save_result = duplicate_ops.save_extracted_text(
            test_user_id, test_username, "resume_3.pdf", sample_text_3
        )
        print(f"Save result: {save_result}")

        # Test 6: Get user statistics
        print("\n📈 Test 6: Getting user statistics...")
        stats = duplicate_ops.get_duplicate_statistics(test_user_id)
        print(f"Statistics: {stats}")

        # Test 7: Get user extracted texts
        print("\n📋 Test 7: Getting user extracted texts...")
        texts = duplicate_ops.get_user_extracted_texts(test_user_id)
        print(f"Total extracted texts for user: {len(texts)}")
        for i, text in enumerate(texts):
            print(f"  {i+1}. {text['filename']} - {text['text_length']} characters")

        print("\n✅ All tests completed successfully!")

        # Cleanup (optional)
        print("\n🧹 Cleanup: Deleting test data...")
        for text in texts:
            delete_result = duplicate_ops.delete_extracted_text(text["_id"])
            print(f"Deleted {text['filename']}: {delete_result}")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_duplicate_detection()
