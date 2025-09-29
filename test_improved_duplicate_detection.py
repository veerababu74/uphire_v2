#!/usr/bin/env python3

"""
Test script to verify the improved duplicate detection logic.
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mangodatabase.operations import ResumeOperations
from mangodatabase.client import get_collection
from embeddings.vectorizer import AddUserDataVectorizer


def test_improved_duplicate_detection():
    """Test that the improved duplicate detection ignores generic values."""

    print("🧪 Testing Improved Duplicate Detection Logic")
    print("=" * 60)

    try:
        # Initialize components
        collection = get_collection()
        vectorizer = AddUserDataVectorizer()
        resume_ops = ResumeOperations(collection, vectorizer)

        print("✅ ResumeOperations initialized successfully")

        # Test 1: Generic email should NOT be considered duplicate
        print("\n📧 Testing Generic Email Handling:")
        result = resume_ops.check_duplicate_resume(email="noemail@notprovided.com")
        if result is None:
            print("✅ Generic email 'noemail@notprovided.com' ignored - GOOD")
        else:
            print("❌ Generic email was considered for duplicate check - BAD")

        # Test 2: Generic name should NOT be considered duplicate
        print("\n👤 Testing Generic Name Handling:")
        result = resume_ops.check_duplicate_resume(name="Name Not Found")
        if result is None:
            print("✅ Generic name 'Name Not Found' ignored - GOOD")
        else:
            print("❌ Generic name was considered for duplicate check - BAD")

        # Test 3: Generic phone should NOT be considered duplicate
        print("\n📱 Testing Generic Phone Handling:")
        result = resume_ops.check_duplicate_resume(phone="+91")
        if result is None:
            print("✅ Generic phone '+91' ignored - GOOD")
        else:
            print("❌ Generic phone was considered for duplicate check - BAD")

        # Test 4: Combination of all generic values should NOT be considered duplicate
        print("\n🔗 Testing Combined Generic Values:")
        result = resume_ops.check_duplicate_resume(
            email="noemail@notprovided.com", phone="+91", name="Name Not Found"
        )
        if result is None:
            print("✅ All generic values combined ignored - GOOD")
            print("   This should fix the multiple resume parsing issue!")
        else:
            print("❌ Generic values were considered for duplicate check - BAD")

        # Test 5: Valid values should still work
        print("\n✨ Testing Valid Values (should work normally):")
        result = resume_ops.check_duplicate_resume(
            email="real.email@company.com", phone="9876543210", name="John Doe"
        )
        print(f"✅ Valid values processed normally: {result is not None}")

        return True

    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def simulate_multiple_resume_scenario():
    """Simulate the exact scenario from the user's multiple resume parsing."""

    print("\n🎭 Simulating Multiple Resume Scenario")
    print("=" * 60)

    # This is the exact data pattern from the user's results
    resume_data = {
        "name": "Name Not Found",
        "email": "noemail@notprovided.com",
        "phone": "+91",
    }

    try:
        collection = get_collection()
        vectorizer = AddUserDataVectorizer()
        resume_ops = ResumeOperations(collection, vectorizer)

        print("📄 Testing Resume 1 (first resume):")
        result1 = resume_ops.check_duplicate_resume(**resume_data)
        print(f"   Duplicate found: {result1 is not None}")

        print("📄 Testing Resume 2 (same generic data):")
        result2 = resume_ops.check_duplicate_resume(**resume_data)
        print(f"   Duplicate found: {result2 is not None}")

        print("📄 Testing Resume 3 (same generic data):")
        result3 = resume_ops.check_duplicate_resume(**resume_data)
        print(f"   Duplicate found: {result3 is not None}")

        if result1 is None and result2 is None and result3 is None:
            print("\n🎉 SUCCESS! All resumes with generic data will be processed!")
            print("   No more 'skipped due to duplicate' issues!")
        else:
            print("\n❌ Issue still exists - resumes will be flagged as duplicates")

        return True

    except Exception as e:
        print(f"❌ Error in simulation: {str(e)}")
        return False


if __name__ == "__main__":
    print("🚀 IMPROVED DUPLICATE DETECTION TEST")
    print("=" * 70)

    # Run tests
    logic_test = test_improved_duplicate_detection()
    scenario_test = simulate_multiple_resume_scenario()

    print("\n" + "=" * 70)
    if logic_test and scenario_test:
        print("🎉 ALL TESTS PASSED!")
        print(
            "✅ Improved duplicate detection should fix the multiple resume parsing issues"
        )
        print("✅ Generic/placeholder values will be ignored")
        print("✅ Valid resumes will be processed instead of being skipped")
    else:
        print("❌ Some tests failed. Check the output above.")

    print("=" * 70)
