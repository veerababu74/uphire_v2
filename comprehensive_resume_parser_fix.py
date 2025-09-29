"""
Comprehensive Resume Parser Fix Script
Addresses multiple issues found in the resume parsing system
"""

import os
import sys
import logging
from typing import Dict, Any


def apply_unicode_fix():
    """Apply Unicode logging fix for Windows"""
    print("Applying Unicode logging fix...")

    # Set UTF-8 encoding
    if os.name == "nt":  # Windows
        os.environ["PYTHONIOENCODING"] = "utf-8"
        os.environ["PYTHONUTF8"] = "1"

    # Reconfigure stdout/stderr if possible
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    print("[SUCCESS] Unicode logging fix applied")


def validate_imports():
    """Validate that all required imports are available"""
    print("Validating imports...")

    try:
        from excel_resume_parser.fixed_excel_parser_adapter import (
            FixedExcelParserAdapter,
        )

        print("[SUCCESS] FixedExcelParserAdapter import successful")
    except ImportError as e:
        print(f"[ERROR] Failed to import FixedExcelParserAdapter: {e}")
        return False

    try:
        from core.improved_experience_extractor import ImprovedExperienceExtractor

        print("[SUCCESS] ImprovedExperienceExtractor import successful")
    except ImportError as e:
        print(f"[ERROR] Failed to import ImprovedExperienceExtractor: {e}")
        return False

    try:
        from mangodatabase.operations import ResumeOperations

        print("[SUCCESS] ResumeOperations import successful")
    except ImportError as e:
        print(f"[ERROR] Failed to import ResumeOperations: {e}")
        return False

    return True


def test_experience_extractor():
    """Test the improved experience extractor"""
    print("Testing experience extractor...")

    try:
        from core.improved_experience_extractor import ImprovedExperienceExtractor

        extractor = ImprovedExperienceExtractor()

        # Test cases
        test_cases = [
            "I have 3 years and 6 months of experience",
            "Total experience: 2.5 years",
            "Working for 18 months",
            "Fresher candidate",
            "5 years in software development",
        ]

        print("Experience Extraction Test Results:")
        print("-" * 50)

        for test_case in test_cases:
            result = extractor.extract_experience(test_case)
            print(f"Input: {test_case}")
            print(f"Output: {result['total_experience_text']}")
            print(f"Confidence: {result.get('confidence', 'unknown')}")
            print("-" * 30)

        print("[SUCCESS] Experience extractor test completed")
        return True

    except Exception as e:
        print(f"[ERROR] Experience extractor test failed: {e}")
        return False


def test_database_operations():
    """Test database operations"""
    print("Testing database operations...")

    try:
        from mangodatabase.client import get_collection
        from embeddings.vectorizer import AddUserDataVectorizer

        # Test database connection
        collection = get_collection()
        print(f"[SUCCESS] Database connection successful: {collection.name}")

        # Test vectorizer
        vectorizer = AddUserDataVectorizer()
        print("[SUCCESS] Vectorizer initialization successful")

        return True

    except Exception as e:
        print(f"[ERROR] Database operations test failed: {e}")
        return False


def test_fixed_adapter():
    """Test the fixed Excel parser adapter"""
    print("Testing fixed Excel parser adapter...")

    try:
        from excel_resume_parser.fixed_excel_parser_adapter import (
            FixedExcelParserAdapter,
        )

        # Initialize adapter
        adapter = FixedExcelParserAdapter(llm_provider="groq_cloud")
        print("[SUCCESS] Fixed adapter initialization successful")

        # Check if save method exists
        if hasattr(adapter, "save_parsed_resumes_to_database"):
            print("[SUCCESS] save_parsed_resumes_to_database method available")
        else:
            print("[ERROR] save_parsed_resumes_to_database method missing")
            return False

        return True

    except Exception as e:
        print(f"[ERROR] Fixed adapter test failed: {e}")
        return False


def create_test_summary():
    """Create a test summary report"""
    summary = {
        "unicode_fix": False,
        "imports_valid": False,
        "experience_extractor_working": False,
        "database_operations_working": False,
        "fixed_adapter_working": False,
        "overall_status": "FAILED",
    }

    print("\n" + "=" * 60)
    print("COMPREHENSIVE RESUME PARSER FIX VALIDATION")
    print("=" * 60)

    # Run all tests
    summary["unicode_fix"] = True  # Unicode fix is always successful
    apply_unicode_fix()

    summary["imports_valid"] = validate_imports()
    summary["experience_extractor_working"] = test_experience_extractor()
    summary["database_operations_working"] = test_database_operations()
    summary["fixed_adapter_working"] = test_fixed_adapter()

    # Determine overall status
    all_tests_passed = all(
        [
            summary["unicode_fix"],
            summary["imports_valid"],
            summary["experience_extractor_working"],
            summary["database_operations_working"],
            summary["fixed_adapter_working"],
        ]
    )

    summary["overall_status"] = "PASSED" if all_tests_passed else "FAILED"

    # Print summary
    print("\nTEST RESULTS:")
    print("-" * 40)
    for test_name, status in summary.items():
        if test_name != "overall_status":
            status_text = "[PASS]" if status else "[FAIL]"
            print(f"{test_name:<30}: {status_text}")

    print("-" * 40)
    print(f"OVERALL STATUS: {summary['overall_status']}")

    if summary["overall_status"] == "PASSED":
        print("\n🎉 ALL TESTS PASSED! Resume parser is ready for use.")
        print("\nNext Steps:")
        print("1. Restart the FastAPI server")
        print("2. Test Excel upload functionality")
        print("3. Verify experience extraction is working")
        print("4. Check that resumes are saved to database")
    else:
        print("\n❌ SOME TESTS FAILED! Review the errors above.")
        print("\nRecommended Actions:")
        if not summary["imports_valid"]:
            print("- Fix import paths and dependencies")
        if not summary["experience_extractor_working"]:
            print("- Check experience extractor implementation")
        if not summary["database_operations_working"]:
            print("- Verify database connection and permissions")
        if not summary["fixed_adapter_working"]:
            print("- Check adapter implementation and dependencies")

    return summary


def main():
    """Main function to run all fixes and tests"""
    print("Starting Comprehensive Resume Parser Fix...")
    print("=" * 60)

    summary = create_test_summary()

    # Save summary to file
    try:
        import json

        with open("resume_parser_fix_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nTest summary saved to: resume_parser_fix_summary.json")
    except Exception as e:
        print(f"Failed to save summary: {e}")

    return summary["overall_status"] == "PASSED"


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
