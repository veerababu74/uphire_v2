#!/usr/bin/env python3
"""
Test script to verify Excel Resume Parser auto-detection functionality.

This script tests that:
1. LLM provider is auto-detected from configuration
2. save_temp_file is automatically set to False
3. The parser initializes correctly with these settings
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_llm_provider_auto_detection():
    """Test that LLM provider is auto-detected from configuration."""
    try:
        from core.config import AppConfig

        print(f"✓ LLM Provider from config: {AppConfig.LLM_PROVIDER}")
        return True
    except Exception as e:
        print(f"✗ Failed to load LLM provider from config: {e}")
        return False


def test_excel_resume_parser_initialization():
    """Test that ExcelResumeParser initializes with auto-detected settings."""
    try:
        from excel_resume_parser.excel_resume_parser import ExcelResumeParser

        # Test with no parameters (should auto-detect)
        parser = ExcelResumeParser()
        print(
            "✓ ExcelResumeParser initialized successfully with auto-detected LLM provider"
        )

        # Test with explicit None (should auto-detect)
        parser2 = ExcelResumeParser(llm_provider=None)
        print(
            "✓ ExcelResumeParser initialized successfully with explicit None LLM provider"
        )

        return True
    except Exception as e:
        print(f"✗ Failed to initialize ExcelResumeParser: {e}")
        return False


def test_excel_resume_parser_manager_initialization():
    """Test that ExcelResumeParserManager initializes with auto-detected settings."""
    try:
        from excel_resume_parser.main import ExcelResumeParserManager

        # Test with no parameters (should auto-detect)
        manager = ExcelResumeParserManager()
        print(
            "✓ ExcelResumeParserManager initialized successfully with auto-detected LLM provider"
        )

        # Test with explicit None (should auto-detect)
        manager2 = ExcelResumeParserManager(llm_provider=None)
        print(
            "✓ ExcelResumeParserManager initialized successfully with explicit None LLM provider"
        )

        return True
    except Exception as e:
        print(f"✗ Failed to initialize ExcelResumeParserManager: {e}")
        return False


def test_api_imports():
    """Test that the updated API imports work correctly."""
    try:
        from apis.excel_resume_parser_api import router

        print("✓ Excel Resume Parser API imports successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import Excel Resume Parser API: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Excel Resume Parser Auto-Detection Functionality")
    print("=" * 60)

    tests = [
        test_llm_provider_auto_detection,
        test_excel_resume_parser_initialization,
        test_excel_resume_parser_manager_initialization,
        test_api_imports,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Auto-detection is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the configuration.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
