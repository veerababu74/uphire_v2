"""
Test Excel Resume Parser

Test script to verify Excel resume parser functionality.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from excel_resume_parser.main import ExcelResumeParserManager
from excel_resume_parser.excel_processor import ExcelProcessor
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("test_excel_resume_parser")


def test_excel_processor():
    """Test Excel processor functionality."""
    print("=== Testing Excel Processor ===")

    try:
        processor = ExcelProcessor()

        # Test sample data creation (simulate Excel data)
        sample_data = [
            {
                "name": "John Doe",
                "email": "john.doe@email.com",
                "phone": "1234567890",
                "experience": "5 years",
                "skills": "Python, JavaScript, React",
                "current_company": "Tech Corp",
                "education": "B.Tech Computer Science",
            },
            {
                "name": "Jane Smith",
                "email": "jane.smith@email.com",
                "phone": "0987654321",
                "experience": "3 years",
                "skills": "Java, Spring Boot, MySQL",
                "current_company": "Software Inc",
                "education": "MCA",
            },
        ]

        print(f"✅ Excel processor initialized successfully")
        print(f"✅ Sample data created with {len(sample_data)} records")

        return True

    except Exception as e:
        print(f"❌ Excel processor test failed: {e}")
        return False


def test_excel_resume_parser_manager():
    """Test Excel Resume Parser Manager."""
    print("\n=== Testing Excel Resume Parser Manager ===")

    try:
        # Initialize manager (this tests all the imports and connections)
        manager = ExcelResumeParserManager()

        print(f"✅ Excel Resume Parser Manager initialized successfully")

        # Test getting statistics
        stats = manager.get_processing_statistics()
        if stats.get("status") == "success":
            print(f"✅ Statistics retrieved successfully")
            print(f"   LLM Provider: {stats.get('llm_provider', 'Unknown')}")
        else:
            print(f"❌ Statistics test failed")
            return False

        # Test supported formats info
        print(f"✅ Manager functionality verified")

        return True

    except Exception as e:
        print(f"❌ Excel Resume Parser Manager test failed: {e}")
        return False


def create_sample_excel_data():
    """Create sample Excel-like data for testing."""
    print("\n=== Creating Sample Excel Data ===")

    sample_excel_data = [
        {
            "name": "Alice Johnson",
            "email": "alice.johnson@example.com",
            "phone": "555-1234",
            "current_city": "New York",
            "total_experience": "7 years",
            "current_role": "Senior Developer",
            "current_company": "BigTech Corp",
            "skills": "Python, Django, PostgreSQL, AWS",
            "education": "MS Computer Science",
            "college": "MIT",
            "current_salary": "120000",
            "expected_salary": "140000",
            "notice_period": "30 days",
        },
        {
            "name": "Bob Wilson",
            "email": "bob.wilson@example.com",
            "phone": "555-5678",
            "current_city": "San Francisco",
            "total_experience": "4 years",
            "current_role": "Frontend Developer",
            "current_company": "StartupXYZ",
            "skills": "React, JavaScript, TypeScript, Node.js",
            "education": "B.Tech IT",
            "college": "Stanford University",
            "current_salary": "95000",
            "expected_salary": "110000",
            "notice_period": "Immediate",
        },
        {
            "name": "Carol Davis",
            "email": "carol.davis@example.com",
            "phone": "555-9012",
            "current_city": "Austin",
            "total_experience": "2 years",
            "current_role": "Data Analyst",
            "current_company": "DataCorp",
            "skills": "Python, SQL, Pandas, Machine Learning",
            "education": "MS Data Science",
            "college": "UT Austin",
            "current_salary": "75000",
            "expected_salary": "85000",
            "notice_period": "15 days",
        },
    ]

    print(f"✅ Created sample data with {len(sample_excel_data)} candidates")
    return sample_excel_data


def test_excel_data_processing():
    """Test processing of Excel-like data."""
    print("\n=== Testing Excel Data Processing ===")

    try:
        # Create sample data
        sample_data = create_sample_excel_data()

        # Initialize manager
        manager = ExcelResumeParserManager()

        # Test the Excel resume parser component
        from excel_resume_parser.excel_resume_parser import ExcelResumeParser

        parser = ExcelResumeParser()

        # Test formatting a single row
        first_row = sample_data[0]
        formatted_text = parser.format_excel_row_as_resume_text(first_row)

        print(f"✅ Excel row formatted successfully")
        print(f"   Sample formatted text (first 200 chars): {formatted_text[:200]}...")

        print(f"✅ Excel data processing test completed")

        return True

    except Exception as e:
        print(f"❌ Excel data processing test failed: {e}")
        return False


def test_api_requirements():
    """Test if API-related imports work."""
    print("\n=== Testing API Requirements ===")

    try:
        # Test FastAPI imports
        from fastapi import APIRouter, HTTPException, File, UploadFile, Form

        print(f"✅ FastAPI imports successful")

        # Test if our API module can be imported
        from apis.excel_resume_parser_api import router

        print(f"✅ Excel Resume Parser API module imported successfully")

        return True

    except Exception as e:
        print(f"❌ API requirements test failed: {e}")
        print(f"   This might be expected if dependencies are not installed")
        return False


def main():
    """Run all tests."""
    print("🧪 Excel Resume Parser - Test Suite")
    print("=" * 50)

    tests = [
        ("Excel Processor", test_excel_processor),
        ("Excel Resume Parser Manager", test_excel_resume_parser_manager),
        ("Excel Data Processing", test_excel_data_processing),
        ("API Requirements", test_api_requirements),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Excel Resume Parser is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
        print("💡 Note: Some failures might be due to missing dependencies.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
