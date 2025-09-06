#!/usr/bin/env python3
"""
Test Excel Fix

Script to test the fixes for Excel resume parser issues:
1. ResumeParser.parse_resume -> ResumeParser.process_resume
2. JSON serialization with NaN values
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from excel_resume_parser.excel_processor import ExcelProcessor
from excel_resume_parser.main import ExcelResumeParserManager
from multipleresumepraser.main import ResumeParser
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("test_excel_fix")


def test_resume_parser_method():
    """Test that ResumeParser has process_resume method."""
    print("\n=== Testing ResumeParser Method ===")
    try:
        parser = ResumeParser()
        print(f"✅ ResumeParser initialized successfully")

        # Check if process_resume method exists
        if hasattr(parser, "process_resume"):
            print(f"✅ process_resume method found")

            # Test with sample text
            result = parser.process_resume("John Doe\nSoftware Engineer\nPython, Java")
            print(f"✅ process_resume method works: {type(result)}")

            return True
        else:
            print(f"❌ process_resume method not found")
            return False

    except Exception as e:
        print(f"❌ Error testing ResumeParser: {e}")
        return False


def test_nan_json_serialization():
    """Test JSON serialization with NaN values."""
    print("\n=== Testing NaN JSON Serialization ===")
    try:
        # Create test data with NaN values
        test_data = {
            "name": "John Doe",
            "age": np.nan,
            "salary": 50000.0,
            "skills": ["Python", None, ""],
            "experience": [
                {"company": "ABC Corp", "years": 3.5},
                {"company": "XYZ Inc", "years": np.nan},
            ],
            "empty_field": None,
            "infinity": np.inf,
            "negative_infinity": -np.inf,
        }

        print(f"Original data: {test_data}")

        # Test with ExcelProcessor cleaning
        processor = ExcelProcessor()
        cleaned_data = processor.clean_nan_values(test_data)

        print(f"Cleaned data: {cleaned_data}")

        # Test JSON serialization
        json_str = json.dumps(cleaned_data)
        print(f"✅ JSON serialization successful: {len(json_str)} characters")

        # Test with ExcelResumeParserManager cleaning
        manager = ExcelResumeParserManager()
        manager_cleaned = manager.clean_for_json_serialization(test_data)

        json_str2 = json.dumps(manager_cleaned)
        print(f"✅ Manager JSON serialization successful: {len(json_str2)} characters")

        return True

    except Exception as e:
        print(f"❌ Error testing JSON serialization: {e}")
        return False


def test_excel_processor():
    """Test Excel processor with NaN values."""
    print("\n=== Testing Excel Processor ===")
    try:
        # Create test DataFrame with NaN values
        df = pd.DataFrame(
            {
                "name": ["John Doe", "Jane Smith", "Bob Johnson"],
                "age": [25, np.nan, 35],
                "salary": [50000, 60000, np.nan],
                "skills": ["Python", "Java", "C++"],
                "experience": [3.5, np.nan, 7.2],
            }
        )

        print(f"Test DataFrame shape: {df.shape}")
        print(f"Contains NaN: {df.isna().any().any()}")

        # Test processor
        processor = ExcelProcessor()
        rows = processor.convert_rows_to_dictionaries(df)

        print(f"✅ Converted {len(rows)} rows to dictionaries")

        # Test JSON serialization of results
        for i, row in enumerate(rows):
            json_str = json.dumps(row)
            print(f"✅ Row {i+1} JSON serialization successful")

        return True

    except Exception as e:
        print(f"❌ Error testing Excel processor: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("TESTING EXCEL RESUME PARSER FIXES")
    print("=" * 50)

    tests = [
        test_resume_parser_method,
        test_nan_json_serialization,
        test_excel_processor,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    success_count = sum(results)
    total_count = len(results)

    print(f"Passed: {success_count}/{total_count}")

    if success_count == total_count:
        print("✅ All tests passed! Excel fixes are working.")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
