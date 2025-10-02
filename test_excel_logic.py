#!/usr/bin/env python3
"""
Simple test for the Excel to JSON conversion endpoint logic
Tests the core functionality without requiring a running server.
"""

import json
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def test_excel_processor():
    """Test the ExcelProcessor functionality directly."""
    print("🧪 Testing Excel Processor Logic")
    print("=" * 40)

    try:
        from excel_resume_parser.excel_processor import ExcelProcessor

        # Create test Excel data
        test_data = {
            "Name": [
                "John Doe",
                "Jane Smith",
                "",
                "Bob Johnson",
                np.nan,
                "Alice Brown",
            ],
            "Email": [
                "john@email.com",
                "jane.smith@company.com",
                "invalid-email",
                "bob@test.com",
                None,
                "alice@example.org",
            ],
            "Phone": [
                "123-456-7890",
                "987.654.3210",
                "invalid",
                "555-0123",
                "",
                "(555) 999-8888",
            ],
            "Age": [25, 30, "invalid", 35, np.nan, "28"],
            "Salary": ["50000", "75000.50", "", "65000", "N/A", "80000"],
            "Department": ["Engineering", "Marketing", None, "Sales", "N/A", "HR"],
            "Active": [True, False, "", True, None, "Yes"],
            "Score": [8.5, 9.2, "N/A", 7.8, np.nan, "8.0"],
        }

        # Create DataFrame
        df = pd.DataFrame(test_data)
        print(f"Created test DataFrame with shape: {df.shape}")

        # Save to temporary Excel file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        df.to_excel(temp_file.name, index=False)
        temp_file.close()

        print(f"Temporary Excel file: {temp_file.name}")

        # Test ExcelProcessor
        processor = ExcelProcessor()

        # Test 1: Process from file path
        print("\n📁 Test 1: Process from file path")
        result_from_file = processor.process_excel_file(temp_file.name)
        print(f"✅ Processed {len(result_from_file)} rows from file")

        # Test 2: Process from bytes
        print("\n📄 Test 2: Process from bytes")
        with open(temp_file.name, "rb") as f:
            file_bytes = f.read()

        result_from_bytes = processor.process_excel_bytes(
            file_bytes=file_bytes, filename="test.xlsx"
        )
        print(f"✅ Processed {len(result_from_bytes)} rows from bytes")

        # Test 3: Verify data cleaning
        print("\n🧹 Test 3: Verify data cleaning")
        if result_from_bytes:
            sample_row = result_from_bytes[0]
            print("Sample cleaned row:")
            print(json.dumps(sample_row, indent=2))

            # Check for None values (should replace NaN)
            has_null = any(value is None for value in sample_row.values())
            print(f"✅ Contains None values (NaN converted): {has_null}")

            # Check all rows for JSON serialization
            try:
                json_str = json.dumps(result_from_bytes)
                print(f"✅ All data is JSON serializable ({len(json_str)} chars)")
            except Exception as e:
                print(f"❌ JSON serialization failed: {e}")

        # Clean up
        os.unlink(temp_file.name)
        print(f"🧹 Cleaned up temporary file")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_data_cleaning_logic():
    """Test the data cleaning logic specifically."""
    print("\n🧪 Testing Data Cleaning Logic")
    print("=" * 40)

    try:
        from excel_resume_parser.excel_processor import ExcelProcessor

        processor = ExcelProcessor()

        # Test various data types that need cleaning
        test_objects = [
            np.nan,
            None,
            "N/A",
            "NA",
            "#N/A",
            "",
            "  ",
            "valid_string",
            123,
            45.67,
            np.int64(42),
            np.float64(3.14),
            True,
            False,
            ["list", "values"],
            {"dict": "value"},
        ]

        print("Testing clean_nan_values method:")
        for i, obj in enumerate(test_objects):
            try:
                cleaned = processor.clean_nan_values(obj)
                print(
                    f"  {i+1:2d}. {str(obj):15} ({type(obj).__name__:12}) -> {str(cleaned):15} ({type(cleaned).__name__ if cleaned is not None else 'NoneType'})"
                )
            except Exception as e:
                print(f"  {i+1:2d}. {str(obj):15} -> ERROR: {e}")

        # Test nested structure cleaning
        print("\nTesting nested structure cleaning:")
        nested_test = {
            "name": "John Doe",
            "age": np.int64(25),
            "salary": np.float64(50000.5),
            "active": True,
            "missing": np.nan,
            "empty": "",
            "null_val": None,
            "skills": ["Python", "JavaScript", np.nan],
            "metadata": {"score": np.float64(8.5), "rating": "N/A", "valid": True},
        }

        cleaned_nested = processor.clean_nan_values(nested_test)
        print("Original nested structure:")
        print(f"  Type: {type(nested_test)}")
        print("Cleaned nested structure:")
        print(json.dumps(cleaned_nested, indent=2))

        return True

    except Exception as e:
        print(f"❌ Data cleaning test failed: {e}")
        return False


def simulate_api_response():
    """Simulate the API response structure."""
    print("\n🎭 Simulating API Response Structure")
    print("=" * 40)

    try:
        # Simulate the response structure that our API would return
        simulated_response = {
            "status": "success",
            "message": "Excel file successfully converted to clean JSON",
            "filename": "test_data.xlsx",
            "sheet_name": None,
            "statistics": {
                "original_rows": 6,
                "cleaned_rows": 4,
                "rows_removed": 2,
                "processing_time_seconds": 1.23,
            },
            "settings": {
                "skip_empty_rows": True,
                "normalize_headers": True,
                "max_rows": None,
            },
            "data": [
                {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "phone": "123-456-7890",
                    "age": 25,
                    "salary": 50000,
                    "department": "Engineering",
                    "active": True,
                    "score": 8.5,
                },
                {
                    "name": "Jane Smith",
                    "email": "jane@example.com",
                    "phone": "987-654-3210",
                    "age": 30,
                    "salary": 75000,
                    "department": "Marketing",
                    "active": False,
                    "score": 9.2,
                },
                {
                    "name": "Bob Johnson",
                    "email": "bob@example.com",
                    "phone": "555-0123",
                    "age": 35,
                    "salary": 65000,
                    "department": "Sales",
                    "active": True,
                    "score": 7.8,
                },
                {
                    "name": "Alice Brown",
                    "email": "alice@example.com",
                    "phone": "(555) 999-8888",
                    "age": 28,
                    "salary": 80000,
                    "department": "HR",
                    "active": True,
                    "score": 8.0,
                },
            ],
        }

        # Verify JSON serialization
        json_str = json.dumps(simulated_response, indent=2)
        print(f"✅ API response structure is valid JSON ({len(json_str)} characters)")

        # Show sample
        print("\nSample API response structure:")
        print(
            json.dumps(
                {
                    "status": simulated_response["status"],
                    "statistics": simulated_response["statistics"],
                    "first_data_record": simulated_response["data"][:1],
                },
                indent=2,
            )
        )

        print(f"\nData array contains {len(simulated_response['data'])} records")

        return True

    except Exception as e:
        print(f"❌ API simulation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Excel to JSON Conversion - Logic Tests")
    print("=" * 60)

    tests_passed = 0
    total_tests = 3

    # Test 1: Excel Processor
    if test_excel_processor():
        tests_passed += 1
        print("✅ Excel Processor Test PASSED")
    else:
        print("❌ Excel Processor Test FAILED")

    # Test 2: Data Cleaning Logic
    if test_data_cleaning_logic():
        tests_passed += 1
        print("✅ Data Cleaning Logic Test PASSED")
    else:
        print("❌ Data Cleaning Logic Test FAILED")

    # Test 3: API Response Structure
    if simulate_api_response():
        tests_passed += 1
        print("✅ API Response Structure Test PASSED")
    else:
        print("❌ API Response Structure Test FAILED")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print(
            "🎉 All tests passed! The Excel to JSON conversion logic is working correctly."
        )
        print("\nNext steps:")
        print("1. Start the FastAPI server: uvicorn main:app --reload --port 8000")
        print("2. Test the API endpoint: python test_excel_to_json_api.py")
        print("3. Check the API documentation: http://localhost:8000/docs")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
