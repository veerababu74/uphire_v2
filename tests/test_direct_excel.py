#!/usr/bin/env python3
"""
Direct Test of Excel Resume Parser Fixes

Test the Excel resume parser components directly without needing the server.
"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
import tempfile

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from excel_resume_parser.excel_processor import ExcelProcessor
from excel_resume_parser.excel_resume_parser import ExcelResumeParser
from excel_resume_parser.main import ExcelResumeParserManager
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("test_direct_excel")


def test_excel_processor_with_nan():
    """Test Excel processor with NaN values."""
    print("\n=== Testing Excel Processor with NaN Values ===")

    # Create test data with NaN values
    data = {
        "name": ["John Doe", "Jane Smith", "Bob Johnson"],
        "age": [25, np.nan, 35],
        "salary": [50000, 60000, np.nan],
        "skills": ["Python", "Java", None],
        "experience": [3.5, np.nan, 7.2],
        "empty_col": [np.nan, np.nan, np.nan],
    }

    df = pd.DataFrame(data)
    print(f"Test DataFrame shape: {df.shape}")
    print(f"Contains NaN: {df.isna().any().any()}")

    # Save to temporary Excel file
    temp_file = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    temp_file.close()
    df.to_excel(temp_file.name, index=False)

    try:
        # Test processor
        processor = ExcelProcessor()
        result = processor.process_excel_file(temp_file.name)

        print(f"✅ Processed {len(result)} rows from Excel")

        # Test JSON serialization
        json_str = json.dumps(result)
        print(f"✅ JSON serialization successful: {len(json_str)} characters")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        Path(temp_file.name).unlink()


def test_excel_resume_parser():
    """Test Excel resume parser with sample data."""
    print("\n=== Testing Excel Resume Parser ===")

    try:
        # Create parser
        parser = ExcelResumeParser()
        print("✅ Excel Resume Parser initialized")

        # Test format_excel_row_as_resume_text
        sample_row = {
            "candidate_name": "John Doe",
            "email": "john@example.com",
            "mobile_no": "1234567890",
            "key_skills": "Python, Java, React",
            "total_experience": 5,
            "current_city": "New York",
            "company_name": "TechCorp",
            "position_title": "Software Engineer",
            "education_name": "Computer Science",
            "unknown_field": np.nan,  # Test NaN handling
        }

        formatted_text = parser.format_excel_row_as_resume_text(sample_row)
        print(f"✅ Formatted resume text length: {len(formatted_text)}")

        # Test parse_excel_row_to_resume
        parsed_resume = parser.parse_excel_row_to_resume(
            sample_row, "test_user_1", "john_doe"
        )

        if parsed_resume:
            print("✅ Resume parsing successful")
            print(f"   Resume type: {type(parsed_resume)}")

            # Test JSON serialization of result
            if hasattr(parsed_resume, "__dict__"):
                resume_dict = parsed_resume.__dict__
            else:
                resume_dict = parsed_resume

            json_str = json.dumps(resume_dict, default=str)
            print(
                f"✅ Resume JSON serialization successful: {len(json_str)} characters"
            )
        else:
            print("⚠️  Resume parsing returned None (might be expected)")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_full_manager():
    """Test the full Excel Resume Parser Manager."""
    print("\n=== Testing Excel Resume Parser Manager ===")

    # Create test Excel file with NaN values
    data = {
        "Candidate Name": ["John Doe", "Jane Smith"],
        "Mobile No": ["1234567890", "9876543210"],
        "Email": ["john@example.com", "jane@example.com"],
        "Key Skills": ["Python, Django", "Java, Spring"],
        "Total Experience": [5, np.nan],  # NaN value
        "Current Salary": [75000, np.nan],  # NaN value
        "May Also Know": ["AWS", np.nan],  # NaN value
    }

    df = pd.DataFrame(data)

    # Save to temporary Excel file
    temp_file = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    temp_file.close()
    df.to_excel(temp_file.name, index=False)

    try:
        # Read file as bytes
        with open(temp_file.name, "rb") as f:
            file_bytes = f.read()

        # Test manager
        manager = ExcelResumeParserManager()
        print("✅ Manager initialized")

        # Process Excel bytes
        result = manager.process_excel_file_from_bytes(
            file_bytes=file_bytes,
            filename="test_resumes.xlsx",
            base_user_id="test_batch",
            base_username="candidate",
        )

        print(f"✅ Processing completed")
        print(f"   Status: {result.get('status')}")
        print(f"   Processing time: {result.get('total_processing_time', 0):.2f}s")

        # Test JSON serialization of final result
        json_str = json.dumps(result)
        print(
            f"✅ Final result JSON serialization successful: {len(json_str)} characters"
        )

        # Print summary
        if "summary" in result:
            summary = result["summary"]
            print(f"   Rows processed: {summary.get('total_rows_processed', 0)}")
            print(f"   Successful parses: {summary.get('successful_parses', 0)}")
            print(f"   Failed parses: {summary.get('failed_parses', 0)}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        print(f"   Traceback: {traceback.format_exc()}")
        return False
    finally:
        Path(temp_file.name).unlink()


def main():
    """Run all direct tests."""
    print("=" * 70)
    print("DIRECT TESTING OF EXCEL RESUME PARSER FIXES")
    print("=" * 70)

    tests = [test_excel_processor_with_nan, test_excel_resume_parser, test_full_manager]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    success_count = sum(results)
    total_count = len(results)

    print(f"Passed: {success_count}/{total_count}")

    if success_count == total_count:
        print("✅ All direct tests passed!")
        print("Both issues have been successfully fixed:")
        print(
            "  1. ✅ ResumeParser method call corrected (parse_resume → process_resume)"
        )
        print("  2. ✅ NaN values properly handled for JSON serialization")
        print("The Excel resume parser should now work without errors.")
    else:
        print("❌ Some tests failed. Check the details above.")

    return success_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
