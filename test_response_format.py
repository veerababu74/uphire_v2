#!/usr/bin/env python3
"""
Test Excel Response Format

Test that sample_data and original_data are removed from the response.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from excel_resume_parser.main import ExcelResumeParserManager
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("test_response_format")


def test_response_format():
    """Test that response doesn't include sample_data and original_data."""
    print("=== Testing Response Format ===")

    try:
        # Create test Excel file with sample data
        data = {
            "Candidate Name": ["John Doe"],
            "Mobile No": ["1234567890"],
            "Email": ["john@example.com"],
            "Key Skills": ["Python, Java"],
            "Total Experience": [5],
            "Current Salary": [75000],
        }

        df = pd.DataFrame(data)

        # Save to temporary Excel file
        temp_file = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        temp_file.close()
        df.to_excel(temp_file.name, index=False)

        # Read file as bytes
        with open(temp_file.name, "rb") as f:
            file_bytes = f.read()

        # Test manager
        manager = ExcelResumeParserManager()

        # Process Excel bytes
        result = manager.process_excel_file_from_bytes(
            file_bytes=file_bytes,
            filename="test_resumes.xlsx",
            base_user_id="test_batch",
            base_username="candidate",
        )

        print(f"✅ Processing completed successfully")

        # Check that sample_data is NOT in excel_processing
        if "sample_data" in result.get("excel_processing", {}):
            print(f"❌ sample_data found in excel_processing")
            return False
        else:
            print(f"✅ sample_data removed from excel_processing")

        # Check that original_data is NOT in parsed resumes
        parsed_resumes = result.get("resume_parsing", {}).get("parsed_resumes", [])

        has_original_data = False
        for resume_data in parsed_resumes:
            if "original_data" in resume_data:
                has_original_data = True
                break

        if has_original_data:
            print(f"❌ original_data found in parsed resumes")
            return False
        else:
            print(f"✅ original_data removed from parsed resumes")

        # Verify JSON serialization still works
        json_str = json.dumps(result)
        print(f"✅ JSON serialization successful: {len(json_str)} characters")

        # Print a summary of what's included
        print(f"\nResponse structure:")
        print(
            f"  - excel_processing keys: {list(result.get('excel_processing', {}).keys())}"
        )
        if parsed_resumes:
            print(f"  - parsed_resume keys: {list(parsed_resumes[0].keys())}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        print(f"   Traceback: {traceback.format_exc()}")
        return False
    finally:
        try:
            Path(temp_file.name).unlink()
        except:
            pass


def main():
    """Run the response format test."""
    print("=" * 60)
    print("TESTING EXCEL RESPONSE FORMAT - NO SAMPLE/ORIGINAL DATA")
    print("=" * 60)

    success = test_response_format()

    print("\n" + "=" * 60)
    if success:
        print("✅ Response format test passed!")
        print("Both sample_data and original_data have been successfully removed.")
    else:
        print("❌ Response format test failed.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
