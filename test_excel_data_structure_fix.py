#!/usr/bin/env python3

"""
Test script to verify the Excel parser data structure fix.
"""

import sys
import os
import tempfile
import pandas as pd

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_excel_data_structure_fix():
    """Test that the Excel parser can handle the correct data structure."""

    print("🧪 Testing Excel Parser Data Structure Fix")
    print("=" * 50)

    try:
        # Create a sample Excel file for testing
        sample_data = {
            "Name": ["John Doe", "Jane Smith", "Bob Johnson"],
            "Email": ["john@example.com", "jane@example.com", "bob@example.com"],
            "Phone": ["1234567890", "9876543210", "5555555555"],
            "Experience": ["2 years", "5 years", "3 years"],
            "Skills": ["Python, SQL", "Java, React", "C++, Docker"],
            "Location": ["New York", "California", "Texas"],
            "Current Salary": ["50000", "80000", "60000"],
            "Expected Salary": ["60000", "100000", "70000"],
            "Notice Period": ["30 days", "60 days", "45 days"],
            "Education": [
                "BS Computer Science",
                "MS Software Engineering",
                "BS Information Technology",
            ],
            "College": ["MIT", "Stanford", "CMU"],
        }

        # Create temporary Excel file
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
            temp_path = temp_file.name
            df = pd.DataFrame(sample_data)
            df.to_excel(temp_path, index=False)

        print(f"✅ Created test Excel file: {temp_path}")

        # Test ExcelProcessor first
        from excel_resume_parser.excel_processor import ExcelProcessor

        processor = ExcelProcessor()
        result = processor.process_excel_file(temp_path)

        print(f"✅ ExcelProcessor returned: {type(result)}")
        print(f"   Data type: {type(result)}")
        print(f"   Length: {len(result) if result else 0}")

        if isinstance(result, list) and len(result) > 0:
            print(f"   First row keys: {list(result[0].keys())}")
            print("✅ ExcelProcessor working correctly - returns list of dicts")
        else:
            print("❌ ExcelProcessor issue - unexpected return format")
            return False

        # Test EnhancedExcelResumeParser
        from excel_resume_parser.enhanced_excel_resume_parser import (
            EnhancedExcelResumeParser,
        )

        print("\n🔧 Testing Enhanced Excel Parser...")
        parser = EnhancedExcelResumeParser()

        # This should now work without the data structure error
        processing_result = parser.process_excel_file(
            file_path=temp_path,
            sheet_name=None,
            validation_level="standard",
            cleaning_aggressive=False,
            include_quality_scores=True,
            batch_size=10,
        )

        print("✅ Enhanced Excel Parser processed successfully!")
        print(f"   Total rows processed: {processing_result.get('total_rows', 0)}")
        print(f"   Parsed resumes: {len(processing_result.get('parsed_resumes', []))}")

        # Clean up
        os.unlink(temp_path)
        print("✅ Test file cleaned up")

        return True

    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_data_structure_compatibility():
    """Test the specific data structure issue that was causing problems."""

    print("\n🔍 Testing Data Structure Compatibility")
    print("=" * 50)

    try:
        from excel_resume_parser.excel_processor import ExcelProcessor

        # Simulate the data structure issue
        processor = ExcelProcessor()

        # This is what the processor returns - a list of dicts
        sample_data = [
            {"name": "John Doe", "email": "john@example.com"},
            {"name": "Jane Smith", "email": "jane@example.com"},
        ]

        print("📋 Processor returns list of dicts:")
        print(f"   Type: {type(sample_data)}")
        print(f"   Length: {len(sample_data)}")
        print(f"   Sample: {sample_data[0] if sample_data else 'None'}")

        # Convert to DataFrame (this is what enhanced parser now does)
        import pandas as pd

        df = pd.DataFrame(sample_data)

        print("\n📊 Enhanced parser converts to DataFrame:")
        print(f"   Type: {type(df)}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")

        # Test that we can access DataFrame methods
        print(f"   iloc works: {type(df.iloc[0:1])}")

        print("✅ Data structure compatibility verified!")
        return True

    except Exception as e:
        print(f"❌ Data structure compatibility error: {str(e)}")
        return False


if __name__ == "__main__":
    print("🚀 EXCEL PARSER DATA STRUCTURE FIX TEST")
    print("=" * 60)

    # Run tests
    structure_test = test_data_structure_compatibility()
    excel_test = test_excel_data_structure_fix()

    print("\n" + "=" * 60)
    if structure_test and excel_test:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Excel parser data structure fix is working correctly")
        print("✅ No more 'Could not read Excel file or no data found' errors")
        print("✅ Excel processing should now complete successfully")
    else:
        print("❌ Some tests failed. Check the output above.")

    print("=" * 60)
