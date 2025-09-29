#!/usr/bin/env python3
"""
Test script to verify the complete Excel parser processing pipeline

This script tests the entire Excel resume parsing flow from file upload
to successful processing without the column mapper method error.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from excel_resume_parser.enhanced_excel_resume_parser import EnhancedExcelResumeParser
from excel_resume_parser.excel_processor import ExcelProcessor
import pandas as pd
import tempfile
import json


def create_test_excel_file():
    """Create a test Excel file for parsing."""
    print("Creating test Excel file...")

    # Test data
    test_data = {
        "Name": ["John Smith", "Jane Doe", "Bob Johnson"],
        "Email": [
            "john.smith@email.com",
            "jane.doe@email.com",
            "bob.johnson@email.com",
        ],
        "Phone": ["+1-555-0123", "+1-555-0456", "+1-555-0789"],
        "Experience": ["5 years", "3 years", "7 years"],
        "Skills": [
            "Python, Java, SQL",
            "React, Node.js, MongoDB",
            "C++, Python, Docker",
        ],
        "Location": ["New York, NY", "San Francisco, CA", "Austin, TX"],
        "Current Salary": ["80000", "75000", "90000"],
        "Expected Salary": ["95000", "85000", "105000"],
        "Notice Period": ["2 weeks", "1 month", "2 weeks"],
        "Education": [
            "BS Computer Science",
            "MS Software Engineering",
            "BS Information Technology",
        ],
        "College": ["MIT", "Stanford", "UT Austin"],
    }

    df = pd.DataFrame(test_data)

    # Create temporary Excel file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    df.to_excel(temp_file.name, index=False)
    temp_file.close()

    print(f"✅ Test Excel file created: {temp_file.name}")
    return temp_file.name


def test_excel_processor():
    """Test the Excel processor component."""
    print("\n=== Testing Excel Processor ===")

    test_file = create_test_excel_file()

    try:
        processor = ExcelProcessor()
        result = processor.process_excel_file(test_file)

        print(f"✅ ExcelProcessor processed successfully")
        print(f"   - Returned type: {type(result)}")
        print(
            f"   - Number of records: {len(result) if isinstance(result, list) else 'N/A'}"
        )

        if isinstance(result, list) and len(result) > 0:
            print(f"   - First record keys: {list(result[0].keys())}")
            return True, test_file, result
        else:
            print("❌ Result is not a list or is empty")
            return False, test_file, None

    except Exception as e:
        print(f"❌ ExcelProcessor failed: {str(e)}")
        return False, test_file, None


def test_enhanced_excel_parser(test_file):
    """Test the enhanced Excel resume parser."""
    print("\n=== Testing Enhanced Excel Resume Parser ===")

    try:
        # Initialize parser
        print("1. Initializing EnhancedExcelResumeParser...")
        parser = EnhancedExcelResumeParser(llm_provider="groq_cloud")
        print("✅ Parser initialized successfully")

        # Test the processing pipeline
        print("2. Testing enhanced processing pipeline...")
        result = parser.process_excel_file(test_file)

        print("✅ Enhanced processing completed successfully!")
        print(f"   - Processing result type: {type(result)}")

        # Check result structure
        if isinstance(result, dict):
            print("3. Analyzing processing result:")
            print(f"   - Total rows: {result.get('total_rows', 'N/A')}")
            print(
                f"   - Successfully processed: {result.get('successfully_processed', 'N/A')}"
            )
            print(f"   - Failed rows: {len(result.get('failed_rows', []))}")

            # Check column analysis
            column_analysis = result.get("column_analysis", {})
            if column_analysis:
                print(
                    f"   - Original columns: {len(column_analysis.get('original_columns', []))}"
                )
                print(
                    f"   - Mapped fields: {len(column_analysis.get('mapped_fields', {}))}"
                )
                print(
                    f"   - Unmapped columns: {len(column_analysis.get('unmapped_columns', []))}"
                )

                # Show some mappings
                mapped_fields = column_analysis.get("mapped_fields", {})
                if mapped_fields:
                    print("   - Sample mappings:")
                    for i, (orig, mapped) in enumerate(mapped_fields.items()):
                        if i < 5:  # Show first 5
                            confidence = column_analysis.get(
                                "mapping_confidence", {}
                            ).get(orig, 0)
                            print(
                                f"     '{orig}' → '{mapped}' (confidence: {confidence:.3f})"
                            )

            # Check parsed resumes
            parsed_resumes = result.get("parsed_resumes", [])
            print(f"   - Parsed resumes: {len(parsed_resumes)}")

            return True

        else:
            print(f"❌ Unexpected result type: {type(result)}")
            return False

    except Exception as e:
        print(f"❌ Enhanced Excel parser failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_column_mapping_integration():
    """Test the column mapping integration specifically."""
    print("\n=== Testing Column Mapping Integration ===")

    try:
        parser = EnhancedExcelResumeParser()

        # Test column mapping directly
        test_columns = ["Name", "Email", "Phone", "Skills", "Experience"]
        mapping_result = parser.column_mapper.map_columns(test_columns)

        print(f"✅ Column mapping integration works")
        print(f"   - Mapped {len(mapping_result)} columns")

        # Test the data structure we use in enhanced_excel_resume_parser
        mapped_fields = {}
        confidence_scores = {}
        unmapped_columns = []

        for col_name, mapping_info in mapping_result.items():
            if mapping_info["mapped_field"]:
                mapped_fields[col_name] = mapping_info["mapped_field"]
                confidence_scores[col_name] = mapping_info["confidence"]
            else:
                unmapped_columns.append(col_name)

        print(f"   - Data structure transformation successful")
        print(f"   - Mapped: {len(mapped_fields)}, Unmapped: {len(unmapped_columns)}")

        return True

    except Exception as e:
        print(f"❌ Column mapping integration failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("Testing Complete Excel Parser Pipeline\n")

    # Test 1: Excel processor
    success1, test_file, processor_result = test_excel_processor()

    # Test 2: Column mapping integration
    success2 = test_column_mapping_integration()

    # Test 3: Enhanced Excel parser (if processor works)
    success3 = False
    if success1 and test_file:
        success3 = test_enhanced_excel_parser(test_file)

    # Cleanup
    if success1 and test_file:
        try:
            os.unlink(test_file)
            print(f"\n✅ Cleaned up test file: {test_file}")
        except:
            pass

    print(f"\n=== Final Test Summary ===")
    print(f"Excel Processor: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Column Mapping Integration: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"Enhanced Excel Parser: {'✅ PASS' if success3 else '❌ FAIL'}")

    if success1 and success2 and success3:
        print("\n🎉 ALL TESTS PASSED! Excel parser pipeline is working correctly.")
        print("🎉 The method name error has been fixed and the system is operational!")
    else:
        print(f"\n❌ Some tests failed. Please check the issues above.")
        sys.exit(1)
