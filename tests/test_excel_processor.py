"""
Simple Excel Resume Parser Test

Basic test script to verify Excel processing functionality without LLM dependencies.
"""

import sys
import os
import pandas as pd
import io
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from excel_resume_parser.excel_processor import ExcelProcessor


def test_excel_processor():
    """Test Excel processor functionality."""
    print("=== Testing Excel Processor ===")

    try:
        processor = ExcelProcessor()
        print(f"✅ Excel processor initialized successfully")

        # Test with sample DataFrame
        sample_data = {
            "Name": ["John Doe", "Jane Smith", "Bob Johnson"],
            "name": ["John Doe2", "Jane Smith2", "Bob Johnson2"],  # Duplicate column
            "Email": ["john@email.com", "jane@email.com", "bob@email.com"],
            "Phone": ["1234567890", "0987654321", "5555555555"],
            "Experience": ["5 years", "3 years", "7 years"],
            "Skills": ["Python, JavaScript", "Java, Spring", "React, Node.js"],
        }

        df = pd.DataFrame(sample_data)
        print(f"✅ Created sample DataFrame with shape: {df.shape}")
        print(f"   Original columns: {list(df.columns)}")

        # Test duplicate header detection
        df_cleaned, duplicates = processor.detect_and_remove_duplicate_headers(df)
        print(f"✅ Duplicate header detection completed")
        print(f"   Cleaned DataFrame shape: {df_cleaned.shape}")
        print(f"   Duplicate columns removed: {duplicates}")

        # Test column name cleaning
        df_final = processor.clean_column_names(df_cleaned)
        print(f"✅ Column name cleaning completed")
        print(f"   Final columns: {list(df_final.columns)}")

        # Test row conversion
        row_dicts = processor.convert_rows_to_dictionaries(df_final)
        print(f"✅ Row conversion completed")
        print(f"   Number of rows converted: {len(row_dicts)}")
        print(f"   Sample row: {row_dicts[0] if row_dicts else 'None'}")

        return True

    except Exception as e:
        print(f"❌ Excel processor test failed: {e}")
        return False


def test_excel_bytes_processing():
    """Test Excel processing from bytes."""
    print("\n=== Testing Excel Bytes Processing ===")

    try:
        processor = ExcelProcessor()

        # Create sample Excel data in memory
        sample_data = {
            "candidate_name": ["Alice Johnson", "Bob Wilson"],
            "email_address": ["alice@example.com", "bob@example.com"],
            "phone_number": ["555-1234", "555-5678"],
            "total_experience": ["7 years", "4 years"],
            "skills": ["Python, Django, AWS", "React, JavaScript, Node.js"],
            "current_city": ["New York", "San Francisco"],
        }

        df = pd.DataFrame(sample_data)

        # Convert to Excel bytes
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine="openpyxl")
        excel_bytes = excel_buffer.getvalue()

        print(f"✅ Created Excel bytes data ({len(excel_bytes)} bytes)")

        # Process from bytes
        row_dicts = processor.process_excel_bytes(
            file_bytes=excel_bytes, filename="test_candidates.xlsx"
        )

        print(f"✅ Excel bytes processing completed")
        print(f"   Number of rows processed: {len(row_dicts)}")
        print(f"   Sample row: {row_dicts[0] if row_dicts else 'None'}")

        return True

    except Exception as e:
        print(f"❌ Excel bytes processing test failed: {e}")
        return False


def test_column_mapping():
    """Test intelligent column mapping."""
    print("\n=== Testing Column Mapping ===")

    try:
        # Test data with various column name formats
        sample_data = {
            "Full Name": ["Test User 1"],
            "Email ID": ["test1@example.com"],
            "Contact Number": ["9876543210"],
            "Years of Experience": ["5 years"],
            "Technical Skills": ["Python, Java, SQL"],
            "Current Location": ["Mumbai"],
            "Current Salary (LPA)": ["12"],
            "Expected CTC": ["15"],
            "Notice Period": ["30 days"],
            "Qualification": ["B.Tech"],
            "College/University": ["IIT Delhi"],
        }

        df = pd.DataFrame(sample_data)

        processor = ExcelProcessor()

        # Clean column names
        df_cleaned = processor.clean_column_names(df)
        print(f"✅ Column name cleaning test")
        print(f"   Original: {list(df.columns)}")
        print(f"   Cleaned: {list(df_cleaned.columns)}")

        # Convert to dictionaries
        row_dicts = processor.convert_rows_to_dictionaries(df_cleaned)
        print(f"✅ Row conversion completed")
        print(f"   Sample data: {row_dicts[0] if row_dicts else 'None'}")

        return True

    except Exception as e:
        print(f"❌ Column mapping test failed: {e}")
        return False


def create_sample_excel_file():
    """Create a sample Excel file for testing."""
    print("\n=== Creating Sample Excel File ===")

    try:
        sample_data = {
            "Name": ["John Doe", "Jane Smith", "Mike Johnson"],
            "Email": ["john@example.com", "jane@example.com", "mike@example.com"],
            "Phone": ["1234567890", "9876543210", "5555555555"],
            "Experience": ["5 years", "3 years", "8 years"],
            "Skills": [
                "Python, Django, PostgreSQL",
                "Java, Spring Boot, MySQL",
                "React, Node.js, MongoDB",
            ],
            "Location": ["New York", "San Francisco", "Austin"],
            "Current Salary": ["120000", "95000", "140000"],
            "Expected Salary": ["140000", "110000", "160000"],
            "Notice Period": ["30 days", "Immediate", "45 days"],
            "Education": ["B.Tech CS", "MS CS", "B.Tech IT"],
            "College": ["MIT", "Stanford", "UC Berkeley"],
        }

        df = pd.DataFrame(sample_data)

        # Save to file
        output_path = Path("sample_resumes.xlsx")
        df.to_excel(output_path, index=False, engine="openpyxl")

        print(f"✅ Sample Excel file created: {output_path}")
        print(f"   File size: {output_path.stat().st_size} bytes")
        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")

        return str(output_path)

    except Exception as e:
        print(f"❌ Sample Excel file creation failed: {e}")
        return None


def test_sample_excel_file():
    """Test processing a real Excel file."""
    print("\n=== Testing Sample Excel File Processing ===")

    try:
        # Create sample file
        file_path = create_sample_excel_file()
        if not file_path:
            print("❌ Could not create sample file")
            return False

        processor = ExcelProcessor()

        # Test file validation
        is_valid = processor.validate_excel_file(file_path)
        print(f"✅ File validation: {is_valid}")

        if not is_valid:
            print("❌ File validation failed")
            return False

        # Test sheet names
        sheet_names = processor.get_sheet_names(file_path)
        print(f"✅ Sheet names: {sheet_names}")

        # Process the file
        row_dicts = processor.process_excel_file(file_path)
        print(f"✅ File processing completed")
        print(f"   Number of rows processed: {len(row_dicts)}")
        print(f"   Sample row: {row_dicts[0] if row_dicts else 'None'}")

        # Cleanup
        Path(file_path).unlink()
        print(f"✅ Cleaned up sample file")

        return True

    except Exception as e:
        print(f"❌ Sample Excel file test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Excel Processor - Test Suite")
    print("=" * 50)

    tests = [
        ("Excel Processor Basic", test_excel_processor),
        ("Excel Bytes Processing", test_excel_bytes_processing),
        ("Column Mapping", test_column_mapping),
        ("Sample Excel File", test_sample_excel_file),
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
        print("🎉 All Excel processor tests passed!")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
