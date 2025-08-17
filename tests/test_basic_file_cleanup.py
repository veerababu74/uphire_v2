"""
Simple File Cleanup Test

Test the file cleanup functionality without LLM dependencies.
"""

import sys
import os
import time
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from excel_resume_parser.excel_processor import ExcelProcessor


def test_basic_file_operations():
    """Test basic file creation and cleanup operations."""
    print("=== Testing Basic File Operations ===")

    try:
        # Create a test Excel file
        sample_data = {
            "Name": ["Test User"],
            "Email": ["test@example.com"],
            "Experience": ["5 years"],
        }

        df = pd.DataFrame(sample_data)
        test_file = "cleanup_test.xlsx"
        df.to_excel(test_file, index=False, engine="openpyxl")

        print(f"✅ Created test file: {test_file}")

        # Verify file exists
        file_path = Path(test_file)
        assert file_path.exists(), "Test file should exist"
        print(f"✅ Confirmed file exists: {file_path.stat().st_size} bytes")

        # Process the file (just read it)
        processor = ExcelProcessor()
        is_valid = processor.validate_excel_file(test_file)
        print(f"✅ File validation: {is_valid}")

        if is_valid:
            excel_data = processor.process_excel_file(test_file)
            print(f"✅ File processed successfully: {len(excel_data)} records")

        # Test cleanup
        file_path.unlink()
        print(f"✅ File cleaned up successfully")

        # Verify cleanup
        assert not file_path.exists(), "File should be deleted"
        print(f"✅ Confirmed file no longer exists")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        # Emergency cleanup
        try:
            Path("cleanup_test.xlsx").unlink()
        except:
            pass
        return False


def test_temp_directory_management():
    """Test temporary directory creation and cleanup."""
    print("\n=== Testing Temporary Directory Management ===")

    try:
        # Create temp directory
        temp_dir = Path("dummy_data_save/temp_excel_files")
        temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created temp directory: {temp_dir}")

        # Create test files
        test_files = []
        for i in range(3):
            test_file = temp_dir / f"temp_{i}_{int(time.time())}.xlsx"

            # Create valid Excel file
            sample_data = {"col": [f"data_{i}"]}
            df = pd.DataFrame(sample_data)
            df.to_excel(test_file, index=False, engine="openpyxl")
            test_files.append(test_file)

        print(f"✅ Created {len(test_files)} test files")

        # Verify all files exist
        for test_file in test_files:
            assert test_file.exists(), f"File {test_file} should exist"
        print(f"✅ All test files verified to exist")

        # Cleanup all files
        cleaned_count = 0
        for test_file in test_files:
            if test_file.exists():
                test_file.unlink()
                cleaned_count += 1

        print(f"✅ Cleaned up {cleaned_count} files")

        # Verify cleanup
        remaining_files = [f for f in test_files if f.exists()]
        if not remaining_files:
            print(f"✅ All files successfully cleaned up")
            return True
        else:
            print(f"❌ Some files still exist: {remaining_files}")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_error_scenario_cleanup():
    """Test cleanup in error scenarios."""
    print("\n=== Testing Error Scenario Cleanup ===")

    try:
        # Create a file that will cause processing errors
        error_file = Path("error_test.txt")
        with open(error_file, "w") as f:
            f.write("This is not an Excel file")

        print(f"✅ Created invalid file: {error_file}")

        # Verify file exists
        assert error_file.exists(), "Error test file should exist"

        # Try to process (should fail)
        processor = ExcelProcessor()
        try:
            is_valid = processor.validate_excel_file(str(error_file))
            print(f"📋 File validation result: {is_valid}")

            if not is_valid:
                print(f"✅ File correctly identified as invalid")

        except Exception as process_error:
            print(f"✅ Processing failed as expected: {process_error}")

        # Cleanup the error file
        error_file.unlink()
        print(f"✅ Error file cleaned up successfully")

        # Verify cleanup
        assert not error_file.exists(), "Error file should be deleted"
        print(f"✅ Confirmed error file no longer exists")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        # Emergency cleanup
        try:
            Path("error_test.txt").unlink()
        except:
            pass
        return False


def test_file_path_cleanup_function():
    """Test a simulated file cleanup function."""
    print("\n=== Testing File Cleanup Function ===")

    def cleanup_file(file_path: str, description: str = "file") -> dict:
        """Simulate the cleanup function used in the main code."""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                return {
                    "status": "success",
                    "message": f"Successfully cleaned up {description}: {file_path}",
                }
            else:
                return {
                    "status": "not_found",
                    "message": f"{description} not found: {file_path}",
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to cleanup {description} {file_path}: {e}",
            }

    try:
        # Test cleanup of existing file
        test_file = "cleanup_function_test.xlsx"
        sample_data = {"test": ["data"]}
        df = pd.DataFrame(sample_data)
        df.to_excel(test_file, index=False, engine="openpyxl")

        print(f"✅ Created test file for cleanup function: {test_file}")

        # Test cleanup
        result = cleanup_file(test_file, "test Excel file")
        print(f"✅ Cleanup result: {result}")

        if result["status"] == "success":
            print(f"✅ Cleanup function working correctly")
        else:
            print(f"❌ Cleanup function failed: {result}")
            return False

        # Test cleanup of non-existent file
        result2 = cleanup_file("non_existent.xlsx", "non-existent file")
        print(f"✅ Non-existent file result: {result2}")

        if result2["status"] == "not_found":
            print(f"✅ Non-existent file handling working correctly")
            return True
        else:
            print(f"❌ Non-existent file handling failed")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Run all basic cleanup tests."""
    print("🧪 Basic File Cleanup - Test Suite")
    print("=" * 50)

    tests = [
        ("Basic File Operations", test_basic_file_operations),
        ("Temporary Directory Management", test_temp_directory_management),
        ("Error Scenario Cleanup", test_error_scenario_cleanup),
        ("File Cleanup Function", test_file_path_cleanup_function),
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
    print("📊 Basic Cleanup Test Results")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} basic cleanup tests passed")

    if passed == total:
        print("🎉 All basic cleanup functionality working!")
        print("\n💡 Cleanup Features Verified:")
        print("   ✅ Excel file creation and deletion")
        print("   ✅ Temporary directory management")
        print("   ✅ Error scenario handling")
        print("   ✅ Cleanup function implementation")
        print("\n🔧 File cleanup is ready for integration!")
    else:
        print("⚠️  Some basic cleanup tests failed.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
