#!/usr/bin/env python3
"""
Excel Resume Parser Test Script
This script demonstrates how to correctly upload Excel files for resume processing.
"""

import requests
import json
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
EXCEL_ENDPOINT = f"{BASE_URL}/excel-resume-parser/upload"
TEST_EXCEL_FILE = "example_resumes.xlsx"


def test_excel_upload():
    """Test uploading an Excel file to the correct endpoint."""

    # Check if test file exists
    if not os.path.exists(TEST_EXCEL_FILE):
        print(f"❌ Test file '{TEST_EXCEL_FILE}' not found!")
        print("Please make sure the example Excel file is in the current directory.")
        return False

    print(f"📁 Found test file: {TEST_EXCEL_FILE}")

    # Prepare the request
    try:
        with open(TEST_EXCEL_FILE, "rb") as file:
            files = {
                "file": (
                    TEST_EXCEL_FILE,
                    file,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            }
            data = {"user_id": "test_user_123", "username": "test_candidate"}

            print(f"🚀 Sending request to: {EXCEL_ENDPOINT}")
            print(f"📝 Data: {data}")

            # Make the request
            response = requests.post(EXCEL_ENDPOINT, files=files, data=data, timeout=30)

            print(f"📡 Response Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print("✅ SUCCESS! Excel file processed successfully!")
                print(f"📊 Response: {json.dumps(result, indent=2)}")
                return True
            else:
                print(f"❌ ERROR! Status Code: {response.status_code}")
                print(f"Response: {response.text}")
                return False

    except requests.exceptions.ConnectionError:
        print("❌ Connection Error! Make sure the FastAPI server is running.")
        print("Start the server with: python main.py")
        return False
    except FileNotFoundError:
        print(f"❌ File not found: {TEST_EXCEL_FILE}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_excel_analyze():
    """Test analyzing an Excel file structure."""

    if not os.path.exists(TEST_EXCEL_FILE):
        print(f"❌ Test file '{TEST_EXCEL_FILE}' not found!")
        return False

    analyze_endpoint = f"{BASE_URL}/excel-resume-parser/analyze"

    try:
        with open(TEST_EXCEL_FILE, "rb") as file:
            files = {
                "file": (
                    TEST_EXCEL_FILE,
                    file,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            }

            print(f"🔍 Analyzing Excel file structure...")
            response = requests.post(analyze_endpoint, files=files, timeout=30)

            if response.status_code == 200:
                result = response.json()
                print("✅ Excel analysis successful!")
                print(f"📋 Structure: {json.dumps(result, indent=2)}")
                return True
            else:
                print(f"❌ Analysis failed! Status: {response.status_code}")
                print(f"Response: {response.text}")
                return False

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False


def test_queue_status():
    """Test getting queue status."""

    queue_endpoint = f"{BASE_URL}/excel-resume-parser/queue-status"

    try:
        response = requests.get(queue_endpoint, timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("✅ Queue status retrieved!")
            print(f"📊 Status: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ Queue status failed! Status: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Queue status error: {e}")
        return False


def main():
    """Main test function."""

    print("🧪 Excel Resume Parser Test Script")
    print("=" * 50)

    print("\n1. Testing Excel file analysis...")
    test_excel_analyze()

    print("\n2. Testing queue status...")
    test_queue_status()

    print("\n3. Testing Excel file upload and processing...")
    success = test_excel_upload()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed! Excel Resume Parser is working correctly.")
        print("\n💡 Usage Tips:")
        print("- Use POST /excel-resume-parser/upload for Excel files")
        print("- Use POST /multiple-resume-parser/upload for PDF/DOCX files")
        print("- Check /docs for complete API documentation")
    else:
        print("❌ Tests failed! Please check the server and file paths.")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure FastAPI server is running: python main.py")
        print("2. Check if example_resumes.xlsx exists in current directory")
        print("3. Verify server is accessible at http://localhost:8000")


if __name__ == "__main__":
    main()
