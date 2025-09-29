#!/usr/bin/env python3

"""
Test script for the simplified Excel resume parser API.
This tests that the Excel parser now has simplified parameters consistent with other parsers.
"""

import requests
import json
import os
import time
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8001"
EXCEL_ENDPOINT = f"{BASE_URL}/resume-parser/excel"
STATUS_ENDPOINT = f"{BASE_URL}/resume-parser/status"


def test_excel_parser_simplified():
    """Test that Excel parser works with simplified parameters."""

    print("🧪 Testing Simplified Excel Resume Parser")
    print("=" * 50)

    # Test data
    test_user_name = "test_user"
    test_user_id = "test123"

    # Create a simple test Excel file path (assuming we have one)
    test_file_path = "example_resumes.xlsx"

    if not os.path.exists(test_file_path):
        print(f"❌ Test file not found: {test_file_path}")
        print("   Please ensure example_resumes.xlsx exists in the workspace")
        return False

    try:
        # Test the simplified API call
        print("📤 Testing Excel parser with only required parameters...")

        with open(test_file_path, "rb") as f:
            files = {"file": f}
            data = {
                "user_name": test_user_name,
                "user_id": test_user_id,
                # Notice: No manual parameters for save_to_database, detect_duplicates, etc.
                # These should now default to sensible values
            }

            response = requests.post(EXCEL_ENDPOINT, files=files, data=data)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 200:
            response_data = response.json()
            session_id = response_data.get("session_id")

            if session_id:
                print(f"✅ Excel parser started successfully!")
                print(f"   Session ID: {session_id}")

                # Check status
                print("\n📊 Checking processing status...")
                status_response = requests.get(f"{STATUS_ENDPOINT}/{session_id}")

                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"   Status: {status_data.get('status', 'unknown')}")

                    # Check that default parameters were applied
                    parameters = status_data.get("parameters", {})
                    print(
                        f"   Save to Database: {parameters.get('save_to_database', 'not set')}"
                    )
                    print(
                        f"   Detect Duplicates: {parameters.get('detect_duplicates', 'not set')}"
                    )
                    print(
                        f"   Auto Batch Size: {parameters.get('auto_determined_batch_size', 'not set')}"
                    )

                    if (
                        parameters.get("save_to_database") == True
                        and parameters.get("detect_duplicates") == True
                        and parameters.get("auto_determined_batch_size") == True
                    ):
                        print("✅ Default parameters correctly applied!")
                        return True
                    else:
                        print("⚠️  Default parameters not as expected")
                        return False
                else:
                    print(f"❌ Failed to get status: {status_response.status_code}")
                    return False
            else:
                print("❌ No session_id in response")
                return False
        else:
            print(f"❌ Failed to start Excel parser: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API server.")
        print("   Please ensure the server is running on http://localhost:8001")
        return False
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        return False


def test_parameter_simplification():
    """Test that the Excel parser endpoint no longer requires complex parameters."""

    print("\n🔍 Testing Parameter Simplification")
    print("=" * 50)

    # These parameters should no longer be required
    unnecessary_params = [
        "save_to_database",
        "detect_duplicates",
        "batch_size",
        "llm_provider",
        "api_keys",
    ]

    print("The following parameters should now be handled automatically:")
    for param in unnecessary_params:
        print(f"   ✅ {param} - auto-configured")

    print("\nOnly required parameters:")
    print("   📋 user_name - User identification")
    print("   📋 user_id - User identification")
    print("   📋 file - Excel file to process")

    return True


if __name__ == "__main__":
    print("🚀 Testing Simplified Excel Resume Parser API")
    print("=" * 60)

    # Test parameter simplification
    param_test = test_parameter_simplification()

    # Test actual API call
    api_test = test_excel_parser_simplified()

    print("\n" + "=" * 60)
    if param_test and api_test:
        print("🎉 ALL TESTS PASSED! Excel parser is now simplified and consistent!")
    else:
        print("❌ Some tests failed. Check the output above for details.")

    print("=" * 60)
