#!/usr/bin/env python3
"""
Test Excel to JSON Conversion API

This script demonstrates how to use the new Excel to JSON conversion endpoint
that converts Excel files to properly cleaned JSON format without resume parsing.
"""

import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys


def create_sample_excel_file():
    """Create a sample Excel file with mixed data types and some messy data."""
    print("Creating sample Excel file...")

    # Sample data with various issues that need cleaning
    data = {
        "Name": ["John Doe", "Jane Smith", "", "Bob Johnson", np.nan, "Alice Brown"],
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
        "Start Date": [
            "2023-01-15",
            "2022-06-30",
            "",
            "2021-12-01",
            "invalid",
            "2023-03-20",
        ],
        "Active": [True, False, "", True, None, "Yes"],
        "Notes": ["Good performer", "", "N/A", "Needs training", None, "Top talent"],
        "Score": [8.5, 9.2, "N/A", 7.8, np.nan, "8.0"],
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add some duplicate headers issue
    df.columns = [
        "Name",
        "Email",
        "Phone",
        "Age",
        "Salary",
        "Department",
        "Start Date",
        "Active",
        "Notes",
        "Score",
    ]

    # Save to temporary Excel file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    df.to_excel(temp_file.name, index=False)
    temp_file.close()

    print(f"Sample Excel file created: {temp_file.name}")
    print(f"Data shape: {df.shape}")
    print("Sample data:")
    print(df.head())

    return temp_file.name


def test_excel_to_json_api(excel_file_path):
    """Test the Excel to JSON conversion API."""
    print("\n" + "=" * 50)
    print("TESTING EXCEL TO JSON CONVERSION API")
    print("=" * 50)

    # API endpoint URL
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/resume-parser/excel-to-json"

    print(f"API Endpoint: {endpoint}")
    print(f"Excel file: {excel_file_path}")

    # Test 1: Basic conversion with default settings
    print("\nTest 1: Basic conversion with default settings")
    print("-" * 40)

    try:
        with open(excel_file_path, "rb") as f:
            files = {
                "file": (
                    "sample_data.xlsx",
                    f,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            }
            data = {"skip_empty_rows": True, "normalize_headers": True}

            response = requests.post(endpoint, files=files, data=data, timeout=30)

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print("✅ API request successful!")
                print(f"Status: {result.get('status')}")
                print(f"Filename: {result.get('filename')}")
                print(
                    f"Original rows: {result.get('statistics', {}).get('original_rows')}"
                )
                print(
                    f"Cleaned rows: {result.get('statistics', {}).get('cleaned_rows')}"
                )
                print(
                    f"Processing time: {result.get('statistics', {}).get('processing_time_seconds')}s"
                )

                print("\nFirst 3 rows of cleaned data:")
                data_rows = result.get("data", [])
                for i, row in enumerate(data_rows[:3]):
                    print(f"Row {i+1}: {json.dumps(row, indent=2)}")

                print(f"\nTotal data rows returned: {len(data_rows)}")

            else:
                print(f"❌ API request failed: {response.status_code}")
                print(f"Response: {response.text}")

    except Exception as e:
        print(f"❌ Error testing basic conversion: {e}")

    # Test 2: Conversion without header normalization
    print("\n\nTest 2: Conversion without header normalization")
    print("-" * 45)

    try:
        with open(excel_file_path, "rb") as f:
            files = {
                "file": (
                    "sample_data.xlsx",
                    f,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            }
            data = {"skip_empty_rows": True, "normalize_headers": False}

            response = requests.post(endpoint, files=files, data=data, timeout=30)

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print("✅ Non-normalized headers test successful!")
                data_rows = result.get("data", [])
                if data_rows:
                    print("Headers in data:")
                    print(list(data_rows[0].keys()))

            else:
                print(f"❌ Non-normalized headers test failed: {response.status_code}")

    except Exception as e:
        print(f"❌ Error testing non-normalized headers: {e}")

    # Test 3: Limited rows conversion
    print("\n\nTest 3: Limited rows conversion (max 3 rows)")
    print("-" * 42)

    try:
        with open(excel_file_path, "rb") as f:
            files = {
                "file": (
                    "sample_data.xlsx",
                    f,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            }
            data = {"skip_empty_rows": True, "normalize_headers": True, "max_rows": 3}

            response = requests.post(endpoint, files=files, data=data, timeout=30)

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print("✅ Limited rows test successful!")
                print(
                    f"Processed rows: {result.get('statistics', {}).get('cleaned_rows')}"
                )
                print(f"Total data returned: {len(result.get('data', []))}")

            else:
                print(f"❌ Limited rows test failed: {response.status_code}")

    except Exception as e:
        print(f"❌ Error testing limited rows: {e}")


def test_invalid_file():
    """Test API with invalid file."""
    print("\n\nTest 4: Invalid file handling")
    print("-" * 30)

    endpoint = "http://localhost:8000/resume-parser/excel-to-json"

    try:
        # Create a text file instead of Excel
        temp_txt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
        temp_txt.write("This is not an Excel file")
        temp_txt.close()

        with open(temp_txt.name, "rb") as f:
            files = {"file": ("invalid.txt", f, "text/plain")}
            data = {"skip_empty_rows": True, "normalize_headers": True}

            response = requests.post(endpoint, files=files, data=data, timeout=30)

            print(f"Status Code: {response.status_code}")

            if response.status_code == 400:
                print("✅ Invalid file correctly rejected!")
                print(f"Error message: {response.json().get('detail')}")
            else:
                print(f"❌ Invalid file not properly handled: {response.status_code}")

        # Clean up
        Path(temp_txt.name).unlink()

    except Exception as e:
        print(f"❌ Error testing invalid file: {e}")


def main():
    """Run all tests for the Excel to JSON conversion API."""
    print("🚀 Starting Excel to JSON Conversion API Tests")
    print("=" * 60)

    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code != 200:
            print("❌ Server doesn't seem to be running at http://localhost:8000")
            print("Please start the server first: python main.py")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Please start the server first: python main.py")
        return False

    print("✅ Server is running!")

    # Create sample Excel file
    excel_file_path = None
    try:
        excel_file_path = create_sample_excel_file()

        # Run tests
        test_excel_to_json_api(excel_file_path)
        test_invalid_file()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review the API response structure")
        print("2. Use the cleaned JSON data in your application")
        print("3. Integrate with your existing workflows")

        return True

    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

    finally:
        # Clean up
        if excel_file_path and Path(excel_file_path).exists():
            try:
                Path(excel_file_path).unlink()
                print(f"🧹 Cleaned up temporary file: {excel_file_path}")
            except Exception as e:
                print(f"⚠️ Could not clean up temporary file: {e}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
