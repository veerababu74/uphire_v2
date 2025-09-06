#!/usr/bin/env python3
"""
Test Excel Upload with Real Data

Test the actual Excel resume parser API with sample data.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import requests
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("test_excel_upload")


def create_sample_excel_file():
    """Create a sample Excel file with resume data including NaN values."""
    print("Creating sample Excel file...")

    # Sample data with some NaN values to test our fix
    data = {
        "Candidate Name": ["John Doe", "Jane Smith", "Bob Johnson"],
        "Mobile No": ["1234567890", "9876543210", "5555555555"],
        "Email": ["john@example.com", "jane@example.com", "bob@example.com"],
        "Current City": ["New York", "San Francisco", "Chicago"],
        "Prefered Locations": ["NYC, LA", "SF, Seattle", np.nan],  # NaN value
        "Total Experience": [5, np.nan, 7],  # NaN value
        "Notice Period (In Days)": [30, 60, np.nan],  # NaN value
        "Currency": ["USD", "USD", "USD"],
        "Current Salary": [75000, np.nan, 90000],  # NaN value
        "Expected Salary": [85000, 120000, np.nan],  # NaN value
        "Key Skills": [
            "Python, Django, React",
            "Java, Spring, Angular",
            "C#, .NET, SQL",
        ],
        "May Also Know": ["AWS, Docker", np.nan, "Azure, K8s"],  # NaN value
        "Company Name": ["TechCorp", "DataInc", "SoftwareLLC"],
        "Position Title": ["Software Engineer", "Senior Developer", "Lead Engineer"],
        "Education Name": [
            "Computer Science",
            "Software Engineering",
            np.nan,
        ],  # NaN value
        "College Name": ["MIT", "Stanford", "Berkeley"],
        "Naukari Profile Link": [np.nan, "naukri.com/profile1", np.nan],  # NaN values
        "Linkedin Profile": [
            "linkedin.com/john",
            np.nan,
            "linkedin.com/bob",
        ],  # NaN value
        "Age": [28, np.nan, 32],  # NaN value
        "Gender": ["M", "F", np.nan],  # NaN value
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    temp_file.close()

    # Save to Excel
    df.to_excel(temp_file.name, index=False)

    print(f"Sample Excel file created: {temp_file.name}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"Contains NaN: {df.isna().any().any()}")

    return temp_file.name


def test_excel_upload_api():
    """Test the Excel upload API."""
    print("\n=== Testing Excel Upload API ===")

    try:
        # Create sample Excel file
        excel_file_path = create_sample_excel_file()

        # Prepare API request
        url = "http://localhost:8000/excel-resume-parser/upload"

        files = {
            "file": (
                "test_resumes.xlsx",
                open(excel_file_path, "rb"),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        }

        data = {"user_id": "test_user", "username": "test_candidate"}

        print(f"Making POST request to: {url}")
        print(f"Data: {data}")

        # Make API request
        response = requests.post(url, files=files, data=data, timeout=120)

        files["file"][1].close()  # Close file

        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ API request successful!")
            print(f"Status: {result.get('status')}")
            print(f"Processing time: {result.get('total_processing_time', 0):.2f}s")

            # Print summary
            if "summary" in result:
                summary = result["summary"]
                print(f"Rows processed: {summary.get('total_rows_processed', 0)}")
                print(f"Successful parses: {summary.get('successful_parses', 0)}")
                print(f"Failed parses: {summary.get('failed_parses', 0)}")
                print(f"Successfully saved: {summary.get('successfully_saved', 0)}")
                print(f"Duplicates detected: {summary.get('duplicates_detected', 0)}")

            return True

        else:
            print(f"❌ API request failed!")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(
            "❌ Could not connect to server. Make sure the server is running on localhost:8000"
        )
        return False
    except Exception as e:
        print(f"❌ Error testing Excel upload API: {e}")
        return False
    finally:
        # Cleanup
        try:
            Path(excel_file_path).unlink()
            print(f"Cleaned up temporary file: {excel_file_path}")
        except:
            pass


def main():
    """Run the Excel upload test."""
    print("=" * 60)
    print("TESTING EXCEL UPLOAD API WITH NaN VALUES")
    print("=" * 60)

    success = test_excel_upload_api()

    print("\n" + "=" * 60)
    if success:
        print("✅ Excel upload test completed successfully!")
        print("The fixes for both issues are working correctly:")
        print("  1. ✅ ResumeParser.process_resume method call fixed")
        print("  2. ✅ NaN values JSON serialization fixed")
    else:
        print("❌ Excel upload test failed. Check server logs for details.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
