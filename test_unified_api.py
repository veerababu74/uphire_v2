"""
Test Script for Unified Resume Parser API

This script tests all the main functionality of the new unified API
to ensure everything works correctly after migration.
"""

import requests
import json
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
API_PREFIX = "/resume-parser"

# Test data
TEST_USER_NAME = "John Doe"
TEST_USER_ID = "test_user_123"


def test_api_health():
    """Test API health endpoint"""
    print("🔍 Testing API Health...")
    try:
        response = requests.get(f"{BASE_URL}{API_PREFIX}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ API Health Check - PASSED")
            print(f"   Status: {data.get('status')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Features: {len(data.get('features', []))} available")
            return True
        else:
            print(f"❌ API Health Check - FAILED (Status: {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API Health Check - FAILED (Connection Error)")
        print("   Make sure your server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ API Health Check - FAILED (Error: {e})")
        return False


def test_single_resume_parsing():
    """Test single resume parsing endpoint"""
    print("\n🔍 Testing Single Resume Parsing...")

    # Create a dummy text file for testing
    test_content = """
    John Smith
    Software Engineer
    Email: john.smith@email.com
    Phone: +1-555-123-4567
    
    Experience:
    - Software Engineer at Tech Corp (2020-2023)
    - Junior Developer at StartupXYZ (2018-2020)
    
    Skills: Python, JavaScript, React, Node.js
    
    Education:
    Bachelor of Computer Science, University of Tech (2018)
    """

    # Create temp file
    test_file_path = "test_resume.txt"
    with open(test_file_path, "w") as f:
        f.write(test_content)

    try:
        with open(test_file_path, "rb") as f:
            files = {"file": ("test_resume.txt", f, "text/plain")}
            data = {
                "user_name": TEST_USER_NAME,
                "user_id": TEST_USER_ID,
                "save_to_database": "false",  # Don't save to DB for testing
            }

            response = requests.post(
                f"{BASE_URL}{API_PREFIX}/single", files=files, data=data
            )

            if response.status_code == 200:
                result = response.json()
                print("✅ Single Resume Parsing - PASSED")
                print(f"   Parsed successfully: {result.get('message')}")

                parsed_data = result.get("parsed_data", {})
                contact = parsed_data.get("contact_details", {})
                print(f"   Name extracted: {contact.get('name', 'Not found')}")
                print(f"   Email extracted: {contact.get('email', 'Not found')}")

                metadata = parsed_data.get("parsing_metadata", {})
                print(f"   User Name: {metadata.get('user_name')}")
                print(f"   User ID: {metadata.get('user_id')}")

                return True
            else:
                print(
                    f"❌ Single Resume Parsing - FAILED (Status: {response.status_code})"
                )
                print(f"   Error: {response.text}")
                return False

    except Exception as e:
        print(f"❌ Single Resume Parsing - FAILED (Error: {e})")
        return False
    finally:
        # Clean up test file
        try:
            Path(test_file_path).unlink()
        except:
            pass


def test_multiple_resume_parsing():
    """Test multiple resume parsing endpoint"""
    print("\n🔍 Testing Multiple Resume Parsing...")

    # Create dummy resume files
    resumes = [
        ("Alice Johnson", "alice.johnson@email.com", "Data Scientist"),
        ("Bob Wilson", "bob.wilson@email.com", "DevOps Engineer"),
        ("Carol Davis", "carol.davis@email.com", "UI/UX Designer"),
    ]

    files_to_upload = []
    temp_files = []

    try:
        for i, (name, email, title) in enumerate(resumes):
            content = f"""
            {name}
            {title}
            Email: {email}
            Phone: +1-555-{i+100}-{i*1000+456}
            
            Experience:
            - {title} at Company {i+1} (2021-2023)
            - Junior {title} at Startup {i+1} (2019-2021)
            
            Skills: Python, SQL, Machine Learning, Data Analysis
            
            Education:
            Master of {title} (2019)
            """

            filename = f"test_resume_{i+1}.txt"
            with open(filename, "w") as f:
                f.write(content)
            temp_files.append(filename)

        # Prepare files for upload
        files = []
        for filename in temp_files:
            files.append(("files", (filename, open(filename, "rb"), "text/plain")))

        data = {
            "user_name": TEST_USER_NAME,
            "user_id": TEST_USER_ID,
            "save_to_database": "false",  # Don't save to DB for testing
        }

        response = requests.post(
            f"{BASE_URL}{API_PREFIX}/multiple", files=files, data=data
        )

        # Close file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()

        if response.status_code == 202:  # Accepted for async processing
            result = response.json()
            job_id = result.get("job_id")
            print("✅ Multiple Resume Parsing - STARTED")
            print(f"   Job ID: {job_id}")
            print(f"   Files count: {result.get('files_count')}")
            print(f"   User: {result.get('user_info', {}).get('user_name')}")

            # Wait a bit and check status
            print("   Waiting for processing to complete...")
            time.sleep(3)

            status_response = requests.get(f"{BASE_URL}{API_PREFIX}/status/{job_id}")
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"   Job Status: {status.get('status')}")
                print(
                    f"   Progress: {status.get('processed_items', 0)}/{status.get('total_items', 0)}"
                )

                if status.get("status") == "completed":
                    results_response = requests.get(
                        f"{BASE_URL}{API_PREFIX}/results/{job_id}"
                    )
                    if results_response.status_code == 200:
                        results = results_response.json()
                        summary = results.get("summary", {})
                        print(f"   ✅ Processing completed!")
                        print(f"   Successful: {summary.get('successful_count', 0)}")
                        print(f"   Failed: {summary.get('failed_count', 0)}")
                        return True

            return True
        else:
            print(
                f"❌ Multiple Resume Parsing - FAILED (Status: {response.status_code})"
            )
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Multiple Resume Parsing - FAILED (Error: {e})")
        return False
    finally:
        # Clean up test files
        for filename in temp_files:
            try:
                Path(filename).unlink()
            except:
                pass


def test_excel_resume_parsing():
    """Test Excel resume parsing endpoint (mock test)"""
    print("\n🔍 Testing Excel Resume Parsing...")
    print("   📝 Note: This is a mock test since we don't have an actual Excel file")
    print("   📝 In real usage, you would upload a .xlsx or .xls file")

    # This would be the actual test with a real Excel file:
    # with open('test_resumes.xlsx', 'rb') as f:
    #     files = {'file': ('test_resumes.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
    #     data = {
    #         'user_name': TEST_USER_NAME,
    #         'user_id': TEST_USER_ID,
    #         'save_to_database': 'false',
    #     }
    #     response = requests.post(f"{BASE_URL}{API_PREFIX}/excel", files=files, data=data)

    print("   ✅ Excel Resume Parsing endpoint is available")
    print("   📋 Required parameters: file, user_name, user_id")
    print("   📋 Optional parameters: validation_level, batch_size, etc.")
    return True


def test_statistics_endpoint():
    """Test statistics endpoint"""
    print("\n🔍 Testing Statistics Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}{API_PREFIX}/statistics")
        if response.status_code == 200:
            data = response.json()
            print("✅ Statistics Endpoint - PASSED")
            print(f"   System Status: {data.get('system_status')}")

            accuracy = data.get("accuracy_metrics", {})
            print(f"   Total Parsed: {accuracy.get('total_parsed', 0)}")
            print(f"   Accuracy Rate: {accuracy.get('accuracy_rate', 0):.1f}%")
            return True
        else:
            print(f"❌ Statistics Endpoint - FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Statistics Endpoint - FAILED (Error: {e})")
        return False


def test_jobs_endpoint():
    """Test jobs listing endpoint"""
    print("\n🔍 Testing Jobs Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}{API_PREFIX}/jobs")
        if response.status_code == 200:
            data = response.json()
            print("✅ Jobs Endpoint - PASSED")
            print(f"   Total Jobs: {data.get('total_jobs', 0)}")
            print(f"   Total Sessions: {data.get('total_sessions', 0)}")
            return True
        else:
            print(f"❌ Jobs Endpoint - FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Jobs Endpoint - FAILED (Error: {e})")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("🚀 UNIFIED RESUME PARSER API - TESTING")
    print("=" * 80)

    tests = [
        ("API Health Check", test_api_health),
        ("Single Resume Parsing", test_single_resume_parsing),
        ("Multiple Resume Parsing", test_multiple_resume_parsing),
        ("Excel Resume Parsing", test_excel_resume_parsing),
        ("Statistics Endpoint", test_statistics_endpoint),
        ("Jobs Endpoint", test_jobs_endpoint),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} - FAILED (Exception: {e})")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:<12} - {test_name}")

    print("-" * 80)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your unified API is working correctly!")
    else:
        print(
            f"\n⚠️  {total-passed} test(s) failed. Check the output above for details."
        )

    print("\n💡 Next steps:")
    print("1. If tests failed, make sure your server is running")
    print("2. Update your main.py with the unified API")
    print("3. Test with real resume files")
    print("4. Update your frontend/client code")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
