"""
Example usage of the updated Multiple Resume Parser API

This shows how to call the API with user_id and username
"""

import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"  # Replace with your actual API URL
MULTIPLE_RESUME_ENDPOINT = f"{BASE_URL}/resume-parser-multiple"
BULK_RESUME_ENDPOINT = f"{BASE_URL}/resume-parser-bulk"


def upload_multiple_resumes_example():
    """
    Example of how to upload multiple resumes with user details
    """

    # User details that will be assigned to all resumes
    user_data = {
        "user_id": "66c8771a20bd68c725758679",  # Your user ID
        "username": "Harsh Gajera",  # Your username
        "max_concurrent": 10,  # Optional: concurrent processing threads
    }

    # Files to upload (replace with actual file paths)
    files = [
        (
            "files",
            ("resume1.pdf", open("path/to/resume1.pdf", "rb"), "application/pdf"),
        ),
        (
            "files",
            (
                "resume2.docx",
                open("path/to/resume2.docx", "rb"),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
        ),
        ("files", ("resume3.txt", open("path/to/resume3.txt", "rb"), "text/plain")),
    ]

    try:
        # Make the API call
        response = requests.post(
            MULTIPLE_RESUME_ENDPOINT,
            data=user_data,  # Form data with user details
            files=files,
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Message: {result['message']}")
            print(
                f"Processing Statistics: {json.dumps(result['processing_statistics'], indent=2)}"
            )
            print(f"Processing Time: {json.dumps(result['processing_time'], indent=2)}")
            print(
                f"Queue Information: {json.dumps(result['queue_information'], indent=2)}"
            )
            print(
                f"Successfully Parsed: {result['processing_statistics']['successfully_parsed']}"
            )
            print(f"Failed: {result['processing_statistics']['failed_to_parse']}")
            print(
                f"Success Rate: {result['processing_statistics']['parsing_success_rate']}"
            )
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")
    finally:
        # Close file handles
        for _, file_tuple in files:
            if len(file_tuple) > 1:
                file_tuple[1].close()


def upload_bulk_resumes_example():
    """
    Example of how to upload bulk resumes with optimized settings
    """

    # User details and bulk processing settings
    user_data = {
        "user_id": "66c8771a20bd68c725758679",  # Your user ID
        "username": "Harsh Gajera",  # Your username
        "max_concurrent": 15,  # Optional: concurrent processing threads
        "batch_size": 25,  # Optional: database batch size
    }

    # Multiple files for bulk processing
    files = []
    for i in range(1, 11):  # Example: 10 resumes
        files.append(
            (
                "files",
                (
                    f"resume{i}.pdf",
                    open(f"path/to/resume{i}.pdf", "rb"),
                    "application/pdf",
                ),
            )
        )

    try:
        # Make the bulk API call
        response = requests.post(
            BULK_RESUME_ENDPOINT,
            data=user_data,  # Form data with user details
            files=files,
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Bulk processing success!")
            print(f"Message: {result['message']}")
            print(
                f"Bulk Statistics: {json.dumps(result['bulk_processing_statistics'], indent=2)}"
            )
            print(f"Processing Time: {json.dumps(result['processing_time'], indent=2)}")
            print(
                f"Queue Information: {json.dumps(result['queue_information'], indent=2)}"
            )
            print(
                f"Successfully Parsed: {result['bulk_processing_statistics']['successfully_parsed']}"
            )
            print(f"Failed: {result['bulk_processing_statistics']['failed_to_parse']}")
            print(
                f"Success Rate: {result['bulk_processing_statistics']['parsing_success_rate']}"
            )
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")
    finally:
        # Close file handles
        for _, file_tuple in files:
            if len(file_tuple) > 1:
                file_tuple[1].close()


def get_queue_status():
    """
    Get current processing queue status
    """
    try:
        response = requests.get(f"{BASE_URL}/queue-status")
        if response.status_code == 200:
            queue_info = response.json()
            print("📊 Queue Status:")
            print(json.dumps(queue_info, indent=2))
            return queue_info
        else:
            print(f"❌ Error getting queue status: {response.status_code}")
    except Exception as e:
        print(f"❌ Exception: {e}")


def get_api_info():
    """
    Get information about the API endpoints
    """
    try:
        response = requests.get(f"{BASE_URL}/info")
        if response.status_code == 200:
            info = response.json()
            print("📋 API Information:")
            print(json.dumps(info, indent=2))
        else:
            print(f"❌ Error getting API info: {response.status_code}")
    except Exception as e:
        print(f"❌ Exception: {e}")


if __name__ == "__main__":
    print("Multiple Resume Parser API Examples")
    print("=" * 50)

    # Get API information
    print("\n1. Getting API Information:")
    get_api_info()

    # Check queue status
    print("\n2. Checking Queue Status:")
    get_queue_status()

    # Example 1: Standard multiple resume upload
    print("\n3. Standard Multiple Resume Upload:")
    print("Note: Replace file paths with actual resume files")
    # upload_multiple_resumes_example()  # Uncomment when you have actual files

    # Example 2: Bulk resume upload
    print("\n4. Bulk Resume Upload:")
    print("Note: Replace file paths with actual resume files")
    # upload_bulk_resumes_example()  # Uncomment when you have actual files

    print("\n📋 Enhanced Features Added:")
    print("- Detailed processing statistics with success/failure rates")
    print("- Processing time tracking (start, end, duration)")
    print("- Queue management and status monitoring")
    print("- Session ID tracking for individual requests")
    print("- Enhanced error handling with timing information")
    print("- New /queue-status endpoint for monitoring")
    print("\n✨ All resumes will be assigned to the provided user_id and username!")
