#!/usr/bin/env python3
"""
Test script for the enhanced multiple resume parser API with automatic embedding generation.

This script demonstrates how to use the new multiple resume parser functionality
that automatically generates embeddings for parsed resume data.
"""

import os
import sys
import requests
import json
from pathlib import Path

# API base URL (adjust as needed)
BASE_URL = "http://localhost:8000"  # Adjust to your FastAPI server URL


def test_single_resume_parsing():
    """Test single resume parsing with embeddings."""
    print("Testing single resume parsing with automatic embedding generation...")

    # Create a sample resume file for testing
    sample_resume_content = """
John Doe
Email: john.doe@example.com
Phone: +1-555-123-4567
Location: New York, NY

EXPERIENCE:
Software Engineer at TechCorp (2020-2023)
- Developed web applications using Python and React
- Led team of 3 developers
- Implemented machine learning algorithms

Senior Developer at StartupXYZ (2018-2020)
- Built scalable microservices
- Worked with AWS and Docker

EDUCATION:
Bachelor of Computer Science, MIT (2014-2018)
Master of Computer Science, Stanford (2018-2020)

SKILLS:
Python, JavaScript, React, AWS, Docker, Machine Learning, SQL, MongoDB
"""

    # Save sample resume to a temporary file
    temp_file = Path("temp_resume.txt")
    with open(temp_file, "w") as f:
        f.write(sample_resume_content)

    try:
        # Test with different LLM providers
        providers = ["ollama", "groq"]  # Add more as needed

        for provider in providers:
            print(f"\n--- Testing with {provider} provider ---")

            # Prepare files for upload
            files = {"file": ("test_resume.txt", open(temp_file, "rb"), "text/plain")}
            data = {"llm_provider": provider} if provider != "default" else {}

            # Make API request
            try:
                response = requests.post(
                    f"{BASE_URL}/multiple_resume_parser/parse-single",
                    files=files,
                    data=data,
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Success with {provider}!")
                    print(
                        f"Embeddings generated: {result.get('embeddings_generated', False)}"
                    )
                    print(f"Provider used: {result.get('provider_used', 'unknown')}")

                    # Show parsed data structure
                    parsed_data = result.get("result", {}).get("parsed_data", {})
                    if parsed_data:
                        print(f"Name: {parsed_data.get('name', 'N/A')}")
                        print(f"Skills count: {len(parsed_data.get('skills', []))}")
                        print(
                            f"Experience count: {len(parsed_data.get('experience', []))}"
                        )
                        print(
                            f"Embeddings available: {list(parsed_data.get('embeddings', {}).keys())}"
                        )

                else:
                    print(f"❌ Error with {provider}: {response.status_code}")
                    print(response.text)

            except requests.exceptions.RequestException as e:
                print(f"❌ Request failed for {provider}: {e}")

            finally:
                files["file"][1].close()

    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()


def test_provider_management():
    """Test LLM provider management endpoints."""
    print("\n" + "=" * 50)
    print("Testing LLM Provider Management")
    print("=" * 50)

    try:
        # Get supported providers
        response = requests.get(
            f"{BASE_URL}/multiple_resume_parser/supported-providers"
        )
        if response.status_code == 200:
            providers_info = response.json()
            print("✅ Supported providers:")
            for provider in providers_info.get("supported_providers", []):
                description = providers_info.get("descriptions", {}).get(
                    provider, "No description"
                )
                print(f"  - {provider}: {description}")
            print(
                f"Current default: {providers_info.get('current_default', 'unknown')}"
            )
        else:
            print(f"❌ Failed to get providers: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")


def test_multiple_resume_parsing():
    """Test multiple resume parsing functionality."""
    print("\n" + "=" * 50)
    print("Testing Multiple Resume Parsing")
    print("=" * 50)

    # Create multiple sample resumes
    resumes = [
        {
            "filename": "resume1.txt",
            "content": """
Alice Smith
alice.smith@email.com
+1-555-111-2222

SKILLS: Python, Data Science, Machine Learning
EXPERIENCE: Data Scientist at DataCorp (2021-2023)
EDUCATION: PhD in Data Science, Harvard (2019-2021)
""",
        },
        {
            "filename": "resume2.txt",
            "content": """
Bob Johnson
bob.johnson@email.com
+1-555-333-4444

SKILLS: Java, Spring Boot, Microservices
EXPERIENCE: Senior Java Developer at Enterprise Inc (2019-2023)
EDUCATION: MS Computer Science, Berkeley (2017-2019)
""",
        },
    ]

    # Save temporary files
    temp_files = []
    for resume in resumes:
        temp_file = Path(resume["filename"])
        with open(temp_file, "w") as f:
            f.write(resume["content"])
        temp_files.append(temp_file)

    try:
        # Prepare files for upload
        files = []
        for temp_file in temp_files:
            files.append(
                ("files", (temp_file.name, open(temp_file, "rb"), "text/plain"))
            )

        # Make API request
        response = requests.post(
            f"{BASE_URL}/multiple_resume_parser/parse-multiple",
            files=files,
            data={"llm_provider": "ollama", "max_concurrent": 2},
            timeout=120,
        )

        # Close file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()

        if response.status_code == 200:
            result = response.json()
            print("✅ Multiple resume parsing successful!")

            stats = result.get("statistics", {})
            print(f"Total files: {stats.get('total_files', 0)}")
            print(f"Successful parses: {stats.get('successful_parses', 0)}")
            print(f"Failed parses: {stats.get('failed_parses', 0)}")
            print(f"Embeddings generated: {stats.get('embeddings_generated', 0)}")
            print(f"Provider used: {stats.get('provider_used', 'unknown')}")

            # Show details for each resume
            for i, resume_result in enumerate(result.get("results", [])):
                print(f"\nResume {i+1}: {resume_result.get('filename', 'unknown')}")
                print(f"  Status: {resume_result.get('status', 'unknown')}")
                if resume_result.get("status") == "success":
                    parsed_data = resume_result.get("parsed_data", {})
                    print(f"  Name: {parsed_data.get('name', 'N/A')}")
                    print(
                        f"  Embeddings: {list(parsed_data.get('embeddings', {}).keys())}"
                    )
        else:
            print(f"❌ Multiple resume parsing failed: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

    finally:
        # Clean up
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()


def main():
    """Main test function."""
    print("Multiple Resume Parser API Test Suite")
    print("=" * 50)

    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code != 200:
            print("❌ FastAPI server doesn't seem to be running at", BASE_URL)
            print("Please start your FastAPI server first.")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to FastAPI server at", BASE_URL)
        print("Please ensure your FastAPI server is running.")
        return

    print("✅ FastAPI server is accessible")

    # Run tests
    test_provider_management()
    test_single_resume_parsing()
    test_multiple_resume_parsing()

    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
