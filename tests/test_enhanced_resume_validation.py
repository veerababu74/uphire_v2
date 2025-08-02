#!/usr/bin/env python3
"""
Test Enhanced Resume Validation
Tests the improved multiple resume parser with intelligent LLM-based content validation.
"""

import os
import sys
import requests
import json
from io import BytesIO

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_valid_resume():
    """Test with valid resume content"""
    valid_resume_content = """
    John Smith
    Software Engineer
    Email: john.smith@email.com
    Phone: (555) 123-4567
    
    PROFESSIONAL EXPERIENCE
    Senior Software Engineer | ABC Tech Company | 2020-Present
    • Developed scalable web applications using Python and React
    • Led a team of 5 developers on multiple projects
    • Implemented CI/CD pipelines reducing deployment time by 50%
    
    Software Developer | XYZ Solutions | 2018-2020
    • Built REST APIs using Flask and Django
    • Collaborated with cross-functional teams
    • Optimized database queries improving performance by 30%
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of Technology | 2014-2018
    
    SKILLS
    Programming Languages: Python, JavaScript, Java, C++
    Frameworks: React, Django, Flask, Spring Boot
    Databases: PostgreSQL, MongoDB, MySQL
    Tools: Git, Docker, Kubernetes, AWS
    """

    print("Testing with valid resume content...")

    files = {
        "files": (
            "john_smith_resume.txt",
            BytesIO(valid_resume_content.encode()),
            "text/plain",
        )
    }

    try:
        response = requests.post(
            "http://localhost:8000/resume-parser-multiple",
            files=files,
            timeout=30,
        )

        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")

        if response.status_code == 200:
            print("✅ Valid resume was processed successfully")
            return True
        else:
            print("❌ Valid resume processing failed")
            return False

    except Exception as e:
        print(f"❌ Error testing valid resume: {e}")
        return False


def test_invalid_content():
    """Test with non-resume content"""
    invalid_content = """
    GROCERY SHOPPING LIST
    
    Fruits:
    - Apples
    - Bananas 
    - Oranges
    
    Vegetables:
    - Carrots
    - Broccoli
    - Spinach
    
    Dairy:
    - Milk
    - Cheese
    - Yogurt
    
    Other:
    - Bread
    - Pasta
    - Rice
    
    Total estimated cost: $45.50
    Don't forget to use the discount coupon!
    """

    print("\nTesting with invalid (non-resume) content...")

    files = {
        "files": ("shopping_list.txt", BytesIO(invalid_content.encode()), "text/plain")
    }

    try:
        response = requests.post(
            "http://localhost:8000/resume-parser-multiple",
            files=files,
            timeout=30,
        )

        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")

        # Check if the LLM properly detected invalid content
        if response.status_code == 200:
            results = result.get("results", [])
            if results and results[0].get("error_type") == "invalid_content":
                print("✅ Invalid content was properly detected by LLM")
                return True
            else:
                print("❌ Invalid content was not detected")
                return False
        else:
            print("❌ Request failed")
            return False

    except Exception as e:
        print(f"❌ Error testing invalid content: {e}")
        return False


def test_borderline_content():
    """Test with borderline content (job description instead of resume)"""
    job_description = """
    SOFTWARE ENGINEER POSITION
    
    We are looking for a talented Software Engineer to join our growing team.
    
    RESPONSIBILITIES:
    • Develop and maintain web applications
    • Write clean, maintainable code
    • Collaborate with team members
    • Participate in code reviews
    
    REQUIREMENTS:
    • Bachelor's degree in Computer Science
    • 3+ years of experience with Python
    • Knowledge of React and Django
    • Strong problem-solving skills
    
    BENEFITS:
    • Competitive salary
    • Health insurance
    • 401k matching
    • Flexible work hours
    
    Apply now by sending your resume to jobs@company.com
    """

    print("\nTesting with borderline content (job description)...")

    files = {
        "files": ("job_posting.txt", BytesIO(job_description.encode()), "text/plain")
    }

    try:
        response = requests.post(
            "http://localhost:8000/resume-parser-multiple",
            files=files,
            timeout=30,
        )

        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")

        # Check if the LLM properly detected this as invalid content
        if response.status_code == 200:
            results = result.get("results", [])
            if results and results[0].get("error_type") == "invalid_content":
                print(
                    "✅ Job description was properly identified as non-resume content"
                )
                return True
            else:
                print("❌ Job description was not properly identified")
                return False
        else:
            print("❌ Request failed")
            return False

    except Exception as e:
        print(f"❌ Error testing borderline content: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Testing Enhanced Resume Validation System")
    print("=" * 50)

    # Check if server is running
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Server is not running. Please start the server first.")
            return
    except:
        print("❌ Cannot connect to server. Please start the server first.")
        return

    print("✅ Server is running")

    # Run tests
    tests_passed = 0
    total_tests = 3

    if test_valid_resume():
        tests_passed += 1

    if test_invalid_content():
        tests_passed += 1

    if test_borderline_content():
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print(
            "🎉 All tests passed! The enhanced validation system is working correctly."
        )
    else:
        print("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
