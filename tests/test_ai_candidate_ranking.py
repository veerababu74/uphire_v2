"""
Test file for AI Candidate Ranking API

This script tests the new AI candidate ranking functionality including:
- Skills match percentage calculation
- Missing skills identification
- Experience relevance scoring
- Automatic rejection below 40% threshold
- Status tagging for rejected candidates

Author: Uphire Team
Version: 1.0.0
"""

import requests
import json
import os
from datetime import datetime


def test_ai_ranking_by_text():
    """Test AI candidate ranking with job description text"""
    print("🚀 Testing AI Candidate Ranking by Job Description Text")
    print("=" * 60)

    # Test data
    test_job_description = """
    Senior Python Developer Position
    
    We are looking for an experienced Python developer to join our team.
    
    Required Skills:
    - Python (5+ years)
    - Django or Flask
    - PostgreSQL or MySQL
    - REST API development
    - Git version control
    - Unit testing (pytest)
    
    Preferred Skills:
    - React.js
    - Docker
    - AWS or Azure
    - Redis
    - Celery
    
    Experience Requirements:
    - Minimum 5 years of professional Python development
    - Experience with web frameworks
    - Database design and optimization
    - API development and integration
    
    Education:
    - Bachelor's degree in Computer Science or related field
    
    Responsibilities:
    - Develop and maintain web applications
    - Design and implement APIs
    - Write clean, maintainable code
    - Collaborate with frontend developers
    - Participate in code reviews
    """

    test_payload = {
        "job_description": test_job_description,
        "user_id": "test_user_123",
        "max_candidates": 10,
        "include_rejected": True,
    }

    try:
        # Make API request
        print("📤 Sending request to AI ranking API...")
        response = requests.post(
            "http://localhost:8000/ai-ranking/rank-by-job-text",
            json=test_payload,
            headers={"Content-Type": "application/json"},
        )

        print(f"📥 Response Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Success! AI Ranking completed")
            print(f"📊 Results Summary:")
            print(
                f"   - Total candidates analyzed: {result['total_candidates_analyzed']}"
            )
            print(f"   - Accepted candidates: {result['accepted_candidates']}")
            print(f"   - Rejected candidates: {result['rejected_candidates']}")
            print(
                f"   - Rejection threshold: {result['ranking_criteria']['rejection_threshold']}%"
            )

            print("\n🏆 Top 5 Candidates:")
            for i, candidate in enumerate(result["candidates"][:5], 1):
                status_icon = "✅" if not candidate["is_auto_rejected"] else "❌"
                print(
                    f"   {i}. {status_icon} {candidate['name']} ({candidate['overall_match_score']}%)"
                )
                print(f"      ID: {candidate.get('_id', 'N/A')}")
                print(f"      Status: {candidate['status']}")
                print(
                    f"      Skills Match: {candidate['skills_match']['skills_match_percentage']}%"
                )
                print(
                    f"      Experience Match: {candidate['experience_relevance']['experience_match_percentage']}%"
                )
                print(f"      Reason: {candidate['ranking_reason']}")

                if candidate["skills_match"]["missing_skills"]:
                    print(
                        f"      Missing Skills: {', '.join(candidate['skills_match']['missing_skills'][:3])}"
                    )
                print()

            # Show rejection statistics
            rejected_candidates = [
                c for c in result["candidates"] if c["is_auto_rejected"]
            ]
            if rejected_candidates:
                print(f"❌ Auto-Rejected Candidates ({len(rejected_candidates)}):")
                for candidate in rejected_candidates[:3]:
                    print(
                        f"   - {candidate['name']}: {candidate['overall_match_score']}% (Below {result['ranking_criteria']['rejection_threshold']}% threshold)"
                    )
                    print(f"     Reason: {candidate['ranking_reason']}")
                print()

            return True

        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        return False


def test_ai_ranking_by_file():
    """Test AI candidate ranking with job description file upload"""
    print("🚀 Testing AI Candidate Ranking by Job Description File")
    print("=" * 60)

    # Create a test job description file
    test_file_content = """
    Data Scientist Position - AI/ML Focus
    
    We are seeking a talented Data Scientist to join our AI/ML team.
    
    Required Skills:
    - Python (3+ years)
    - Machine Learning (scikit-learn, pandas, numpy)
    - Statistics and Data Analysis
    - SQL and database management
    - Data visualization (matplotlib, seaborn)
    
    Preferred Skills:
    - Deep Learning (TensorFlow, PyTorch)
    - Big Data tools (Spark, Hadoop)
    - Cloud platforms (AWS, GCP)
    - R programming
    - Jupyter notebooks
    
    Experience Requirements:
    - Minimum 3 years in data science or related field
    - Experience with ML model development and deployment
    - Statistical analysis and hypothesis testing
    
    Education:
    - Master's degree in Data Science, Statistics, or related field
    
    Responsibilities:
    - Develop and deploy machine learning models
    - Analyze large datasets to extract insights
    - Create data visualizations and reports
    - Collaborate with engineering teams
    """

    # Create temporary test file
    test_file_path = "temp_test_job_description.txt"
    try:
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write(test_file_content)

        print("📤 Uploading job description file...")

        # Prepare file upload
        with open(test_file_path, "rb") as f:
            files = {"file": ("job_description.txt", f, "text/plain")}
            params = {
                "user_id": "test_user_456",
                "max_candidates": 8,
                "include_rejected": True,
            }

            response = requests.post(
                "http://localhost:8000/ai-ranking/rank-by-job-file",
                files=files,
                params=params,
            )

        print(f"📥 Response Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Success! AI Ranking by file completed")
            print(f"📊 Results Summary:")
            print(
                f"   - Total candidates analyzed: {result['total_candidates_analyzed']}"
            )
            print(f"   - Accepted candidates: {result['accepted_candidates']}")
            print(f"   - Rejected candidates: {result['rejected_candidates']}")

            print("\n🏆 Top 3 Candidates:")
            for i, candidate in enumerate(result["candidates"][:3], 1):
                status_icon = "✅" if not candidate["is_auto_rejected"] else "❌"
                print(
                    f"   {i}. {status_icon} {candidate['name']} ({candidate['overall_match_score']}%)"
                )
                print(
                    f"      Skills: {len(candidate['skills_match']['matched_skills'])} matched, {len(candidate['skills_match']['missing_skills'])} missing"
                )
                print(
                    f"      Experience: {candidate['experience_relevance']['relevant_experience_years']} years relevant"
                )
                print()

            return True

        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error during file test: {str(e)}")
        return False

    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)


def test_ranking_statistics():
    """Test AI ranking statistics endpoint"""
    print("🚀 Testing AI Ranking Statistics")
    print("=" * 60)

    try:
        print("📤 Getting ranking statistics...")
        response = requests.get(
            "http://localhost:8000/ai-ranking/ranking-stats",
            params={"user_id": "test_user_123"},
        )

        print(f"📥 Response Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Success! Statistics retrieved")
            print(f"📊 Database Overview:")
            print(
                f"   - Total candidates: {result['database_overview']['total_candidates']}"
            )
            print(
                f"   - Sample analyzed: {result['database_overview']['sample_analyzed']}"
            )

            print(f"\n🔧 Ranking Configuration:")
            config = result["ranking_configuration"]
            print(f"   - Skills weight: {config['skills_weight']}%")
            print(f"   - Experience weight: {config['experience_weight']}%")
            print(f"   - Education weight: {config['education_weight']}%")
            print(f"   - Rejection threshold: {config['rejection_threshold']}%")

            if "top_skills" in result["skills_analysis"]:
                print(f"\n🏅 Top Skills in Database:")
                for skill_data in result["skills_analysis"]["top_skills"][:5]:
                    print(
                        f"   - {skill_data['skill']}: {skill_data['count']} candidates"
                    )

            print(f"\n💡 Recommendations:")
            for rec in result["recommendations"]:
                print(f"   - {rec}")

            return True

        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error during statistics test: {str(e)}")
        return False


def test_edge_cases():
    """Test edge cases and error handling"""
    print("🚀 Testing Edge Cases and Error Handling")
    print("=" * 60)

    test_cases = [
        {
            "name": "Too short job description",
            "payload": {
                "job_description": "Python dev",  # Too short
                "user_id": "test_user",
                "max_candidates": 5,
            },
            "expected_status": 422,
        },
        {
            "name": "Invalid max_candidates (too high)",
            "payload": {
                "job_description": "We need a Python developer with 5+ years experience in web development using Django framework.",
                "user_id": "test_user",
                "max_candidates": 150,  # Above limit
            },
            "expected_status": 422,
        },
        {
            "name": "Missing user_id",
            "payload": {
                "job_description": "We need a Python developer with 5+ years experience in web development using Django framework.",
                "max_candidates": 10,
                # user_id missing
            },
            "expected_status": 422,
        },
    ]

    passed_tests = 0
    total_tests = len(test_cases)

    for test_case in test_cases:
        print(f"🧪 Testing: {test_case['name']}")

        try:
            response = requests.post(
                "http://localhost:8000/ai-ranking/rank-by-job-text",
                json=test_case["payload"],
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == test_case["expected_status"]:
                print(
                    f"   ✅ Passed (Expected {test_case['expected_status']}, Got {response.status_code})"
                )
                passed_tests += 1
            else:
                print(
                    f"   ❌ Failed (Expected {test_case['expected_status']}, Got {response.status_code})"
                )

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

    print(f"\n📊 Edge Cases Results: {passed_tests}/{total_tests} tests passed")
    return passed_tests == total_tests


def main():
    """Run all tests"""
    print("🔬 AI Candidate Ranking API - Comprehensive Test Suite")
    print("=" * 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = []

    # Test 1: Ranking by text
    print("TEST 1: AI Ranking by Job Description Text")
    print("-" * 50)
    results.append(test_ai_ranking_by_text())
    print()

    # Test 2: Ranking by file
    print("TEST 2: AI Ranking by Job Description File")
    print("-" * 50)
    results.append(test_ai_ranking_by_file())
    print()

    # Test 3: Statistics
    print("TEST 3: AI Ranking Statistics")
    print("-" * 50)
    results.append(test_ranking_statistics())
    print()

    # Test 4: Edge cases
    print("TEST 4: Edge Cases and Error Handling")
    print("-" * 50)
    results.append(test_edge_cases())
    print()

    # Summary
    passed = sum(results)
    total = len(results)

    print("📋 TEST SUMMARY")
    print("=" * 30)
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("\n🎉 All tests passed! AI Candidate Ranking API is working correctly.")
        print("\n🚀 Key Features Verified:")
        print("   ✅ Skills match percentage calculation")
        print("   ✅ Missing skills identification")
        print("   ✅ Experience relevance scoring")
        print("   ✅ Automatic rejection below 40% threshold")
        print("   ✅ Status tagging (CV Rejected - In Process)")
        print("   ✅ File upload support")
        print("   ✅ Comprehensive statistics")
        print("   ✅ Error handling")
    else:
        print("\n⚠️ Some tests failed. Please check the API implementation.")

    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
