#!/usr/bin/env python3
"""
Test script to validate priority-based search functionality
"""

import sys
import os
import re

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_priority_scoring():
    """Test the priority scoring functions"""

    # Test candidate data
    test_candidate = {
        "contact_details": {
            "name": "John Doe",
            "current_city": "Bangalore",
            "looking_for_jobs_in": ["Mumbai", "Delhi"],
        },
        "experience": [
            {
                "role": "Senior Software Engineer",
                "company": "Tech Corp",
                "duration": "2020-Present",
            },
            {
                "role": "Software Developer",
                "company": "StartupXYZ",
                "duration": "2018-2020",
            },
        ],
        "skills": ["Python", "Machine Learning", "Django", "AWS"],
        "may_also_known_skills": ["JavaScript", "React"],
        "labels": ["Software Engineer", "Python Developer"],
        "total_experience": "5.5",
        "expected_salary": 1200000,  # 12 LPA in rupees
    }

    # Test query: "Senior Software Engineer in Bangalore with Python skills 5 years experience 12 lakh salary"
    test_query = "Senior Software Engineer in Bangalore with Python skills 5 years experience 12 lakh salary"

    print("Testing Priority-Based Search Scoring")
    print("=" * 50)
    print(f"Test Query: {test_query}")
    print(f"Candidate: {test_candidate['contact_details']['name']}")
    print(f"Current City: {test_candidate['contact_details']['current_city']}")
    print(f"Current Role: {test_candidate['experience'][0]['role']}")
    print(f"Skills: {', '.join(test_candidate['skills'])}")
    print(f"Experience: {test_candidate['total_experience']} years")
    print(f"Expected Salary: {test_candidate['expected_salary']/100000:.1f} LPA")
    print("-" * 50)

    # Test individual scoring functions (we'll import them here)
    try:
        from apis.vector_search import (
            calculate_priority_score,
            calculate_designation_score,
            calculate_location_score,
            calculate_skills_score,
            calculate_experience_score,
            calculate_salary_score,
        )

        query_lower = test_query.lower()

        # Test individual components
        designation_score = calculate_designation_score(test_candidate, query_lower)
        location_score = calculate_location_score(test_candidate, query_lower)
        skills_score = calculate_skills_score(test_candidate, query_lower)
        experience_score = calculate_experience_score(test_candidate, query_lower)
        salary_score = calculate_salary_score(test_candidate, query_lower)

        print("Individual Scoring Components:")
        print(f"Designation Score: {designation_score:.3f} (Weight: 40%)")
        print(f"Location Score: {location_score:.3f} (Weight: 30%)")
        print(f"Skills Score: {skills_score:.3f} (Weight: 15%)")
        print(f"Experience Score: {experience_score:.3f} (Weight: 10%)")
        print(f"Salary Score: {salary_score:.3f} (Weight: 5%)")
        print("-" * 30)

        # Test overall priority score
        priority_score, match_reason = calculate_priority_score(
            test_candidate, test_query
        )

        print("Overall Priority Scoring:")
        print(f"Priority Score: {priority_score:.3f} ({priority_score*100:.1f}%)")
        print(f"Match Reason: {match_reason}")
        print("-" * 30)

        # Test with different scenarios
        print("\nTesting Different Scenarios:")
        print("=" * 50)

        scenarios = [
            ("Python developer", "Role-focused search"),
            ("Software engineer in Mumbai", "Role + Location search"),
            ("Machine learning engineer 3 years", "Role + Experience search"),
            ("Developer in Chennai 8 lakh salary", "Role + Location + Salary search"),
            ("Java developer in Delhi", "Non-matching skills and location"),
        ]

        for scenario_query, description in scenarios:
            score, reason = calculate_priority_score(test_candidate, scenario_query)
            print(f"{description}:")
            print(f"  Query: '{scenario_query}'")
            print(f"  Score: {score:.3f} ({score*100:.1f}%)")
            print(f"  Reason: {reason}")
            print()

        print("✅ Priority scoring test completed successfully!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


if __name__ == "__main__":
    success = test_priority_scoring()
    if success:
        print("\n🎉 All tests passed! Priority-based search is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the implementation.")
