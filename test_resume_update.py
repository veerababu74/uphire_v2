#!/usr/bin/env python3
"""
Test script to validate resume update functionality with vector regeneration
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

# Mock test data similar to your example
SAMPLE_RESUME_DATA = {
    "user_id": "66c8771a20bd68c725758679",
    "username": "Harsh Gajera",
    "contact_details": {
        "email": "harsh@example.com",
        "phone": "+91-9876543210",
        "address": "Mumbai, India",
    },
    "total_experience": "2 years",
    "notice_period": "30",
    "currency": "INR (Lacs)",
    "pay_duration": "Yearly",
    "current_salary": 2,
    "hike": 2,
    "expected_salary": 2.04,
    "skills": ["bootstrap", "react.js", "python", "javascript"],
    "may_also_known_skills": ["string", "vsam", "bootstr"],
    "labels": ["Frontend Developer"],
    "experience": [
        {
            "title": "Software Developer",
            "company": "TechCorp",
            "from_date": "2023-01-01",
            "until": "Present",
            "description": "Working on web applications",
        }
    ],
    "academic_details": [
        {
            "education": "B.Tech Computer Science",
            "college": "ABC University",
            "pass_year": "2022",
            "percentage": "85%",
        }
    ],
    "source": "By Recruiter",
    "last_working_day": "2025-07-10",
    "is_tier1_mba": False,
    "is_tier1_engineering": False,
    "comment": "no comment",
    "exit_reason": "no reason",
    "combined_resume": """
RESUME

PERSONAL INFORMATION
-------------------
Name: Harsh Gajera
Contact: harsh@example.com, +91-9876543210
Address: Mumbai, India

EXPERIENCE
----------
Software Developer at TechCorp (2023-Present)
- Developed web applications using React.js and Python
- Implemented responsive designs with Bootstrap
- Worked with JavaScript for frontend functionality

EDUCATION
---------
B.Tech Computer Science from ABC University (2022)
Percentage: 85%

SKILLS
------
Bootstrap, React.js, Python, JavaScript, HTML, CSS
""",
}

UPDATED_RESUME_DATA = {
    "user_id": "66c8771a20bd68c725758679",
    "username": "Harsh Gajera",
    "contact_details": {
        "email": "harsh.updated@example.com",  # Updated email
        "phone": "+91-9876543210",
        "address": "Pune, India",  # Updated address
    },
    "total_experience": "3 years",  # Updated experience
    "notice_period": "15",  # Updated notice period
    "currency": "INR (Lacs)",
    "pay_duration": "Yearly",
    "current_salary": 3,  # Updated salary
    "hike": 3,
    "expected_salary": 3.5,  # Updated expected salary
    "skills": [
        "bootstrap",
        "react.js",
        "python",
        "javascript",
        "nodejs",
        "mongodb",
    ],  # Added new skills
    "may_also_known_skills": ["string", "vsam", "bootstr", "docker"],  # Added docker
    "labels": ["Full Stack Developer"],  # Updated label
    "experience": [
        {
            "title": "Senior Software Developer",  # Updated title
            "company": "TechCorp",
            "from_date": "2023-01-01",
            "until": "Present",
            "description": "Leading web application development and mentoring junior developers",  # Updated description
        }
    ],
    "academic_details": [
        {
            "education": "B.Tech Computer Science",
            "college": "ABC University",
            "pass_year": "2022",
            "percentage": "85%",
        }
    ],
    "source": "By Recruiter",
    "last_working_day": "2025-07-10",
    "is_tier1_mba": False,
    "is_tier1_engineering": False,
    "comment": "Updated profile with new skills and experience",  # Updated comment
    "exit_reason": "Career growth",  # Updated exit reason
    "combined_resume": """
RESUME

PERSONAL INFORMATION
-------------------
Name: Harsh Gajera
Contact: harsh.updated@example.com, +91-9876543210
Address: Pune, India

EXPERIENCE
----------
Senior Software Developer at TechCorp (2023-Present)
- Leading web application development using React.js, Node.js and Python
- Implemented responsive designs with Bootstrap
- Worked with JavaScript, MongoDB for full-stack development
- Mentoring junior developers and code review

EDUCATION
---------
B.Tech Computer Science from ABC University (2022)
Percentage: 85%

SKILLS
------
Bootstrap, React.js, Python, JavaScript, Node.js, MongoDB, Docker, HTML, CSS
""",
}


def test_vector_fields_presence(resume_data: Dict[str, Any]) -> bool:
    """Test if all required vector fields are present"""
    required_vector_fields = [
        "experience_text_vector",
        "education_text_vector",
        "skills_vector",
        "combined_resume_vector",
        "total_resume_text",
        "total_resume_vector",
    ]

    missing_fields = []
    for field in required_vector_fields:
        if field not in resume_data:
            missing_fields.append(field)

    if missing_fields:
        print(f"❌ Missing vector fields: {missing_fields}")
        return False
    else:
        print("✅ All required vector fields are present")
        return True


def test_timestamp_update(old_timestamp: datetime, new_timestamp: datetime) -> bool:
    """Test if timestamp was properly updated"""
    if new_timestamp > old_timestamp:
        print(f"✅ Timestamp updated: {old_timestamp} → {new_timestamp}")
        return True
    else:
        print(f"❌ Timestamp not updated: {old_timestamp} = {new_timestamp}")
        return False


def test_vector_changes(old_vectors: Dict, new_vectors: Dict) -> bool:
    """Test if vectors were regenerated (should be different)"""
    vector_fields = [
        "experience_text_vector",
        "education_text_vector",
        "skills_vector",
        "combined_resume_vector",
        "total_resume_vector",
    ]

    changes_detected = 0
    for field in vector_fields:
        if field in old_vectors and field in new_vectors:
            if old_vectors[field] != new_vectors[field]:
                changes_detected += 1
                print(f"✅ {field} was regenerated")
            else:
                print(f"⚠️  {field} unchanged")

    if changes_detected > 0:
        print(f"✅ {changes_detected} vector fields were regenerated")
        return True
    else:
        print("❌ No vector fields were regenerated")
        return False


def simulate_resume_operations():
    """Simulate the resume operations to test the functionality"""
    print("🧪 Testing Resume Update Functionality\n")

    # Test 1: Create resume with vectors
    print("1. Testing Resume Creation:")
    print("Creating resume with initial data...")
    # In real scenario, this would call the AddUserDataVectorizer
    from embeddings.manager import AddUserDataVectorizer, EmbeddingManager
    from datetime import datetime, timezone

    manager = EmbeddingManager()
    vectorizer = AddUserDataVectorizer(manager)

    # Simulate creating initial resume
    initial_resume = vectorizer.generate_resume_embeddings(SAMPLE_RESUME_DATA)
    initial_resume["created_at"] = datetime.now(timezone.utc)

    # Test vector field presence
    test_vector_fields_presence(initial_resume)

    print(f"Initial created_at: {initial_resume['created_at']}")
    print(
        f"Initial total_resume_text length: {len(initial_resume.get('total_resume_text', ''))}"
    )
    print(
        f"Initial skills_vector length: {len(initial_resume.get('skills_vector', []))}"
    )

    # Wait a moment to ensure timestamp difference
    import time

    time.sleep(1)

    print("\n2. Testing Resume Update:")
    print("Updating resume with new data...")

    # Simulate updating resume
    updated_resume = vectorizer.generate_resume_embeddings(UPDATED_RESUME_DATA)
    updated_resume["created_at"] = datetime.now(timezone.utc)

    # Test vector field presence
    test_vector_fields_presence(updated_resume)

    # Test timestamp update
    test_timestamp_update(initial_resume["created_at"], updated_resume["created_at"])

    # Test vector changes
    test_vector_changes(initial_resume, updated_resume)

    print(f"Updated created_at: {updated_resume['created_at']}")
    print(
        f"Updated total_resume_text length: {len(updated_resume.get('total_resume_text', ''))}"
    )
    print(
        f"Updated skills_vector length: {len(updated_resume.get('skills_vector', []))}"
    )

    print("\n3. Comparing Text Changes:")
    print("Initial total_resume_text preview:")
    print(initial_resume.get("total_resume_text", "")[:200] + "...")
    print("\nUpdated total_resume_text preview:")
    print(updated_resume.get("total_resume_text", "")[:200] + "...")

    print("\n4. Skills Comparison:")
    initial_skills = SAMPLE_RESUME_DATA.get("skills", []) + SAMPLE_RESUME_DATA.get(
        "may_also_known_skills", []
    )
    updated_skills = UPDATED_RESUME_DATA.get("skills", []) + UPDATED_RESUME_DATA.get(
        "may_also_known_skills", []
    )

    print(f"Initial skills: {initial_skills}")
    print(f"Updated skills: {updated_skills}")

    new_skills = set(updated_skills) - set(initial_skills)
    if new_skills:
        print(f"✅ New skills detected: {new_skills}")
    else:
        print("ℹ️  No new skills added")

    print("\n🎉 Test completed!")


if __name__ == "__main__":
    simulate_resume_operations()
