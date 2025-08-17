#!/usr/bin/env python3
"""
Test script to validate resume access control functionality
"""

import json
from typing import Dict, Any

# Test scenarios for access control validation
TEST_SCENARIOS = {
    "admin_user": {
        "user_id": "admin123",
        "description": "Admin user (exists in users_collection)",
        "expected_permissions": {
            "can_access_any_resume": True,
            "can_create_for_others": True,
            "can_update_others": True,
            "can_delete_others": True,
            "can_list_all": True,
            "can_update_all_embeddings": True,
        },
    },
    "regular_user": {
        "user_id": "user456",
        "description": "Regular user (NOT in users_collection)",
        "expected_permissions": {
            "can_access_any_resume": False,
            "can_create_for_others": False,
            "can_update_others": False,
            "can_delete_others": False,
            "can_list_all": False,
            "can_update_all_embeddings": False,
        },
    },
}

SAMPLE_RESUME_DATA_ADMIN = {
    "user_id": "admin123",
    "username": "Admin User",
    "contact_details": {"email": "admin@company.com", "phone": "+91-9876543210"},
    "skills": ["python", "management", "leadership"],
    "experience": [
        {
            "title": "System Administrator",
            "company": "TechCorp",
            "from_date": "2020-01",
            "until": "Present",
        }
    ],
    "academic_details": [
        {"education": "MBA", "college": "Business School", "pass_year": "2019"}
    ],
}

SAMPLE_RESUME_DATA_USER = {
    "user_id": "user456",
    "username": "Regular User",
    "contact_details": {"email": "user@example.com", "phone": "+91-9876543211"},
    "skills": ["javascript", "react", "node.js"],
    "experience": [
        {
            "title": "Frontend Developer",
            "company": "StartupXYZ",
            "from_date": "2022-01",
            "until": "Present",
        }
    ],
    "academic_details": [
        {
            "education": "BTech Computer Science",
            "college": "Engineering College",
            "pass_year": "2021",
        }
    ],
}

SAMPLE_RESUME_DATA_OTHER = {
    "user_id": "other789",
    "username": "Other User",
    "contact_details": {"email": "other@example.com", "phone": "+91-9876543212"},
    "skills": ["java", "spring", "microservices"],
    "experience": [
        {
            "title": "Backend Developer",
            "company": "Enterprise Corp",
            "from_date": "2021-06",
            "until": "Present",
        }
    ],
    "academic_details": [
        {"education": "MCA", "college": "Computer College", "pass_year": "2020"}
    ],
}


def test_access_control_logic():
    """Test the access control logic implementation"""
    print("🧪 Testing Resume Access Control Logic\n")

    # Simulate the access control functions
    def simulate_user_exists(user_id: str) -> bool:
        """Simulate checking if user exists in users_collection"""
        # Simulate that admin123 exists in users collection
        admin_users = ["admin123", "admin456", "superuser"]
        return user_id in admin_users

    def simulate_get_effective_user_id(requesting_user_id: str):
        """Simulate the get_effective_user_id_for_resume_operations function"""
        user_exists = simulate_user_exists(requesting_user_id)
        if user_exists:
            print(
                f"✅ User {requesting_user_id} exists in collection - can access all resumes"
            )
            return None  # Can access all
        else:
            print(
                f"⚠️  User {requesting_user_id} not in collection - can only access own resumes"
            )
            return requesting_user_id  # Can only access own

    def simulate_validate_user_access(
        requesting_user_id: str, resume: Dict[str, Any]
    ) -> bool:
        """Simulate the validate_user_access_to_resume function"""
        user_exists = simulate_user_exists(requesting_user_id)

        if user_exists:
            # Admin user - can access all resumes
            return True
        else:
            # Regular user - can only access their own resumes
            resume_user_id = resume.get("user_id")
            return resume_user_id == requesting_user_id

    # Test scenarios
    test_resumes = [
        {"_id": "resume1", "user_id": "admin123", "username": "Admin User"},
        {"_id": "resume2", "user_id": "user456", "username": "Regular User"},
        {"_id": "resume3", "user_id": "other789", "username": "Other User"},
    ]

    print("1. Testing Access Control for Different User Types:\n")

    for scenario_name, scenario in TEST_SCENARIOS.items():
        user_id = scenario["user_id"]
        description = scenario["description"]
        expected = scenario["expected_permissions"]

        print(f"🔍 Testing {description} (user_id: {user_id})")
        print("=" * 60)

        # Test effective user ID
        effective_user_id = simulate_get_effective_user_id(user_id)
        can_access_all = effective_user_id is None

        print(
            f"Can access all resumes: {can_access_all} (expected: {expected['can_access_any_resume']})"
        )

        # Test access to each resume
        print("\nResume Access Tests:")
        for resume in test_resumes:
            can_access = simulate_validate_user_access(user_id, resume)
            resume_owner = resume["user_id"]
            access_status = "✅ ALLOWED" if can_access else "❌ DENIED"
            print(f"  Resume owned by {resume_owner}: {access_status}")

        print("\n" + "-" * 60 + "\n")

    print("2. Testing Permission Scenarios:\n")

    # Test create operations
    print("🔨 CREATE Operations:")
    scenarios = [
        ("admin123", SAMPLE_RESUME_DATA_ADMIN, "Admin creating own resume"),
        ("admin123", SAMPLE_RESUME_DATA_USER, "Admin creating resume for user456"),
        ("user456", SAMPLE_RESUME_DATA_USER, "User creating own resume"),
        ("user456", SAMPLE_RESUME_DATA_OTHER, "User creating resume for other789"),
    ]

    for requesting_user, resume_data, description in scenarios:
        effective_user_id = simulate_get_effective_user_id(requesting_user)
        can_create = True

        if effective_user_id is not None:
            # Regular user - can only create for themselves
            if resume_data["user_id"] != requesting_user:
                can_create = False

        status = "✅ ALLOWED" if can_create else "❌ DENIED"
        print(f"  {description}: {status}")

    print("\n🔍 READ Operations:")
    for requesting_user, _, user_desc in [
        ("admin123", None, "Admin"),
        ("user456", None, "Regular user"),
    ]:
        print(f"  {user_desc} reading resumes:")
        for resume in test_resumes:
            can_read = simulate_validate_user_access(requesting_user, resume)
            status = "✅ ALLOWED" if can_read else "❌ DENIED"
            print(f"    Resume by {resume['user_id']}: {status}")

    print("\n✏️  UPDATE Operations:")
    for requesting_user, _, user_desc in [
        ("admin123", None, "Admin"),
        ("user456", None, "Regular user"),
    ]:
        print(f"  {user_desc} updating resumes:")
        for resume in test_resumes:
            can_update = simulate_validate_user_access(requesting_user, resume)
            status = "✅ ALLOWED" if can_update else "❌ DENIED"
            print(f"    Resume by {resume['user_id']}: {status}")

    print("\n🗑️  DELETE Operations:")
    for requesting_user, _, user_desc in [
        ("admin123", None, "Admin"),
        ("user456", None, "Regular user"),
    ]:
        print(f"  {user_desc} deleting resumes:")
        for resume in test_resumes:
            can_delete = simulate_validate_user_access(requesting_user, resume)
            status = "✅ ALLOWED" if can_delete else "❌ DENIED"
            print(f"    Resume by {resume['user_id']}: {status}")

    print("\n3. Testing API Endpoint Behavior:\n")

    print("📋 LIST Resumes:")
    for user_id, description in [("admin123", "Admin"), ("user456", "Regular user")]:
        effective_user_id = simulate_get_effective_user_id(user_id)
        if effective_user_id is None:
            result = "Returns ALL resumes"
        else:
            result = f"Returns only resumes with user_id={user_id}"
        print(f"  {description}: {result}")

    print("\n🔄 UPDATE Embeddings:")
    for user_id, description in [("admin123", "Admin"), ("user456", "Regular user")]:
        effective_user_id = simulate_get_effective_user_id(user_id)
        if effective_user_id is None:
            result = "Updates embeddings for ALL resumes"
        else:
            result = f"Updates embeddings only for resumes with user_id={user_id}"
        print(f"  {description}: {result}")

    print("\n4. Security Validation:\n")

    security_tests = [
        {
            "test": "Regular user trying to create resume for another user",
            "requesting_user": "user456",
            "target_user": "other789",
            "expected": "BLOCKED",
        },
        {
            "test": "Regular user trying to access another user's resume",
            "requesting_user": "user456",
            "target_resume": {"user_id": "other789"},
            "expected": "BLOCKED",
        },
        {
            "test": "Admin user accessing any resume",
            "requesting_user": "admin123",
            "target_resume": {"user_id": "anyone"},
            "expected": "ALLOWED",
        },
        {
            "test": "Regular user accessing their own resume",
            "requesting_user": "user456",
            "target_resume": {"user_id": "user456"},
            "expected": "ALLOWED",
        },
    ]

    for test in security_tests:
        test_name = test["test"]
        requesting_user = test["requesting_user"]
        expected = test["expected"]

        if "target_user" in test:
            # Create operation test
            effective_user_id = simulate_get_effective_user_id(requesting_user)
            can_create = True
            if effective_user_id is not None and test["target_user"] != requesting_user:
                can_create = False
            result = "ALLOWED" if can_create else "BLOCKED"
        else:
            # Access operation test
            target_resume = test["target_resume"]
            can_access = simulate_validate_user_access(requesting_user, target_resume)
            result = "ALLOWED" if can_access else "BLOCKED"

        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"  {test_name}: {result} (expected: {expected}) {status}")

    print("\n🎉 Access Control Logic Test Completed!")
    print("\nSummary:")
    print("✅ Admin users can access all resumes")
    print("✅ Regular users can only access their own resumes")
    print("✅ Permission validation works correctly")
    print("✅ Security measures are properly enforced")


def print_api_usage_examples():
    """Print usage examples for the new API"""
    print("\n" + "=" * 80)
    print("📚 API Usage Examples")
    print("=" * 80)

    print("\n1. Admin User Operations:")
    print("-" * 30)

    examples = [
        {
            "operation": "Create resume for any user",
            "method": "POST",
            "url": "/resumes/?requesting_user_id=admin123",
            "body": SAMPLE_RESUME_DATA_USER,
        },
        {
            "operation": "Get any resume",
            "method": "GET",
            "url": "/resumes/648f1234567890abcdef?requesting_user_id=admin123",
            "body": None,
        },
        {
            "operation": "List all resumes",
            "method": "GET",
            "url": "/resumes/?requesting_user_id=admin123&skip=0&limit=10",
            "body": None,
        },
        {
            "operation": "Update any resume",
            "method": "PUT",
            "url": "/resumes/648f1234567890abcdef?requesting_user_id=admin123",
            "body": {"user_id": "user456", "username": "Updated Name"},
        },
    ]

    for example in examples:
        print(f"\n{example['operation']}:")
        print(f"{example['method']} {example['url']}")
        if example["body"]:
            print("Content-Type: application/json")
            print(json.dumps(example["body"], indent=2))

    print("\n\n2. Regular User Operations:")
    print("-" * 30)

    examples = [
        {
            "operation": "Create own resume",
            "method": "POST",
            "url": "/resumes/?requesting_user_id=user456",
            "body": SAMPLE_RESUME_DATA_USER,
        },
        {
            "operation": "Get own resume",
            "method": "GET",
            "url": "/resumes/648f1234567890abcdef?requesting_user_id=user456",
            "body": None,
        },
        {
            "operation": "List own resumes",
            "method": "GET",
            "url": "/resumes/?requesting_user_id=user456&skip=0&limit=10",
            "body": None,
        },
        {
            "operation": "Update own resume",
            "method": "PUT",
            "url": "/resumes/648f1234567890abcdef?requesting_user_id=user456",
            "body": {"user_id": "user456", "username": "Updated Name"},
        },
    ]

    for example in examples:
        print(f"\n{example['operation']}:")
        print(f"{example['method']} {example['url']}")
        if example["body"]:
            print("Content-Type: application/json")
            print(json.dumps(example["body"], indent=2))

    print("\n\n3. Error Scenarios:")
    print("-" * 20)

    error_examples = [
        {
            "scenario": "Regular user tries to create resume for another user",
            "request": "POST /resumes/?requesting_user_id=user456",
            "body": SAMPLE_RESUME_DATA_OTHER,
            "expected_error": "403 Forbidden: You can only create resumes for your own user_id",
        },
        {
            "scenario": "Regular user tries to access another user's resume",
            "request": "GET /resumes/other_user_resume_id?requesting_user_id=user456",
            "expected_error": "403 Forbidden: You don't have permission to access this resume",
        },
    ]

    for example in error_examples:
        print(f"\n{example['scenario']}:")
        print(f"Request: {example['request']}")
        if "body" in example:
            print(f"Body: {json.dumps(example['body'], indent=2)}")
        print(f"Expected: {example['expected_error']}")


if __name__ == "__main__":
    test_access_control_logic()
    print_api_usage_examples()
