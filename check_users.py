#!/usr/bin/env python3
"""
Check what users exist in the users collection
"""

import sys

sys.path.append(".")

from mangodatabase.client import get_users_collection

try:
    users_collection = get_users_collection()

    # Get a few sample users
    users = list(users_collection.find({}, {"user_id": 1, "_id": 0}).limit(10))

    print(f"Found {len(users)} users in the collection:")
    for user in users:
        print(f"  - {user.get('user_id', 'No user_id field')}")

    # Get total count
    total_count = users_collection.count_documents({})
    print(f"\nTotal users in collection: {total_count}")

    # Check if there are any documents in the resumes collection
    from mangodatabase.client import get_collection

    resumes_collection = get_collection()
    resume_count = resumes_collection.count_documents({})
    print(f"Total resumes in collection: {resume_count}")

    # Get a sample of user_ids from resumes
    sample_resume_users = list(
        resumes_collection.find({}, {"user_id": 1, "_id": 0}).limit(5)
    )
    print(f"\nSample user_ids from resumes:")
    for resume in sample_resume_users:
        print(f"  - {resume.get('user_id', 'No user_id field')}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
