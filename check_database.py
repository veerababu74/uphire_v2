#!/usr/bin/env python3
"""
Simple script to check what's in the database
"""

from mangodatabase.client import get_collection, get_users_collection
import json


def check_database():
    """Check the database contents"""
    try:
        # Get collections
        resumes_collection = get_collection()
        users_collection = get_users_collection()

        # Count documents
        resume_count = resumes_collection.count_documents({})
        user_count = users_collection.count_documents({})

        print(f"📊 Database Status:")
        print(f"  - Resume Collection: {resume_count} documents")
        print(f"  - Users Collection: {user_count} documents")

        if resume_count > 0:
            print(f"\n📄 Sample Resume Data:")
            sample_resume = resumes_collection.find_one({})
            if sample_resume:
                # Remove the _id for cleaner output
                if "_id" in sample_resume:
                    del sample_resume["_id"]
                # Remove vector data for cleaner output
                if "combined_resume_vector" in sample_resume:
                    del sample_resume["combined_resume_vector"]
                print(json.dumps(sample_resume, indent=2, default=str)[:1000] + "...")

        if user_count > 0:
            print(f"\n👥 Sample User Data:")
            sample_user = users_collection.find_one({})
            if sample_user:
                if "_id" in sample_user:
                    del sample_user["_id"]
                print(json.dumps(sample_user, indent=2, default=str))

        # Check for any specific user_id in resumes
        print(f"\n🔍 Checking for different user_ids in resumes:")
        user_ids = resumes_collection.distinct("user_id")
        print(
            f"  - Found {len(user_ids)} unique user_ids: {user_ids[:10]}"
        )  # Show first 10

    except Exception as e:
        print(f"❌ Error checking database: {str(e)}")


if __name__ == "__main__":
    check_database()
