#!/usr/bin/env python3
"""
Detailed debugging script for manual search with keyword matching analysis
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
from mangodatabase.client import get_collection, get_users_collection
from mangodatabase.user_operations import UserOperations
import json
from datetime import datetime


def analyze_search_issue():
    """Comprehensive analysis of the manual search issue"""

    print("🔍 MANUAL SEARCH DEBUG ANALYSIS")
    print("=" * 60)

    # Initialize database connections
    resumes_collection = get_collection()
    users_collection = get_users_collection()
    user_ops = UserOperations(users_collection)

    # Test user ID from the request
    test_userid = "66c8771a20bd68c725758679"

    # Test search criteria
    search_criteria = {
        "experience_titles": [
            "frontend developer",
            "software developer",
            "python developer",
        ],
        "skills": ["cobol"],
        "min_education": ["10th Pass"],
        "min_experience": "6 Months",
        "max_experience": "1 Year",
        "locations": ["adwani"],
        "min_salary": 1.0,
        "max_salary": 2.0,
        "relevant_score": 40.0,
    }

    print(f"📋 Testing with User ID: {test_userid}")
    print(f"📋 Search Criteria: {json.dumps(search_criteria, indent=2)}")
    print()

    # Step 1: Check user existence
    print("STEP 1: Check User Existence")
    print("-" * 30)
    try:
        user_exists = user_ops.user_exists(test_userid)
        print(f"User exists in users collection: {user_exists}")

        if user_exists:
            effective_user_id = None  # Can search all documents
            print("User is admin - can search all documents")
        else:
            effective_user_id = test_userid  # Can only search own documents
            print("User is regular - can only search own documents")

        print(f"Effective user_id for search: {effective_user_id}")
    except Exception as e:
        print(f"Error checking user existence: {e}")
        effective_user_id = test_userid

    print()

    # Step 2: Check total available candidates
    print("STEP 2: Total Available Candidates")
    print("-" * 35)
    base_query = {}
    if effective_user_id is not None:
        base_query["user_id"] = effective_user_id

    total_candidates = resumes_collection.count_documents(base_query)
    print(f"Total candidates available: {total_candidates}")

    if total_candidates == 0:
        print("❌ No candidates found for this user!")
        return

    print()

    # Step 3: Get sample data to understand structure
    print("STEP 3: Sample Resume Data Analysis")
    print("-" * 37)

    sample_resumes = list(resumes_collection.find(base_query).limit(5))
    print(f"Found {len(sample_resumes)} sample resumes")
    print()

    # Analyze each sample resume for potential matches
    all_matched_keywords = {
        "experience_titles": set(),
        "skills": set(),
        "education": set(),
        "locations": set(),
        "salary_ranges": [],
        "experience_ranges": [],
    }

    for i, resume in enumerate(sample_resumes, 1):
        print(f"RESUME {i} ANALYSIS:")
        print(f"User ID: {resume.get('user_id', 'N/A')}")
        print(f"Name: {resume.get('contact_details', {}).get('name', 'N/A')}")

        # Analyze experience titles
        print("\n🏢 Experience Titles:")
        experience_titles = []
        for exp in resume.get("experience", []):
            title = exp.get("title", "")
            if title:
                experience_titles.append(title)
                # Check for matches with search criteria
                for search_title in search_criteria["experience_titles"]:
                    if search_title.lower() in title.lower():
                        all_matched_keywords["experience_titles"].add(search_title)
                        print(f"  ✅ MATCH: '{title}' contains '{search_title}'")
                    else:
                        print(f"  ❌ NO MATCH: '{title}' vs '{search_title}'")

        if not experience_titles:
            print("  No experience titles found")

        # Analyze skills
        print("\n🛠️ Skills:")
        skills = resume.get("skills", [])
        may_also_known_skills = resume.get("may_also_known_skills", [])
        all_skills = skills + may_also_known_skills

        print(f"  Skills: {skills}")
        print(f"  May also known skills: {may_also_known_skills}")

        for search_skill in search_criteria["skills"]:
            skill_found = False
            for skill in all_skills:
                if search_skill.lower() in skill.lower():
                    all_matched_keywords["skills"].add(search_skill)
                    print(f"  ✅ SKILL MATCH: '{skill}' contains '{search_skill}'")
                    skill_found = True
            if not skill_found:
                print(f"  ❌ NO SKILL MATCH for: '{search_skill}'")

        # Analyze education
        print("\n🎓 Education:")
        academic_details = resume.get("academic_details", [])
        for edu in academic_details:
            education = edu.get("education", "")
            print(f"  Education: {education}")
            for search_edu in search_criteria["min_education"]:
                if search_edu.lower() in education.lower():
                    all_matched_keywords["education"].add(search_edu)
                    print(
                        f"  ✅ EDUCATION MATCH: '{education}' contains '{search_edu}'"
                    )

        # Analyze locations
        print("\n📍 Locations:")
        current_city = resume.get("contact_details", {}).get("current_city", "")
        looking_for_jobs_in = resume.get("contact_details", {}).get(
            "looking_for_jobs_in", []
        )

        print(f"  Current city: {current_city}")
        print(f"  Looking for jobs in: {looking_for_jobs_in}")

        for search_location in search_criteria["locations"]:
            if current_city and search_location.lower() in current_city.lower():
                all_matched_keywords["locations"].add(search_location)
                print(
                    f"  ✅ LOCATION MATCH: '{current_city}' contains '{search_location}'"
                )

            for job_location in looking_for_jobs_in:
                if job_location and search_location.lower() in job_location.lower():
                    all_matched_keywords["locations"].add(search_location)
                    print(
                        f"  ✅ JOB LOCATION MATCH: '{job_location}' contains '{search_location}'"
                    )

        # Analyze salary
        print("\n💰 Salary:")
        expected_salary = resume.get("expected_salary", 0)
        current_salary = resume.get("current_salary", 0)
        print(f"  Expected salary: {expected_salary}")
        print(f"  Current salary: {current_salary}")

        candidate_salary = (
            expected_salary
            if expected_salary and expected_salary > 0
            else current_salary
        )
        if candidate_salary:
            all_matched_keywords["salary_ranges"].append(candidate_salary)
            if (
                search_criteria["min_salary"]
                <= candidate_salary
                <= search_criteria["max_salary"]
            ):
                print(
                    f"  ✅ SALARY MATCH: {candidate_salary} is within {search_criteria['min_salary']}-{search_criteria['max_salary']}"
                )
            else:
                print(
                    f"  ❌ SALARY NO MATCH: {candidate_salary} not in {search_criteria['min_salary']}-{search_criteria['max_salary']}"
                )

        # Analyze total experience
        print("\n⏱️ Experience:")
        total_experience = resume.get("total_experience", "")
        print(f"  Total experience: {total_experience}")

        if total_experience and total_experience != "N/A":
            all_matched_keywords["experience_ranges"].append(total_experience)

        print("\n" + "=" * 50)

    # Step 4: Summary of all matched keywords
    print("\n📊 MATCHED KEYWORDS SUMMARY")
    print("-" * 35)

    print(
        f"Experience titles that match: {list(all_matched_keywords['experience_titles'])}"
    )
    print(f"Skills that match: {list(all_matched_keywords['skills'])}")
    print(f"Education that matches: {list(all_matched_keywords['education'])}")
    print(f"Locations that match: {list(all_matched_keywords['locations'])}")
    print(f"Salary ranges found: {all_matched_keywords['salary_ranges']}")
    print(f"Experience ranges found: {all_matched_keywords['experience_ranges']}")

    # Step 5: Test actual MongoDB query
    print("\n🔍 TESTING MONGODB QUERY")
    print("-" * 30)

    # Build OR conditions like in the actual code
    or_conditions = []

    # Experience titles
    for title in search_criteria["experience_titles"]:
        title_pattern = re.compile(f".*{re.escape(title)}.*", re.IGNORECASE)
        or_conditions.append({"experience.title": {"$regex": title_pattern}})

    # Skills
    for skill in search_criteria["skills"]:
        skill_pattern = re.compile(f".*{re.escape(skill)}.*", re.IGNORECASE)
        or_conditions.extend(
            [
                {"skills": {"$regex": skill_pattern}},
                {"may_also_known_skills": {"$regex": skill_pattern}},
            ]
        )

    # Education
    for edu in search_criteria["min_education"]:
        education_pattern = re.compile(f".*{re.escape(edu)}.*", re.IGNORECASE)
        or_conditions.append(
            {"academic_details.education": {"$regex": education_pattern}}
        )

    # Locations
    for location in search_criteria["locations"]:
        location_pattern = re.compile(f".*{re.escape(location)}.*", re.IGNORECASE)
        or_conditions.extend(
            [
                {"contact_details.current_city": {"$regex": location_pattern}},
                {"contact_details.looking_for_jobs_in": {"$regex": location_pattern}},
            ]
        )

    if or_conditions:
        final_query = {"$and": [base_query, {"$or": or_conditions}]}
    else:
        final_query = base_query

    print(f"MongoDB Query: {json.dumps(final_query, default=str, indent=2)}")

    try:
        query_results = list(resumes_collection.find(final_query))
        print(f"Query returned {len(query_results)} results")

        if query_results:
            print("\n✅ Found matching resumes!")
            for result in query_results:
                name = result.get("contact_details", {}).get("name", "N/A")
                print(f"  - {name}")
        else:
            print("\n❌ No results from MongoDB query")

            # Let's try individual queries to see which parts are failing
            print("\n🔍 Testing individual query components:")

            # Test experience titles
            for title in search_criteria["experience_titles"]:
                title_pattern = re.compile(f".*{re.escape(title)}.*", re.IGNORECASE)
                title_query = {
                    "$and": [
                        base_query,
                        {"experience.title": {"$regex": title_pattern}},
                    ]
                }
                title_results = resumes_collection.count_documents(title_query)
                print(f"  Experience title '{title}': {title_results} matches")

            # Test skills
            for skill in search_criteria["skills"]:
                skill_pattern = re.compile(f".*{re.escape(skill)}.*", re.IGNORECASE)
                skill_query1 = {
                    "$and": [base_query, {"skills": {"$regex": skill_pattern}}]
                }
                skill_query2 = {
                    "$and": [
                        base_query,
                        {"may_also_known_skills": {"$regex": skill_pattern}},
                    ]
                }
                skill_results1 = resumes_collection.count_documents(skill_query1)
                skill_results2 = resumes_collection.count_documents(skill_query2)
                print(f"  Skill '{skill}' in skills: {skill_results1} matches")
                print(
                    f"  Skill '{skill}' in may_also_known_skills: {skill_results2} matches"
                )

            # Test education
            for edu in search_criteria["min_education"]:
                edu_pattern = re.compile(f".*{re.escape(edu)}.*", re.IGNORECASE)
                edu_query = {
                    "$and": [
                        base_query,
                        {"academic_details.education": {"$regex": edu_pattern}},
                    ]
                }
                edu_results = resumes_collection.count_documents(edu_query)
                print(f"  Education '{edu}': {edu_results} matches")

            # Test locations
            for location in search_criteria["locations"]:
                location_pattern = re.compile(
                    f".*{re.escape(location)}.*", re.IGNORECASE
                )
                loc_query1 = {
                    "$and": [
                        base_query,
                        {"contact_details.current_city": {"$regex": location_pattern}},
                    ]
                }
                loc_query2 = {
                    "$and": [
                        base_query,
                        {
                            "contact_details.looking_for_jobs_in": {
                                "$regex": location_pattern
                            }
                        },
                    ]
                }
                loc_results1 = resumes_collection.count_documents(loc_query1)
                loc_results2 = resumes_collection.count_documents(loc_query2)
                print(
                    f"  Location '{location}' in current_city: {loc_results1} matches"
                )
                print(
                    f"  Location '{location}' in looking_for_jobs_in: {loc_results2} matches"
                )

    except Exception as e:
        print(f"Error executing MongoDB query: {e}")


def create_matched_keywords_file():
    """Create a file with all matched keywords for analysis"""

    print("\n📝 CREATING MATCHED KEYWORDS FILE")
    print("-" * 40)

    # Initialize database connection
    resumes_collection = get_collection()

    # Get all resumes
    all_resumes = list(resumes_collection.find({}))

    # Extract all unique keywords
    all_keywords = {
        "experience_titles": set(),
        "skills": set(),
        "education": set(),
        "locations_current": set(),
        "locations_preference": set(),
        "salary_ranges": [],
        "experience_ranges": [],
    }

    for resume in all_resumes:
        # Extract experience titles
        for exp in resume.get("experience", []):
            title = exp.get("title", "").strip()
            if title:
                all_keywords["experience_titles"].add(title.lower())

        # Extract skills
        for skill in resume.get("skills", []):
            if skill.strip():
                all_keywords["skills"].add(skill.lower().strip())

        for skill in resume.get("may_also_known_skills", []):
            if skill.strip():
                all_keywords["skills"].add(skill.lower().strip())

        # Extract education
        for edu in resume.get("academic_details", []):
            education = edu.get("education", "").strip()
            if education:
                all_keywords["education"].add(education.lower())

        # Extract locations
        current_city = resume.get("contact_details", {}).get("current_city", "")
        if current_city and current_city != "N/A":
            all_keywords["locations_current"].add(current_city.lower().strip())

        looking_for_jobs_in = resume.get("contact_details", {}).get(
            "looking_for_jobs_in", []
        )
        for location in looking_for_jobs_in:
            if location and location != "N/A":
                all_keywords["locations_preference"].add(location.lower().strip())

        # Extract salary ranges
        expected_salary = resume.get("expected_salary", 0)
        current_salary = resume.get("current_salary", 0)
        if expected_salary and expected_salary > 0:
            all_keywords["salary_ranges"].append(expected_salary)
        if current_salary and current_salary > 0:
            all_keywords["salary_ranges"].append(current_salary)

        # Extract experience ranges
        total_experience = resume.get("total_experience", "")
        if total_experience and total_experience != "N/A":
            all_keywords["experience_ranges"].append(total_experience)

    # Convert sets to sorted lists
    keywords_data = {
        "experience_titles": sorted(list(all_keywords["experience_titles"])),
        "skills": sorted(list(all_keywords["skills"])),
        "education": sorted(list(all_keywords["education"])),
        "locations_current": sorted(list(all_keywords["locations_current"])),
        "locations_preference": sorted(list(all_keywords["locations_preference"])),
        "salary_ranges": sorted(list(set(all_keywords["salary_ranges"]))),
        "experience_ranges": list(set(all_keywords["experience_ranges"])),
    }

    # Create the file
    with open(
        "d:\\UPH\\uphire_v2\\matched_keywords_analysis.json", "w", encoding="utf-8"
    ) as f:
        json.dump(keywords_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Created matched_keywords_analysis.json with:")
    print(f"  - {len(keywords_data['experience_titles'])} unique experience titles")
    print(f"  - {len(keywords_data['skills'])} unique skills")
    print(f"  - {len(keywords_data['education'])} unique education levels")
    print(f"  - {len(keywords_data['locations_current'])} unique current cities")
    print(
        f"  - {len(keywords_data['locations_preference'])} unique job preference locations"
    )
    print(f"  - {len(keywords_data['salary_ranges'])} unique salary ranges")
    print(f"  - {len(keywords_data['experience_ranges'])} unique experience ranges")


if __name__ == "__main__":
    try:
        analyze_search_issue()
        create_matched_keywords_file()
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()
