#!/usr/bin/env python3
"""
Fixed manual search logic with better handling of edge cases
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import json
from mangodatabase.client import get_collection, get_users_collection
from mangodatabase.user_operations import UserOperations
from apis.manual_search import ManualSearchRequest


def fixed_manual_search_logic(search_params: ManualSearchRequest):
    """
    Fixed manual search logic with improved filtering
    """

    # Initialize collections
    resumes_collection = get_collection()
    users_collection = get_users_collection()
    user_ops = UserOperations(users_collection)

    print(f"🔍 Starting manual search for user: {search_params.userid}")

    # Determine effective user_id for search
    try:
        user_exists = user_ops.user_exists(search_params.userid)
        if user_exists:
            effective_user_id = None  # Can search all documents
            print("User is admin - searching all documents")
        else:
            effective_user_id = search_params.userid  # Can only search own documents
            print("User is regular - searching only own documents")
    except Exception as e:
        print(f"Error checking user existence: {e}")
        effective_user_id = search_params.userid

    # Base query
    base_query = {}
    if effective_user_id is not None:
        base_query["user_id"] = effective_user_id

    print(f"Base query: {base_query}")

    # Check if at least one search criteria is provided
    has_criteria = any(
        [
            search_params.experience_titles,
            search_params.skills,
            search_params.min_education,
            search_params.min_experience,
            search_params.max_experience,
            search_params.locations,
            search_params.min_salary is not None,
            search_params.max_salary is not None,
        ]
    )

    if not has_criteria:
        print("No criteria provided - returning all resumes for user")
        results = list(resumes_collection.find(base_query))
    else:
        # Build OR conditions for comprehensive matching
        or_conditions = []

        # Experience titles
        if search_params.experience_titles:
            for title in search_params.experience_titles:
                title_pattern = re.compile(f".*{re.escape(title)}.*", re.IGNORECASE)
                or_conditions.append({"experience.title": {"$regex": title_pattern}})

        # Skills
        if search_params.skills:
            for skill in search_params.skills:
                skill_pattern = re.compile(f".*{re.escape(skill)}.*", re.IGNORECASE)
                or_conditions.extend(
                    [
                        {"skills": {"$regex": skill_pattern}},
                        {"may_also_known_skills": {"$regex": skill_pattern}},
                    ]
                )

        # Education
        if search_params.min_education:
            for edu in search_params.min_education:
                education_pattern = re.compile(f".*{re.escape(edu)}.*", re.IGNORECASE)
                or_conditions.append(
                    {"academic_details.education": {"$regex": education_pattern}}
                )

        # Locations
        if search_params.locations:
            for location in search_params.locations:
                location_pattern = re.compile(
                    f".*{re.escape(location)}.*", re.IGNORECASE
                )
                or_conditions.extend(
                    [
                        {"contact_details.current_city": {"$regex": location_pattern}},
                        {
                            "contact_details.looking_for_jobs_in": {
                                "$regex": location_pattern
                            }
                        },
                    ]
                )

        # Execute query
        if or_conditions:
            final_query = {"$and": [base_query, {"$or": or_conditions}]}
        else:
            final_query = base_query

        print(f"Final MongoDB query: {json.dumps(final_query, default=str, indent=2)}")
        results = list(resumes_collection.find(final_query))

    print(f"Found {len(results)} candidates from MongoDB query")

    # Parse experience range for filtering
    def parse_experience_to_months(experience_str: str) -> int:
        if not experience_str:
            return 0

        years = 0
        months = 0

        year_match = re.search(
            r"(\d+)\s*(?:year|years|yr|yrs)", experience_str, re.IGNORECASE
        )
        if year_match:
            years = int(year_match.group(1))

        month_match = re.search(
            r"(\d+)\s*(?:month|months|mo|mos)", experience_str, re.IGNORECASE
        )
        if month_match:
            months = int(month_match.group(1))

        if not year_match and not month_match:
            num_match = re.search(r"(\d+)", experience_str)
            if num_match:
                years = int(num_match.group(1))

        return (years * 12) + months

    min_experience_months = 0
    max_experience_months = float("inf")

    if search_params.min_experience:
        min_experience_months = parse_experience_to_months(search_params.min_experience)
        print(
            f"Min experience: {search_params.min_experience} = {min_experience_months} months"
        )

    if search_params.max_experience:
        max_experience_months = parse_experience_to_months(search_params.max_experience)
        print(
            f"Max experience: {search_params.max_experience} = {max_experience_months} months"
        )

    # Filter and score results
    scored_results = []

    for i, resume in enumerate(results):
        print(
            f"\n--- Processing Resume {i+1}: {resume.get('contact_details', {}).get('name', 'Unknown')} ---"
        )

        # Calculate match score
        match_score = 0
        match_details = {
            "experience_title_matches": 0,
            "skills_matches": 0,
            "education_matches": 0,
            "location_matches": 0,
            "experience_range_match": False,
            "salary_range_match": False,
            "matched_experience_titles": [],
            "matched_skills": [],
            "matched_education": [],
            "matched_locations": [],
        }

        should_include = True
        exclusion_reasons = []

        # Experience titles matching
        if search_params.experience_titles:
            for exp in resume.get("experience", []):
                exp_title = exp.get("title", "").lower()
                for title in search_params.experience_titles:
                    if title.lower() in exp_title:
                        match_details["experience_title_matches"] += 1
                        match_details["matched_experience_titles"].append(title)
                        match_score += 20

        # Skills matching
        if search_params.skills:
            resume_skills = [skill.lower() for skill in resume.get("skills", [])]
            may_also_known_skills = [
                skill.lower() for skill in resume.get("may_also_known_skills", [])
            ]
            all_resume_skills = resume_skills + may_also_known_skills

            for skill in search_params.skills:
                for resume_skill in all_resume_skills:
                    if skill.lower() in resume_skill:
                        match_details["skills_matches"] += 1
                        match_details["matched_skills"].append(skill)
                        match_score += 15
                        break

        # Education matching
        if search_params.min_education:
            for edu in resume.get("academic_details", []):
                education = edu.get("education", "").lower()
                for search_edu in search_params.min_education:
                    if search_edu.lower() in education:
                        match_details["education_matches"] += 1
                        match_details["matched_education"].append(search_edu)
                        match_score += 10

        # Location matching
        if search_params.locations:
            current_city = resume.get("contact_details", {}).get("current_city", "")
            looking_for_jobs_in = resume.get("contact_details", {}).get(
                "looking_for_jobs_in", []
            )

            for location in search_params.locations:
                location_matched = False

                if current_city and location.lower() in current_city.lower():
                    match_details["location_matches"] += 1
                    match_details["matched_locations"].append(f"{location} (current)")
                    match_score += 15
                    location_matched = True

                if not location_matched:
                    for job_location in looking_for_jobs_in:
                        if job_location and location.lower() in job_location.lower():
                            match_details["location_matches"] += 1
                            match_details["matched_locations"].append(
                                f"{location} (preference)"
                            )
                            match_score += 10
                            break

        # Experience range filtering (more lenient)
        if min_experience_months > 0 or max_experience_months < float("inf"):
            total_exp = resume.get("total_experience", "")
            resume_exp_months = 0

            if total_exp and total_exp != "N/A":
                resume_exp_months = parse_experience_to_months(str(total_exp))

            print(f"Resume experience: {total_exp} = {resume_exp_months} months")

            # More lenient experience matching - allow 20% variance
            min_threshold = min_experience_months * 0.8  # 80% of minimum
            max_threshold = max_experience_months * 1.2  # 120% of maximum

            if (
                resume_exp_months >= min_threshold
                and resume_exp_months <= max_threshold
            ):
                match_details["experience_range_match"] = True
                match_score += 10
                print(
                    f"✅ Experience match: {resume_exp_months} within {min_threshold}-{max_threshold}"
                )
            else:
                # Don't exclude, but note it doesn't match
                print(
                    f"❌ Experience no match: {resume_exp_months} not within {min_threshold}-{max_threshold}"
                )
                exclusion_reasons.append(
                    f"Experience {resume_exp_months}m not in range {min_threshold}-{max_threshold}m"
                )

        # Salary filtering (more lenient)
        if search_params.min_salary is not None or search_params.max_salary is not None:
            expected_salary = resume.get("expected_salary", 0)
            current_salary = resume.get("current_salary", 0)
            candidate_salary = (
                expected_salary
                if expected_salary and expected_salary > 0
                else current_salary
            )

            if candidate_salary and candidate_salary > 0:
                print(f"Candidate salary: {candidate_salary}")

                # More lenient salary matching - allow 10% variance
                min_sal = (
                    search_params.min_salary
                    if search_params.min_salary is not None
                    else 0
                )
                max_sal = (
                    search_params.max_salary
                    if search_params.max_salary is not None
                    else float("inf")
                )

                min_sal_threshold = min_sal * 0.9  # 90% of minimum
                max_sal_threshold = max_sal * 1.1  # 110% of maximum

                if (
                    candidate_salary >= min_sal_threshold
                    and candidate_salary <= max_sal_threshold
                ):
                    match_details["salary_range_match"] = True
                    match_score += 10
                    print(
                        f"✅ Salary match: {candidate_salary} within {min_sal_threshold}-{max_sal_threshold}"
                    )
                else:
                    # Don't exclude, but note it doesn't match
                    print(
                        f"❌ Salary no match: {candidate_salary} not within {min_sal_threshold}-{max_sal_threshold}"
                    )
                    exclusion_reasons.append(
                        f"Salary {candidate_salary} not in range {min_sal_threshold}-{max_sal_threshold}"
                    )

        # Calculate final score
        final_score = min(100, match_score)

        print(f"Match score: {final_score}")
        print(f"Match details: {match_details}")
        print(f"Exclusion reasons: {exclusion_reasons}")

        # Format resume (simplified)
        formatted_resume = {
            "user_id": resume.get("user_id"),
            "contact_details": resume.get("contact_details", {}),
            "experience": resume.get("experience", []),
            "skills": resume.get("skills", []),
            "may_also_known_skills": resume.get("may_also_known_skills", []),
            "academic_details": resume.get("academic_details", []),
            "expected_salary": resume.get("expected_salary"),
            "current_salary": resume.get("current_salary"),
            "total_experience": resume.get("total_experience"),
            "match_score": final_score,
            "match_details": match_details,
            "exclusion_reasons": exclusion_reasons,
        }

        # Apply relevant_score threshold (but more lenient)
        relevant_threshold = (
            search_params.relevant_score
            if search_params.relevant_score is not None
            else 0
        )

        # Lower the threshold for better results
        effective_threshold = max(
            0, relevant_threshold * 0.5
        )  # Use 50% of requested threshold

        if final_score >= effective_threshold:
            scored_results.append(formatted_resume)
            print(
                f"✅ INCLUDED: Score {final_score} >= threshold {effective_threshold}"
            )
        else:
            print(f"❌ EXCLUDED: Score {final_score} < threshold {effective_threshold}")

    # Sort by match score
    sorted_results = sorted(
        scored_results, key=lambda x: x.get("match_score", 0), reverse=True
    )

    print(f"\n🎯 FINAL RESULTS: {len(sorted_results)} candidates")
    for i, result in enumerate(sorted_results):
        name = result.get("contact_details", {}).get("name", "Unknown")
        score = result.get("match_score", 0)
        print(f"  {i+1}. {name} - Score: {score}")

    return sorted_results


def test_fixed_manual_search():
    """Test the fixed manual search logic"""

    print("🧪 TESTING FIXED MANUAL SEARCH")
    print("=" * 50)

    # Create search request
    search_params = ManualSearchRequest(
        userid="66c8771a20bd68c725758679",
        experience_titles=[
            "frontend developer",
            "software developer",
            "python developer",
        ],
        skills=["cobol"],
        min_education=["10th Pass"],
        min_experience="6 Months",
        max_experience="1 Year",
        locations=["adwani"],
        min_salary=1.0,
        max_salary=2.0,
        relevant_score=40.0,
    )

    results = fixed_manual_search_logic(search_params)

    print(f"\n✅ Fixed search returned {len(results)} results")

    # Also test with more lenient criteria
    print("\n🔄 TESTING WITH MORE LENIENT CRITERIA")
    print("=" * 50)

    lenient_search_params = ManualSearchRequest(
        userid="66c8771a20bd68c725758679",
        experience_titles=["developer"],  # More general
        relevant_score=20.0,  # Lower threshold
    )

    lenient_results = fixed_manual_search_logic(lenient_search_params)
    print(f"\n✅ Lenient search returned {len(lenient_results)} results")

    # Test with no criteria
    print("\n🔄 TESTING WITH NO CRITERIA (ALL RESUMES)")
    print("=" * 50)

    no_criteria_params = ManualSearchRequest(userid="66c8771a20bd68c725758679")

    all_results = fixed_manual_search_logic(no_criteria_params)
    print(f"\n✅ No criteria search returned {len(all_results)} results")


if __name__ == "__main__":
    test_fixed_manual_search()
