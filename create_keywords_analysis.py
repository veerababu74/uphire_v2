#!/usr/bin/env python3
"""
Create comprehensive matched keywords analysis for manual search optimization
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from mangodatabase.client import get_collection
from collections import defaultdict


def create_comprehensive_keywords_analysis():
    """Create a comprehensive analysis of all keywords in the database"""

    print("🔍 CREATING COMPREHENSIVE KEYWORDS ANALYSIS")
    print("=" * 60)

    # Initialize database connection
    resumes_collection = get_collection()

    # Get all resumes
    all_resumes = list(resumes_collection.find({}))
    print(f"Analyzing {len(all_resumes)} total resumes")

    # Initialize keyword collections
    keywords_analysis = {
        "experience_titles": {
            "unique_titles": set(),
            "title_frequency": defaultdict(int),
            "common_words": defaultdict(int),
        },
        "skills": {
            "unique_skills": set(),
            "skill_frequency": defaultdict(int),
            "skill_categories": defaultdict(list),
        },
        "education": {
            "unique_education": set(),
            "education_frequency": defaultdict(int),
            "education_levels": defaultdict(list),
        },
        "locations": {
            "current_cities": set(),
            "job_preferences": set(),
            "city_frequency": defaultdict(int),
            "states_countries": defaultdict(list),
        },
        "salary_analysis": {
            "salary_ranges": [],
            "expected_salary_stats": {},
            "current_salary_stats": {},
        },
        "experience_analysis": {
            "experience_ranges": [],
            "parsed_experience_months": [],
            "experience_patterns": defaultdict(int),
        },
    }

    # Parse experience to months helper function
    def parse_experience_to_months(experience_str: str) -> int:
        import re

        if not experience_str or experience_str == "N/A":
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

    # Analyze each resume
    for resume in all_resumes:
        # Extract and analyze experience titles
        for exp in resume.get("experience", []):
            title = exp.get("title", "").strip()
            if title and title.lower() != "null":
                title_lower = title.lower()
                keywords_analysis["experience_titles"]["unique_titles"].add(title_lower)
                keywords_analysis["experience_titles"]["title_frequency"][
                    title_lower
                ] += 1

                # Extract common words from titles
                words = title_lower.replace("-", " ").replace("/", " ").split()
                for word in words:
                    if len(word) > 2:  # Ignore very short words
                        keywords_analysis["experience_titles"]["common_words"][
                            word
                        ] += 1

        # Extract and analyze skills
        all_skills = resume.get("skills", []) + resume.get("may_also_known_skills", [])
        for skill in all_skills:
            if skill and skill.strip():
                skill_clean = skill.lower().strip()
                keywords_analysis["skills"]["unique_skills"].add(skill_clean)
                keywords_analysis["skills"]["skill_frequency"][skill_clean] += 1

                # Categorize skills by type
                if any(
                    word in skill_clean
                    for word in [
                        "python",
                        "java",
                        "javascript",
                        "c++",
                        "c#",
                        "php",
                        "ruby",
                        "go",
                        "rust",
                    ]
                ):
                    keywords_analysis["skills"]["skill_categories"][
                        "programming_languages"
                    ].append(skill_clean)
                elif any(
                    word in skill_clean
                    for word in [
                        "react",
                        "angular",
                        "vue",
                        "node",
                        "express",
                        "django",
                        "flask",
                        "spring",
                    ]
                ):
                    keywords_analysis["skills"]["skill_categories"][
                        "frameworks"
                    ].append(skill_clean)
                elif any(
                    word in skill_clean
                    for word in ["aws", "azure", "gcp", "docker", "kubernetes", "cloud"]
                ):
                    keywords_analysis["skills"]["skill_categories"][
                        "cloud_devops"
                    ].append(skill_clean)
                elif any(
                    word in skill_clean
                    for word in [
                        "sql",
                        "mysql",
                        "postgresql",
                        "mongodb",
                        "redis",
                        "database",
                    ]
                ):
                    keywords_analysis["skills"]["skill_categories"]["databases"].append(
                        skill_clean
                    )
                else:
                    keywords_analysis["skills"]["skill_categories"]["others"].append(
                        skill_clean
                    )

        # Extract and analyze education
        for edu in resume.get("academic_details", []):
            education = edu.get("education", "").strip()
            if education and education.lower() != "unknown":
                education_lower = education.lower()
                keywords_analysis["education"]["unique_education"].add(education_lower)
                keywords_analysis["education"]["education_frequency"][
                    education_lower
                ] += 1

                # Categorize education levels
                if any(
                    word in education_lower for word in ["10th", "ssc", "high school"]
                ):
                    keywords_analysis["education"]["education_levels"][
                        "secondary"
                    ].append(education_lower)
                elif any(
                    word in education_lower
                    for word in ["12th", "hsc", "intermediate", "higher secondary"]
                ):
                    keywords_analysis["education"]["education_levels"][
                        "higher_secondary"
                    ].append(education_lower)
                elif any(
                    word in education_lower for word in ["diploma", "certificate"]
                ):
                    keywords_analysis["education"]["education_levels"][
                        "diploma"
                    ].append(education_lower)
                elif any(
                    word in education_lower
                    for word in [
                        "bachelor",
                        "btech",
                        "be",
                        "bsc",
                        "ba",
                        "bcom",
                        "degree",
                    ]
                ):
                    keywords_analysis["education"]["education_levels"][
                        "bachelors"
                    ].append(education_lower)
                elif any(
                    word in education_lower
                    for word in ["master", "mtech", "me", "msc", "ma", "mcom", "mba"]
                ):
                    keywords_analysis["education"]["education_levels"][
                        "masters"
                    ].append(education_lower)
                elif any(word in education_lower for word in ["phd", "doctorate"]):
                    keywords_analysis["education"]["education_levels"][
                        "doctorate"
                    ].append(education_lower)
                else:
                    keywords_analysis["education"]["education_levels"]["others"].append(
                        education_lower
                    )

        # Extract and analyze locations
        current_city = resume.get("contact_details", {}).get("current_city", "")
        if (
            current_city
            and current_city != "N/A"
            and current_city.lower() != "location not specified"
        ):
            city_clean = current_city.lower().strip()
            keywords_analysis["locations"]["current_cities"].add(city_clean)
            keywords_analysis["locations"]["city_frequency"][city_clean] += 1

            # Extract state/country info
            if "," in current_city:
                parts = [part.strip() for part in current_city.split(",")]
                if len(parts) > 1:
                    keywords_analysis["locations"]["states_countries"]["states"].extend(
                        parts[1:]
                    )

        looking_for_jobs_in = resume.get("contact_details", {}).get(
            "looking_for_jobs_in", []
        )
        for location in looking_for_jobs_in:
            if (
                location
                and location != "N/A"
                and location.lower() != "location not specified"
            ):
                location_clean = location.lower().strip()
                keywords_analysis["locations"]["job_preferences"].add(location_clean)
                keywords_analysis["locations"]["city_frequency"][location_clean] += 1

        # Analyze salary data
        expected_salary = resume.get("expected_salary", 0)
        current_salary = resume.get("current_salary", 0)

        if expected_salary and expected_salary > 0:
            keywords_analysis["salary_analysis"]["salary_ranges"].append(
                ("expected", expected_salary)
            )

        if current_salary and current_salary > 0:
            keywords_analysis["salary_analysis"]["salary_ranges"].append(
                ("current", current_salary)
            )

        # Analyze experience data
        total_experience = resume.get("total_experience", "")
        if total_experience and total_experience != "N/A":
            keywords_analysis["experience_analysis"]["experience_ranges"].append(
                total_experience
            )

            # Parse to months for statistics
            exp_months = parse_experience_to_months(total_experience)
            if exp_months > 0:
                keywords_analysis["experience_analysis"][
                    "parsed_experience_months"
                ].append(exp_months)

            # Pattern analysis
            if "year" in total_experience.lower():
                keywords_analysis["experience_analysis"]["experience_patterns"][
                    "years_format"
                ] += 1
            if "month" in total_experience.lower():
                keywords_analysis["experience_analysis"]["experience_patterns"][
                    "months_format"
                ] += 1
            if "+" in total_experience:
                keywords_analysis["experience_analysis"]["experience_patterns"][
                    "plus_format"
                ] += 1

    # Calculate salary statistics
    all_salaries = [
        salary for _, salary in keywords_analysis["salary_analysis"]["salary_ranges"]
    ]
    if all_salaries:
        keywords_analysis["salary_analysis"]["expected_salary_stats"] = {
            "min": min(all_salaries),
            "max": max(all_salaries),
            "avg": sum(all_salaries) / len(all_salaries),
            "count": len(all_salaries),
        }

    # Calculate experience statistics
    exp_months = keywords_analysis["experience_analysis"]["parsed_experience_months"]
    if exp_months:
        keywords_analysis["experience_analysis"]["experience_stats"] = {
            "min_months": min(exp_months),
            "max_months": max(exp_months),
            "avg_months": sum(exp_months) / len(exp_months),
            "min_years": min(exp_months) / 12,
            "max_years": max(exp_months) / 12,
            "avg_years": sum(exp_months) / len(exp_months) / 12,
            "count": len(exp_months),
        }

    # Convert sets to sorted lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, set):
            return sorted(list(obj))
        elif isinstance(obj, defaultdict):
            return dict(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    final_analysis = convert_for_json(keywords_analysis)

    # Add summary statistics
    final_analysis["summary"] = {
        "total_resumes_analyzed": len(all_resumes),
        "unique_experience_titles": len(
            keywords_analysis["experience_titles"]["unique_titles"]
        ),
        "unique_skills": len(keywords_analysis["skills"]["unique_skills"]),
        "unique_education_levels": len(
            keywords_analysis["education"]["unique_education"]
        ),
        "unique_current_cities": len(keywords_analysis["locations"]["current_cities"]),
        "unique_job_preference_cities": len(
            keywords_analysis["locations"]["job_preferences"]
        ),
        "total_salary_data_points": len(
            keywords_analysis["salary_analysis"]["salary_ranges"]
        ),
        "total_experience_data_points": len(
            keywords_analysis["experience_analysis"]["experience_ranges"]
        ),
    }

    # Get top items for quick reference
    final_analysis["top_items"] = {
        "most_common_titles": sorted(
            keywords_analysis["experience_titles"]["title_frequency"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:20],
        "most_common_skills": sorted(
            keywords_analysis["skills"]["skill_frequency"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:30],
        "most_common_education": sorted(
            keywords_analysis["education"]["education_frequency"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:15],
        "most_common_cities": sorted(
            keywords_analysis["locations"]["city_frequency"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:20],
        "most_common_title_words": sorted(
            keywords_analysis["experience_titles"]["common_words"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:30],
    }

    # Save the analysis
    output_file = "d:\\UPH\\uphire_v2\\comprehensive_keywords_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_analysis, f, indent=2, ensure_ascii=False)

    print(f"✅ Created comprehensive_keywords_analysis.json")
    print(f"📊 Analysis Summary:")
    print(f"   - {final_analysis['summary']['total_resumes_analyzed']} total resumes")
    print(
        f"   - {final_analysis['summary']['unique_experience_titles']} unique job titles"
    )
    print(f"   - {final_analysis['summary']['unique_skills']} unique skills")
    print(
        f"   - {final_analysis['summary']['unique_education_levels']} unique education levels"
    )
    print(f"   - {final_analysis['summary']['unique_current_cities']} unique cities")

    print(f"\n🏆 Top Job Titles:")
    for title, count in final_analysis["top_items"]["most_common_titles"][:10]:
        print(f"   - {title}: {count} candidates")

    print(f"\n🛠️ Top Skills:")
    for skill, count in final_analysis["top_items"]["most_common_skills"][:10]:
        print(f"   - {skill}: {count} candidates")

    print(f"\n📍 Top Cities:")
    for city, count in final_analysis["top_items"]["most_common_cities"][:10]:
        print(f"   - {city}: {count} candidates")

    return final_analysis


if __name__ == "__main__":
    try:
        analysis = create_comprehensive_keywords_analysis()
        print("\n🎉 Keyword analysis created successfully!")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()
