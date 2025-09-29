#!/usr/bin/env python3
"""
Create a simple, user-friendly list of all matched keywords for manual search optimization
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from mangodatabase.client import get_collection


def create_simple_keywords_list():
    """Create a simple, easy-to-read list of all available keywords"""

    print("📝 CREATING SIMPLE KEYWORDS LIST")
    print("=" * 50)

    # Initialize database connection
    resumes_collection = get_collection()

    # Get all resumes
    all_resumes = list(resumes_collection.find({}))
    print(f"Analyzing {len(all_resumes)} resumes for keywords")

    # Collect all keywords
    all_keywords = {
        "job_titles": set(),
        "skills": set(),
        "education_levels": set(),
        "cities": set(),
        "experience_ranges": set(),
        "salary_ranges": set(),
    }

    for resume in all_resumes:
        # Job titles
        for exp in resume.get("experience", []):
            title = exp.get("title", "").strip()
            if title and title.lower() not in ["null", "n/a", ""]:
                all_keywords["job_titles"].add(title.lower())

        # Skills
        for skill_list in [
            resume.get("skills", []),
            resume.get("may_also_known_skills", []),
        ]:
            for skill in skill_list:
                if skill and skill.strip():
                    all_keywords["skills"].add(skill.lower().strip())

        # Education
        for edu in resume.get("academic_details", []):
            education = edu.get("education", "").strip()
            if education and education.lower() not in ["unknown", "n/a", ""]:
                all_keywords["education_levels"].add(education.lower())

        # Cities
        current_city = resume.get("contact_details", {}).get("current_city", "")
        if current_city and current_city.lower() not in [
            "n/a",
            "location not specified",
            "",
        ]:
            all_keywords["cities"].add(current_city.lower())

        for location in resume.get("contact_details", {}).get(
            "looking_for_jobs_in", []
        ):
            if location and location.lower() not in [
                "n/a",
                "location not specified",
                "",
            ]:
                all_keywords["cities"].add(location.lower())

        # Experience ranges
        total_exp = resume.get("total_experience", "")
        if total_exp and total_exp.lower() not in ["n/a", ""]:
            all_keywords["experience_ranges"].add(total_exp)

        # Salary ranges
        for salary_field in ["expected_salary", "current_salary"]:
            salary = resume.get(salary_field, 0)
            if salary and salary > 0:
                all_keywords["salary_ranges"].add(salary)

    # Convert to sorted lists
    keywords_list = {
        "job_titles": sorted(list(all_keywords["job_titles"])),
        "skills": sorted(list(all_keywords["skills"])),
        "education_levels": sorted(list(all_keywords["education_levels"])),
        "cities": sorted(list(all_keywords["cities"])),
        "experience_ranges": sorted(list(all_keywords["experience_ranges"])),
        "salary_ranges": sorted(list(all_keywords["salary_ranges"])),
    }

    # Create a user-friendly text file
    output_file = "d:\\UPH\\uphire_v2\\available_keywords_for_search.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("AVAILABLE KEYWORDS FOR MANUAL SEARCH\\n")
        f.write("====================================\\n\\n")
        f.write("Use these keywords for better search results in manual search API.\\n")
        f.write("Copy and paste relevant keywords into your search criteria.\\n\\n")

        # Job Titles Section
        f.write(
            "🏢 JOB TITLES ({} available)\\n".format(len(keywords_list["job_titles"]))
        )
        f.write("-" * 40 + "\\n")
        f.write("Experience titles to search for:\\n\\n")

        for title in keywords_list["job_titles"]:
            f.write(f'  - "{title}"\\n')

        # Skills Section
        f.write("\\n\\n🛠️ SKILLS ({} available)\\n".format(len(keywords_list["skills"])))
        f.write("-" * 40 + "\\n")
        f.write("Technical and soft skills to search for:\\n\\n")

        # Group skills by first letter for better organization
        skills_by_letter = {}
        for skill in keywords_list["skills"]:
            first_letter = skill[0].upper() if skill else "A"
            if first_letter not in skills_by_letter:
                skills_by_letter[first_letter] = []
            skills_by_letter[first_letter].append(skill)

        for letter in sorted(skills_by_letter.keys()):
            f.write(f"\\n[{letter}]\\n")
            for skill in skills_by_letter[letter][:10]:  # Show top 10 per letter
                f.write(f'  - "{skill}"\\n')

        # Education Section
        f.write(
            "\\n\\n🎓 EDUCATION LEVELS ({} available)\\n".format(
                len(keywords_list["education_levels"])
            )
        )
        f.write("-" * 40 + "\\n")
        f.write("Education qualifications to search for:\\n\\n")

        for edu in keywords_list["education_levels"]:
            f.write(f'  - "{edu}"\\n')

        # Cities Section
        f.write(
            "\\n\\n📍 CITIES ({} available)\\n".format(len(keywords_list["cities"]))
        )
        f.write("-" * 40 + "\\n")
        f.write("Locations to search for:\\n\\n")

        for city in keywords_list["cities"]:
            f.write(f'  - "{city}"\\n')

        # Experience Ranges
        f.write(
            "\\n\\n⏱️ EXPERIENCE RANGES ({} available)\\n".format(
                len(keywords_list["experience_ranges"])
            )
        )
        f.write("-" * 40 + "\\n")
        f.write("Experience patterns found in database:\\n\\n")

        for exp in keywords_list["experience_ranges"]:
            f.write(f'  - "{exp}"\\n')

        # Salary Ranges
        f.write(
            "\\n\\n💰 SALARY RANGES ({} available)\\n".format(
                len(keywords_list["salary_ranges"])
            )
        )
        f.write("-" * 40 + "\\n")
        f.write("Salary values found in database:\\n\\n")

        # Group salaries by range for better readability
        salary_ranges = {
            "0-2 Lakhs": [],
            "2-5 Lakhs": [],
            "5-10 Lakhs": [],
            "10+ Lakhs": [],
        }

        for salary in keywords_list["salary_ranges"]:
            if salary <= 2:
                salary_ranges["0-2 Lakhs"].append(salary)
            elif salary <= 5:
                salary_ranges["2-5 Lakhs"].append(salary)
            elif salary <= 10:
                salary_ranges["5-10 Lakhs"].append(salary)
            else:
                salary_ranges["10+ Lakhs"].append(salary)

        for range_name, salaries in salary_ranges.items():
            if salaries:
                f.write(f"\\n{range_name}:\\n")
                for salary in sorted(salaries)[:10]:  # Show top 10 per range
                    f.write(f"  - {salary} lakhs\\n")

        # Usage Examples
        f.write("\\n\\n📋 USAGE EXAMPLES\\n")
        f.write("-" * 40 + "\\n")
        f.write("Example search payloads for better results:\\n\\n")

        f.write("1. BROAD DEVELOPER SEARCH:\\n")
        f.write("{\\n")
        f.write('  "userid": "your_user_id",\\n')
        f.write('  "experience_titles": ["developer", "engineer", "programmer"],\\n')
        f.write('  "skills": ["python", "java", "javascript"],\\n')
        f.write('  "relevant_score": 20.0\\n')
        f.write("}\\n\\n")

        f.write("2. LOCATION-BASED SEARCH:\\n")
        f.write("{\\n")
        f.write('  "userid": "your_user_id",\\n')
        f.write('  "locations": ["ahmedabad", "mumbai", "pune"],\\n')
        f.write('  "relevant_score": 15.0\\n')
        f.write("}\\n\\n")

        f.write("3. SKILLS-FOCUSED SEARCH:\\n")
        f.write("{\\n")
        f.write('  "userid": "your_user_id",\\n')
        f.write('  "skills": ["python", "sql", "javascript", "html", "css"],\\n')
        f.write('  "relevant_score": 25.0\\n')
        f.write("}\\n\\n")

        f.write("💡 TIPS FOR BETTER RESULTS:\\n")
        f.write("- Use broader terms like 'developer' instead of specific titles\\n")
        f.write("- Include multiple skills for better coverage\\n")
        f.write("- Set relevance_score between 15-30 for balanced results\\n")
        f.write("- Allow salary/experience flexibility with ±20% variance\\n")
        f.write("- Use multiple location options for wider coverage\\n")

    # Also save as JSON for programmatic use
    json_output = "d:\\UPH\\uphire_v2\\available_keywords_for_search.json"
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(keywords_list, f, indent=2, ensure_ascii=False)

    print(f"✅ Created keyword files:")
    print(f"   📄 {output_file}")
    print(f"   📄 {json_output}")
    print(f"\\n📊 Summary:")
    print(f"   - {len(keywords_list['job_titles'])} job titles")
    print(f"   - {len(keywords_list['skills'])} skills")
    print(f"   - {len(keywords_list['education_levels'])} education levels")
    print(f"   - {len(keywords_list['cities'])} cities")
    print(f"   - {len(keywords_list['experience_ranges'])} experience patterns")
    print(f"   - {len(keywords_list['salary_ranges'])} salary values")

    return keywords_list


if __name__ == "__main__":
    try:
        create_simple_keywords_list()
        print("\\n🎉 Keyword list created successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
