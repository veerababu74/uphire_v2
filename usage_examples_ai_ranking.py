"""
AI Candidate Ranking - Usage Examples

This script demonstrates how to use the AI Candidate Ranking feature
with practical examples and different use cases.

Author: Uphire Team
Version: 1.0.0
"""

import requests
import json
import time
from datetime import datetime


def example_1_python_developer_ranking():
    """
    Example 1: Rank candidates for a Python Developer position
    Demonstrates skills matching, experience relevance, and automatic rejection
    """
    print("🐍 Example 1: Python Developer Position Ranking")
    print("=" * 60)

    job_description = """
    Senior Python Developer - Full Stack
    
    We are looking for an experienced Python developer to join our growing team.
    
    Required Skills:
    - Python (4+ years)
    - Django or Flask framework
    - PostgreSQL or MySQL
    - REST API development
    - Git version control
    - Unit testing
    
    Preferred Skills:
    - React.js or Vue.js
    - Docker containerization
    - AWS or Azure cloud services
    - Redis caching
    - Celery task queues
    
    Experience:
    - Minimum 4 years of Python development
    - Full-stack web development experience
    - Database design and optimization
    - API development and integration
    
    Education:
    - Bachelor's degree in Computer Science or related field
    
    Location: Remote-friendly, preference for candidates in major cities
    """

    payload = {
        "job_description": job_description,
        "user_id": "hiring_manager_001",
        "max_candidates": 15,
        "include_rejected": True,
    }

    try:
        print("📤 Sending ranking request...")
        response = requests.post(
            "http://localhost:8000/ai-ranking/rank-by-job-text", json=payload
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Ranking completed successfully!")
            print(f"📊 Analysis Results:")
            print(
                f"   • Total candidates analyzed: {result['total_candidates_analyzed']}"
            )
            print(f"   • Qualified candidates: {result['accepted_candidates']}")
            print(f"   • Auto-rejected: {result['rejected_candidates']}")

            print(f"\n🏆 Top Qualified Candidates:")
            qualified = [c for c in result["candidates"] if not c["is_auto_rejected"]]
            for i, candidate in enumerate(qualified[:5], 1):
                print(
                    f"   {i}. {candidate['name']} - {candidate['overall_match_score']}%"
                )
                print(
                    f"      Skills Match: {candidate['skills_match']['skills_match_percentage']}%"
                )
                print(
                    f"      Experience: {candidate['experience_relevance']['experience_match_percentage']}%"
                )
                print(
                    f"      Matched Skills: {', '.join(candidate['skills_match']['matched_skills'][:4])}"
                )
                if candidate["skills_match"]["missing_skills"]:
                    print(
                        f"      Missing: {', '.join(candidate['skills_match']['missing_skills'][:3])}"
                    )
                print()

            print(
                f"❌ Auto-Rejected Candidates (Below {result['ranking_criteria']['rejection_threshold']}%):"
            )
            rejected = [c for c in result["candidates"] if c["is_auto_rejected"]]
            for candidate in rejected[:3]:
                print(f"   • {candidate['name']}: {candidate['overall_match_score']}%")
                print(f"     Reason: {candidate['ranking_reason']}")

        else:
            print(f"❌ Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


def example_2_data_scientist_file_upload():
    """
    Example 2: Upload job description file for Data Scientist position
    Demonstrates file upload, AI parsing, and comprehensive analysis
    """
    print("📊 Example 2: Data Scientist Position (File Upload)")
    print("=" * 60)

    # Create a job description file
    job_description_content = """
    Data Scientist - Machine Learning Focus
    
    We are seeking a talented Data Scientist to join our AI/ML team and drive
    data-driven decision making across the organization.
    
    Core Requirements:
    • Python programming (3+ years)
    • Machine Learning frameworks (scikit-learn, pandas, numpy)
    • Statistical analysis and hypothesis testing
    • SQL and database management (PostgreSQL, MongoDB)
    • Data visualization (matplotlib, seaborn, plotly)
    
    Advanced Skills:
    • Deep Learning (TensorFlow, PyTorch, Keras)
    • Big Data tools (Spark, Hadoop, Dask)
    • Cloud platforms (AWS SageMaker, Google AI Platform)
    • MLOps and model deployment
    • R programming language
    
    Experience Requirements:
    • Minimum 3 years in data science or analytics
    • Experience with end-to-end ML model development
    • Statistical modeling and experimentation
    • Business stakeholder collaboration
    
    Education:
    • Master's degree in Data Science, Statistics, Computer Science, or related field
    • PhD preferred for senior positions
    
    Key Responsibilities:
    • Develop predictive models and algorithms
    • Analyze large datasets to extract actionable insights
    • Create data visualizations and reports for stakeholders
    • Collaborate with engineering teams on model deployment
    • Stay current with latest ML research and techniques
    """

    # Save to temporary file
    filename = "data_scientist_job_description.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(job_description_content)

    try:
        print("📤 Uploading job description file...")

        with open(filename, "rb") as f:
            files = {"file": (filename, f, "text/plain")}
            params = {
                "user_id": "hr_team_002",
                "max_candidates": 12,
                "include_rejected": True,
            }

            response = requests.post(
                "http://localhost:8000/ai-ranking/rank-by-job-file",
                files=files,
                params=params,
            )

        if response.status_code == 200:
            result = response.json()
            print("✅ File processing and ranking completed!")

            print(f"\n📈 Skills Analysis:")
            all_matched_skills = []
            all_missing_skills = []

            for candidate in result["candidates"]:
                all_matched_skills.extend(candidate["skills_match"]["matched_skills"])
                all_missing_skills.extend(candidate["skills_match"]["missing_skills"])

            # Count skill frequencies
            from collections import Counter

            matched_counter = Counter(all_matched_skills)
            missing_counter = Counter(all_missing_skills)

            print(f"   Most Common Matched Skills:")
            for skill, count in matched_counter.most_common(5):
                print(f"     • {skill}: {count} candidates")

            print(f"   Most Common Missing Skills:")
            for skill, count in missing_counter.most_common(5):
                print(f"     • {skill}: {count} candidates missing")

            print(f"\n🎯 Scoring Breakdown:")
            weights = result["ranking_criteria"]
            print(f"   • Skills: {weights['skills_weight']}%")
            print(f"   • Experience: {weights['experience_weight']}%")
            print(f"   • Education: {weights['education_weight']}%")
            print(f"   • Location: {weights['location_weight']}%")
            print(f"   • Salary: {weights['salary_weight']}%")

        else:
            print(f"❌ Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

    finally:
        # Clean up
        import os

        if os.path.exists(filename):
            os.remove(filename)


def example_3_ranking_statistics():
    """
    Example 3: Get comprehensive ranking statistics
    Demonstrates analytics and insights about the candidate database
    """
    print("📊 Example 3: Ranking Statistics and Analytics")
    print("=" * 60)

    try:
        print("📤 Retrieving ranking statistics...")

        response = requests.get(
            "http://localhost:8000/ai-ranking/ranking-stats",
            params={"user_id": "analytics_team"},
        )

        if response.status_code == 200:
            stats = response.json()
            print("✅ Statistics retrieved successfully!")

            print(f"\n🗄️ Database Overview:")
            db_overview = stats["database_overview"]
            print(f"   • Total candidates: {db_overview['total_candidates']:,}")
            print(f"   • Sample analyzed: {db_overview['sample_analyzed']}")
            print(f"   • Last updated: {db_overview['last_updated']}")

            print(f"\n🛠️ Skills Analysis:")
            skills_analysis = stats["skills_analysis"]
            print(
                f"   • Unique skills in database: {skills_analysis['total_unique_skills']:,}"
            )
            print(
                f"   • Average skills per candidate: {skills_analysis['average_skills_per_candidate']}"
            )

            print(f"   Top Skills in Database:")
            for skill_data in skills_analysis["top_skills"][:8]:
                print(f"     • {skill_data['skill']}: {skill_data['count']} candidates")

            print(f"\n📈 Experience Distribution:")
            exp_dist = stats["experience_distribution"]
            total_candidates = sum(exp_dist.values())
            for range_name, count in exp_dist.items():
                percentage = (
                    (count / total_candidates) * 100 if total_candidates > 0 else 0
                )
                print(
                    f"   • {range_name} years: {count} candidates ({percentage:.1f}%)"
                )

            print(f"\n⚙️ Current Configuration:")
            config = stats["ranking_configuration"]
            print(f"   • Rejection threshold: {config['rejection_threshold']}%")
            print(f"   • Skills weight: {config['skills_weight']}%")
            print(f"   • Experience weight: {config['experience_weight']}%")

            print(f"\n💡 Recommendations:")
            for rec in stats["recommendations"]:
                print(f"   • {rec}")

        else:
            print(f"❌ Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


def example_4_skills_gap_analysis():
    """
    Example 4: Perform skills gap analysis for workforce planning
    Demonstrates how to use the API for strategic insights
    """
    print("🔍 Example 4: Skills Gap Analysis for Strategic Planning")
    print("=" * 60)

    # Multiple job descriptions to analyze different roles
    job_roles = {
        "Frontend Developer": """
        Frontend Developer - React Specialist
        Required: React.js, JavaScript, HTML5, CSS3, TypeScript, Redux
        Preferred: Next.js, GraphQL, Jest, Webpack
        Experience: 3+ years frontend development
        """,
        "DevOps Engineer": """
        DevOps Engineer - Cloud Infrastructure
        Required: AWS, Docker, Kubernetes, Jenkins, Terraform, Python
        Preferred: Helm, Prometheus, Grafana, GitLab CI/CD
        Experience: 4+ years DevOps/Infrastructure
        """,
        "Machine Learning Engineer": """
        ML Engineer - Production Systems
        Required: Python, TensorFlow, scikit-learn, Docker, Kubernetes, AWS
        Preferred: MLflow, Kubeflow, Apache Spark, MongoDB
        Experience: 3+ years ML engineering
        """,
    }

    gap_analysis = {}

    for role_name, job_desc in job_roles.items():
        print(f"\n🔍 Analyzing {role_name} position...")

        payload = {
            "job_description": job_desc,
            "user_id": "workforce_planning",
            "max_candidates": 20,
            "include_rejected": True,
        }

        try:
            response = requests.post(
                "http://localhost:8000/ai-ranking/rank-by-job-text", json=payload
            )

            if response.status_code == 200:
                result = response.json()

                # Analyze skills gaps
                all_missing_skills = []
                qualified_candidates = 0

                for candidate in result["candidates"]:
                    if not candidate["is_auto_rejected"]:
                        qualified_candidates += 1
                    all_missing_skills.extend(
                        candidate["skills_match"]["missing_skills"]
                    )

                from collections import Counter

                missing_skills_counter = Counter(all_missing_skills)

                gap_analysis[role_name] = {
                    "total_analyzed": result["total_candidates_analyzed"],
                    "qualified_candidates": qualified_candidates,
                    "qualification_rate": (
                        qualified_candidates / result["total_candidates_analyzed"]
                    )
                    * 100,
                    "top_missing_skills": missing_skills_counter.most_common(5),
                }

                print(
                    f"   ✅ Qualification rate: {gap_analysis[role_name]['qualification_rate']:.1f}%"
                )
                print(
                    f"   📊 Qualified candidates: {qualified_candidates}/{result['total_candidates_analyzed']}"
                )

            else:
                print(f"   ❌ Error analyzing {role_name}")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

        time.sleep(1)  # Rate limiting

    # Summary of gap analysis
    print(f"\n📋 Skills Gap Analysis Summary")
    print("=" * 40)

    for role_name, analysis in gap_analysis.items():
        print(f"\n{role_name}:")
        print(f"   • Qualification Rate: {analysis['qualification_rate']:.1f}%")
        print(f"   • Qualified Candidates: {analysis['qualified_candidates']}")
        print(f"   • Top Missing Skills:")

        for skill, count in analysis["top_missing_skills"]:
            print(f"     - {skill}: {count} candidates missing")

    # Strategic recommendations
    print(f"\n🎯 Strategic Recommendations:")
    all_missing = []
    for analysis in gap_analysis.values():
        for skill, count in analysis["top_missing_skills"]:
            all_missing.append(skill)

    from collections import Counter

    overall_gaps = Counter(all_missing)

    print(f"   • Priority Training Areas:")
    for skill, frequency in overall_gaps.most_common(5):
        print(f"     - {skill} (appears in {frequency} role analyses)")

    # Calculate average qualification rate
    avg_qualification = sum(
        a["qualification_rate"] for a in gap_analysis.values()
    ) / len(gap_analysis)
    print(f"   • Overall Talent Pool Quality: {avg_qualification:.1f}%")

    if avg_qualification < 30:
        print(
            f"   • Recommendation: Focus on talent acquisition and upskilling programs"
        )
    elif avg_qualification < 60:
        print(f"   • Recommendation: Targeted training for specific skill gaps")
    else:
        print(
            f"   • Recommendation: Strong talent pool, focus on retention and advanced skills"
        )


def main():
    """Run all usage examples"""
    print("🚀 AI Candidate Ranking - Comprehensive Usage Examples")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    examples = [
        ("Example 1: Python Developer Ranking", example_1_python_developer_ranking),
        ("Example 2: Data Scientist File Upload", example_2_data_scientist_file_upload),
        ("Example 3: Ranking Statistics", example_3_ranking_statistics),
        ("Example 4: Skills Gap Analysis", example_4_skills_gap_analysis),
    ]

    for example_name, example_func in examples:
        try:
            print(f"\n🔄 Running {example_name}")
            print("-" * 60)
            example_func()
            print(f"✅ {example_name} completed successfully!")

        except Exception as e:
            print(f"❌ {example_name} failed: {str(e)}")

        print("\n" + "=" * 80 + "\n")
        time.sleep(2)  # Pause between examples

    print("🎉 All examples completed!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n💡 Key Takeaways:")
    print("• AI Candidate Ranking provides objective, consistent evaluation")
    print("• Automatic rejection saves time by filtering unqualified candidates")
    print("• Skills gap analysis helps with strategic workforce planning")
    print("• File upload support enables easy integration with existing workflows")
    print("• Comprehensive statistics provide insights for data-driven decisions")


if __name__ == "__main__":
    main()
