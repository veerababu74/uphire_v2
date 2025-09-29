"""
Detailed accuracy analysis for all parsers
"""

import sys

sys.path.append(".")

from core.enhanced_resume_parser import EnhancedResumeParser
from excel_resume_parser.excel_resume_parser import ExcelResumeParser
from multipleresumepraser.main import ResumeParser
import json


def detailed_enhanced_parser_test():
    print("=== DETAILED ENHANCED PARSER ANALYSIS ===")
    parser = EnhancedResumeParser()

    sample_text = """John Doe
Email: john.doe@techcorp.com  
Phone: +1-555-123-4567
Alternative Phone: +1-555-987-6543
Location: San Francisco, CA
LinkedIn: https://linkedin.com/in/johndoe

PROFESSIONAL EXPERIENCE:

Senior Software Engineer at TechCorp Inc
January 2021 - Present (3 years 8 months)
- Led development of microservices architecture
- Managed team of 5 developers
- Implemented CI/CD pipelines using Jenkins and Docker

Software Developer at StartupXYZ  
June 2019 - December 2020 (1 year 6 months)
- Built REST APIs using Python and Django
- Worked with PostgreSQL and Redis databases
- Developed frontend using React and TypeScript

TECHNICAL SKILLS:
Programming: Python, Java, JavaScript, TypeScript, C++
Web Technologies: React, Angular, Node.js, Django, Flask
Databases: PostgreSQL, MongoDB, Redis, MySQL
Cloud & DevOps: AWS, Docker, Kubernetes, Jenkins
Tools: Git, JIRA, Postman

EDUCATION:
Master of Computer Science
University of California, Berkeley
Graduated: 2018
GPA: 3.8/4.0

Bachelor of Engineering (Computer Science)
Stanford University
Graduated: 2016
GPA: 3.6/4.0

CERTIFICATIONS:
- AWS Solutions Architect Professional (2022)
- Certified Kubernetes Administrator (2021)

PROJECTS:
E-commerce Platform (2020)
- Built scalable microservices architecture
- Handled 10M+ daily transactions

Current Salary: $120,000 USD
Expected Salary: $150,000 USD
Notice Period: 2 weeks"""

    result = parser.parse_resume(sample_text, use_llm=False)  # Test rule-based first

    print("CONTACT DETAILS ANALYSIS:")
    contact = result.get("contact_details", {})
    print(f"  Name: '{contact.get('name', 'NOT_FOUND')}'")
    print(f"  Email: '{contact.get('email', 'NOT_FOUND')}'")
    print(f"  Phone: '{contact.get('phone', 'NOT_FOUND')}'")
    print(f"  Alt Phone: '{contact.get('alternative_phone', 'NOT_FOUND')}'")
    print(f"  City: '{contact.get('current_city', 'NOT_FOUND')}'")
    print(f"  LinkedIn: '{contact.get('linkedin_profile', 'NOT_FOUND')}'")

    print("\nEXPERIENCE ANALYSIS:")
    experiences = result.get("experience", [])
    print(f"  Total experiences found: {len(experiences)}")
    for i, exp in enumerate(experiences):
        if isinstance(exp, dict):
            print(f"  Experience {i+1}:")
            print(f"    Company: '{exp.get('company', 'NOT_FOUND')}'")
            print(f"    Title: '{exp.get('title', 'NOT_FOUND')}'")
            print(f"    From: '{exp.get('from_date', 'NOT_FOUND')}'")
            print(f"    To: '{exp.get('to_date', 'NOT_FOUND')}'")
            print(f"    Duration: {exp.get('duration_months', 'NOT_FOUND')} months")

    print(f"\nTOTAL EXPERIENCE: {result.get('total_experience', 'NOT_FOUND')}")
    print(f"TOTAL MONTHS: {result.get('total_experience_months', 'NOT_FOUND')}")

    print("\nSKILLS ANALYSIS:")
    skills = result.get("skills", [])
    print(f"  Skills found: {len(skills)}")
    print(f"  Skills: {skills}")

    print("\nEDUCATION ANALYSIS:")
    education = result.get("academic_details", [])
    print(f"  Education entries: {len(education)}")
    for i, edu in enumerate(education):
        if isinstance(edu, dict):
            print(f"  Education {i+1}:")
            print(f"    Degree: '{edu.get('education', 'NOT_FOUND')}'")
            print(f"    College: '{edu.get('college', 'NOT_FOUND')}'")
            print(f"    Year: {edu.get('pass_year', 'NOT_FOUND')}")

    print("\nSALARY ANALYSIS:")
    print(f"  Current Salary: {result.get('current_salary', 'NOT_FOUND')}")
    print(f"  Expected Salary: {result.get('expected_salary', 'NOT_FOUND')}")
    print(f"  Currency: {result.get('currency', 'NOT_FOUND')}")

    return result


def detailed_excel_parser_test():
    print("\n=== DETAILED EXCEL PARSER ANALYSIS ===")
    parser = ExcelResumeParser()

    # Comprehensive Excel row data
    sample_row = {
        # Personal Info
        "candidate_name": "Jane Smith",
        "email": "jane.smith@datacorp.com",
        "mobile_no": "+91-9876543210",
        "alternative_phone": "+91-9876543211",
        "current_city": "Mumbai",
        "linkedin": "https://linkedin.com/in/janesmith",
        # Professional Info
        "total_experience": "5 years 2 months",
        "current_company": "DataCorp Solutions",
        "designation": "Senior Data Scientist",
        "previous_company": "AnalyticsPro",
        "previous_designation": "Data Analyst",
        # Skills
        "key_skills": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Tableau, Power BI",
        "technical_skills": "Pandas, NumPy, Scikit-learn, Keras, Apache Spark",
        # Education
        "education": "M.Tech in Computer Science",
        "college": "IIT Delhi",
        "graduation_year": 2019,
        "undergraduate": "B.Tech Computer Science",
        "undergraduate_college": "NIT Trichy",
        "undergraduate_year": 2017,
        # Salary
        "current_salary": "15.5 LPA",
        "expected_salary": "22 LPA",
        "currency": "INR",
        # Additional
        "notice_period": "2 months",
        "preferred_location": "Mumbai, Pune, Bangalore",
        "work_mode": "Hybrid",
        # Experience Details
        "company1": "DataCorp Solutions",
        "role1": "Senior Data Scientist",
        "duration1": "Jan 2022 - Present",
        "company2": "AnalyticsPro",
        "role2": "Data Analyst",
        "duration2": "Jul 2019 - Dec 2021",
        # Projects
        "project1": "Customer Churn Prediction using ML",
        "project2": "Real-time Fraud Detection System",
    }

    # Test text formatting
    formatted_text = parser.format_excel_row_as_resume_text(sample_row)
    print(f"FORMATTED TEXT ({len(formatted_text)} chars):")
    print("=" * 40)
    print(formatted_text)
    print("=" * 40)

    # Test parsing
    parsed_resume = parser.parse_excel_row_to_resume(
        sample_row, "test_user_456", "jane_smith"
    )

    if parsed_resume and isinstance(parsed_resume, dict):
        print("\nPARSED RESULTS ANALYSIS:")

        # Contact details
        contact = parsed_resume.get("contact_details", {})
        print(f"Name: '{contact.get('name', 'NOT_FOUND')}'")
        print(f"Email: '{contact.get('email', 'NOT_FOUND')}'")
        print(f"Phone: '{contact.get('phone', 'NOT_FOUND')}'")
        print(f"City: '{contact.get('current_city', 'NOT_FOUND')}'")

        # Experience
        experiences = parsed_resume.get("experience", [])
        print(f"Experience entries: {len(experiences)}")

        # Skills
        skills = parsed_resume.get("skills", [])
        print(f"Skills extracted: {len(skills)}")
        print(f"Skills: {skills[:10]}...")  # Show first 10

        # Education
        education = parsed_resume.get("academic_details", [])
        print(f"Education entries: {len(education)}")

        # Total experience
        print(f"Total experience: {parsed_resume.get('total_experience', 'NOT_FOUND')}")
        print(
            f"Total months: {parsed_resume.get('total_experience_months', 'NOT_FOUND')}"
        )

    return parsed_resume


def analyze_accuracy_issues():
    print("\n=== ACCURACY ISSUE ANALYSIS ===")

    # Test phone number extraction
    parser = EnhancedResumeParser()
    phone_test_cases = [
        "+1-555-123-4567",
        "+91-9876543210",
        "555-123-4567",
        "(555) 123-4567",
        "555.123.4567",
        "5551234567",
    ]

    print("PHONE NUMBER EXTRACTION TEST:")
    for phone in phone_test_cases:
        matches = parser.phone_pattern.findall(phone)
        print(f"  '{phone}' -> {matches}")

    # Test experience extraction
    print("\nEXPERIENCE EXTRACTION TEST:")
    exp_text = """
    Senior Engineer at TechCorp
    Jan 2021 - Present
    
    Developer at StartupXYZ
    Jun 2019 - Dec 2020
    """

    result = parser._extract_experience_rule_based(exp_text)
    print(f"  Experiences found: {len(result)}")
    for exp in result:
        print(f"    {exp}")


def main():
    # Run detailed tests
    enhanced_result = detailed_enhanced_parser_test()
    excel_result = detailed_excel_parser_test()

    # Analyze specific issues
    analyze_accuracy_issues()

    print("\n=== SUMMARY OF ISSUES FOUND ===")
    print("1. Phone number extraction incomplete")
    print("2. Experience calculation discrepancies")
    print("3. Skills extraction could be more comprehensive")
    print("4. Date parsing needs improvement")
    print("5. Excel text formatting could be more structured")


if __name__ == "__main__":
    main()
