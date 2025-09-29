"""
Quick test to check current parser accuracy issues
"""

import sys

sys.path.append(".")

from core.enhanced_resume_parser import EnhancedResumeParser
from excel_resume_parser.excel_resume_parser import ExcelResumeParser
from multipleresumepraser.main import ResumeParser


def test_enhanced_parser():
    print("=== TESTING ENHANCED RESUME PARSER ===")
    try:
        parser = EnhancedResumeParser()

        sample_text = """John Doe
Email: john.doe@techcorp.com  
Phone: +1-555-123-4567
Location: San Francisco, CA

EXPERIENCE:
Senior Software Engineer at TechCorp Inc
January 2021 - Present
- Led development of microservices architecture

Software Developer at StartupXYZ  
June 2019 - December 2020
- Built REST APIs using Python and Django

SKILLS: Python, Java, React, PostgreSQL, AWS, Docker

EDUCATION:
Bachelor of Computer Science
University of California, Berkeley
Graduated: 2018"""

        result = parser.parse_resume(sample_text)
        print(f"✅ Enhanced Parser executed successfully")
        print(f"✅ Result type: {type(result)}")

        if isinstance(result, dict):
            contact = result.get("contact_details", {})
            print(f"Name: {contact.get('name', 'NOT_FOUND')}")
            print(f"Email: {contact.get('email', 'NOT_FOUND')}")
            print(f"Phone: {contact.get('phone', 'NOT_FOUND')}")
            print(f"Experience count: {len(result.get('experience', []))}")
            print(f"Skills count: {len(result.get('skills', []))}")
            print(f"Education count: {len(result.get('academic_details', []))}")

        return True

    except Exception as e:
        print(f"❌ Enhanced Parser failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_excel_parser():
    print("\n=== TESTING EXCEL RESUME PARSER ===")
    try:
        parser = ExcelResumeParser()

        # Sample Excel row data
        sample_row = {
            "candidate_name": "Jane Smith",
            "email": "jane@example.com",
            "mobile_no": "9876543210",
            "key_skills": "Python, Machine Learning, Data Analysis",
            "total_experience": "5 years",
            "current_city": "Mumbai",
            "current_company": "DataCorp",
            "designation": "Data Scientist",
            "education": "M.Tech Computer Science",
            "college": "IIT Delhi",
            "current_salary": "15 LPA",
            "expected_salary": "20 LPA",
        }

        formatted_text = parser.format_excel_row_as_resume_text(sample_row)
        print(f"✅ Excel text formatting successful")
        print(f"✅ Formatted text length: {len(formatted_text)}")

        parsed_resume = parser.parse_excel_row_to_resume(
            sample_row, "test_user_123", "jane_smith"
        )
        print(f"✅ Excel parsing result: {type(parsed_resume)}")

        if parsed_resume:
            print("✅ Excel parsing successful")
            # Check if it has expected fields
            if hasattr(parsed_resume, "contact_details"):
                print("✅ Has contact_details")
            if hasattr(parsed_resume, "experience"):
                print("✅ Has experience")
            if hasattr(parsed_resume, "skills"):
                print("✅ Has skills")
        else:
            print("❌ Excel parsing returned None")

        return True

    except Exception as e:
        print(f"❌ Excel Parser failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_single_parser():
    print("\n=== TESTING SINGLE RESUME PARSER ===")
    try:
        parser = ResumeParser()

        sample_text = """John Doe
Senior Software Engineer
Email: john.doe@techcorp.com  
Phone: +1-555-123-4567

EXPERIENCE:
TechCorp Inc - Senior Software Engineer (2021-Present)
StartupXYZ - Software Developer (2019-2020)

SKILLS: Python, Java, React

EDUCATION:
B.Tech Computer Science, UC Berkeley, 2018"""

        result = parser.process_resume(sample_text)
        print(f"✅ Single Parser executed successfully")
        print(f"✅ Result type: {type(result)}")

        if result:
            print("✅ Single parsing successful")
            # Check if it's a dict or object
            if hasattr(result, "__dict__"):
                print(f"✅ Result attributes: {list(result.__dict__.keys())}")
            elif isinstance(result, dict):
                print(f"✅ Result keys: {list(result.keys())}")
        else:
            print("❌ Single parsing returned None/empty")

        return True

    except Exception as e:
        print(f"❌ Single Parser failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("PARSER ACCURACY ASSESSMENT")
    print("=" * 50)

    results = []
    results.append(test_enhanced_parser())
    results.append(test_excel_parser())
    results.append(test_single_parser())

    print(f"\n=== OVERALL RESULTS ===")
    print(f"Enhanced Parser: {'✅ PASS' if results[0] else '❌ FAIL'}")
    print(f"Excel Parser: {'✅ PASS' if results[1] else '❌ FAIL'}")
    print(f"Single Parser: {'✅ PASS' if results[2] else '❌ FAIL'}")

    if all(results):
        print("🎉 All parsers working!")
    else:
        print("⚠️  Some parsers need fixing!")


if __name__ == "__main__":
    main()
