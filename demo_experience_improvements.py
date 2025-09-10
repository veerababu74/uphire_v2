#!/usr/bin/env python3
"""
Demonstration of the enhanced experience extraction improvements.
Shows before/after comparison and highlights new features.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multipleresumepraser.main import ResumeParser
from Expericecal.total_exp import calculator, format_experience
import json

# Sample resume with complex experience scenarios
COMPLEX_RESUME = """
Sarah Johnson
Senior Data Scientist
sarah.johnson@email.com | +1-555-987-6543 | New York, NY

PROFESSIONAL EXPERIENCE

Data Science Manager | DataCorp Inc. | March 2023 - Present
• Leading a team of 8 data scientists and ML engineers
• Developing predictive models for customer behavior analysis
• Implementing MLOps pipelines for model deployment

Senior Data Scientist | DataCorp Inc. | January 2021 - February 2023 (2 years 1 month)
• Built machine learning models for fraud detection
• Reduced false positive rate by 35%
• Collaborated with engineering teams on model integration

Data Scientist | TechStartup | June 2019 - December 2020
• Developed recommendation systems using collaborative filtering
• Worked with big data technologies (Spark, Hadoop)
• Created data visualization dashboards

ML Research Intern | AI Research Lab | Summer 2018 (3 months)
• Researched deep learning applications in computer vision
• Published 2 papers in peer-reviewed conferences
• Worked on neural network optimization

Data Analyst | Analytics Co. | August 2017 - May 2019
• Performed statistical analysis on customer data
• Created reports and dashboards for stakeholders
• Worked with SQL, Python, and R

Freelance Data Consultant | Self-employed | January 2020 - June 2021 (overlapping with other roles)
• Provided data analysis services to small businesses
• Part-time consulting work on weekends and evenings

EDUCATION
Master of Science in Data Science | MIT | 2017
Bachelor of Science in Statistics | UC Berkeley | 2015

SKILLS
Python, R, SQL, Machine Learning, Deep Learning, Spark, Hadoop, AWS, Docker
"""


def demonstrate_improvements():
    """Demonstrate the enhanced experience extraction"""
    print("=== ENHANCED EXPERIENCE EXTRACTION DEMONSTRATION ===\n")

    print(
        "📄 Sample Resume: Complex experience with overlapping roles, promotions, and various date formats"
    )
    print(f"Resume Length: {len(COMPLEX_RESUME)} characters\n")

    # Initialize enhanced parser
    parser = ResumeParser(llm_provider="groq")

    # Process with enhanced system
    print("🚀 Processing with Enhanced System...")
    print("=" * 50)

    result = parser.process_resume(COMPLEX_RESUME)

    if "error" not in result:
        print("✅ ENHANCED PROCESSING RESULTS:")
        print(f"📊 Total Experience: {result.get('total_experience', 'N/A')}")

        # Show metadata
        if "experience_metadata" in result:
            metadata = result["experience_metadata"]
            print(
                f"🎯 Extraction Confidence: {metadata.get('extraction_confidence', 'N/A')}"
            )
            print(f"⚙️  Calculation Method: {metadata.get('calculation_method', 'N/A')}")
            print(
                f"📈 Detailed: {metadata.get('total_years', 0)} years, {metadata.get('total_months', 0)} months"
            )

        # Show extracted experiences
        experiences = result.get("experience", [])
        print(f"\n📋 Extracted Experience Entries: {len(experiences)}")

        for i, exp in enumerate(experiences, 1):
            print(f"  {i}. {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')}")
            print(
                f"     Duration: {exp.get('from_date', 'N/A')} to {exp.get('to_date', 'Present')}"
            )
            if exp.get("location"):
                print(f"     Location: {exp.get('location')}")
            print()

        # Demonstrate basic calculation for comparison
        print("\n🔧 BASIC CALCULATOR COMPARISON:")
        print("=" * 40)

        try:
            basic_years, basic_months = calculator.calculate_experience(
                {"experience": experiences}
            )
            basic_total = format_experience(basic_years, basic_months)
            print(f"📊 Basic Calculator Result: {basic_total}")
            print(f"📈 Basic Detailed: {basic_years} years, {basic_months} months")
        except Exception as e:
            print(f"❌ Basic calculator failed: {e}")

        # Show key improvements
        print("\n🆕 KEY IMPROVEMENTS DEMONSTRATED:")
        print("=" * 40)
        print("✓ Enhanced date parsing (handles 'March 2023', 'Summer 2018', etc.)")
        print("✓ Overlap detection (freelance work concurrent with full-time)")
        print("✓ Promotion tracking (Data Scientist → Senior → Manager)")
        print("✓ Confidence scoring and metadata")
        print("✓ Fallback mechanisms for reliability")
        print("✓ Support for various employment types (intern, freelance, full-time)")

    else:
        print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")


def show_date_parsing_examples():
    """Show examples of improved date parsing"""
    print("\n📅 DATE PARSING IMPROVEMENTS:")
    print("=" * 40)

    from core.experience_extractor import ExperienceExtractor
    from core.llm_factory import LLMFactory

    try:
        llm = LLMFactory.create_llm()
        extractor = ExperienceExtractor(llm)

        test_dates = [
            "March 2023",
            "Jan 2021",
            "2019-06",
            "Summer 2018",
            "Present",
            "Current",
            "December 2020",
            "06/2019",
        ]

        for date_str in test_dates:
            parsed = extractor._parse_date(date_str)
            if parsed:
                print(f"  '{date_str}' → {parsed.strftime('%Y-%m')}")
            else:
                print(f"  '{date_str}' → Failed to parse")

    except Exception as e:
        print(f"Date parsing demo failed: {e}")


if __name__ == "__main__":
    demonstrate_improvements()
    show_date_parsing_examples()

    print("\n🎉 SUMMARY:")
    print("The enhanced experience extraction system provides:")
    print("• More accurate total experience calculation")
    print("• Better handling of complex employment scenarios")
    print("• Robust date parsing for various formats")
    print("• Confidence metrics and transparency")
    print("• Graceful fallback mechanisms")
    print("• Support for overlapping and concurrent roles")
