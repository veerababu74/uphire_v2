#!/usr/bin/env python3
"""
Test script for enhanced experience extraction in resume parsers.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multipleresumepraser.main import ResumeParser as MultipleResumeParser
from GroqcloudLLM.main import ResumeParser as SingleResumeParser
from core.experience_extractor import ExperienceExtractor
import json

# Sample resume text with complex experience data
SAMPLE_RESUME = """
John Doe
Software Engineer
Email: john.doe@email.com
Phone: +1-555-123-4567
Location: San Francisco, CA

EXPERIENCE:

Senior Software Engineer | Tech Corp | Jan 2022 - Present
- Led development of microservices architecture
- Managed team of 5 engineers
- Worked on cloud migration projects

Software Engineer | StartupXYZ | June 2019 - December 2021 (2 years 6 months)
- Developed web applications using React and Node.js
- Implemented CI/CD pipelines
- Collaborated with cross-functional teams

Junior Developer | WebSolutions | August 2018 - May 2019
- Built responsive websites
- Worked with HTML, CSS, JavaScript
- Gained experience in agile methodology

Intern Software Developer | DevCompany | June 2017 - August 2018 (1 year 2 months)
- Assisted in mobile app development
- Learning Android and iOS development
- Participated in code reviews

EDUCATION:
Bachelor of Science in Computer Science
University of California, Berkeley
Graduated: 2017

SKILLS:
Python, JavaScript, React, Node.js, AWS, Docker, Kubernetes, Git, Agile
"""


def test_experience_extraction():
    """Test the enhanced experience extraction"""
    print("=== Testing Enhanced Experience Extraction ===\n")

    try:
        # Test with Multiple Resume Parser
        print("1. Testing Multiple Resume Parser...")
        multiple_parser = MultipleResumeParser(llm_provider="groq")

        if (
            hasattr(multiple_parser, "experience_extractor")
            and multiple_parser.experience_extractor
        ):
            print("✓ Experience extractor initialized successfully")

            # Test experience extraction
            exp_result = multiple_parser.experience_extractor.extract_experience(
                SAMPLE_RESUME
            )
            print(f"Experience extraction result:")
            print(f"  - Total: {exp_result.get('total_experience_text', 'N/A')}")
            print(f"  - Confidence: {exp_result.get('extraction_confidence', 'N/A')}")
            print(f"  - Method: {exp_result.get('calculation_method', 'N/A')}")
            print(
                f"  - Years: {exp_result.get('total_years', 0)}, Months: {exp_result.get('total_months', 0)}"
            )
            print(f"  - Experiences found: {len(exp_result.get('experiences', []))}")
        else:
            print("✗ Experience extractor not initialized")

    except Exception as e:
        print(f"✗ Multiple Resume Parser test failed: {e}")

    print()

    try:
        # Test with Single Resume Parser (GroqcloudLLM)
        print("2. Testing Single Resume Parser (GroqcloudLLM)...")
        single_parser = SingleResumeParser(llm_provider="groq")

        if (
            hasattr(single_parser, "experience_extractor")
            and single_parser.experience_extractor
        ):
            print("✓ Experience extractor initialized successfully")

            # Test full resume processing
            result = single_parser.process_resume(SAMPLE_RESUME)

            if "error" not in result:
                print(f"✓ Resume processed successfully")
                print(f"  - Total Experience: {result.get('total_experience', 'N/A')}")
                if "experience_metadata" in result:
                    metadata = result["experience_metadata"]
                    print(
                        f"  - Confidence: {metadata.get('extraction_confidence', 'N/A')}"
                    )
                    print(f"  - Method: {metadata.get('calculation_method', 'N/A')}")
                    print(
                        f"  - Years: {metadata.get('total_years', 0)}, Months: {metadata.get('total_months', 0)}"
                    )

                experience_list = result.get("experience", [])
                print(f"  - Experience entries: {len(experience_list)}")

                for i, exp in enumerate(experience_list[:3]):  # Show first 3
                    print(
                        f"    {i+1}. {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')} ({exp.get('from_date', 'N/A')} - {exp.get('to_date', 'Present')})"
                    )
            else:
                print(
                    f"✗ Resume processing failed: {result.get('error', 'Unknown error')}"
                )
        else:
            print("✗ Experience extractor not initialized")

    except Exception as e:
        print(f"✗ Single Resume Parser test failed: {e}")


if __name__ == "__main__":
    test_experience_extraction()
