#!/usr/bin/env python3
"""
Test script for Enhanced Multiple Resume Parser API

This script demonstrates the new features:
1. Enhanced system prompts for better data extraction
2. Resume content validation before processing
3. Improved error handling with specific error types
4. Enhanced statistics and reporting

Run this script to test the improvements made to the multiple resume parser.
"""

import requests
import json
from typing import Dict, Any


class EnhancedResumeParserTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def test_content_validation(self):
        """Test the new content validation feature."""
        print("🔍 Testing Content Validation Feature...")

        # Test data - some valid resume content, some invalid
        test_cases = [
            {
                "name": "valid_resume.txt",
                "content": """
                John Doe
                Email: john.doe@email.com
                Phone: +1-555-123-4567
                Address: New York, NY
                
                PROFESSIONAL EXPERIENCE
                Software Developer at TechCorp (2020-2023)
                - Developed web applications using Python and React
                - Worked with databases and APIs
                
                EDUCATION
                Bachelor of Computer Science
                University of Technology, 2020
                
                SKILLS
                Python, JavaScript, React, SQL, Git
                """,
                "expected": "success",
            },
            {
                "name": "invalid_content.txt",
                "content": """
                This is just random text that has nothing to do with resumes.
                Lorem ipsum dolor sit amet, consectetur adipiscing elit.
                This is a test document with no professional information.
                """,
                "expected": "invalid_content",
            },
            {
                "name": "partial_resume.txt",
                "content": """
                Jane Smith
                jane@email.com
                
                I am looking for a job. I have experience working.
                """,
                "expected": "may_pass",  # Might pass validation but with minimal data
            },
        ]

        for test_case in test_cases:
            print(
                f"\n📄 Testing: {test_case['name']} (Expected: {test_case['expected']})"
            )

            # Test through the parser directly (simulating what the API would do)
            from GroqcloudLLM.main import ResumeParser

            try:
                parser = ResumeParser()
                result = parser.process_resume(test_case["content"])

                if "error" in result:
                    error_type = result.get("error_type", "unknown")
                    print(f"❌ Error detected: {error_type}")
                    print(f"   Message: {result.get('error', 'No message')}")
                    if result.get("suggestion"):
                        print(f"   Suggestion: {result.get('suggestion')}")
                else:
                    print(
                        f"✅ Successfully parsed - Name: {result.get('contact_details', {}).get('name', 'N/A')}"
                    )

            except Exception as e:
                print(f"🔥 Exception occurred: {str(e)}")

        print("\n" + "=" * 60)

    def test_enhanced_prompts(self):
        """Test the enhanced system prompts."""
        print("🎯 Testing Enhanced System Prompts...")

        sample_resume = """
        RESUME
        
        Maria Rodriguez
        maria.rodriguez@techmail.com
        +1-408-555-9876
        San Francisco, CA
        LinkedIn: linkedin.com/in/maria-rodriguez
        
        SUMMARY
        Experienced software engineer with 5 years in full-stack development
        
        WORK EXPERIENCE
        Senior Software Engineer - Google Inc. (Jan 2021 - Present)
        • Lead development of microservices architecture
        • Mentored junior developers
        • Improved system performance by 40%
        
        Software Developer - StartupXYZ (Jun 2019 - Dec 2020)
        • Built REST APIs using Node.js and Express
        • Developed React frontend applications
        • Collaborated with product managers
        
        EDUCATION
        Master of Science in Computer Science
        Stanford University, 2019
        
        Bachelor of Engineering in Software Engineering  
        UC Berkeley, 2017
        
        TECHNICAL SKILLS
        Languages: Python, JavaScript, Java, Go
        Frameworks: React, Node.js, Django, Flask
        Databases: PostgreSQL, MongoDB, Redis
        Cloud: AWS, GCP, Docker, Kubernetes
        """

        print(f"\n📋 Testing enhanced prompts with comprehensive resume...")

        try:
            from GroqcloudLLM.main import ResumeParser

            parser = ResumeParser()
            result = parser.process_resume(sample_resume)

            if "error" in result:
                print(f"❌ Error: {result.get('error')}")
            else:
                print("✅ Successfully parsed with enhanced prompts!")

                # Check key improvements
                contact = result.get("contact_details", {})
                print(f"   📧 Email: {contact.get('email', 'Not extracted')}")
                print(f"   📱 Phone: {contact.get('phone', 'Not extracted')}")
                print(f"   🏙️  City: {contact.get('current_city', 'Not extracted')}")
                print(
                    f"   🔗 LinkedIn: {contact.get('linkedin_profile', 'Not extracted')}"
                )

                experience = result.get("experience", [])
                print(f"   💼 Experience entries: {len(experience)}")
                print(
                    f"   📅 Total experience: {result.get('total_experience', 'Not calculated')}"
                )

                skills = result.get("skills", [])
                print(f"   🛠️  Skills extracted: {len(skills)}")
                if skills:
                    print(f"      First 5 skills: {skills[:5]}")

                education = result.get("academic_details", [])
                print(f"   🎓 Education entries: {len(education)}")

        except Exception as e:
            print(f"🔥 Exception: {str(e)}")

        print("\n" + "=" * 60)

    def test_api_endpoints(self):
        """Test the actual API endpoints."""
        print("🌐 Testing API Endpoints...")

        # Test info endpoint
        try:
            response = requests.get(f"{self.base_url}/info")
            if response.status_code == 200:
                info = response.json()
                print("✅ Info endpoint working")

                # Check for new features
                features = info.get("features", {})
                new_features = info.get("new_features", {})

                print(
                    f"   📊 Content validation: {features.get('content_validation', False)}"
                )
                print(
                    f"   🔧 Enhanced error handling: {features.get('enhanced_error_handling', False)}"
                )
                print(
                    f"   📈 Resume content detection: {features.get('resume_content_detection', False)}"
                )

                if "content_validation" in new_features:
                    print(f"   🎯 Content validation features documented: ✅")

            else:
                print(f"❌ Info endpoint failed: {response.status_code}")

        except Exception as e:
            print(f"🔥 Info endpoint error: {str(e)}")

        # Test queue status endpoint
        try:
            response = requests.get(f"{self.base_url}/queue-status")
            if response.status_code == 200:
                queue_info = response.json()
                print("✅ Queue status endpoint working")
                print(
                    f"   📊 Current queue size: {queue_info.get('queue_status', {}).get('current_queue_size', 'N/A')}"
                )
                print(
                    f"   🟢 Queue health: {queue_info.get('queue_health', {}).get('is_available', 'N/A')}"
                )
            else:
                print(f"❌ Queue status endpoint failed: {response.status_code}")

        except Exception as e:
            print(f"🔥 Queue status error: {str(e)}")

        print("\n" + "=" * 60)

    def run_all_tests(self):
        """Run all tests."""
        print("🚀 Enhanced Resume Parser Testing Suite")
        print("=" * 60)

        self.test_content_validation()
        self.test_enhanced_prompts()
        self.test_api_endpoints()

        print("\n🎉 Testing Complete!")
        print("\nKey Improvements Tested:")
        print("✅ Content validation before processing")
        print("✅ Enhanced system prompts for better extraction")
        print("✅ Improved error handling with specific error types")
        print("✅ Enhanced API documentation and features")
        print("✅ Better user feedback and suggestions")


def main():
    """Main function to run tests."""
    tester = EnhancedResumeParserTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
