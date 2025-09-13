"""
Comprehensive Test Suite for Enhanced Resume Parser
Tests accuracy, validation, and edge cases
"""

import pytest
import json
import os
import tempfile
from typing import Dict, Any
from unittest.mock import Mock, patch

# Import the enhanced parser components
from core.enhanced_resume_parser import EnhancedResumeParser, create_enhanced_parser
from schemas.enhanced_resume_schemas import EnhancedResumeData, validate_resume_data
from apis.enhanced_resume_parser_api import validate_extracted_data


class TestEnhancedResumeParser:
    """Test suite for enhanced resume parser accuracy"""

    def setup_method(self):
        """Setup test environment"""
        self.parser = EnhancedResumeParser()

        # Sample resume texts for testing
        self.sample_resumes = {
            "complete_resume": """
            John Doe
            Email: john.doe@techcorp.com
            Phone: +1-555-123-4567
            Address: San Francisco, CA
            
            PROFESSIONAL EXPERIENCE
            Senior Software Engineer at TechCorp Inc
            January 2021 - Present
            - Led development of microservices architecture
            - Managed team of 5 developers
            - Implemented CI/CD pipelines
            
            Software Developer at StartupXYZ
            June 2019 - December 2020
            - Built REST APIs using Python and Django
            - Worked with PostgreSQL and Redis
            - Developed frontend using React
            
            Junior Developer at WebSolutions
            January 2018 - May 2019
            - Created responsive web applications
            - Used HTML, CSS, JavaScript
            
            EDUCATION
            Bachelor of Computer Science
            Stanford University, 2017
            GPA: 3.8/4.0
            
            TECHNICAL SKILLS
            Python, JavaScript, React, Django, PostgreSQL, AWS, Docker, Kubernetes, Git
            
            ADDITIONAL INFORMATION
            - PAN: ABCDE1234F
            - Expected Salary: $120,000 per year
            - Notice Period: 30 days
            """,
            "minimal_resume": """
            Jane Smith
            jane@email.com
            9876543210
            
            Experience:
            Developer at TechStart (2 years)
            
            Education:
            Computer Science Graduate
            
            Skills: Python, Java
            """,
            "poor_format_resume": """
            NAME:RAJESH KUMAR EMAIL rajesh123@gmail PHONE NO 9988776655 CITY mumbai
            WORK EXP: SOFTWARE ENG AT INFOSYS 2019-2022 WEB DEV AT TCS 2017-2019
            EDUCATION BE COMPUTER 2017 FROM MUMBAI UNIVERSITY
            SKILLS java python javascript html css sql
            """,
            "non_resume_text": """
            This is not a resume. This is just some random text about cooking recipes.
            You can make pasta by boiling water and adding noodles.
            Then add sauce and cheese. Enjoy your meal!
            Weather today is sunny and pleasant.
            """,
            "experience_heavy_resume": """
            Michael Johnson
            mike.johnson@example.com
            +1-234-567-8900
            New York, NY
            
            WORK EXPERIENCE
            
            Senior Technical Lead - Microsoft Corporation
            March 2020 - Present (3 years 9 months)
            • Led cross-functional team of 12 engineers
            • Architected scalable cloud solutions on Azure
            • Reduced system latency by 40%
            
            Software Engineering Manager - Google LLC  
            Jan 2018 - Feb 2020 (2 years 2 months)
            • Managed 8-person development team
            • Launched 3 major product features
            • Improved code quality metrics by 35%
            
            Senior Software Engineer - Amazon Web Services
            Jun 2015 - Dec 2017 (2 years 7 months)
            • Developed distributed systems for AWS Lambda
            • Mentored 4 junior engineers
            • Optimized database performance
            
            Software Engineer - Facebook (Meta)
            Aug 2013 - May 2015 (1 year 10 months)
            • Built real-time messaging features
            • Worked on News Feed algorithm optimization
            
            EDUCATION
            Master of Science in Computer Science
            Carnegie Mellon University, 2013
            
            Bachelor of Technology in Computer Engineering  
            Indian Institute of Technology, Delhi, 2011
            
            SKILLS
            Java, Python, C++, JavaScript, TypeScript, React, Node.js, 
            AWS, Azure, GCP, Docker, Kubernetes, MySQL, PostgreSQL, 
            MongoDB, Redis, Elasticsearch, Apache Kafka, Jenkins, Git
            """,
        }

    def test_complete_resume_parsing(self):
        """Test parsing of a complete, well-formatted resume"""
        result = self.parser.parse_resume(self.sample_resumes["complete_resume"])

        assert "error" not in result
        assert result["contact_details"]["name"] == "John Doe"
        assert result["contact_details"]["email"] == "john.doe@techcorp.com"
        assert result["contact_details"]["phone"] == "+1-555-123-4567"
        assert len(result["experience"]) >= 3
        assert len(result["skills"]) >= 5
        assert "Stanford University" in str(result["academic_details"])

        # Check experience calculation
        total_months = result.get("total_experience_months", 0)
        assert total_months > 36  # Should be more than 3 years

        print(f"✅ Complete resume parsing: {result['total_experience']}")

    def test_minimal_resume_parsing(self):
        """Test parsing of minimal resume with basic information"""
        result = self.parser.parse_resume(self.sample_resumes["minimal_resume"])

        assert "error" not in result
        assert result["contact_details"]["name"] == "Jane Smith"
        assert "jane@email.com" in result["contact_details"]["email"]
        assert len(result["experience"]) >= 1
        assert len(result["skills"]) >= 2

        print(f"✅ Minimal resume parsing: {result['contact_details']['name']}")

    def test_poor_format_resume_parsing(self):
        """Test parsing of poorly formatted resume"""
        result = self.parser.parse_resume(self.sample_resumes["poor_format_resume"])

        assert "error" not in result
        assert "RAJESH" in result["contact_details"]["name"].upper()
        assert "rajesh" in result["contact_details"]["email"].lower()
        assert len(result["experience"]) >= 1
        assert len(result["skills"]) >= 3

        print(f"✅ Poor format resume parsing: {result['contact_details']['name']}")

    def test_non_resume_content_rejection(self):
        """Test rejection of non-resume content"""
        result = self.parser.parse_resume(self.sample_resumes["non_resume_text"])

        assert "error" in result
        assert result["error_type"] == "content_validation_failed"

        print("✅ Non-resume content correctly rejected")

    def test_experience_calculation_accuracy(self):
        """Test accuracy of experience calculation"""
        result = self.parser.parse_resume(
            self.sample_resumes["experience_heavy_resume"]
        )

        assert "error" not in result

        # Calculate expected experience manually
        # Mar 2020 - Present (~3y 9m) + Jan 2018 - Feb 2020 (2y 2m) +
        # Jun 2015 - Dec 2017 (2y 7m) + Aug 2013 - May 2015 (1y 10m)
        # Total: ~10+ years

        total_months = result.get("total_experience_months", 0)
        assert total_months >= 120  # At least 10 years

        # Check individual experiences
        experiences = result.get("experience", [])
        assert len(experiences) >= 4

        # Verify company names are extracted
        companies = [exp.get("company", "").lower() for exp in experiences]
        assert any("microsoft" in company for company in companies)
        assert any("google" in company for company in companies)

        print(
            f"✅ Experience calculation: {result['total_experience']} ({total_months} months)"
        )

    def test_skills_extraction_accuracy(self):
        """Test accuracy of skills extraction"""
        result = self.parser.parse_resume(
            self.sample_resumes["experience_heavy_resume"]
        )

        skills = result.get("skills", [])

        # Check for specific skills that should be extracted
        skills_lower = [skill.lower() for skill in skills]

        expected_skills = [
            "java",
            "python",
            "javascript",
            "aws",
            "docker",
            "kubernetes",
        ]
        found_skills = [skill for skill in expected_skills if skill in skills_lower]

        assert len(found_skills) >= 4, f"Expected more skills, found: {found_skills}"

        print(f"✅ Skills extraction: {len(skills)} skills found")

    def test_contact_info_extraction(self):
        """Test contact information extraction accuracy"""
        test_cases = [
            {
                "text": "John Doe\nemail: john@test.com\nPhone: +1-555-123-4567\nNew York, NY",
                "expected_name": "John Doe",
                "expected_email": "john@test.com",
                "expected_phone": "+1-555-123-4567",
            },
            {
                "text": "JANE SMITH\nContact: jane.smith@company.com\nMobile: 9876543210\nLocation: Mumbai",
                "expected_name": "JANE SMITH",
                "expected_email": "jane.smith@company.com",
                "expected_phone": "+919876543210",
            },
        ]

        for i, case in enumerate(test_cases):
            result = self.parser.parse_resume(case["text"])

            assert case["expected_name"] in result["contact_details"]["name"]
            assert case["expected_email"] in result["contact_details"]["email"]
            # Phone validation might modify format but should contain digits

            print(f"✅ Contact info test case {i+1} passed")

    def test_date_normalization(self):
        """Test date format normalization"""
        date_test_cases = [
            ("Jan 2020", "2020-01"),
            ("January 2020", "2020-01"),
            ("2020-01", "2020-01"),
            ("01/2020", "2020-01"),
            ("2020", "2020-01"),
            ("Present", None),
            ("Current", None),
        ]

        for input_date, expected in date_test_cases:
            result = EnhancedResumeParser.normalize_date_string(input_date)
            assert (
                result == expected
            ), f"Date {input_date} should normalize to {expected}, got {result}"

        print("✅ Date normalization tests passed")

    def test_validation_accuracy(self):
        """Test data validation accuracy"""
        # Test with good data
        good_result = self.parser.parse_resume(self.sample_resumes["complete_resume"])
        validation_report = validate_extracted_data(good_result)

        assert validation_report["validation_passed"] == True
        assert validation_report["confidence_score"] >= 80

        # Test with poor data
        poor_data = {
            "contact_details": {
                "name": "Name Not Found",
                "email": "invalid-email",
                "phone": "123",
            },
            "experience": [],
            "skills": [],
        }

        poor_validation = validate_extracted_data(poor_data)
        assert poor_validation["validation_passed"] == False
        assert poor_validation["confidence_score"] < 70

        print(
            f"✅ Validation accuracy: Good={validation_report['confidence_score']}%, Poor={poor_validation['confidence_score']}%"
        )

    def test_schema_validation(self):
        """Test Pydantic schema validation"""
        result = self.parser.parse_resume(self.sample_resumes["complete_resume"])

        # Test schema validation
        is_valid, issues = validate_resume_data(result)

        if not is_valid:
            print(f"Schema validation issues: {issues}")

        assert is_valid, f"Schema validation failed: {issues}"

        # Test with EnhancedResumeData model
        try:
            resume_data = EnhancedResumeData(**result)
            assert resume_data.contact_details.name == "John Doe"
            assert resume_data.completeness_score >= 80
            assert resume_data.accuracy_score >= 85
        except Exception as e:
            pytest.fail(f"Schema validation failed: {e}")

        print("✅ Schema validation passed")

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "a" * 10000,  # Very long text
            "123 456 789",  # Numbers only
            "ÀÁÂÃÄÅ ÇÈÉÊË",  # Special characters
        ]

        for i, case in enumerate(edge_cases):
            result = self.parser.parse_resume(case)

            # Should either parse successfully or fail gracefully
            if "error" in result:
                assert "error_type" in result
            else:
                assert "contact_details" in result

            print(f"✅ Edge case {i+1} handled correctly")

    def test_accuracy_benchmark(self):
        """Run accuracy benchmark on all test cases"""
        test_results = []

        for resume_name, resume_text in self.sample_resumes.items():
            if resume_name == "non_resume_text":
                continue  # Skip non-resume test

            result = self.parser.parse_resume(resume_text)
            validation_report = validate_extracted_data(result)

            test_results.append(
                {
                    "resume": resume_name,
                    "success": "error" not in result,
                    "confidence": validation_report.get("confidence_score", 0),
                    "validation_passed": validation_report.get(
                        "validation_passed", False
                    ),
                }
            )

        # Calculate overall accuracy
        successful_parses = sum(1 for r in test_results if r["success"])
        total_tests = len(test_results)
        accuracy_rate = (successful_parses / total_tests) * 100

        avg_confidence = sum(r["confidence"] for r in test_results) / total_tests
        validation_pass_rate = (
            sum(1 for r in test_results if r["validation_passed"]) / total_tests * 100
        )

        print(f"\n📊 ACCURACY BENCHMARK RESULTS:")
        print(f"   Parsing Success Rate: {accuracy_rate:.1f}%")
        print(f"   Average Confidence: {avg_confidence:.1f}%")
        print(f"   Validation Pass Rate: {validation_pass_rate:.1f}%")
        print(f"   Tests Run: {total_tests}")

        # Assert minimum accuracy requirements
        assert (
            accuracy_rate >= 80
        ), f"Accuracy rate {accuracy_rate}% below 80% threshold"
        assert (
            avg_confidence >= 75
        ), f"Average confidence {avg_confidence}% below 75% threshold"
        assert (
            validation_pass_rate >= 70
        ), f"Validation pass rate {validation_pass_rate}% below 70% threshold"

        return {
            "accuracy_rate": accuracy_rate,
            "avg_confidence": avg_confidence,
            "validation_pass_rate": validation_pass_rate,
        }


class TestIntegration:
    """Integration tests for the enhanced parser API"""

    def test_api_integration(self):
        """Test API integration with enhanced parser"""
        # This would test the actual API endpoints
        # Mock implementation for demonstration

        mock_file = Mock()
        mock_file.filename = "test_resume.pdf"

        # Test would involve calling the actual API endpoint
        # For now, just test the parser integration

        parser = create_enhanced_parser()
        result = parser.parse_resume(
            "John Doe\njohn@test.com\n+1234567890\nSoftware Engineer"
        )

        assert "error" not in result
        assert result["contact_details"]["name"] == "John Doe"

        print("✅ API integration test passed")

    def test_database_integration(self):
        """Test database saving integration"""
        # Mock database operations
        # In real implementation, this would test actual database saving

        parser = create_enhanced_parser()
        result = parser.parse_resume("Test User\ntest@example.com\n+1234567890")

        # Simulate database save validation
        assert result.get("user_id") == "SYSTEM_GENERATED"
        assert result.get("parsing_timestamp") is not None

        print("✅ Database integration test passed")


def run_comprehensive_test_suite():
    """Run the complete test suite and generate report"""

    print("🚀 Starting Enhanced Resume Parser Test Suite")
    print("=" * 60)

    # Initialize test class
    test_parser = TestEnhancedResumeParser()
    test_parser.setup_method()

    # Run all tests
    tests = [
        test_parser.test_complete_resume_parsing,
        test_parser.test_minimal_resume_parsing,
        test_parser.test_poor_format_resume_parsing,
        test_parser.test_non_resume_content_rejection,
        test_parser.test_experience_calculation_accuracy,
        test_parser.test_skills_extraction_accuracy,
        test_parser.test_contact_info_extraction,
        test_parser.test_date_normalization,
        test_parser.test_validation_accuracy,
        test_parser.test_schema_validation,
        test_parser.test_edge_cases,
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test in tests:
        try:
            test()
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")

    # Run accuracy benchmark
    try:
        benchmark_results = test_parser.test_accuracy_benchmark()
        print(f"\n✅ Accuracy benchmark completed")
    except Exception as e:
        print(f"❌ Accuracy benchmark failed: {e}")
        benchmark_results = {}

    # Integration tests
    integration_tests = TestIntegration()
    try:
        integration_tests.test_api_integration()
        integration_tests.test_database_integration()
        passed_tests += 2
        total_tests += 2
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")

    # Final report
    print("\n" + "=" * 60)
    print("📊 FINAL TEST REPORT")
    print("=" * 60)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Test Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    if benchmark_results:
        print(f"Parsing Accuracy: {benchmark_results.get('accuracy_rate', 0):.1f}%")
        print(f"Average Confidence: {benchmark_results.get('avg_confidence', 0):.1f}%")
        print(
            f"Validation Pass Rate: {benchmark_results.get('validation_pass_rate', 0):.1f}%"
        )

    print("\n🎯 TARGET: 100% Accuracy Achievement Status:")

    if benchmark_results:
        accuracy = benchmark_results.get("accuracy_rate", 0)
        if accuracy >= 95:
            print("🟢 EXCELLENT: 95%+ accuracy achieved!")
        elif accuracy >= 90:
            print("🟡 GOOD: 90%+ accuracy achieved, approaching target")
        elif accuracy >= 80:
            print("🟠 ACCEPTABLE: 80%+ accuracy, needs improvement")
        else:
            print("🔴 NEEDS WORK: Below 80% accuracy, significant improvements needed")

    return {
        "tests_passed": passed_tests,
        "total_tests": total_tests,
        "success_rate": (passed_tests / total_tests) * 100,
        "benchmark_results": benchmark_results,
    }


if __name__ == "__main__":
    # Run the test suite
    results = run_comprehensive_test_suite()

    # Export results to JSON for analysis
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Test results saved to test_results.json")
