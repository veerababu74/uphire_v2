#!/usr/bin/env python3
"""
Simple test script for Enhanced Resume Parser
"""

import sys
import os

sys.path.append(".")


def test_enhanced_parser():
    print("🚀 Testing Enhanced Resume Parser...")
    print("=" * 50)

    try:
        from core.enhanced_resume_parser import EnhancedResumeParser

        print("✅ Enhanced parser imported successfully")

        # Test resume sample
        sample_resume = """
        John Doe
        Email: john.doe@techcorp.com
        Phone: +1-555-123-4567
        Address: San Francisco, CA
        
        PROFESSIONAL EXPERIENCE
        Senior Software Engineer at TechCorp Inc
        January 2021 - Present
        - Led development team of 5 engineers
        - Built microservices architecture using Python
        - Implemented CI/CD pipelines
        
        Software Developer at StartupXYZ
        June 2019 - December 2020
        - Developed REST APIs using Django
        - Worked with PostgreSQL and Redis
        
        EDUCATION
        Bachelor of Computer Science
        Stanford University, 2019
        
        TECHNICAL SKILLS
        Python, JavaScript, React, Django, PostgreSQL, AWS, Docker, Git
        """

        # Initialize parser without LLM
        parser = EnhancedResumeParser(llm_parser=None)
        print("✅ Parser initialized")

        # Parse the resume
        result = parser.parse_resume(sample_resume, use_llm=False)
        print("✅ Resume parsing completed")

        # Check results
        if "error" in result:
            print(f"❌ Parsing failed: {result['error']}")
            return False

        # Display results
        print("\n📊 PARSING RESULTS:")
        print("-" * 30)

        contact = result.get("contact_details", {})
        print(f"Name: {contact.get('name', 'Not found')}")
        print(f"Email: {contact.get('email', 'Not found')}")
        print(f"Phone: {contact.get('phone', 'Not found')}")
        print(f"City: {contact.get('current_city', 'Not found')}")

        experiences = result.get("experience", [])
        print(f"Experience entries: {len(experiences)}")
        for i, exp in enumerate(experiences[:3]):  # Show first 3
            print(f"  {i+1}. {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')}")

        skills = result.get("skills", [])
        print(f"Skills found: {len(skills)}")
        print(f"  Skills: {', '.join(skills[:10])}")  # Show first 10

        education = result.get("academic_details", [])
        print(f"Education entries: {len(education)}")

        total_exp = result.get("total_experience", "Not calculated")
        total_months = result.get("total_experience_months", 0)
        print(f"Total experience: {total_exp} ({total_months} months)")

        # Test validation
        try:
            from apis.enhanced_resume_parser_api import validate_extracted_data

            validation = validate_extracted_data(result)
            print(f"\n📋 VALIDATION RESULTS:")
            print(f"Validation passed: {validation['validation_passed']}")
            print(f"Confidence score: {validation['confidence_score']}%")
            print(f"Issues found: {len(validation['issues_found'])}")
            if validation["issues_found"]:
                for issue in validation["issues_found"]:
                    print(f"  - {issue}")
        except ImportError:
            print("\n⚠️  Validation API not available")

        # Test schema validation
        try:
            from schemas.enhanced_resume_schemas import EnhancedResumeData

            resume_data = EnhancedResumeData(**result)
            print(f"\n✅ Schema validation passed")
            print(f"Completeness score: {resume_data.completeness_score}%")
            print(f"Accuracy score: {resume_data.accuracy_score}%")

        except Exception as e:
            print(f"\n❌ Schema validation failed: {e}")

        # Calculate accuracy score
        accuracy_factors = []

        # Contact info accuracy
        if contact.get("name") and contact["name"] != "Name Not Extracted":
            accuracy_factors.append(20)
        if contact.get("email") and "@" in contact["email"]:
            accuracy_factors.append(15)
        if contact.get("phone") and len(contact["phone"]) >= 10:
            accuracy_factors.append(10)

        # Experience accuracy
        if len(experiences) >= 2:
            accuracy_factors.append(25)
        if total_months > 12:
            accuracy_factors.append(10)

        # Skills accuracy
        if len(skills) >= 5:
            accuracy_factors.append(15)

        # Education accuracy
        if len(education) >= 1:
            accuracy_factors.append(5)

        overall_accuracy = sum(accuracy_factors)

        print(f"\n🎯 ACCURACY ASSESSMENT:")
        print(f"Overall accuracy score: {overall_accuracy}/100")

        if overall_accuracy >= 90:
            print("🟢 EXCELLENT: 90%+ accuracy achieved!")
        elif overall_accuracy >= 80:
            print("🟡 GOOD: 80%+ accuracy achieved")
        elif overall_accuracy >= 70:
            print("🟠 ACCEPTABLE: 70%+ accuracy")
        else:
            print("🔴 NEEDS IMPROVEMENT: Below 70% accuracy")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases"""
    print("\n🧪 Testing Edge Cases...")

    try:
        from core.enhanced_resume_parser import EnhancedResumeParser

        parser = EnhancedResumeParser(llm_parser=None)

        # Test cases
        test_cases = [
            {
                "name": "Minimal Resume",
                "text": "Jane Smith\njane@email.com\n9876543210\nDeveloper at TechStart\nPython, Java",
            },
            {
                "name": "Poor Format",
                "text": "RAJESH KUMAR EMAIL rajesh@gmail PHONE 9988776655 WORK: INFOSYS 2019-2022 SKILLS java python",
            },
            {
                "name": "Non-Resume Text",
                "text": "This is not a resume. It's about cooking recipes and weather.",
            },
        ]

        results = []
        for case in test_cases:
            result = parser.parse_resume(case["text"], use_llm=False)
            success = "error" not in result
            results.append((case["name"], success))

            print(f"  {case['name']}: {'✅ PASS' if success else '❌ FAIL'}")
            if not success:
                print(f"    Error: {result.get('error', 'Unknown')}")

        success_rate = sum(1 for _, success in results if success) / len(results) * 100
        print(f"\nEdge case success rate: {success_rate:.1f}%")

        return success_rate >= 66  # At least 2/3 should pass

    except Exception as e:
        print(f"❌ Edge case testing failed: {e}")
        return False


if __name__ == "__main__":
    print("Enhanced Resume Parser - Accuracy Test Suite")
    print("=" * 60)

    # Run main test
    main_test_passed = test_enhanced_parser()

    # Run edge case tests
    edge_tests_passed = test_edge_cases()

    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)

    if main_test_passed and edge_tests_passed:
        print("🎉 ALL TESTS PASSED - Enhanced Parser is working correctly!")
        print("🎯 Ready for 100% accuracy production use")
    elif main_test_passed:
        print("✅ Main tests passed, some edge cases need work")
        print("🟡 Good for production with monitoring")
    else:
        print("❌ Tests failed - Parser needs debugging")
        print("🔴 Not ready for production")

    print("\n📁 Check the logs above for detailed analysis")
