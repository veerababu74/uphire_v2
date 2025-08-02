#!/usr/bin/env python3
"""
Simple validation test for import and basic structure checks.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test all important imports"""
    print("Testing imports...")

    try:
        # Test core imports
        from core.config import AppConfig

        print("✅ Core config import successful")

        from core.llm_config import LLMConfigManager, LLMProvider

        print("✅ LLM config import successful")

        from core.custom_logger import CustomLogger

        print("✅ Custom logger import successful")

        # Test main parser import
        from GroqcloudLLM.main import (
            ResumeParser,
            Resume,
            Experience,
            Education,
            ContactDetails,
        )

        print("✅ ResumeParser and models import successful")

        # Test API imports
        from apis.multiple_resume_parser_api import router

        print("✅ FastAPI router import successful")

        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def test_model_validation():
    """Test Pydantic model validation"""
    print("\nTesting Pydantic models...")

    try:
        from GroqcloudLLM.main import ContactDetails, Experience, Education

        # Test ContactDetails validation
        contact = ContactDetails(
            name="John Doe",
            email="john@example.com",
            phone="+1234567890",
            current_city="New York",
            looking_for_jobs_in=["Remote", "New York"],
            pan_card="ABCDE1234F",
        )
        print("✅ ContactDetails model validation successful")

        # Test Experience validation
        exp = Experience(
            company="Tech Corp", title="Software Engineer", from_date="2020-01"
        )
        print("✅ Experience model validation successful")

        # Test Education validation
        edu = Education(
            education="Bachelor of Science", college="Tech University", pass_year=2020
        )
        print("✅ Education model validation successful")

        return True

    except Exception as e:
        print(f"❌ Model validation error: {e}")
        return False


def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")

    try:
        from core.llm_config import LLMConfigManager

        # This should work without requiring actual LLM connections
        manager = LLMConfigManager()
        print(
            f"✅ LLM Config Manager initialized with provider: {manager.provider.value}"
        )

        return True

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def check_environment():
    """Check environment variables and settings"""
    print("\nChecking environment...")

    # Check for important environment variables
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")
    print(f"📋 LLM Provider: {llm_provider}")

    groq_key = os.getenv("GROQ_API_KEY", "not_set")
    if groq_key != "not_set":
        print("✅ Groq API key found")
    else:
        print("⚠️  Groq API key not set")

    openai_key = os.getenv("OPENAI_API_KEY", "not_set")
    if openai_key != "not_set":
        print("✅ OpenAI API key found")
    else:
        print("⚠️  OpenAI API key not set")

    return True


def main():
    """Run all validation tests"""
    print("🔍 Enhanced Resume Parser - Validation Tests")
    print("=" * 50)

    tests_passed = 0
    total_tests = 4

    if test_imports():
        tests_passed += 1

    if test_model_validation():
        tests_passed += 1

    if test_configuration():
        tests_passed += 1

    if check_environment():
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Validation Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All validation tests passed!")
        print("\n📝 Next steps:")
        print("1. Ensure your chosen LLM provider is running/configured")
        print("2. Test with: python test_enhanced_resume_validation.py")
        print("3. Start the API server: python main.py")
    else:
        print("⚠️  Some validation tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
