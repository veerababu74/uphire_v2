"""
Configuration examples for enhanced experience extraction.
Shows how to use the improved features in different scenarios.
"""


# Example 1: Using enhanced multiple resume parser
def example_multiple_resume_parser():
    """Example using the enhanced multiple resume parser"""
    from multipleresumepraser.main import ResumeParser

    # Initialize with different LLM providers
    groq_parser = ResumeParser(llm_provider="groq")
    ollama_parser = ResumeParser(llm_provider="ollama")

    # Process resume with enhanced experience extraction
    resume_text = "Your resume text here..."
    result = groq_parser.process_resume(resume_text)

    # Access enhanced experience data
    total_experience = result.get("total_experience")
    experience_metadata = result.get("experience_metadata", {})

    print(f"Total Experience: {total_experience}")
    print(f"Confidence: {experience_metadata.get('extraction_confidence')}")
    print(f"Method: {experience_metadata.get('calculation_method')}")

    return result


# Example 2: Using enhanced single resume parser (GroqcloudLLM)
def example_single_resume_parser():
    """Example using the enhanced GroqcloudLLM parser"""
    from GroqcloudLLM.main import ResumeParser

    # Initialize parser
    parser = ResumeParser(llm_provider="groq")

    # Process resume
    resume_text = "Your resume text here..."
    result = parser.process_resume(resume_text)

    # Check if enhanced extraction was successful
    if "experience_metadata" in result:
        metadata = result["experience_metadata"]
        print(f"Enhanced extraction used: {metadata['calculation_method']}")
        print(f"Confidence level: {metadata['extraction_confidence']}")

    return result


# Example 3: Direct experience extraction
def example_direct_experience_extraction():
    """Example using the experience extractor directly"""
    from core.experience_extractor import ExperienceExtractor
    from core.llm_factory import LLMFactory

    # Create LLM instance
    llm = LLMFactory.create_llm(force_provider="groq")

    # Initialize experience extractor
    extractor = ExperienceExtractor(llm)

    # Extract experience
    resume_text = "Your resume text here..."
    experience_data = extractor.extract_experience(resume_text)

    print(f"Total: {experience_data.get('total_experience_text')}")
    print(f"Years: {experience_data.get('total_years')}")
    print(f"Months: {experience_data.get('total_months')}")
    print(f"Experiences: {len(experience_data.get('experiences', []))}")

    return experience_data


# Example 4: Error handling and fallbacks
def example_error_handling():
    """Example showing error handling and fallback mechanisms"""
    from multipleresumepraser.main import ResumeParser

    parser = ResumeParser(llm_provider="groq")
    result = parser.process_resume("Invalid text here")

    # Check extraction confidence
    metadata = result.get("experience_metadata", {})
    confidence = metadata.get("extraction_confidence", "unknown")

    if confidence == "low":
        print("⚠️  Low confidence extraction - manual review recommended")
    elif confidence == "medium":
        print("✓ Medium confidence extraction - generally reliable")
    elif confidence == "high":
        print("✅ High confidence extraction - very reliable")

    # Check calculation method
    method = metadata.get("calculation_method", "unknown")
    if method == "enhanced_llm":
        print("Used enhanced LLM extraction")
    elif method == "manual_validation":
        print("Used manual validation (LLM result corrected)")
    elif method == "basic_calculator":
        print("Fell back to basic calculator")
    elif method == "regex_fallback":
        print("Used regex fallback method")


# Example 5: Custom configuration
def example_custom_configuration():
    """Example showing custom configuration options"""
    # For different providers
    providers_to_try = ["groq", "ollama", "openai"]

    for provider in providers_to_try:
        try:
            from multipleresumepraser.main import ResumeParser

            parser = ResumeParser(llm_provider=provider)

            # Check if experience extractor is available
            if hasattr(parser, "experience_extractor") and parser.experience_extractor:
                print(f"✓ {provider} - Experience extractor available")
            else:
                print(f"❌ {provider} - Experience extractor not available")

        except Exception as e:
            print(f"❌ {provider} - Failed to initialize: {e}")


# Configuration settings for optimal performance
RECOMMENDED_SETTINGS = {
    "llm_provider": "groq",  # Best for accuracy and speed
    "fallback_provider": "ollama",  # Local fallback
    "extraction_confidence_threshold": "medium",  # Minimum acceptable confidence
    "enable_manual_validation": True,  # Cross-validate LLM results
    "enable_regex_fallback": True,  # Use regex as last resort
}

# Common date formats supported
SUPPORTED_DATE_FORMATS = [
    "January 2023",
    "Jan 2023",
    "01/2023",
    "2023-01",
    "2023",
    "01-2023",
    "January 15, 2023",
    "15 January 2023",
    "15/01/2023",
    "15-01-2023",
    "Jan 15, 2023",
    "01/23",
    "Present",
    "Current",
    "Till date",
    "Ongoing",
]

# Experience types that are handled
SUPPORTED_EXPERIENCE_TYPES = [
    "Full-time employment",
    "Part-time work",
    "Internships",
    "Contract work",
    "Freelance projects",
    "Consulting",
    "Volunteer work",
    "Academic positions",
    "Research roles",
]
