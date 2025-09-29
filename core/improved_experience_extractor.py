"""
Improved Experience Extractor for better experience calculation
"""

import re
from typing import Dict, Any, Optional
from core.custom_logger import CustomLogger

logger_manager = CustomLogger()
logger = logger_manager.get_logger("improved_experience_extractor")


class ImprovedExperienceExtractor:
    """
    Enhanced experience extractor with better pattern matching and fallback logic
    """

    def __init__(self):
        # Common experience patterns
        self.experience_patterns = [
            # Standard patterns
            r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:and\s*)?(?:(\d+)\s*(?:months?|mons?))?",
            r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
            r"(\d+)\s*(?:months?|mons?)",
            # Excel common formats
            r"(\d+(?:\.\d+)?)\s*(?:year|yr)\s*(?:(\d+)\s*(?:month|mon))?",
            r"(\d+)\s*(?:to|-)?\s*(\d+)\s*(?:years?|yrs?)",
            r"(\d+)\s*(?:to|-)?\s*(\d+)\s*(?:months?|mons?)",
            # Range patterns
            r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
            r"(\d+)\s*-\s*(\d+)\s*(?:months?|mons?)",
            # Fresher patterns
            r"(?:fresher|fresh|0\s*years?)",
            r"(?:no\s*experience|new\s*graduate)",
        ]

        # Month patterns for separate extraction
        self.month_patterns = [
            r"(\d+)\s*(?:months?|mons?)",
            r"(\d+)\s*(?:month|mon)",
        ]

    def extract_experience(self, resume_text: str) -> Dict[str, Any]:
        """
        Extract experience from resume text with improved patterns

        Args:
            resume_text: Raw resume text to analyze

        Returns:
            Dictionary with experience details
        """
        if not resume_text or not isinstance(resume_text, str):
            return self._get_default_result()

        # Clean and normalize text
        text = self._clean_text(resume_text)
        logger.info(f"Analyzing text for experience: {text[:200]}...")

        # Try to extract experience from different sections
        experience_data = self._extract_from_patterns(text)

        if experience_data["total_years"] == 0 and experience_data["total_months"] == 0:
            # Try contextual extraction
            experience_data = self._contextual_extraction(text)

        # Format the result
        result = self._format_result(experience_data)

        logger.info(f"Experience extraction result: {result['total_experience_text']}")
        return result

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better pattern matching"""
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Normalize common variations
        text = text.replace("yrs", "years")
        text = text.replace("yr", "year")
        text = text.replace("mons", "months")
        text = text.replace("mon", "month")

        return text.strip()

    def _extract_from_patterns(self, text: str) -> Dict[str, int]:
        """Extract experience using regex patterns"""
        total_years = 0
        total_months = 0

        # Try each pattern
        for pattern in self.experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)

            if matches:
                for match in matches:
                    try:
                        if isinstance(match, tuple):
                            # Handle tuple matches
                            if len(match) >= 2 and match[0] and match[1]:
                                # Years and months
                                years = (
                                    float(match[0])
                                    if match[0] and match[0].replace(".", "").isdigit()
                                    else 0
                                )
                                months = (
                                    int(match[1])
                                    if match[1] and match[1].isdigit()
                                    else 0
                                )
                                total_years = max(total_years, years)
                                total_months = max(total_months, months)
                            elif len(match) >= 1 and match[0]:
                                # Only years or only months
                                if match[0].replace(".", "").isdigit():
                                    value = float(match[0])
                                    if "month" in pattern:
                                        total_months = max(total_months, int(value))
                                    else:
                                        total_years = max(total_years, value)
                        else:
                            # Single value match
                            if str(match).replace(".", "").isdigit():
                                if "month" in pattern:
                                    total_months = max(total_months, int(match))
                                else:
                                    total_years = max(total_years, float(match))
                    except (ValueError, TypeError):
                        # Skip invalid numeric conversions
                        continue

        # Check for fresher indicators
        if any(
            keyword in text
            for keyword in [
                "fresher",
                "fresh graduate",
                "no experience",
                "new graduate",
            ]
        ):
            total_years = 0
            total_months = 0

        return {"total_years": total_years, "total_months": total_months}

    def _contextual_extraction(self, text: str) -> Dict[str, int]:
        """Try contextual extraction from work history sections"""
        total_years = 0
        total_months = 0

        # Look for work history patterns
        work_patterns = [
            r"work(?:ing)?\s*(?:experience|history)",
            r"professional\s*experience",
            r"employment\s*history",
            r"experience\s*summary",
            r"work\s*background",
        ]

        for pattern in work_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Found work section, extract dates
                date_ranges = self._extract_date_ranges(text)
                if date_ranges:
                    years, months = self._calculate_total_experience(date_ranges)
                    total_years = max(total_years, years)
                    total_months = max(total_months, months)

        return {"total_years": total_years, "total_months": total_months}

    def _extract_date_ranges(self, text: str) -> list:
        """Extract date ranges from text"""
        # Simple date range patterns
        date_patterns = [
            r"(\d{4})\s*(?:to|-)\s*(\d{4})",
            r"(\d{4})\s*(?:to|-)\s*(?:present|current)",
            r"(\d{1,2})/(\d{4})\s*(?:to|-)\s*(\d{1,2})/(\d{4})",
        ]

        ranges = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            ranges.extend(matches)

        return ranges

    def _calculate_total_experience(self, date_ranges: list) -> tuple:
        """Calculate total experience from date ranges"""
        total_months = 0
        current_year = 2024  # You can make this dynamic

        for date_range in date_ranges:
            if len(date_range) >= 2:
                try:
                    start_year = int(date_range[0])
                    if (
                        "present" in str(date_range[1]).lower()
                        or "current" in str(date_range[1]).lower()
                    ):
                        end_year = current_year
                    else:
                        end_year = int(date_range[1])

                    years_diff = end_year - start_year
                    total_months += years_diff * 12
                except (ValueError, IndexError):
                    continue

        total_years = total_months // 12
        remaining_months = total_months % 12

        return total_years, remaining_months

    def _format_result(self, experience_data: Dict[str, int]) -> Dict[str, Any]:
        """Format the experience result"""
        years = int(experience_data["total_years"])
        months = int(experience_data["total_months"])

        # Convert excess months to years
        if months >= 12:
            additional_years = months // 12
            years += additional_years
            months = months % 12

        # Create formatted text
        if years == 0 and months == 0:
            experience_text = "0 years 0 months"
            total_months_value = 0
        else:
            parts = []
            if years > 0:
                parts.append(f"{years} year{'s' if years != 1 else ''}")
            if months > 0:
                parts.append(f"{months} month{'s' if months != 1 else ''}")

            experience_text = " ".join(parts) if parts else "0 years 0 months"
            total_months_value = (years * 12) + months

        return {
            "total_experience_years": years,
            "total_experience_months": months,
            "total_experience_text": experience_text,
            "total_months": total_months_value,
            "extraction_method": "improved_pattern_matching",
            "confidence": "high" if years > 0 or months > 0 else "low",
        }

    def _get_default_result(self) -> Dict[str, Any]:
        """Return default result for invalid input"""
        return {
            "total_experience_years": 0,
            "total_experience_months": 0,
            "total_experience_text": "0 years 0 months",
            "total_months": 0,
            "extraction_method": "default",
            "confidence": "low",
        }


def test_experience_extractor():
    """Test the improved experience extractor"""
    extractor = ImprovedExperienceExtractor()

    test_cases = [
        "I have 3 years and 6 months of experience in software development",
        "Experience: 2.5 years in Python development",
        "Total experience: 5 years",
        "Working for 18 months as a developer",
        "Fresher candidate with no prior experience",
        "2019 to present - Software Engineer at TechCorp",
        "1.5 years experience in web development",
        "0.5 years in data science",
    ]

    for test_case in test_cases:
        result = extractor.extract_experience(test_case)
        print(f"Input: {test_case}")
        print(f"Result: {result['total_experience_text']}")
        print("-" * 50)


if __name__ == "__main__":
    test_experience_extractor()
