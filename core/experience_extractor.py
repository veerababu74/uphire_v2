"""
Enhanced Experience Extraction Module for Resume Parsing
Provides specialized chains for extracting and calculating work experience.
"""

import re
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("experience_extractor")


class WorkExperience(BaseModel):
    """Enhanced work experience model with better date parsing"""

    company: str = Field(description="Company name")
    title: str = Field(description="Job title/position")
    from_date: str = Field(
        description="Start date in YYYY-MM format or descriptive format"
    )
    to_date: Optional[str] = Field(
        default=None,
        description="End date in YYYY-MM format, 'Present', 'Current', or None",
    )
    duration: Optional[str] = Field(
        default=None,
        description="Explicit duration mentioned (e.g., '2 years 3 months')",
    )
    description: Optional[str] = Field(
        default=None, description="Job description or responsibilities"
    )
    location: Optional[str] = Field(default=None, description="Job location")


class ExperienceSummary(BaseModel):
    """Summary of extracted experience information"""

    experiences: List[WorkExperience] = Field(description="List of work experiences")
    total_years: int = Field(description="Total years of experience")
    total_months: int = Field(description="Additional months of experience")
    total_experience_text: str = Field(description="Human readable total experience")
    extraction_confidence: str = Field(
        description="Confidence level: high, medium, low"
    )
    calculation_method: str = Field(description="Method used for calculation")


class ExperienceExtractor:
    """Enhanced experience extraction with specialized LLM chain"""

    def __init__(self, llm):
        """Initialize with an LLM instance"""
        self.llm = llm
        self.experience_chain = self._setup_experience_chain()
        self.date_formats = [
            "%B %Y",  # January 2023
            "%b %Y",  # Jan 2023
            "%m/%Y",  # 01/2023
            "%Y-%m",  # 2023-01
            "%Y",  # 2023
            "%m-%Y",  # 01-2023
            "%B %d, %Y",  # January 15, 2023
            "%d %B %Y",  # 15 January 2023
            "%d/%m/%Y",  # 15/01/2023
            "%d-%m-%Y",  # 15-01-2023
            "%b %d, %Y",  # Jan 15, 2023
            "%m/%y",  # 01/23
        ]

    def _setup_experience_chain(self):
        """Setup specialized chain for experience extraction"""
        parser = JsonOutputParser(pydantic_object=ExperienceSummary)

        prompt_template = """You are an expert resume parser specialized in extracting work experience information with precision.

TASK: Extract ALL work experience entries from the resume text and calculate total experience accurately.

CRITICAL INSTRUCTIONS:
1. Extract each job/role as a separate experience entry
2. Convert all dates to YYYY-MM format when possible
3. Handle various date formats: "Jan 2020", "January 2020", "2020-01", "01/2020", etc.
4. For "Present", "Current", "Till date" use current date for calculation
5. Calculate overlapping experience correctly (don't double count)
6. Include internships, part-time, contract work, and freelance projects
7. If explicit duration is mentioned (e.g., "2 years 3 months"), use it for validation

DATE PARSING EXAMPLES:
- "Jan 2020" → "2020-01"
- "January 2020" → "2020-01" 
- "2020" → "2020-01" (assume January)
- "01/2020" → "2020-01"
- "2020-01" → "2020-01"
- "Present" → use current date for calculation
- "Current" → use current date for calculation

EXPERIENCE EXTRACTION RULES:
- Look for company names (usually capitalized or in title case)
- Extract job titles/positions (Software Engineer, Manager, Developer, etc.)
- Find start and end dates for each position
- Include job descriptions if available
- Extract location if mentioned
- Handle overlapping positions (e.g., promotion within same company)

CALCULATION RULES:
- Sum total experience across all positions
- Handle overlapping periods (if someone has two jobs simultaneously, don't double count the time)
- Account for gaps between jobs appropriately
- Provide confidence level based on how clear the dates and information are

TOTAL EXPERIENCE CALCULATION:
- Add up all non-overlapping work periods
- Convert to years and months format
- Be precise with month calculations
- Use current date for ongoing positions

CONFIDENCE LEVELS:
- "high": Clear dates, well-structured experience section, explicit durations
- "medium": Most dates clear, some ambiguity in formatting
- "low": Unclear dates, poor formatting, or missing information

{format_instructions}

RESUME TEXT TO ANALYZE:
{resume_text}

IMPORTANT: Return ONLY valid JSON with extracted experience and calculated totals. Focus on accuracy in date parsing and total experience calculation.
"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["resume_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        return prompt | self.llm | parser

    def extract_experience(self, resume_text: str) -> Dict[str, Any]:
        """Extract experience using specialized LLM chain"""
        try:
            logger.info("Starting specialized experience extraction")

            # Use the specialized chain
            result = self.experience_chain.invoke({"resume_text": resume_text})

            if isinstance(result, dict):
                # Validate and post-process the result
                validated_result = self._validate_and_enhance_result(
                    result, resume_text
                )
                logger.info(
                    f"Experience extraction completed. Total: {validated_result.get('total_experience_text', 'Unknown')}"
                )
                return validated_result
            else:
                logger.warning(
                    "LLM returned non-dict result, falling back to manual extraction"
                )
                return self._fallback_experience_extraction(resume_text)

        except Exception as e:
            logger.error(f"Experience extraction failed: {e}")
            return self._fallback_experience_extraction(resume_text)

    def _validate_and_enhance_result(
        self, result: Dict[str, Any], resume_text: str
    ) -> Dict[str, Any]:
        """Validate and enhance the LLM result with manual calculations"""
        try:
            # Recalculate experience using manual method for validation
            manual_total_years, manual_total_months = (
                self._calculate_total_experience_manual(result.get("experiences", []))
            )

            llm_years = result.get("total_years", 0)
            llm_months = result.get("total_months", 0)

            # Use manual calculation if LLM calculation seems off
            if abs(manual_total_years - llm_years) > 1:  # More than 1 year difference
                logger.warning(
                    f"LLM calculation ({llm_years}y {llm_months}m) differs from manual ({manual_total_years}y {manual_total_months}m), using manual"
                )
                result["total_years"] = manual_total_years
                result["total_months"] = manual_total_months
                result["total_experience_text"] = self._format_experience(
                    manual_total_years, manual_total_months
                )
                result["calculation_method"] = "manual_validation"
                result["extraction_confidence"] = "medium"

            # Ensure proper formatting
            if "total_experience_text" not in result:
                result["total_experience_text"] = self._format_experience(
                    result.get("total_years", 0), result.get("total_months", 0)
                )

            return result

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return self._fallback_experience_extraction(resume_text)

    def _calculate_total_experience_manual(
        self, experiences: List[Dict]
    ) -> Tuple[int, int]:
        """Manually calculate total experience from experience list"""
        total_months = 0
        experience_periods = []

        for exp in experiences:
            start_date = self._parse_date(exp.get("from_date", ""))
            end_date = self._parse_date(exp.get("to_date", "") or "Present")

            if start_date and end_date:
                # Calculate months for this experience
                diff = relativedelta(end_date, start_date)
                months = diff.years * 12 + diff.months

                # Store period for overlap detection
                experience_periods.append(
                    {"start": start_date, "end": end_date, "months": months}
                )

        # Handle overlapping periods
        if experience_periods:
            # Sort by start date
            experience_periods.sort(key=lambda x: x["start"])

            # Merge overlapping periods
            merged_periods = []
            current_period = experience_periods[0]

            for period in experience_periods[1:]:
                if period["start"] <= current_period["end"]:
                    # Overlapping, extend current period
                    current_period["end"] = max(current_period["end"], period["end"])
                else:
                    # No overlap, add current to merged and start new
                    merged_periods.append(current_period)
                    current_period = period

            merged_periods.append(current_period)

            # Calculate total from merged periods
            for period in merged_periods:
                diff = relativedelta(period["end"], period["start"])
                total_months += diff.years * 12 + diff.months

        total_years = total_months // 12
        remaining_months = total_months % 12

        return total_years, remaining_months

    def _parse_date(self, date_string: str) -> Optional[datetime]:
        """Enhanced date parsing with multiple format support"""
        if not date_string:
            return None

        date_string = str(date_string).strip()

        # Handle special cases
        if date_string.lower() in [
            "present",
            "current",
            "till date",
            "ongoing",
            "now",
            "none",
        ]:
            return datetime.now()

        # Clean the date string
        date_string = re.sub(r"[^\w\s/-]", "", date_string)

        # Try different formats
        for fmt in self.date_formats:
            try:
                parsed_date = datetime.strptime(date_string, fmt)
                return parsed_date
            except ValueError:
                continue

        # Try to extract year and month using regex
        year_match = re.search(r"\b(19|20)\d{2}\b", date_string)
        month_patterns = [
            (r"\b(jan|january)\b", 1),
            (r"\b(feb|february)\b", 2),
            (r"\b(mar|march)\b", 3),
            (r"\b(apr|april)\b", 4),
            (r"\b(may)\b", 5),
            (r"\b(jun|june)\b", 6),
            (r"\b(jul|july)\b", 7),
            (r"\b(aug|august)\b", 8),
            (r"\b(sep|september)\b", 9),
            (r"\b(oct|october)\b", 10),
            (r"\b(nov|november)\b", 11),
            (r"\b(dec|december)\b", 12),
        ]

        if year_match:
            year = int(year_match.group())
            month = 1  # Default to January

            for pattern, month_num in month_patterns:
                if re.search(pattern, date_string.lower()):
                    month = month_num
                    break

            try:
                return datetime(year, month, 1)
            except ValueError:
                pass

        return None

    def _format_experience(self, years: int, months: int) -> str:
        """Format experience in human readable form"""
        if years == 0 and months == 0:
            return "No experience"

        parts = []
        if years > 0:
            parts.append(f"{years} {'year' if years == 1 else 'years'}")
        if months > 0:
            parts.append(f"{months} {'month' if months == 1 else 'months'}")

        if not parts:
            return "No experience"

        return " and ".join(parts)

    def _fallback_experience_extraction(self, resume_text: str) -> Dict[str, Any]:
        """Fallback method using regex patterns"""
        logger.info("Using fallback regex-based experience extraction")

        # Simple regex patterns for experience extraction
        experiences = []

        # Pattern to match experience sections
        experience_patterns = [
            r"EXPERIENCE(.+?)(?=EDUCATION|SKILLS|PROJECTS|$)",
            r"WORK EXPERIENCE(.+?)(?=EDUCATION|SKILLS|PROJECTS|$)",
            r"PROFESSIONAL EXPERIENCE(.+?)(?=EDUCATION|SKILLS|PROJECTS|$)",
            r"EMPLOYMENT(.+?)(?=EDUCATION|SKILLS|PROJECTS|$)",
        ]

        experience_text = ""
        for pattern in experience_patterns:
            match = re.search(pattern, resume_text.upper(), re.DOTALL)
            if match:
                experience_text = match.group(1)
                break

        if not experience_text:
            experience_text = resume_text  # Use full text if no section found

        # Extract companies and titles
        company_patterns = [
            r"(?:^|\n)([A-Z][A-Za-z\s&.,]+)(?:\s*[-–]\s*|\n)([A-Za-z\s]+)(?:\s*\n|\s*•)",
            r"([A-Z][A-Za-z\s&.,]+)\s*,\s*([A-Za-z\s]+)",
        ]

        # Extract date ranges
        date_patterns = [
            r"(\w+\s+\d{4})\s*[-–]\s*(\w+\s+\d{4}|Present|Current)",
            r"(\d{1,2}/\d{4})\s*[-–]\s*(\d{1,2}/\d{4}|Present|Current)",
            r"(\d{4})\s*[-–]\s*(\d{4}|Present|Current)",
        ]

        # Find experience entries
        date_matches = []
        for pattern in date_patterns:
            matches = re.findall(pattern, experience_text, re.IGNORECASE)
            date_matches.extend(matches)

        # Calculate total experience from date ranges
        total_months = 0
        for start_str, end_str in date_matches:
            start_date = self._parse_date(start_str)
            end_date = self._parse_date(end_str)

            if start_date and end_date:
                diff = relativedelta(end_date, start_date)
                total_months += diff.years * 12 + diff.months

        total_years = total_months // 12
        remaining_months = total_months % 12

        return {
            "experiences": [],
            "total_years": total_years,
            "total_months": remaining_months,
            "total_experience_text": self._format_experience(
                total_years, remaining_months
            ),
            "extraction_confidence": "low",
            "calculation_method": "regex_fallback",
        }
