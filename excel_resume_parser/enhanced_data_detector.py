"""
Enhanced Data Type Detector and Validator for Excel Resume Parser

This module provides comprehensive data type detection, validation, and normalization
for various field types commonly found in resume data.
"""

import re
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, date
from dateutil import parser as date_parser
from email_validator import validate_email, EmailNotValidError
import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberFormat
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("enhanced_data_detector")


class EnhancedDataTypeDetector:
    """
    Advanced data type detector and validator for resume fields.
    Handles detection, validation, and normalization of various data types.
    """

    def __init__(self):
        """Initialize the data type detector."""
        self.currency_patterns = self._initialize_currency_patterns()
        self.experience_patterns = self._initialize_experience_patterns()
        self.duration_patterns = self._initialize_duration_patterns()
        self.skill_separators = self._initialize_skill_separators()
        self.common_domains = self._initialize_common_domains()

    def _initialize_currency_patterns(self) -> Dict[str, str]:
        """Initialize currency detection patterns."""
        return {
            "indian_rupee": r"(?:₹|rs\.?|inr|rupees?)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)|(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:₹|rs\.?|inr|rupees?)",
            "us_dollar": r"(?:\$|usd|dollars?)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)|(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:\$|usd|dollars?)",
            "euro": r"(?:€|eur|euros?)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)|(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:€|eur|euros?)",
            "plain_number": r"^\d+(?:,\d{3})*(?:\.\d{2})?$",
            "lakhs_crores": r"(\d+(?:\.\d+)?)\s*(?:lakh|lac|cr|crore)s?",
            "k_format": r"(\d+(?:\.\d+)?)\s*[kK]",
        }

    def _initialize_experience_patterns(self) -> Dict[str, str]:
        """Initialize experience detection patterns."""
        return {
            "years_months": r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y)\s*(?:and\s*)?(?:(\d+)\s*(?:months?|mon?|m))?|(\d+)\s*(?:months?|mon?|m)\s*(?:and\s*)?(?:(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y))?",
            "decimal_years": r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y)(?:\s*(?:of\s*)?(?:experience|exp))?",
            "months_only": r"(\d+)\s*(?:months?|mon?|m)(?:\s*(?:of\s*)?(?:experience|exp))?",
            "range_format": r"(\d+)\s*[-to]\s*(\d+)\s*(?:years?|yrs?|y)",
            "fresher": r"(?:fresher|fresh|0|no\s*experience|entry\s*level)",
            "plus_format": r"(\d+)\+\s*(?:years?|yrs?|y)",
        }

    def _initialize_duration_patterns(self) -> Dict[str, str]:
        """Initialize duration/notice period patterns."""
        return {
            "days": r"(\d+)\s*(?:days?|d)",
            "weeks": r"(\d+)\s*(?:weeks?|w)",
            "months": r"(\d+)\s*(?:months?|mon?|m)",
            "immediate": r"(?:immediate|immediately|asap|0|zero)",
            "serving": r"(?:serving|currently\s*serving|in\s*notice)",
            "negotiable": r"(?:negotiable|flexible|discuss)",
        }

    def _initialize_skill_separators(self) -> List[str]:
        """Initialize skill separation patterns."""
        return [",", ";", "|", "/", "&", " and ", " or ", "\n", "\r\n", "•", "→", ">>"]

    def _initialize_common_domains(self) -> Set[str]:
        """Initialize common email domains for validation."""
        return {
            "gmail.com",
            "yahoo.com",
            "hotmail.com",
            "outlook.com",
            "rediffmail.com",
            "company.com",
            "corporate.com",
            "tech.com",
            "business.com",
            "professional.com",
        }

    def detect_data_type(
        self, value: Any, field_hint: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Detect the data type of a given value.

        Args:
            value: Value to analyze
            field_hint: Optional hint about expected field type

        Returns:
            Tuple of (detected_type, confidence_score)
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            return "empty", 1.0

        value_str = str(value).strip()

        # Use field hint to guide detection
        if field_hint:
            type_result = self._detect_with_hint(value_str, field_hint)
            if type_result[1] > 0.7:  # High confidence with hint
                return type_result

        # Try different type detections in order of specificity
        detections = [
            self._detect_email(value_str),
            self._detect_phone(value_str),
            self._detect_currency(value_str),
            self._detect_experience(value_str),
            self._detect_percentage(value_str),
            self._detect_date(value_str),
            self._detect_year(value_str),
            self._detect_duration(value_str),
            self._detect_list(value_str),
            self._detect_number(value_str),
            self._detect_text(value_str),
        ]

        # Return the detection with highest confidence
        best_detection = max(detections, key=lambda x: x[1])
        return best_detection

    def _detect_with_hint(self, value_str: str, field_hint: str) -> Tuple[str, float]:
        """Detect data type using field hint guidance."""
        hint_lower = field_hint.lower()

        if "email" in hint_lower:
            return self._detect_email(value_str)
        elif "phone" in hint_lower:
            return self._detect_phone(value_str)
        elif "currency" in hint_lower or "salary" in hint_lower:
            return self._detect_currency(value_str)
        elif "experience" in hint_lower:
            return self._detect_experience(value_str)
        elif "date" in hint_lower:
            return self._detect_date(value_str)
        elif "year" in hint_lower:
            return self._detect_year(value_str)
        elif "percentage" in hint_lower:
            return self._detect_percentage(value_str)
        elif "duration" in hint_lower or "notice" in hint_lower:
            return self._detect_duration(value_str)
        elif "list" in hint_lower or "skills" in hint_lower:
            return self._detect_list(value_str)
        elif "number" in hint_lower:
            return self._detect_number(value_str)
        else:
            return self._detect_text(value_str)

    def _detect_email(self, value_str: str) -> Tuple[str, float]:
        """Detect email addresses."""
        try:
            # Basic email pattern check
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            if re.match(email_pattern, value_str):
                # Try to validate with email-validator
                try:
                    validate_email(value_str)
                    return "email", 0.95
                except EmailNotValidError:
                    return "email", 0.7  # Looks like email but may be invalid
            return "text", 0.1
        except Exception:
            return "text", 0.1

    def _detect_phone(self, value_str: str) -> Tuple[str, float]:
        """Detect phone numbers."""
        try:
            # Clean the value for phone detection
            cleaned_phone = re.sub(r"[^\d+()-\s]", "", value_str)

            # Basic patterns
            phone_patterns = [
                r"^\+?91[-\s]?\d{10}$",  # Indian format
                r"^\+?1[-\s]?\d{10}$",  # US format
                r"^\d{10}$",  # 10 digit
                r"^\+?\d{8,15}$",  # General international
            ]

            for pattern in phone_patterns:
                if re.match(pattern, cleaned_phone.replace(" ", "").replace("-", "")):
                    # Try to parse with phonenumbers library
                    try:
                        phone_number = phonenumbers.parse(cleaned_phone, None)
                        if phonenumbers.is_valid_number(phone_number):
                            return "phone", 0.95
                        else:
                            return "phone", 0.7
                    except NumberParseException:
                        return "phone", 0.6

            return "text", 0.1
        except Exception:
            return "text", 0.1

    def _detect_currency(self, value_str: str) -> Tuple[str, float]:
        """Detect currency values."""
        try:
            value_lower = value_str.lower()

            for currency_type, pattern in self.currency_patterns.items():
                match = re.search(pattern, value_lower)
                if match:
                    confidence = 0.9 if currency_type != "plain_number" else 0.6
                    return "currency", confidence

            return "text", 0.1
        except Exception:
            return "text", 0.1

    def _detect_experience(self, value_str: str) -> Tuple[str, float]:
        """Detect experience values."""
        try:
            value_lower = value_str.lower()

            for exp_type, pattern in self.experience_patterns.items():
                if re.search(pattern, value_lower):
                    confidence = 0.9 if exp_type != "decimal_years" else 0.8
                    return "experience", confidence

            return "text", 0.1
        except Exception:
            return "text", 0.1

    def _detect_percentage(self, value_str: str) -> Tuple[str, float]:
        """Detect percentage values."""
        try:
            # Check for explicit percentage
            if "%" in value_str:
                return "percentage", 0.95

            # Check for CGPA/GPA patterns
            cgpa_patterns = [
                r"^\d+(?:\.\d+)?$",  # Simple decimal
                r"^\d+(?:\.\d+)?/\d+(?:\.\d+)?$",  # Fraction format
            ]

            for pattern in cgpa_patterns:
                if re.match(pattern, value_str.strip()):
                    try:
                        num = float(value_str.split("/")[0])
                        if 0 <= num <= 100:  # Percentage range
                            return "percentage", 0.7
                        elif 0 <= num <= 10:  # CGPA range
                            return "percentage", 0.8
                    except ValueError:
                        pass

            return "text", 0.1
        except Exception:
            return "text", 0.1

    def _detect_date(self, value_str: str) -> Tuple[str, float]:
        """Detect date values."""
        try:
            # Try to parse as date
            try:
                parsed_date = date_parser.parse(value_str, fuzzy=True)

                # Check if it's a reasonable date
                current_year = datetime.now().year
                if 1900 <= parsed_date.year <= current_year + 5:
                    return "date", 0.9
                else:
                    return "date", 0.6
            except (ValueError, TypeError):
                pass

            # Check common date patterns
            date_patterns = [
                r"\d{1,2}/\d{1,2}/\d{4}",
                r"\d{1,2}-\d{1,2}-\d{4}",
                r"\d{4}-\d{1,2}-\d{1,2}",
                r"\d{1,2}\s+\w+\s+\d{4}",
            ]

            for pattern in date_patterns:
                if re.search(pattern, value_str):
                    return "date", 0.8

            return "text", 0.1
        except Exception:
            return "text", 0.1

    def _detect_year(self, value_str: str) -> Tuple[str, float]:
        """Detect year values."""
        try:
            if re.match(r"^\d{4}$", value_str.strip()):
                year = int(value_str.strip())
                current_year = datetime.now().year

                if 1950 <= year <= current_year + 10:
                    return "year", 0.9
                else:
                    return "year", 0.6

            return "text", 0.1
        except Exception:
            return "text", 0.1

    def _detect_duration(self, value_str: str) -> Tuple[str, float]:
        """Detect duration/notice period values."""
        try:
            value_lower = value_str.lower()

            for duration_type, pattern in self.duration_patterns.items():
                if re.search(pattern, value_lower):
                    return "duration", 0.9

            return "text", 0.1
        except Exception:
            return "text", 0.1

    def _detect_list(self, value_str: str) -> Tuple[str, float]:
        """Detect list/comma-separated values."""
        try:
            # Check for common separators
            separator_count = 0
            for separator in self.skill_separators:
                if separator in value_str:
                    separator_count += value_str.count(separator)

            if separator_count >= 1:
                return "list", 0.8

            # Check for multiple words that might be skills
            words = value_str.split()
            if len(words) >= 3 and all(len(word) >= 2 for word in words):
                return "list", 0.6

            return "text", 0.1
        except Exception:
            return "text", 0.1

    def _detect_number(self, value_str: str) -> Tuple[str, float]:
        """Detect numeric values."""
        try:
            # Try to parse as number
            try:
                float(value_str.replace(",", ""))
                return "number", 0.9
            except ValueError:
                pass

            # Check for numeric patterns
            if re.match(r"^\d+(?:,\d{3})*(?:\.\d+)?$", value_str):
                return "number", 0.8

            return "text", 0.1
        except Exception:
            return "text", 0.1

    def _detect_text(self, value_str: str) -> Tuple[str, float]:
        """Default text detection."""
        return "text", 0.5

    def validate_and_normalize(
        self, value: Any, data_type: str, field_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate and normalize a value based on its detected data type.

        Args:
            value: Value to validate and normalize
            data_type: Detected data type
            field_name: Optional field name for context

        Returns:
            Dictionary with validation results and normalized value
        """
        result = {
            "original_value": value,
            "normalized_value": None,
            "is_valid": False,
            "data_type": data_type,
            "field_name": field_name,
            "errors": [],
            "metadata": {},
        }

        if value is None or (isinstance(value, str) and not value.strip()):
            result["normalized_value"] = None
            result["is_valid"] = True
            return result

        value_str = str(value).strip()

        try:
            if data_type == "email":
                result.update(self._validate_email(value_str))
            elif data_type == "phone":
                result.update(self._validate_phone(value_str))
            elif data_type == "currency":
                result.update(self._validate_currency(value_str))
            elif data_type == "experience":
                result.update(self._validate_experience(value_str))
            elif data_type == "percentage":
                result.update(self._validate_percentage(value_str))
            elif data_type == "date":
                result.update(self._validate_date(value_str))
            elif data_type == "year":
                result.update(self._validate_year(value_str))
            elif data_type == "duration":
                result.update(self._validate_duration(value_str))
            elif data_type == "list":
                result.update(self._validate_list(value_str))
            elif data_type == "number":
                result.update(self._validate_number(value_str))
            else:  # text
                result.update(self._validate_text(value_str))

        except Exception as e:
            result["errors"].append(f"Validation error: {str(e)}")
            result["normalized_value"] = value_str
            result["is_valid"] = False

        return result

    def _validate_email(self, value_str: str) -> Dict[str, Any]:
        """Validate and normalize email."""
        try:
            valid_email = validate_email(value_str)
            return {
                "normalized_value": valid_email.email.lower(),
                "is_valid": True,
                "metadata": {
                    "domain": valid_email.email.split("@")[1],
                    "local_part": valid_email.email.split("@")[0],
                },
            }
        except EmailNotValidError as e:
            return {
                "normalized_value": value_str.lower(),
                "is_valid": False,
                "errors": [f"Invalid email: {str(e)}"],
            }

    def _validate_phone(self, value_str: str) -> Dict[str, Any]:
        """Validate and normalize phone number."""
        try:
            # Clean the phone number
            cleaned_phone = re.sub(r"[^\d+()-\s]", "", value_str)

            # Try to parse and format
            phone_number = phonenumbers.parse(cleaned_phone, None)
            if phonenumbers.is_valid_number(phone_number):
                formatted_number = phonenumbers.format_number(
                    phone_number, PhoneNumberFormat.E164
                )
                return {
                    "normalized_value": formatted_number,
                    "is_valid": True,
                    "metadata": {
                        "country_code": phone_number.country_code,
                        "national_number": phone_number.national_number,
                        "formatted_national": phonenumbers.format_number(
                            phone_number, PhoneNumberFormat.NATIONAL
                        ),
                        "formatted_international": phonenumbers.format_number(
                            phone_number, PhoneNumberFormat.INTERNATIONAL
                        ),
                    },
                }
            else:
                return {
                    "normalized_value": cleaned_phone,
                    "is_valid": False,
                    "errors": ["Invalid phone number format"],
                }
        except NumberParseException as e:
            return {
                "normalized_value": value_str,
                "is_valid": False,
                "errors": [f"Phone parsing error: {str(e)}"],
            }

    def _validate_currency(self, value_str: str) -> Dict[str, Any]:
        """Validate and normalize currency value."""
        try:
            value_lower = value_str.lower()
            amount = None
            currency = "INR"  # Default

            # Extract amount and currency
            for currency_type, pattern in self.currency_patterns.items():
                match = re.search(pattern, value_lower)
                if match:
                    # Extract numeric value
                    groups = match.groups()
                    amount_str = next((g for g in groups if g), "0")

                    # Remove commas and convert to float
                    amount = float(re.sub(r"[,\s]", "", amount_str))

                    # Detect currency type
                    if any(
                        symbol in value_lower for symbol in ["₹", "rs", "inr", "rupee"]
                    ):
                        currency = "INR"
                    elif any(
                        symbol in value_lower for symbol in ["$", "usd", "dollar"]
                    ):
                        currency = "USD"
                    elif any(symbol in value_lower for symbol in ["€", "eur", "euro"]):
                        currency = "EUR"

                    # Handle special formats
                    if "lakh" in value_lower or "lac" in value_lower:
                        amount *= 100000
                    elif "crore" in value_lower or "cr" in value_lower:
                        amount *= 10000000
                    elif re.search(r"\d+\s*[kK]", value_lower):
                        amount *= 1000

                    break

            if amount is not None:
                return {
                    "normalized_value": {
                        "amount": amount,
                        "currency": currency,
                        "formatted": f"{currency} {amount:,.2f}",
                    },
                    "is_valid": True,
                    "metadata": {
                        "original_format": value_str,
                        "detected_currency": currency,
                        "amount_numeric": amount,
                    },
                }
            else:
                return {
                    "normalized_value": value_str,
                    "is_valid": False,
                    "errors": ["Could not extract currency amount"],
                }

        except Exception as e:
            return {
                "normalized_value": value_str,
                "is_valid": False,
                "errors": [f"Currency validation error: {str(e)}"],
            }

    def _validate_experience(self, value_str: str) -> Dict[str, Any]:
        """Validate and normalize experience value."""
        try:
            value_lower = value_str.lower()
            years = 0.0
            months = 0

            # Check for fresher
            if re.search(self.experience_patterns["fresher"], value_lower):
                return {
                    "normalized_value": {
                        "years": 0.0,
                        "months": 0,
                        "total_months": 0,
                        "formatted": "Fresher (0 years)",
                    },
                    "is_valid": True,
                    "metadata": {"type": "fresher"},
                }

            # Extract years and months
            for exp_type, pattern in self.experience_patterns.items():
                match = re.search(pattern, value_lower)
                if match:
                    groups = match.groups()

                    if exp_type == "years_months":
                        if groups[0]:  # Years first
                            years = float(groups[0])
                        if groups[1]:  # Months after years
                            months = int(groups[1])
                        if groups[2] and not groups[0]:  # Months first
                            months = int(groups[2])
                        if groups[3] and not groups[1]:  # Years after months
                            years = float(groups[3])
                    elif exp_type == "decimal_years":
                        years = float(groups[0])
                    elif exp_type == "months_only":
                        months = int(groups[0])
                    elif exp_type == "range_format":
                        # Take average of range
                        years = (float(groups[0]) + float(groups[1])) / 2
                    elif exp_type == "plus_format":
                        years = float(groups[0])

                    break

            total_months = int(years * 12) + months
            normalized_years = total_months / 12

            return {
                "normalized_value": {
                    "years": round(normalized_years, 1),
                    "months": months,
                    "total_months": total_months,
                    "formatted": (
                        f"{normalized_years:.1f} years"
                        if normalized_years >= 1
                        else f"{months} months"
                    ),
                },
                "is_valid": True,
                "metadata": {
                    "original_format": value_str,
                    "extracted_years": years,
                    "extracted_months": months,
                },
            }

        except Exception as e:
            return {
                "normalized_value": value_str,
                "is_valid": False,
                "errors": [f"Experience validation error: {str(e)}"],
            }

    def _validate_percentage(self, value_str: str) -> Dict[str, Any]:
        """Validate and normalize percentage/grade value."""
        try:
            # Handle percentage with % symbol
            if "%" in value_str:
                numeric_part = value_str.replace("%", "").strip()
                percentage = float(numeric_part)

                return {
                    "normalized_value": {
                        "percentage": percentage,
                        "type": "percentage",
                        "formatted": f"{percentage}%",
                    },
                    "is_valid": 0 <= percentage <= 100,
                    "metadata": {"original_symbol": "%"},
                }

            # Handle fraction format (e.g., "8.5/10")
            if "/" in value_str:
                parts = value_str.split("/")
                if len(parts) == 2:
                    numerator = float(parts[0].strip())
                    denominator = float(parts[1].strip())

                    if denominator > 0:
                        percentage = (numerator / denominator) * 100
                        return {
                            "normalized_value": {
                                "percentage": round(percentage, 2),
                                "type": "fraction",
                                "numerator": numerator,
                                "denominator": denominator,
                                "formatted": f"{numerator}/{denominator} ({percentage:.1f}%)",
                            },
                            "is_valid": True,
                            "metadata": {"original_format": "fraction"},
                        }

            # Handle plain number (assume CGPA if <= 10, percentage if > 10)
            try:
                number = float(value_str.strip())
                if number <= 10:
                    # Likely CGPA
                    percentage = (number / 10) * 100
                    return {
                        "normalized_value": {
                            "percentage": round(percentage, 2),
                            "type": "cgpa",
                            "cgpa": number,
                            "formatted": f"{number} CGPA ({percentage:.1f}%)",
                        },
                        "is_valid": 0 <= number <= 10,
                        "metadata": {"original_format": "cgpa"},
                    }
                else:
                    # Likely percentage
                    return {
                        "normalized_value": {
                            "percentage": number,
                            "type": "percentage",
                            "formatted": f"{number}%",
                        },
                        "is_valid": 0 <= number <= 100,
                        "metadata": {"original_format": "percentage"},
                    }
            except ValueError:
                pass

            return {
                "normalized_value": value_str,
                "is_valid": False,
                "errors": ["Could not parse percentage/grade"],
            }

        except Exception as e:
            return {
                "normalized_value": value_str,
                "is_valid": False,
                "errors": [f"Percentage validation error: {str(e)}"],
            }

    def _validate_date(self, value_str: str) -> Dict[str, Any]:
        """Validate and normalize date value."""
        try:
            parsed_date = date_parser.parse(value_str, fuzzy=True)

            return {
                "normalized_value": {
                    "date": parsed_date.date().isoformat(),
                    "year": parsed_date.year,
                    "month": parsed_date.month,
                    "day": parsed_date.day,
                    "formatted": parsed_date.strftime("%B %d, %Y"),
                },
                "is_valid": True,
                "metadata": {
                    "original_format": value_str,
                    "parsed_datetime": parsed_date.isoformat(),
                },
            }

        except Exception as e:
            return {
                "normalized_value": value_str,
                "is_valid": False,
                "errors": [f"Date validation error: {str(e)}"],
            }

    def _validate_year(self, value_str: str) -> Dict[str, Any]:
        """Validate and normalize year value."""
        try:
            year = int(value_str.strip())
            current_year = datetime.now().year

            return {
                "normalized_value": year,
                "is_valid": 1950 <= year <= current_year + 10,
                "metadata": {
                    "is_future": year > current_year,
                    "years_ago": current_year - year if year <= current_year else 0,
                },
            }

        except Exception as e:
            return {
                "normalized_value": value_str,
                "is_valid": False,
                "errors": [f"Year validation error: {str(e)}"],
            }

    def _validate_duration(self, value_str: str) -> Dict[str, Any]:
        """Validate and normalize duration/notice period value."""
        try:
            value_lower = value_str.lower()
            days = 0
            duration_type = "unknown"

            # Check for immediate
            if re.search(self.duration_patterns["immediate"], value_lower):
                return {
                    "normalized_value": {
                        "days": 0,
                        "type": "immediate",
                        "formatted": "Immediate",
                    },
                    "is_valid": True,
                    "metadata": {"type": "immediate"},
                }

            # Check for serving notice
            if re.search(self.duration_patterns["serving"], value_lower):
                return {
                    "normalized_value": {
                        "days": -1,  # Special value for serving notice
                        "type": "serving",
                        "formatted": "Currently Serving Notice",
                    },
                    "is_valid": True,
                    "metadata": {"type": "serving"},
                }

            # Check for negotiable
            if re.search(self.duration_patterns["negotiable"], value_lower):
                return {
                    "normalized_value": {
                        "days": -2,  # Special value for negotiable
                        "type": "negotiable",
                        "formatted": "Negotiable",
                    },
                    "is_valid": True,
                    "metadata": {"type": "negotiable"},
                }

            # Extract numeric durations
            for dur_type, pattern in self.duration_patterns.items():
                match = re.search(pattern, value_lower)
                if match and dur_type in ["days", "weeks", "months"]:
                    number = int(match.group(1))

                    if dur_type == "days":
                        days = number
                    elif dur_type == "weeks":
                        days = number * 7
                    elif dur_type == "months":
                        days = number * 30  # Approximate

                    duration_type = dur_type
                    break

            if days > 0:
                return {
                    "normalized_value": {
                        "days": days,
                        "type": duration_type,
                        "formatted": (
                            f"{days} days" if days < 30 else f"{days//30} months"
                        ),
                    },
                    "is_valid": True,
                    "metadata": {
                        "original_format": value_str,
                        "duration_type": duration_type,
                    },
                }

            return {
                "normalized_value": value_str,
                "is_valid": False,
                "errors": ["Could not parse duration"],
            }

        except Exception as e:
            return {
                "normalized_value": value_str,
                "is_valid": False,
                "errors": [f"Duration validation error: {str(e)}"],
            }

    def _validate_list(self, value_str: str) -> Dict[str, Any]:
        """Validate and normalize list/comma-separated values."""
        try:
            # Find the best separator
            best_separator = ","
            max_splits = 0

            for separator in self.skill_separators:
                if separator in value_str:
                    splits = len(value_str.split(separator))
                    if splits > max_splits:
                        max_splits = splits
                        best_separator = separator

            # Split and clean items
            items = []
            raw_items = value_str.split(best_separator)

            for item in raw_items:
                cleaned_item = item.strip()
                if (
                    cleaned_item and len(cleaned_item) > 1
                ):  # Filter out empty or single character items
                    items.append(cleaned_item)

            return {
                "normalized_value": items,
                "is_valid": len(items) > 0,
                "metadata": {
                    "original_format": value_str,
                    "separator_used": best_separator,
                    "item_count": len(items),
                    "raw_items": raw_items,
                },
            }

        except Exception as e:
            return {
                "normalized_value": [value_str],
                "is_valid": False,
                "errors": [f"List validation error: {str(e)}"],
            }

    def _validate_number(self, value_str: str) -> Dict[str, Any]:
        """Validate and normalize numeric value."""
        try:
            # Remove commas and convert to float
            cleaned_number = value_str.replace(",", "").replace(" ", "")
            number = float(cleaned_number)

            # Check if it's an integer
            is_integer = number.is_integer()

            return {
                "normalized_value": int(number) if is_integer else number,
                "is_valid": True,
                "metadata": {
                    "original_format": value_str,
                    "is_integer": is_integer,
                    "has_commas": "," in value_str,
                },
            }

        except Exception as e:
            return {
                "normalized_value": value_str,
                "is_valid": False,
                "errors": [f"Number validation error: {str(e)}"],
            }

    def _validate_text(self, value_str: str) -> Dict[str, Any]:
        """Validate and normalize text value."""
        try:
            # Basic text cleaning
            cleaned_text = re.sub(r"\s+", " ", value_str.strip())

            return {
                "normalized_value": cleaned_text,
                "is_valid": len(cleaned_text) > 0,
                "metadata": {
                    "original_length": len(value_str),
                    "cleaned_length": len(cleaned_text),
                    "word_count": len(cleaned_text.split()),
                    "has_special_chars": bool(re.search(r"[^\w\s-.]", cleaned_text)),
                },
            }

        except Exception as e:
            return {
                "normalized_value": value_str,
                "is_valid": False,
                "errors": [f"Text validation error: {str(e)}"],
            }
