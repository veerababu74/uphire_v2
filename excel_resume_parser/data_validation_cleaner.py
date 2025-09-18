"""
Data Validation and Cleaning Engine for Excel Resume Parser

This module provides comprehensive data validation, cleaning, and quality
assurance routines to ensure high-quality resume data extraction.
"""

import re
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from datetime import datetime, date
from collections import defaultdict, Counter
import logging
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("data_validation_cleaner")


class DataValidationCleaner:
    """
    Comprehensive data validation and cleaning engine for resume data.
    Provides validation rules, data cleaning, quality scoring, and error correction.
    """

    def __init__(self):
        """Initialize the data validation and cleaning engine."""
        self.validation_rules = self._initialize_validation_rules()
        self.cleaning_rules = self._initialize_cleaning_rules()
        self.quality_metrics = self._initialize_quality_metrics()
        self.common_errors = self._initialize_common_errors()
        self.field_constraints = self._initialize_field_constraints()

    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize validation rules for different field types.

        Returns:
            Dictionary mapping field types to validation rules
        """
        return {
            "name": {
                "min_length": 2,
                "max_length": 100,
                "required_pattern": r"^[a-zA-Z\s\.\'-]+$",
                "forbidden_patterns": [r"\d+", r'[!@#$%^&*()_+={}[\]|\\:";,.<>?/]'],
                "must_contain": ["letter"],
                "quality_checks": ["proper_case", "no_numbers", "reasonable_length"],
            },
            "email": {
                "required_pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                "forbidden_patterns": [r"\s", r"[<>()]"],
                "must_contain": ["@", "."],
                "quality_checks": ["valid_format", "common_domain", "no_spaces"],
            },
            "phone": {
                "min_digits": 10,
                "max_digits": 15,
                "required_pattern": r"^[\+]?[\d\s\-\(\)]{10,20}$",
                "forbidden_patterns": [r"[a-zA-Z]"],
                "quality_checks": ["sufficient_digits", "valid_format", "country_code"],
            },
            "experience": {
                "min_value": 0,
                "max_value": 50,  # 50 years max experience
                "data_type": "numeric_or_structured",
                "quality_checks": ["reasonable_range", "consistent_format"],
            },
            "age": {
                "min_value": 16,
                "max_value": 80,
                "data_type": "integer",
                "quality_checks": ["reasonable_range", "integer_value"],
            },
            "salary": {
                "min_value": 1000,
                "max_value": 10000000,  # 1 crore max
                "data_type": "currency",
                "quality_checks": ["reasonable_range", "currency_format"],
            },
            "year": {
                "min_value": 1950,
                "max_value": datetime.now().year + 5,
                "data_type": "integer",
                "quality_checks": ["reasonable_range", "four_digits"],
            },
            "percentage": {
                "min_value": 0,
                "max_value": 100,
                "data_type": "numeric",
                "quality_checks": ["percentage_range", "reasonable_precision"],
            },
            "skills": {
                "min_items": 1,
                "max_items": 50,
                "item_min_length": 2,
                "item_max_length": 50,
                "data_type": "list",
                "quality_checks": ["no_duplicates", "reasonable_items", "valid_skills"],
            },
            "location": {
                "min_length": 2,
                "max_length": 100,
                "required_pattern": r"^[a-zA-Z\s\,\-\.]+$",
                "quality_checks": ["proper_case", "no_numbers", "reasonable_format"],
            },
        }

    def _initialize_cleaning_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Initialize cleaning rules for different data issues.

        Returns:
            Dictionary mapping cleaning categories to rules
        """
        return {
            "whitespace": [
                {
                    "pattern": r"^\s+|\s+$",
                    "replacement": "",
                    "description": "trim whitespace",
                },
                {
                    "pattern": r"\s+",
                    "replacement": " ",
                    "description": "normalize multiple spaces",
                },
                {
                    "pattern": r"\n\s*\n",
                    "replacement": "\n",
                    "description": "remove empty lines",
                },
            ],
            "punctuation": [
                {
                    "pattern": r"\.{2,}",
                    "replacement": ".",
                    "description": "normalize multiple dots",
                },
                {
                    "pattern": r",{2,}",
                    "replacement": ",",
                    "description": "normalize multiple commas",
                },
                {
                    "pattern": r";{2,}",
                    "replacement": ";",
                    "description": "normalize multiple semicolons",
                },
            ],
            "case_normalization": [
                {"field_types": ["name", "location"], "action": "title_case"},
                {"field_types": ["email"], "action": "lower_case"},
                {"field_types": ["skills"], "action": "preserve_technical_case"},
            ],
            "format_standardization": [
                {"field_type": "phone", "action": "standardize_phone"},
                {"field_type": "email", "action": "standardize_email"},
                {"field_type": "skills", "action": "standardize_skills_list"},
            ],
            "data_correction": [
                {"field_type": "experience", "action": "correct_experience_format"},
                {"field_type": "salary", "action": "correct_salary_format"},
                {"field_type": "percentage", "action": "correct_percentage_format"},
            ],
        }

    def _initialize_quality_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize quality scoring metrics.

        Returns:
            Dictionary mapping quality aspects to scoring rules
        """
        return {
            "completeness": {
                "weight": 0.3,
                "critical_fields": ["name", "email", "phone", "experience"],
                "important_fields": ["current_role", "skills", "education"],
                "optional_fields": ["age", "location", "salary"],
            },
            "accuracy": {
                "weight": 0.25,
                "validation_pass_rate": True,
                "format_consistency": True,
                "data_type_match": True,
            },
            "consistency": {
                "weight": 0.2,
                "field_format_consistency": True,
                "cross_field_validation": True,
                "naming_consistency": True,
            },
            "richness": {
                "weight": 0.15,
                "field_diversity": True,
                "detailed_information": True,
                "additional_fields": True,
            },
            "reliability": {
                "weight": 0.1,
                "confidence_scores": True,
                "extraction_method_quality": True,
                "source_reliability": True,
            },
        }

    def _initialize_common_errors(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Initialize common error patterns and their corrections.

        Returns:
            Dictionary mapping error types to correction rules
        """
        return {
            "name_errors": [
                {
                    "pattern": r"^(\w+)\s+(\w+)\s+(\w+)$",
                    "correction": "possible_full_name",
                },
                {"pattern": r"^[A-Z]{2,}$", "correction": "all_caps_name"},
                {"pattern": r"^[a-z]+$", "correction": "all_lowercase_name"},
                {"pattern": r"\d", "correction": "name_with_numbers"},
            ],
            "email_errors": [
                {"pattern": r"^[^@]+$", "correction": "missing_at_symbol"},
                {"pattern": r"^[^@]+@[^.]+$", "correction": "missing_domain_extension"},
                {"pattern": r".*\s.*", "correction": "email_with_spaces"},
                {"pattern": r".*[,;].*", "correction": "multiple_emails_in_field"},
            ],
            "phone_errors": [
                {"pattern": r"^\d{11,}$", "correction": "possible_country_code"},
                {"pattern": r"^\d{1,9}$", "correction": "incomplete_phone_number"},
                {"pattern": r".*[a-zA-Z].*", "correction": "phone_with_letters"},
                {"pattern": r"^0+", "correction": "leading_zeros"},
            ],
            "experience_errors": [
                {"pattern": r"^\d+\.\d{3,}$", "correction": "excessive_decimal_places"},
                {"pattern": r"^[5-9]\d+$", "correction": "possibly_months_not_years"},
                {
                    "pattern": r".*[a-zA-Z].*\d+.*",
                    "correction": "mixed_text_and_numbers",
                },
            ],
            "skills_errors": [
                {"pattern": r"^.{200,}$", "correction": "overly_long_skills_text"},
                {
                    "pattern": r"^[^,;|/&\n]*$",
                    "correction": "possible_unseparated_skills",
                },
                {"pattern": r"(.+?)(\1)+", "correction": "duplicate_skills"},
            ],
        }

    def _initialize_field_constraints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize field-specific constraints and business rules.

        Returns:
            Dictionary mapping fields to their constraints
        """
        return {
            "cross_field_validation": {
                "experience_age": {
                    "rule": "experience should not exceed (age - 18) years",
                    "fields": ["experience_years", "age"],
                    "tolerance": 2,  # years tolerance
                },
                "graduation_age": {
                    "rule": "graduation year should be reasonable for age",
                    "fields": ["graduation_year", "age"],
                    "min_graduation_age": 16,
                    "max_graduation_age": 35,
                },
                "salary_experience": {
                    "rule": "salary should be reasonable for experience level",
                    "fields": ["current_salary", "experience_years"],
                    "min_salary_per_year": 100000,  # INR
                    "max_salary_per_year": 5000000,  # INR
                },
            },
            "business_rules": {
                "notice_period_limits": {
                    "min_days": 0,
                    "max_days": 365,
                    "common_values": [0, 15, 30, 45, 60, 90],
                },
                "experience_increments": {
                    "precision": 0.5,  # Experience should be in 0.5 year increments
                    "max_precision": 1,  # Max 1 decimal place
                },
                "percentage_precision": {
                    "max_decimal_places": 2,
                    "cgpa_max": 10.0,
                    "percentage_max": 100.0,
                },
            },
        }

    def validate_field(
        self,
        field_name: str,
        field_value: Any,
        field_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate a single field against its validation rules.

        Args:
            field_name: Name of the field
            field_value: Value to validate
            field_metadata: Optional metadata about the field

        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            "field_name": field_name,
            "original_value": field_value,
            "is_valid": True,
            "validation_score": 1.0,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "quality_issues": [],
        }

        if field_value is None or (
            isinstance(field_value, str) and not field_value.strip()
        ):
            validation_result["is_valid"] = True  # Empty is valid
            validation_result["validation_score"] = 0.0
            return validation_result

        # Determine field type for validation
        field_type = self._determine_field_type(field_name)
        if not field_type:
            return validation_result  # No specific validation rules

        validation_rules = self.validation_rules.get(field_type, {})

        try:
            # Basic format validation
            if "required_pattern" in validation_rules:
                pattern = validation_rules["required_pattern"]
                if not re.match(pattern, str(field_value)):
                    validation_result["errors"].append(
                        f"Does not match required format: {pattern}"
                    )
                    validation_result["is_valid"] = False

            # Forbidden patterns
            if "forbidden_patterns" in validation_rules:
                for pattern in validation_rules["forbidden_patterns"]:
                    if re.search(pattern, str(field_value)):
                        validation_result["errors"].append(
                            f"Contains forbidden pattern: {pattern}"
                        )
                        validation_result["is_valid"] = False

            # Length validation
            field_length = len(str(field_value))
            if (
                "min_length" in validation_rules
                and field_length < validation_rules["min_length"]
            ):
                validation_result["errors"].append(
                    f"Too short (minimum {validation_rules['min_length']} characters)"
                )
                validation_result["is_valid"] = False

            if (
                "max_length" in validation_rules
                and field_length > validation_rules["max_length"]
            ):
                validation_result["warnings"].append(
                    f"Very long (maximum {validation_rules['max_length']} characters)"
                )
                validation_result["validation_score"] *= 0.8

            # Numeric range validation
            if field_type in ["experience", "age", "salary", "year", "percentage"]:
                numeric_result = self._validate_numeric_field(
                    field_value, validation_rules, field_type
                )
                validation_result["errors"].extend(numeric_result["errors"])
                validation_result["warnings"].extend(numeric_result["warnings"])
                if not numeric_result["is_valid"]:
                    validation_result["is_valid"] = False
                validation_result["validation_score"] *= numeric_result[
                    "score_multiplier"
                ]

            # List validation
            if field_type == "skills" and isinstance(field_value, list):
                list_result = self._validate_list_field(field_value, validation_rules)
                validation_result["errors"].extend(list_result["errors"])
                validation_result["warnings"].extend(list_result["warnings"])
                if not list_result["is_valid"]:
                    validation_result["is_valid"] = False
                validation_result["validation_score"] *= list_result["score_multiplier"]

            # Quality checks
            quality_checks = validation_rules.get("quality_checks", [])
            for check in quality_checks:
                quality_result = self._perform_quality_check(
                    field_value, check, field_type
                )
                if quality_result["issues"]:
                    validation_result["quality_issues"].extend(quality_result["issues"])
                    validation_result["validation_score"] *= quality_result[
                        "score_multiplier"
                    ]

            # Common error detection
            error_patterns = self.common_errors.get(f"{field_type}_errors", [])
            for error_pattern in error_patterns:
                if re.search(error_pattern["pattern"], str(field_value)):
                    validation_result["suggestions"].append(
                        f"Possible issue: {error_pattern['correction']}"
                    )
                    validation_result["validation_score"] *= 0.9

        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["is_valid"] = False
            validation_result["validation_score"] = 0.0

        return validation_result

    def _determine_field_type(self, field_name: str) -> Optional[str]:
        """Determine the validation type for a field based on its name."""
        field_name_lower = field_name.lower()

        if "name" in field_name_lower:
            return "name"
        elif "email" in field_name_lower:
            return "email"
        elif "phone" in field_name_lower:
            return "phone"
        elif "experience" in field_name_lower:
            return "experience"
        elif "age" in field_name_lower:
            return "age"
        elif "salary" in field_name_lower:
            return "salary"
        elif "year" in field_name_lower:
            return "year"
        elif "percentage" in field_name_lower or "cgpa" in field_name_lower:
            return "percentage"
        elif "skill" in field_name_lower:
            return "skills"
        elif any(loc in field_name_lower for loc in ["city", "location", "address"]):
            return "location"

        return None

    def _validate_numeric_field(
        self, value: Any, rules: Dict[str, Any], field_type: str
    ) -> Dict[str, Any]:
        """Validate numeric fields with range and format checks."""
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "score_multiplier": 1.0,
        }

        try:
            # Extract numeric value
            if isinstance(value, dict):
                if field_type == "experience" and "total_months" in value:
                    numeric_value = value["total_months"] / 12
                elif field_type == "salary" and "amount" in value:
                    numeric_value = value["amount"]
                elif field_type == "percentage" and "percentage" in value:
                    numeric_value = value["percentage"]
                else:
                    return result  # Can't extract numeric value
            elif isinstance(value, (int, float)):
                numeric_value = float(value)
            else:
                # Try to extract number from string
                numeric_str = re.sub(r"[^\d\.]", "", str(value))
                if numeric_str:
                    numeric_value = float(numeric_str)
                else:
                    result["errors"].append("Could not extract numeric value")
                    result["is_valid"] = False
                    return result

            # Range validation
            min_value = rules.get("min_value")
            max_value = rules.get("max_value")

            if min_value is not None and numeric_value < min_value:
                result["errors"].append(
                    f"Value {numeric_value} is below minimum {min_value}"
                )
                result["is_valid"] = False

            if max_value is not None and numeric_value > max_value:
                result["errors"].append(
                    f"Value {numeric_value} is above maximum {max_value}"
                )
                result["is_valid"] = False

            # Reasonable range warnings
            if field_type == "experience":
                if numeric_value > 40:
                    result["warnings"].append("Very high experience value")
                    result["score_multiplier"] *= 0.9
                elif numeric_value < 0:
                    result["errors"].append("Negative experience not allowed")
                    result["is_valid"] = False

            elif field_type == "age":
                if numeric_value > 70:
                    result["warnings"].append("Unusually high age")
                    result["score_multiplier"] *= 0.9
                elif numeric_value < 18:
                    result["warnings"].append("Age below typical working age")
                    result["score_multiplier"] *= 0.95

        except ValueError as e:
            result["errors"].append(f"Invalid numeric format: {str(e)}")
            result["is_valid"] = False

        return result

    def _validate_list_field(
        self, value: List[Any], rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate list fields (like skills)."""
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "score_multiplier": 1.0,
        }

        # Check list length
        min_items = rules.get("min_items", 0)
        max_items = rules.get("max_items", 1000)

        if len(value) < min_items:
            result["errors"].append(f"Too few items (minimum {min_items})")
            result["is_valid"] = False

        if len(value) > max_items:
            result["warnings"].append(f"Very many items (maximum {max_items})")
            result["score_multiplier"] *= 0.9

        # Check individual items
        item_min_length = rules.get("item_min_length", 1)
        item_max_length = rules.get("item_max_length", 100)

        short_items = 0
        long_items = 0
        empty_items = 0

        for item in value:
            item_str = str(item).strip()
            if not item_str:
                empty_items += 1
            elif len(item_str) < item_min_length:
                short_items += 1
            elif len(item_str) > item_max_length:
                long_items += 1

        if empty_items > 0:
            result["warnings"].append(f"{empty_items} empty items found")
            result["score_multiplier"] *= 0.95

        if short_items > len(value) * 0.3:  # More than 30% short items
            result["warnings"].append(f"{short_items} very short items")
            result["score_multiplier"] *= 0.9

        if long_items > 0:
            result["warnings"].append(f"{long_items} very long items")
            result["score_multiplier"] *= 0.95

        # Check for duplicates
        unique_items = set(str(item).lower().strip() for item in value)
        if len(unique_items) < len(value):
            duplicates = len(value) - len(unique_items)
            result["warnings"].append(f"{duplicates} duplicate items found")
            result["score_multiplier"] *= 0.85

        return result

    def _perform_quality_check(
        self, value: Any, check: str, field_type: str
    ) -> Dict[str, Any]:
        """Perform specific quality checks on field values."""
        result = {"issues": [], "score_multiplier": 1.0}

        value_str = str(value).strip()

        if check == "proper_case":
            if value_str.isupper():
                result["issues"].append("All uppercase text")
                result["score_multiplier"] = 0.9
            elif value_str.islower():
                result["issues"].append("All lowercase text")
                result["score_multiplier"] = 0.9

        elif check == "no_numbers":
            if re.search(r"\d", value_str):
                result["issues"].append("Contains numbers")
                result["score_multiplier"] = 0.8

        elif check == "reasonable_length":
            if len(value_str) < 3:
                result["issues"].append("Very short value")
                result["score_multiplier"] = 0.7

        elif check == "valid_format":
            if field_type == "email" and not re.match(
                r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", value_str
            ):
                result["issues"].append("Invalid email format")
                result["score_multiplier"] = 0.3
            elif field_type == "phone" and not re.match(
                r"^[\+]?[\d\s\-\(\)]{10,20}$", value_str
            ):
                result["issues"].append("Invalid phone format")
                result["score_multiplier"] = 0.4

        elif check == "no_spaces" and " " in value_str:
            result["issues"].append("Contains spaces")
            result["score_multiplier"] = 0.6

        elif check == "sufficient_digits":
            digit_count = len(re.findall(r"\d", value_str))
            if digit_count < 10:
                result["issues"].append("Insufficient digits for phone number")
                result["score_multiplier"] = 0.5

        elif check == "reasonable_range":
            # Handled in numeric validation
            pass

        elif check == "no_duplicates":
            if isinstance(value, list):
                if len(set(str(item).lower() for item in value)) < len(value):
                    result["issues"].append("Contains duplicate items")
                    result["score_multiplier"] = 0.85

        return result

    def clean_field_value(
        self, field_name: str, field_value: Any, cleaning_aggressive: bool = False
    ) -> Dict[str, Any]:
        """
        Clean and normalize a field value.

        Args:
            field_name: Name of the field
            field_value: Value to clean
            cleaning_aggressive: Whether to apply aggressive cleaning

        Returns:
            Dictionary containing cleaning results
        """
        cleaning_result = {
            "field_name": field_name,
            "original_value": field_value,
            "cleaned_value": field_value,
            "cleaning_applied": [],
            "issues_fixed": [],
            "cleaning_score": 1.0,
        }

        if field_value is None or (
            isinstance(field_value, str) and not field_value.strip()
        ):
            return cleaning_result

        field_type = self._determine_field_type(field_name)
        cleaned_value = field_value

        try:
            # Apply whitespace cleaning
            if isinstance(cleaned_value, str):
                original_cleaned = cleaned_value

                # Basic whitespace normalization
                cleaned_value = re.sub(r"^\s+|\s+$", "", cleaned_value)  # Trim
                cleaned_value = re.sub(r"\s+", " ", cleaned_value)  # Normalize spaces

                if original_cleaned != cleaned_value:
                    cleaning_result["cleaning_applied"].append(
                        "whitespace_normalization"
                    )
                    cleaning_result["issues_fixed"].append("Normalized whitespace")

            # Apply field-specific cleaning
            if field_type == "name":
                cleaned_value = self._clean_name_field(
                    cleaned_value, cleaning_aggressive
                )
                if cleaned_value != field_value:
                    cleaning_result["cleaning_applied"].append("name_formatting")
                    cleaning_result["issues_fixed"].append("Improved name formatting")

            elif field_type == "email":
                cleaned_value = self._clean_email_field(
                    cleaned_value, cleaning_aggressive
                )
                if cleaned_value != field_value:
                    cleaning_result["cleaning_applied"].append("email_formatting")
                    cleaning_result["issues_fixed"].append("Standardized email format")

            elif field_type == "phone":
                cleaned_value = self._clean_phone_field(
                    cleaned_value, cleaning_aggressive
                )
                if cleaned_value != field_value:
                    cleaning_result["cleaning_applied"].append("phone_formatting")
                    cleaning_result["issues_fixed"].append("Standardized phone format")

            elif field_type == "skills":
                cleaned_value = self._clean_skills_field(
                    cleaned_value, cleaning_aggressive
                )
                if cleaned_value != field_value:
                    cleaning_result["cleaning_applied"].append("skills_normalization")
                    cleaning_result["issues_fixed"].append("Normalized skills list")

            elif field_type in ["experience", "salary", "percentage"]:
                cleaned_value = self._clean_numeric_field(
                    cleaned_value, field_type, cleaning_aggressive
                )
                if cleaned_value != field_value:
                    cleaning_result["cleaning_applied"].append(
                        f"{field_type}_normalization"
                    )
                    cleaning_result["issues_fixed"].append(
                        f"Normalized {field_type} format"
                    )

            elif field_type == "location":
                cleaned_value = self._clean_location_field(
                    cleaned_value, cleaning_aggressive
                )
                if cleaned_value != field_value:
                    cleaning_result["cleaning_applied"].append("location_formatting")
                    cleaning_result["issues_fixed"].append(
                        "Improved location formatting"
                    )

            # Calculate cleaning score
            if len(cleaning_result["cleaning_applied"]) > 0:
                cleaning_result["cleaning_score"] = min(
                    1.0, 0.8 + 0.1 * len(cleaning_result["issues_fixed"])
                )

            cleaning_result["cleaned_value"] = cleaned_value

        except Exception as e:
            logger.error(f"Error cleaning field {field_name}: {e}")
            cleaning_result["issues_fixed"].append(f"Cleaning error: {str(e)}")
            cleaning_result["cleaning_score"] = 0.5

        return cleaning_result

    def _clean_name_field(self, value: Any, aggressive: bool = False) -> str:
        """Clean name field values."""
        if not isinstance(value, str):
            return str(value)

        cleaned = value.strip()

        # Fix case issues
        if cleaned.isupper() or cleaned.islower():
            cleaned = cleaned.title()

        # Remove extra spaces
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Remove numbers if aggressive cleaning
        if aggressive:
            cleaned = re.sub(r"\d+", "", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Fix common patterns
        cleaned = re.sub(
            r"\b([A-Z])\.([A-Z])\b", r"\1. \2", cleaned
        )  # Fix "A.B" to "A. B"

        return cleaned

    def _clean_email_field(self, value: Any, aggressive: bool = False) -> str:
        """Clean email field values."""
        if not isinstance(value, str):
            return str(value)

        cleaned = value.strip().lower()

        # Remove spaces
        cleaned = re.sub(r"\s+", "", cleaned)

        # Extract email if it's in a longer string
        if aggressive:
            email_match = re.search(
                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", cleaned
            )
            if email_match:
                cleaned = email_match.group(0)

        return cleaned

    def _clean_phone_field(self, value: Any, aggressive: bool = False) -> str:
        """Clean phone field values."""
        if not isinstance(value, str):
            return str(value)

        cleaned = value.strip()

        # Remove common separators and keep only digits, +, (), -
        cleaned = re.sub(r"[^\d\+\(\)\-\s]", "", cleaned)

        # Normalize spaces
        cleaned = re.sub(r"\s+", " ", cleaned)

        # If aggressive, try to format standard patterns
        if aggressive:
            # Remove all non-digits except +
            digits_only = re.sub(r"[^\d\+]", "", cleaned)

            if digits_only.startswith("+91") and len(digits_only) == 13:
                # Indian format: +91 XXXXX XXXXX
                cleaned = f"+91 {digits_only[3:8]} {digits_only[8:]}"
            elif digits_only.startswith("91") and len(digits_only) == 12:
                # Indian format without +: 91 XXXXX XXXXX
                cleaned = f"+91 {digits_only[2:7]} {digits_only[7:]}"
            elif len(digits_only) == 10:
                # 10 digit number: XXXXX XXXXX
                cleaned = f"{digits_only[:5]} {digits_only[5:]}"

        return cleaned

    def _clean_skills_field(self, value: Any, aggressive: bool = False) -> Any:
        """Clean skills field values."""
        if isinstance(value, list):
            cleaned_skills = []
            seen_skills = set()

            for skill in value:
                if skill and isinstance(skill, str):
                    skill_clean = skill.strip()

                    # Skip empty or very short skills
                    if len(skill_clean) < 2:
                        continue

                    # Remove duplicates (case-insensitive)
                    skill_lower = skill_clean.lower()
                    if skill_lower in seen_skills:
                        continue

                    seen_skills.add(skill_lower)
                    cleaned_skills.append(skill_clean)

            return cleaned_skills

        elif isinstance(value, str):
            # Try to split skills string
            skills_text = value.strip()

            # Common separators
            for separator in [",", ";", "|", "/", "&", "\n"]:
                if separator in skills_text:
                    skills_list = [s.strip() for s in skills_text.split(separator)]
                    skills_list = [s for s in skills_list if s and len(s) >= 2]
                    return skills_list

            # If no separators found, check if it looks like a list
            if aggressive and len(skills_text) > 50:
                # Try to split by spaces for potential skill keywords
                words = skills_text.split()
                if len(words) > 5:
                    return words

            return skills_text

        return value

    def _clean_numeric_field(
        self, value: Any, field_type: str, aggressive: bool = False
    ) -> Any:
        """Clean numeric field values."""
        if isinstance(value, dict):
            return value  # Already structured

        if isinstance(value, (int, float)):
            return value

        if isinstance(value, str):
            value_str = value.strip()

            # Remove currency symbols for salary
            if field_type == "salary":
                value_str = re.sub(r"[₹$€£]", "", value_str)
                value_str = re.sub(
                    r"\b(INR|USD|EUR|GBP|Rs\.?|rupees?)\b",
                    "",
                    value_str,
                    flags=re.IGNORECASE,
                )

            # Handle percentage
            if field_type == "percentage":
                if "%" in value_str:
                    value_str = value_str.replace("%", "").strip()
                    try:
                        return float(value_str)
                    except ValueError:
                        pass

            # Extract numeric value
            numeric_match = re.search(r"(\d+(?:\.\d+)?)", value_str)
            if numeric_match:
                try:
                    numeric_value = float(numeric_match.group(1))

                    # Handle special cases
                    if field_type == "salary":
                        # Convert lakhs/crores
                        if "lakh" in value_str.lower() or "lac" in value_str.lower():
                            numeric_value *= 100000
                        elif "crore" in value_str.lower():
                            numeric_value *= 10000000
                        elif "k" in value_str.lower():
                            numeric_value *= 1000

                    return numeric_value
                except ValueError:
                    pass

        return value

    def _clean_location_field(self, value: Any, aggressive: bool = False) -> str:
        """Clean location field values."""
        if not isinstance(value, str):
            return str(value)

        cleaned = value.strip()

        # Fix case
        if cleaned.isupper() or cleaned.islower():
            cleaned = cleaned.title()

        # Remove extra spaces and normalize punctuation
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r",\s*,", ",", cleaned)
        cleaned = re.sub(r",\s*$", "", cleaned)

        return cleaned

    def validate_cross_field_consistency(
        self, extracted_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate consistency across multiple fields.

        Args:
            extracted_fields: Dictionary of all extracted fields

        Returns:
            Dictionary containing cross-field validation results
        """
        validation_result = {
            "is_consistent": True,
            "consistency_score": 1.0,
            "issues": [],
            "warnings": [],
            "field_conflicts": [],
        }

        try:
            # Experience vs Age validation
            if "experience_years" in extracted_fields and "age" in extracted_fields:
                experience = float(extracted_fields["experience_years"])
                age = int(extracted_fields["age"])

                max_possible_experience = age - 16  # Assuming work starts at 16
                if experience > max_possible_experience + 2:  # 2 years tolerance
                    validation_result["issues"].append(
                        f"Experience ({experience} years) seems too high for age ({age} years)"
                    )
                    validation_result["is_consistent"] = False
                    validation_result["consistency_score"] *= 0.7

            # Graduation year vs Age validation
            if "graduation_year" in extracted_fields and "age" in extracted_fields:
                grad_year = int(extracted_fields["graduation_year"])
                age = int(extracted_fields["age"])
                current_year = datetime.now().year

                graduation_age = age - (current_year - grad_year)
                if graduation_age < 16 or graduation_age > 35:
                    validation_result["warnings"].append(
                        f"Graduation age ({graduation_age}) seems unusual"
                    )
                    validation_result["consistency_score"] *= 0.9

            # Salary vs Experience validation
            if (
                "current_salary" in extracted_fields
                and "experience_years" in extracted_fields
            ):
                salary_data = extracted_fields["current_salary"]
                experience = float(extracted_fields["experience_years"])

                if isinstance(salary_data, dict) and "amount" in salary_data:
                    salary_amount = salary_data["amount"]

                    # Very rough salary validation (Indian context)
                    min_expected = experience * 200000  # 2 lakhs per year of experience
                    max_expected = (
                        experience * 2000000
                    )  # 20 lakhs per year of experience

                    if salary_amount < min_expected * 0.5:  # 50% below minimum
                        validation_result["warnings"].append(
                            f"Salary seems low for {experience} years of experience"
                        )
                        validation_result["consistency_score"] *= 0.95
                    elif salary_amount > max_expected * 2:  # 200% above maximum
                        validation_result["warnings"].append(
                            f"Salary seems very high for {experience} years of experience"
                        )
                        validation_result["consistency_score"] *= 0.95

            # Name consistency validation
            name_fields = ["name", "first_name", "last_name", "display_name"]
            present_name_fields = {
                field: extracted_fields[field]
                for field in name_fields
                if field in extracted_fields
            }

            if len(present_name_fields) > 1:
                # Check if names are consistent
                full_name = present_name_fields.get("name", "")
                first_name = present_name_fields.get("first_name", "")
                last_name = present_name_fields.get("last_name", "")

                if full_name and first_name and last_name:
                    expected_full = f"{first_name} {last_name}"
                    if full_name.lower() != expected_full.lower():
                        validation_result["field_conflicts"].append(
                            {
                                "type": "name_inconsistency",
                                "fields": ["name", "first_name", "last_name"],
                                "issue": f"Full name '{full_name}' doesn't match '{expected_full}'",
                            }
                        )
                        validation_result["consistency_score"] *= 0.9

            # Email domain vs company validation
            if "email" in extracted_fields and "current_company" in extracted_fields:
                email = str(extracted_fields["email"]).lower()
                company = str(extracted_fields["current_company"]).lower()

                if "@" in email:
                    email_domain = email.split("@")[1]
                    # Check if email domain relates to company
                    company_words = re.findall(r"\w+", company)
                    domain_words = re.findall(r"\w+", email_domain.split(".")[0])

                    # This is a basic check - could be enhanced
                    if any(
                        word in email_domain for word in company_words if len(word) > 3
                    ):
                        validation_result[
                            "consistency_score"
                        ] *= 1.05  # Slight bonus for consistency

        except Exception as e:
            logger.error(f"Error in cross-field validation: {e}")
            validation_result["issues"].append(f"Validation error: {str(e)}")

        return validation_result

    def calculate_data_quality_score(
        self,
        extracted_fields: Dict[str, Any],
        validation_results: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate overall data quality score.

        Args:
            extracted_fields: Dictionary of extracted fields
            validation_results: List of individual field validation results

        Returns:
            Dictionary containing quality score and breakdown
        """
        quality_result = {
            "overall_quality_score": 0.0,
            "quality_breakdown": {},
            "grade": "F",
            "recommendations": [],
        }

        try:
            metrics = self.quality_metrics

            # Completeness Score
            completeness_score = self._calculate_completeness_score(
                extracted_fields, metrics["completeness"]
            )

            # Accuracy Score
            accuracy_score = (
                self._calculate_accuracy_score(validation_results, metrics["accuracy"])
                if validation_results
                else 0.8
            )

            # Consistency Score
            consistency_result = self.validate_cross_field_consistency(extracted_fields)
            consistency_score = consistency_result["consistency_score"]

            # Richness Score
            richness_score = self._calculate_richness_score(
                extracted_fields, metrics["richness"]
            )

            # Reliability Score (based on extraction confidence)
            reliability_score = self._calculate_reliability_score(
                extracted_fields, metrics["reliability"]
            )

            # Weighted overall score
            overall_score = (
                completeness_score * metrics["completeness"]["weight"]
                + accuracy_score * metrics["accuracy"]["weight"]
                + consistency_score * metrics["consistency"]["weight"]
                + richness_score * metrics["richness"]["weight"]
                + reliability_score * metrics["reliability"]["weight"]
            )

            quality_result["overall_quality_score"] = round(overall_score * 100, 1)
            quality_result["quality_breakdown"] = {
                "completeness": round(completeness_score * 100, 1),
                "accuracy": round(accuracy_score * 100, 1),
                "consistency": round(consistency_score * 100, 1),
                "richness": round(richness_score * 100, 1),
                "reliability": round(reliability_score * 100, 1),
            }

            # Assign grade
            if overall_score >= 0.9:
                quality_result["grade"] = "A+"
            elif overall_score >= 0.8:
                quality_result["grade"] = "A"
            elif overall_score >= 0.7:
                quality_result["grade"] = "B"
            elif overall_score >= 0.6:
                quality_result["grade"] = "C"
            elif overall_score >= 0.5:
                quality_result["grade"] = "D"
            else:
                quality_result["grade"] = "F"

            # Generate recommendations
            quality_result["recommendations"] = self._generate_quality_recommendations(
                completeness_score,
                accuracy_score,
                consistency_score,
                richness_score,
                reliability_score,
            )

        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            quality_result["overall_quality_score"] = 50.0
            quality_result["grade"] = "D"
            quality_result["recommendations"] = [
                "Error in quality calculation - manual review recommended"
            ]

        return quality_result

    def _calculate_completeness_score(
        self, fields: Dict[str, Any], config: Dict[str, Any]
    ) -> float:
        """Calculate completeness score based on field presence."""
        critical_fields = config.get("critical_fields", [])
        important_fields = config.get("important_fields", [])
        optional_fields = config.get("optional_fields", [])

        critical_present = sum(
            1 for field in critical_fields if field in fields and fields[field]
        )
        important_present = sum(
            1 for field in important_fields if field in fields and fields[field]
        )
        optional_present = sum(
            1 for field in optional_fields if field in fields and fields[field]
        )

        critical_score = (
            critical_present / len(critical_fields) if critical_fields else 1.0
        )
        important_score = (
            important_present / len(important_fields) if important_fields else 1.0
        )
        optional_score = (
            optional_present / len(optional_fields) if optional_fields else 1.0
        )

        # Weighted completeness
        completeness = (
            critical_score * 0.6 + important_score * 0.3 + optional_score * 0.1
        )
        return min(completeness, 1.0)

    def _calculate_accuracy_score(
        self, validation_results: List[Dict[str, Any]], config: Dict[str, Any]
    ) -> float:
        """Calculate accuracy score based on validation results."""
        if not validation_results:
            return 0.8  # Default score

        valid_count = sum(
            1 for result in validation_results if result.get("is_valid", False)
        )
        total_count = len(validation_results)

        if total_count == 0:
            return 0.8

        base_accuracy = valid_count / total_count

        # Factor in validation scores
        avg_validation_score = (
            sum(result.get("validation_score", 0.5) for result in validation_results)
            / total_count
        )

        return base_accuracy * 0.7 + avg_validation_score * 0.3

    def _calculate_richness_score(
        self, fields: Dict[str, Any], config: Dict[str, Any]
    ) -> float:
        """Calculate richness score based on field diversity and detail."""
        total_fields = len(fields)

        # Base score from field count
        base_score = min(total_fields / 15, 1.0)  # Normalize to 15 fields

        # Bonus for detailed fields
        detailed_bonus = 0.0

        # Check for structured data
        structured_fields = sum(
            1 for value in fields.values() if isinstance(value, dict)
        )
        if structured_fields > 0:
            detailed_bonus += 0.1

        # Check for list fields
        list_fields = sum(
            1 for value in fields.values() if isinstance(value, list) and len(value) > 1
        )
        if list_fields > 0:
            detailed_bonus += 0.1

        return min(base_score + detailed_bonus, 1.0)

    def _calculate_reliability_score(
        self, fields: Dict[str, Any], config: Dict[str, Any]
    ) -> float:
        """Calculate reliability score based on extraction confidence."""
        # This would typically use extraction confidence scores
        # For now, return a base score
        return 0.8

    def _generate_quality_recommendations(
        self,
        completeness: float,
        accuracy: float,
        consistency: float,
        richness: float,
        reliability: float,
    ) -> List[str]:
        """Generate recommendations based on quality scores."""
        recommendations = []

        if completeness < 0.7:
            recommendations.append(
                "Add missing critical fields like name, email, phone, or experience"
            )

        if accuracy < 0.7:
            recommendations.append("Improve data accuracy by fixing validation errors")

        if consistency < 0.8:
            recommendations.append(
                "Resolve cross-field inconsistencies (age vs experience, name conflicts)"
            )

        if richness < 0.6:
            recommendations.append(
                "Add more detailed information like skills, projects, or certifications"
            )

        if reliability < 0.7:
            recommendations.append(
                "Verify extracted data and improve source data quality"
            )

        if not recommendations:
            recommendations.append(
                "Data quality is good - consider adding more optional fields for completeness"
            )

        return recommendations
