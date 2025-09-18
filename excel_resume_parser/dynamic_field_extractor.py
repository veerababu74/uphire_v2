"""
Dynamic Field Extractor for Excel Resume Parser

This module provides intelligent field extraction capabilities that can identify
and extract resume-relevant data from any column structure, including handling
nested data, composite fields, and intelligent value extraction.
"""

import re
import json
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from collections import defaultdict
import logging
from core.custom_logger import CustomLogger
from .enhanced_column_mapper import EnhancedColumnMapper
from .enhanced_data_detector import EnhancedDataTypeDetector

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("dynamic_field_extractor")


class DynamicFieldExtractor:
    """
    Advanced field extractor that can dynamically identify and extract
    resume-relevant data from various column structures and data formats.
    """

    def __init__(self):
        """Initialize the dynamic field extractor."""
        self.column_mapper = EnhancedColumnMapper()
        self.data_detector = EnhancedDataTypeDetector()
        self.composite_field_patterns = self._initialize_composite_patterns()
        self.extraction_rules = self._initialize_extraction_rules()
        self.priority_fields = self._initialize_priority_fields()

    def _initialize_composite_patterns(self) -> Dict[str, List[str]]:
        """
        Initialize patterns for composite fields that contain multiple data types.

        Returns:
            Dictionary mapping composite field types to extraction patterns
        """
        return {
            "name_email": [
                r"([a-zA-Z\s]+)\s*[-(<]\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\s*[>)-]",
                r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\s*[-(<]\s*([a-zA-Z\s]+)\s*[>)-]",
            ],
            "name_phone": [
                r"([a-zA-Z\s]+)\s*[-(<]\s*(\+?[\d\s()-]+)\s*[>)-]",
                r"(\+?[\d\s()-]+)\s*[-(<]\s*([a-zA-Z\s]+)\s*[>)-]",
            ],
            "location_details": [
                r"([a-zA-Z\s]+),\s*([a-zA-Z\s]+),\s*([a-zA-Z\s]+)(?:,\s*(\d{5,6}))?",  # City, State, Country, PIN
                r"([a-zA-Z\s]+),\s*([a-zA-Z\s]+)(?:,\s*(\d{5,6}))?",  # City, State, PIN
                r"([a-zA-Z\s]+)\s*-\s*(\d{5,6})",  # City - PIN
            ],
            "experience_company": [
                r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:at|with|in)\s*(.+)",
                r"(.+)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
            ],
            "education_details": [
                r"([a-zA-Z\s.]+)\s*(?:from|in)\s*([a-zA-Z\s.]+)(?:\s*[-–]\s*(\d{4}))?",
                r"([a-zA-Z\s.]+)\s*,\s*([a-zA-Z\s.]+)(?:\s*,\s*(\d{4}))?",
            ],
            "salary_currency": [
                r"([₹$€£])\s*(\d+(?:,\d{3})*(?:\.\d{2})?)",
                r"(\d+(?:,\d{3})*(?:\.\d{2})?)\s*([₹$€£]|INR|USD|EUR|GBP)",
            ],
            "skill_rating": [
                r"([a-zA-Z\s+#.]+)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(?:/\s*\d+|%?)",
                r"([a-zA-Z\s+#.]+)\s*\(\s*(\d+(?:\.\d+)?)\s*(?:/\s*\d+|%?)\s*\)",
            ],
        }

    def _initialize_extraction_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize extraction rules for different field types.

        Returns:
            Dictionary mapping field types to their extraction rules
        """
        return {
            "personal_info": {
                "required_confidence": 0.8,
                "max_instances": 1,
                "validation_required": True,
                "fields": [
                    "name",
                    "first_name",
                    "last_name",
                    "email",
                    "phone",
                    "age",
                    "gender",
                ],
            },
            "contact_info": {
                "required_confidence": 0.7,
                "max_instances": 2,  # Primary and secondary
                "validation_required": True,
                "fields": [
                    "email",
                    "phone",
                    "alternate_phone",
                    "address",
                    "city",
                    "state",
                    "country",
                    "pincode",
                ],
            },
            "professional_info": {
                "required_confidence": 0.6,
                "max_instances": 3,  # Current, previous, etc.
                "validation_required": False,
                "fields": [
                    "current_role",
                    "current_company",
                    "previous_company",
                    "experience",
                    "skills",
                    "technical_skills",
                ],
            },
            "educational_info": {
                "required_confidence": 0.6,
                "max_instances": 5,  # Multiple degrees
                "validation_required": False,
                "fields": [
                    "education",
                    "degree",
                    "college",
                    "graduation_year",
                    "percentage",
                ],
            },
            "financial_info": {
                "required_confidence": 0.7,
                "max_instances": 2,  # Current and expected
                "validation_required": True,
                "fields": ["current_salary", "expected_salary", "last_salary"],
            },
            "preferences": {
                "required_confidence": 0.5,
                "max_instances": 1,
                "validation_required": False,
                "fields": ["notice_period", "preferred_location", "work_mode"],
            },
            "additional_info": {
                "required_confidence": 0.4,
                "max_instances": 10,
                "validation_required": False,
                "fields": [
                    "languages",
                    "certifications",
                    "projects",
                    "achievements",
                    "hobbies",
                    "references",
                ],
            },
        }

    def _initialize_priority_fields(self) -> Dict[str, int]:
        """
        Initialize priority scores for different fields.
        Higher scores indicate higher priority.

        Returns:
            Dictionary mapping field names to priority scores
        """
        return {
            # Critical fields (highest priority)
            "name": 100,
            "email": 95,
            "phone": 90,
            # Important professional fields
            "experience": 85,
            "current_role": 80,
            "skills": 75,
            "current_company": 70,
            # Important personal fields
            "city": 65,
            "education": 60,
            # Moderate priority fields
            "current_salary": 55,
            "expected_salary": 50,
            "degree": 45,
            "college": 40,
            "notice_period": 35,
            # Lower priority fields
            "age": 30,
            "graduation_year": 25,
            "previous_company": 20,
            "state": 15,
            "country": 10,
            # Lowest priority fields
            "gender": 5,
            "marital_status": 5,
            "hobbies": 5,
        }

    def extract_fields_from_row(
        self,
        row_data: Dict[str, Any],
        column_mapping: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Extract and normalize fields from a single row of data.

        Args:
            row_data: Dictionary containing row data
            column_mapping: Optional pre-computed column mapping

        Returns:
            Dictionary containing extracted and normalized fields
        """
        logger.info(f"Extracting fields from row with {len(row_data)} columns")

        # Generate column mapping if not provided
        if column_mapping is None:
            column_names = list(row_data.keys())
            column_mapping = self.column_mapper.map_columns(column_names)

        extracted_fields = {}
        field_metadata = {}
        extraction_stats = {
            "total_columns": len(row_data),
            "mapped_columns": 0,
            "extracted_fields": 0,
            "validation_passed": 0,
            "validation_failed": 0,
        }

        # Process each column
        for column_name, value in row_data.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                continue

            column_info = column_mapping.get(column_name, {})
            mapped_field = column_info.get("mapped_field")

            if mapped_field:
                extraction_stats["mapped_columns"] += 1

                # Extract and validate the field
                field_result = self._extract_single_field(
                    value=value,
                    field_name=mapped_field,
                    column_name=column_name,
                    column_info=column_info,
                )

                if field_result["success"]:
                    extraction_stats["extracted_fields"] += 1

                    # Handle field conflicts (multiple columns mapping to same field)
                    if mapped_field in extracted_fields:
                        conflict_result = self._resolve_field_conflict(
                            existing_data=extracted_fields[mapped_field],
                            new_data=field_result,
                            field_name=mapped_field,
                        )
                        extracted_fields[mapped_field] = conflict_result[
                            "resolved_data"
                        ]
                        field_metadata[mapped_field] = conflict_result["metadata"]
                    else:
                        extracted_fields[mapped_field] = field_result[
                            "normalized_value"
                        ]
                        field_metadata[mapped_field] = field_result["metadata"]

                    if field_result["validation_passed"]:
                        extraction_stats["validation_passed"] += 1
                    else:
                        extraction_stats["validation_failed"] += 1
            else:
                # Try to extract information from unmapped columns
                unmapped_result = self._extract_from_unmapped_column(column_name, value)
                if unmapped_result["extracted_fields"]:
                    for field_name, field_data in unmapped_result[
                        "extracted_fields"
                    ].items():
                        if field_name not in extracted_fields:
                            extracted_fields[field_name] = field_data["value"]
                            field_metadata[field_name] = field_data["metadata"]
                            extraction_stats["extracted_fields"] += 1

        # Post-process extracted fields
        processed_fields = self._post_process_fields(extracted_fields, field_metadata)

        # Generate derived fields
        derived_fields = self._generate_derived_fields(processed_fields)
        processed_fields.update(derived_fields)

        logger.info(
            f"Field extraction completed: {extraction_stats['extracted_fields']} fields extracted from {extraction_stats['mapped_columns']} mapped columns"
        )

        return {
            "extracted_fields": processed_fields,
            "field_metadata": field_metadata,
            "extraction_stats": extraction_stats,
            "column_mapping_used": column_mapping,
        }

    def _extract_single_field(
        self, value: Any, field_name: str, column_name: str, column_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract and validate a single field value.

        Args:
            value: Raw value to extract from
            field_name: Target field name
            column_name: Original column name
            column_info: Column mapping information

        Returns:
            Dictionary containing extraction results
        """
        try:
            # Detect data type
            expected_type = column_info.get("data_type", "text")
            detected_type, type_confidence = self.data_detector.detect_data_type(
                value, expected_type
            )

            # Validate and normalize
            validation_result = self.data_detector.validate_and_normalize(
                value=value, data_type=detected_type, field_name=field_name
            )

            # Extract composite data if applicable
            composite_result = self._extract_composite_data(value, field_name)

            return {
                "success": True,
                "original_value": value,
                "normalized_value": validation_result["normalized_value"],
                "validation_passed": validation_result["is_valid"],
                "detected_type": detected_type,
                "type_confidence": type_confidence,
                "field_name": field_name,
                "column_name": column_name,
                "composite_data": composite_result,
                "metadata": {
                    "extraction_method": "direct_mapping",
                    "mapping_confidence": column_info.get("confidence", 0.0),
                    "validation_errors": validation_result.get("errors", []),
                    "data_type_match": detected_type == expected_type,
                },
            }

        except Exception as e:
            logger.error(
                f"Error extracting field {field_name} from column {column_name}: {e}"
            )
            return {
                "success": False,
                "original_value": value,
                "normalized_value": None,
                "validation_passed": False,
                "error": str(e),
                "metadata": {"extraction_method": "failed"},
            }

    def _extract_composite_data(self, value: Any, field_name: str) -> Dict[str, Any]:
        """
        Extract additional data from composite fields.

        Args:
            value: Raw value to analyze
            field_name: Field name for context

        Returns:
            Dictionary containing extracted composite data
        """
        if not isinstance(value, str) or not value.strip():
            return {}

        value_str = str(value).strip()
        composite_data = {}

        # Check each composite pattern type
        for pattern_type, patterns in self.composite_field_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, value_str, re.IGNORECASE)
                if match:
                    groups = match.groups()

                    if pattern_type == "name_email" and field_name in ["name", "email"]:
                        if len(groups) >= 2:
                            composite_data["extracted_name"] = groups[0].strip()
                            composite_data["extracted_email"] = groups[1].strip()

                    elif pattern_type == "name_phone" and field_name in [
                        "name",
                        "phone",
                    ]:
                        if len(groups) >= 2:
                            composite_data["extracted_name"] = groups[0].strip()
                            composite_data["extracted_phone"] = groups[1].strip()

                    elif pattern_type == "location_details" and field_name in [
                        "address",
                        "city",
                        "location",
                    ]:
                        if len(groups) >= 2:
                            composite_data["extracted_city"] = groups[0].strip()
                            composite_data["extracted_state"] = groups[1].strip()
                            if len(groups) >= 3 and groups[2]:
                                composite_data["extracted_country"] = groups[2].strip()
                            if len(groups) >= 4 and groups[3]:
                                composite_data["extracted_pincode"] = groups[3].strip()

                    elif pattern_type == "experience_company" and field_name in [
                        "experience",
                        "current_company",
                    ]:
                        if len(groups) >= 2:
                            composite_data["extracted_experience"] = groups[0].strip()
                            composite_data["extracted_company"] = groups[1].strip()

                    elif pattern_type == "education_details" and field_name in [
                        "education",
                        "degree",
                        "college",
                    ]:
                        if len(groups) >= 2:
                            composite_data["extracted_degree"] = groups[0].strip()
                            composite_data["extracted_college"] = groups[1].strip()
                            if len(groups) >= 3 and groups[2]:
                                composite_data["extracted_year"] = groups[2].strip()

                    elif pattern_type == "salary_currency" and "salary" in field_name:
                        if len(groups) >= 2:
                            # Determine which group is currency and which is amount
                            if groups[0] in [
                                "₹",
                                "$",
                                "€",
                                "£",
                                "INR",
                                "USD",
                                "EUR",
                                "GBP",
                            ]:
                                composite_data["extracted_currency"] = groups[0]
                                composite_data["extracted_amount"] = groups[1]
                            else:
                                composite_data["extracted_amount"] = groups[0]
                                composite_data["extracted_currency"] = groups[1]

                    elif pattern_type == "skill_rating" and field_name in [
                        "skills",
                        "technical_skills",
                    ]:
                        if len(groups) >= 2:
                            composite_data["extracted_skill"] = groups[0].strip()
                            composite_data["extracted_rating"] = groups[1].strip()

                    break  # Use first matching pattern

        return composite_data

    def _extract_from_unmapped_column(
        self, column_name: str, value: Any
    ) -> Dict[str, Any]:
        """
        Try to extract useful information from unmapped columns.

        Args:
            column_name: Original column name
            value: Column value

        Returns:
            Dictionary containing extraction results
        """
        extracted_fields = {}

        if not value or (isinstance(value, str) and not value.strip()):
            return {"extracted_fields": extracted_fields}

        value_str = str(value).strip()

        # Use fuzzy matching with lower confidence to find potential fields
        potential_matches = self.column_mapper.fuzzy_match_field(
            column_name, confidence_threshold=0.4
        )

        if potential_matches:
            # Try the best match
            best_field, confidence = potential_matches[0]

            # Detect and validate data type
            detected_type, type_confidence = self.data_detector.detect_data_type(
                value_str
            )
            validation_result = self.data_detector.validate_and_normalize(
                value_str, detected_type, best_field
            )

            if validation_result["is_valid"] and confidence > 0.4:
                extracted_fields[best_field] = {
                    "value": validation_result["normalized_value"],
                    "metadata": {
                        "extraction_method": "unmapped_fuzzy_match",
                        "column_name": column_name,
                        "field_confidence": confidence,
                        "type_confidence": type_confidence,
                        "validation_passed": True,
                    },
                }

        # Try to extract composite data
        composite_data = self._extract_composite_data(value_str, "unknown")
        for key, extracted_value in composite_data.items():
            if key.startswith("extracted_"):
                field_name = key.replace("extracted_", "")
                if field_name not in extracted_fields:
                    # Validate extracted composite data
                    detected_type, _ = self.data_detector.detect_data_type(
                        extracted_value
                    )
                    validation_result = self.data_detector.validate_and_normalize(
                        extracted_value, detected_type, field_name
                    )

                    if validation_result["is_valid"]:
                        extracted_fields[field_name] = {
                            "value": validation_result["normalized_value"],
                            "metadata": {
                                "extraction_method": "composite_extraction",
                                "column_name": column_name,
                                "validation_passed": True,
                            },
                        }

        return {"extracted_fields": extracted_fields}

    def _resolve_field_conflict(
        self, existing_data: Any, new_data: Dict[str, Any], field_name: str
    ) -> Dict[str, Any]:
        """
        Resolve conflicts when multiple columns map to the same field.

        Args:
            existing_data: Previously extracted data for this field
            new_data: New extraction result
            field_name: Name of the conflicting field

        Returns:
            Dictionary containing resolved data and metadata
        """
        # Get priority of the field
        field_priority = self.priority_fields.get(field_name, 0)

        # Compare quality metrics
        existing_quality = self._calculate_data_quality(existing_data, field_name)
        new_quality = self._calculate_data_quality(
            new_data["normalized_value"], field_name
        )

        # Decide which data to keep
        if new_quality > existing_quality:
            resolved_data = new_data["normalized_value"]
            resolution_reason = "higher_quality"
        elif new_quality == existing_quality:
            # For same quality, prefer the one with higher mapping confidence
            new_confidence = new_data["metadata"].get("mapping_confidence", 0.0)

            # We don't have existing confidence readily available, so prefer new if it's high
            if new_confidence > 0.8:
                resolved_data = new_data["normalized_value"]
                resolution_reason = "higher_confidence"
            else:
                resolved_data = existing_data
                resolution_reason = "kept_existing"
        else:
            resolved_data = existing_data
            resolution_reason = "existing_higher_quality"

        return {
            "resolved_data": resolved_data,
            "metadata": {
                "conflict_resolution": resolution_reason,
                "existing_quality": existing_quality,
                "new_quality": new_quality,
                "field_priority": field_priority,
            },
        }

    def _calculate_data_quality(self, data: Any, field_name: str) -> float:
        """
        Calculate quality score for a piece of data.

        Args:
            data: Data to evaluate
            field_name: Field name for context

        Returns:
            Quality score between 0 and 1
        """
        if data is None:
            return 0.0

        quality_score = 0.0

        # Base score for having data
        quality_score += 0.2

        # Length/completeness score
        if isinstance(data, str):
            if len(data.strip()) > 0:
                quality_score += 0.2
            if len(data.strip()) > 5:  # Reasonable length
                quality_score += 0.2
        elif isinstance(data, (int, float)):
            if data > 0:
                quality_score += 0.4
        elif isinstance(data, list):
            if len(data) > 0:
                quality_score += 0.2
            if len(data) > 1:
                quality_score += 0.2
        elif isinstance(data, dict):
            if data:
                quality_score += 0.4

        # Field-specific quality checks
        if field_name == "email" and isinstance(data, str):
            if "@" in data and "." in data:
                quality_score += 0.4
        elif field_name == "phone" and isinstance(data, str):
            digit_count = len(re.findall(r"\d", data))
            if digit_count >= 10:
                quality_score += 0.4
        elif field_name in ["name", "first_name", "last_name"] and isinstance(
            data, str
        ):
            if data.strip() and not any(char.isdigit() for char in data):
                quality_score += 0.4

        return min(quality_score, 1.0)

    def _post_process_fields(
        self,
        extracted_fields: Dict[str, Any],
        field_metadata: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Post-process extracted fields to improve data quality.

        Args:
            extracted_fields: Dictionary of extracted fields
            field_metadata: Metadata for each field

        Returns:
            Dictionary of post-processed fields
        """
        processed_fields = extracted_fields.copy()

        # Name consolidation
        if (
            "first_name" in processed_fields
            and "last_name" in processed_fields
            and "name" not in processed_fields
        ):
            full_name = f"{processed_fields['first_name']} {processed_fields['last_name']}".strip()
            if full_name:
                processed_fields["name"] = full_name

        # Split name if we have full name but not first/last
        if "name" in processed_fields and "first_name" not in processed_fields:
            name_parts = str(processed_fields["name"]).strip().split()
            if len(name_parts) >= 2:
                processed_fields["first_name"] = name_parts[0]
                processed_fields["last_name"] = " ".join(name_parts[1:])

        # Location consolidation
        location_parts = []
        for field in ["city", "state", "country"]:
            if field in processed_fields and processed_fields[field]:
                location_parts.append(str(processed_fields[field]))

        if location_parts and "location" not in processed_fields:
            processed_fields["location"] = ", ".join(location_parts)

        # Skill normalization
        if "skills" in processed_fields:
            skills_data = processed_fields["skills"]
            if isinstance(skills_data, list):
                # Remove duplicates and empty items
                normalized_skills = []
                seen_skills = set()

                for skill in skills_data:
                    if isinstance(skill, str):
                        skill_clean = skill.strip().lower()
                        if skill_clean and skill_clean not in seen_skills:
                            seen_skills.add(skill_clean)
                            normalized_skills.append(skill.strip())

                processed_fields["skills"] = normalized_skills

        # Experience normalization
        if "experience" in processed_fields:
            exp_data = processed_fields["experience"]
            if isinstance(exp_data, dict) and "total_months" in exp_data:
                # Add additional computed fields
                years = exp_data["total_months"] / 12
                processed_fields["experience_years"] = round(years, 1)
                processed_fields["experience_months"] = exp_data["total_months"]

        return processed_fields

    def _generate_derived_fields(
        self, processed_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate derived fields based on extracted data.

        Args:
            processed_fields: Dictionary of processed fields

        Returns:
            Dictionary of derived fields
        """
        derived_fields = {}

        # Generate display name
        if "name" in processed_fields:
            derived_fields["display_name"] = processed_fields["name"]
        elif "first_name" in processed_fields:
            display_name = processed_fields["first_name"]
            if "last_name" in processed_fields:
                display_name += f" {processed_fields['last_name']}"
            derived_fields["display_name"] = display_name

        # Generate contact summary
        contact_info = []
        if "email" in processed_fields:
            contact_info.append(f"Email: {processed_fields['email']}")
        if "phone" in processed_fields:
            contact_info.append(f"Phone: {processed_fields['phone']}")
        if "city" in processed_fields:
            contact_info.append(f"Location: {processed_fields['city']}")

        if contact_info:
            derived_fields["contact_summary"] = " | ".join(contact_info)

        # Generate professional summary
        prof_info = []
        if "current_role" in processed_fields:
            prof_info.append(processed_fields["current_role"])
        if "current_company" in processed_fields:
            prof_info.append(f"at {processed_fields['current_company']}")
        if "experience" in processed_fields:
            exp_data = processed_fields["experience"]
            if isinstance(exp_data, dict) and "formatted" in exp_data:
                prof_info.append(f"({exp_data['formatted']} experience)")
            elif isinstance(exp_data, str):
                prof_info.append(f"({exp_data} experience)")

        if prof_info:
            derived_fields["professional_summary"] = " ".join(prof_info)

        # Generate completeness score
        important_fields = [
            "name",
            "email",
            "phone",
            "experience",
            "current_role",
            "skills",
            "education",
        ]
        present_fields = sum(
            1
            for field in important_fields
            if field in processed_fields and processed_fields[field]
        )
        completeness_score = (present_fields / len(important_fields)) * 100
        derived_fields["profile_completeness"] = round(completeness_score, 1)

        # Generate field counts
        derived_fields["total_fields_extracted"] = len(processed_fields)
        derived_fields["skills_count"] = len(
            processed_fields.get("skills", [])
            if isinstance(processed_fields.get("skills"), list)
            else 0
        )

        return derived_fields

    def extract_batch_fields(
        self, rows_data: List[Dict[str, Any]], use_common_mapping: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract fields from multiple rows efficiently.

        Args:
            rows_data: List of row dictionaries
            use_common_mapping: Whether to use a common column mapping for all rows

        Returns:
            List of extraction results for each row
        """
        logger.info(f"Starting batch field extraction for {len(rows_data)} rows")

        results = []
        common_mapping = None

        # Generate common mapping if requested
        if use_common_mapping and rows_data:
            # Use column names from first row as template
            first_row_columns = list(rows_data[0].keys())
            common_mapping = self.column_mapper.map_columns(first_row_columns)
            logger.info(
                f"Generated common mapping for {len(first_row_columns)} columns"
            )

        # Process each row
        for idx, row_data in enumerate(rows_data):
            try:
                extraction_result = self.extract_fields_from_row(
                    row_data=row_data, column_mapping=common_mapping
                )

                extraction_result["row_index"] = idx
                results.append(extraction_result)

                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(rows_data)} rows")

            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                results.append(
                    {
                        "row_index": idx,
                        "error": str(e),
                        "extracted_fields": {},
                        "extraction_stats": {
                            "total_columns": len(row_data),
                            "error": True,
                        },
                    }
                )

        logger.info(f"Batch extraction completed: {len(results)} rows processed")
        return results

    def get_extraction_summary(
        self, extraction_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for batch extraction results.

        Args:
            extraction_results: List of extraction results from extract_batch_fields

        Returns:
            Dictionary containing summary statistics
        """
        total_rows = len(extraction_results)
        successful_rows = sum(
            1 for result in extraction_results if "error" not in result
        )
        failed_rows = total_rows - successful_rows

        # Field extraction statistics
        field_counts = defaultdict(int)
        total_fields_extracted = 0

        for result in extraction_results:
            if "extracted_fields" in result:
                extracted_fields = result["extracted_fields"]
                total_fields_extracted += len(extracted_fields)

                for field_name in extracted_fields:
                    field_counts[field_name] += 1

        # Most common fields
        most_common_fields = sorted(
            field_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # Average completeness
        completeness_scores = []
        for result in extraction_results:
            if "extracted_fields" in result:
                completeness = result["extracted_fields"].get("profile_completeness", 0)
                completeness_scores.append(completeness)

        avg_completeness = (
            sum(completeness_scores) / len(completeness_scores)
            if completeness_scores
            else 0
        )

        return {
            "total_rows": total_rows,
            "successful_rows": successful_rows,
            "failed_rows": failed_rows,
            "success_rate": (
                (successful_rows / total_rows) * 100 if total_rows > 0 else 0
            ),
            "total_fields_extracted": total_fields_extracted,
            "avg_fields_per_row": (
                total_fields_extracted / successful_rows if successful_rows > 0 else 0
            ),
            "unique_fields_found": len(field_counts),
            "most_common_fields": most_common_fields,
            "average_completeness": round(avg_completeness, 1),
            "field_distribution": dict(field_counts),
        }
