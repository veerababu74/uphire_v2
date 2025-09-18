"""
Enhanced Text Formatting Engine for Excel Resume Parser

This module provides intelligent text formatting capabilities to create
well-structured resume text from various data formats and field types.
"""

import re
import json
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from datetime import datetime
from collections import defaultdict
import logging
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("enhanced_text_formatter")


class EnhancedTextFormatter:
    """
    Advanced text formatter that creates structured resume text from
    normalized field data with intelligent formatting and organization.
    """

    def __init__(self):
        """Initialize the enhanced text formatter."""
        self.section_templates = self._initialize_section_templates()
        self.field_priorities = self._initialize_field_priorities()
        self.formatting_rules = self._initialize_formatting_rules()
        self.section_order = self._initialize_section_order()

    def _initialize_section_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize templates for different resume sections.

        Returns:
            Dictionary mapping section names to their templates
        """
        return {
            "personal_information": {
                "title": "PERSONAL INFORMATION",
                "required_fields": ["name"],
                "optional_fields": [
                    "email",
                    "phone",
                    "age",
                    "gender",
                    "marital_status",
                    "date_of_birth",
                ],
                "template": "{name}\n{contact_info}\n{personal_details}",
                "format_style": "header",
            },
            "contact_information": {
                "title": "CONTACT INFORMATION",
                "required_fields": [],
                "optional_fields": [
                    "email",
                    "phone",
                    "alternate_phone",
                    "address",
                    "city",
                    "state",
                    "country",
                    "pincode",
                    "location",
                ],
                "template": "{contact_details}",
                "format_style": "list",
            },
            "professional_summary": {
                "title": "PROFESSIONAL SUMMARY",
                "required_fields": [],
                "optional_fields": [
                    "experience",
                    "current_role",
                    "current_company",
                    "professional_summary",
                ],
                "template": "{summary_text}",
                "format_style": "paragraph",
            },
            "work_experience": {
                "title": "WORK EXPERIENCE",
                "required_fields": [],
                "optional_fields": [
                    "current_role",
                    "current_company",
                    "previous_company",
                    "experience",
                    "experience_years",
                ],
                "template": "{experience_details}",
                "format_style": "structured",
            },
            "skills_expertise": {
                "title": "SKILLS & EXPERTISE",
                "required_fields": [],
                "optional_fields": ["skills", "technical_skills", "soft_skills"],
                "template": "{skills_list}",
                "format_style": "categories",
            },
            "education": {
                "title": "EDUCATION",
                "required_fields": [],
                "optional_fields": [
                    "education",
                    "degree",
                    "college",
                    "graduation_year",
                    "percentage",
                    "highest_qualification",
                ],
                "template": "{education_details}",
                "format_style": "structured",
            },
            "salary_preferences": {
                "title": "SALARY & PREFERENCES",
                "required_fields": [],
                "optional_fields": [
                    "current_salary",
                    "expected_salary",
                    "last_salary",
                    "notice_period",
                    "preferred_location",
                    "work_mode",
                ],
                "template": "{salary_details}\n{preferences}",
                "format_style": "list",
            },
            "additional_information": {
                "title": "ADDITIONAL INFORMATION",
                "required_fields": [],
                "optional_fields": [
                    "languages",
                    "certifications",
                    "projects",
                    "achievements",
                    "hobbies",
                    "references",
                ],
                "template": "{additional_details}",
                "format_style": "categories",
            },
        }

    def _initialize_field_priorities(self) -> Dict[str, int]:
        """
        Initialize priority order for fields within sections.

        Returns:
            Dictionary mapping field names to priority scores
        """
        return {
            # Personal Information (highest priority)
            "name": 100,
            "display_name": 99,
            "email": 95,
            "phone": 90,
            "alternate_phone": 85,
            "address": 80,
            "city": 75,
            "location": 70,
            "state": 65,
            "country": 60,
            "pincode": 55,
            # Professional Information
            "current_role": 95,
            "current_company": 90,
            "experience": 85,
            "experience_years": 83,
            "professional_summary": 80,
            "previous_company": 75,
            "skills": 70,
            "technical_skills": 68,
            # Education
            "education": 80,
            "highest_qualification": 78,
            "degree": 75,
            "college": 70,
            "graduation_year": 65,
            "percentage": 60,
            # Salary & Preferences
            "current_salary": 70,
            "expected_salary": 68,
            "last_salary": 65,
            "notice_period": 60,
            "preferred_location": 55,
            "work_mode": 50,
            # Additional Information
            "certifications": 60,
            "languages": 55,
            "projects": 50,
            "achievements": 45,
            "hobbies": 40,
            # Personal Details
            "age": 40,
            "date_of_birth": 35,
            "gender": 30,
            "marital_status": 25,
            "references": 20,
        }

    def _initialize_formatting_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize formatting rules for different field types.

        Returns:
            Dictionary mapping field types to formatting rules
        """
        return {
            "name": {
                "format": "title_case",
                "prefix": "",
                "suffix": "",
                "separator": "",
                "style": "bold",
            },
            "email": {
                "format": "lowercase",
                "prefix": "Email: ",
                "suffix": "",
                "separator": "",
                "style": "normal",
            },
            "phone": {
                "format": "as_is",
                "prefix": "Phone: ",
                "suffix": "",
                "separator": " | ",
                "style": "normal",
            },
            "experience": {
                "format": "smart_format",
                "prefix": "Experience: ",
                "suffix": "",
                "separator": "",
                "style": "normal",
            },
            "skills": {
                "format": "list",
                "prefix": "",
                "suffix": "",
                "separator": ", ",
                "style": "normal",
            },
            "salary": {
                "format": "currency",
                "prefix": "",
                "suffix": "",
                "separator": " | ",
                "style": "normal",
            },
            "location": {
                "format": "title_case",
                "prefix": "Location: ",
                "suffix": "",
                "separator": ", ",
                "style": "normal",
            },
            "education": {
                "format": "structured",
                "prefix": "",
                "suffix": "",
                "separator": " - ",
                "style": "normal",
            },
            "date": {
                "format": "date_format",
                "prefix": "",
                "suffix": "",
                "separator": "",
                "style": "normal",
            },
            "list_items": {
                "format": "bullet_list",
                "prefix": "• ",
                "suffix": "",
                "separator": "\n",
                "style": "normal",
            },
        }

    def _initialize_section_order(self) -> List[str]:
        """
        Initialize the default order of resume sections.

        Returns:
            List of section names in display order
        """
        return [
            "personal_information",
            "contact_information",
            "professional_summary",
            "work_experience",
            "skills_expertise",
            "education",
            "salary_preferences",
            "additional_information",
        ]

    def format_resume_text(
        self, extracted_fields: Dict[str, Any], format_style: str = "comprehensive"
    ) -> str:
        """
        Format extracted fields into a well-structured resume text.

        Args:
            extracted_fields: Dictionary of extracted and normalized fields
            format_style: Formatting style ('comprehensive', 'compact', 'minimal')

        Returns:
            Formatted resume text
        """
        logger.info(
            f"Formatting resume text with {len(extracted_fields)} fields using {format_style} style"
        )

        # Organize fields into sections
        section_data = self._organize_fields_into_sections(extracted_fields)

        # Generate formatted sections
        formatted_sections = []
        for section_name in self.section_order:
            if section_name in section_data and section_data[section_name]:
                section_text = self._format_section(
                    section_name=section_name,
                    section_fields=section_data[section_name],
                    format_style=format_style,
                )

                if section_text and section_text.strip():
                    formatted_sections.append(section_text)

        # Combine sections based on format style
        if format_style == "comprehensive":
            resume_text = "\n\n".join(formatted_sections)
        elif format_style == "compact":
            resume_text = "\n".join(formatted_sections)
        else:  # minimal
            # Only include most important sections
            important_sections = (
                formatted_sections[:4]
                if len(formatted_sections) > 4
                else formatted_sections
            )
            resume_text = "\n".join(important_sections)

        # Post-process the text
        final_text = self._post_process_text(resume_text, format_style)

        logger.info(
            f"Resume text formatted: {len(final_text)} characters across {len(formatted_sections)} sections"
        )
        return final_text

    def _organize_fields_into_sections(
        self, extracted_fields: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Organize extracted fields into appropriate resume sections.

        Args:
            extracted_fields: Dictionary of extracted fields

        Returns:
            Dictionary mapping section names to their fields
        """
        section_data = defaultdict(dict)

        # Map each field to its appropriate section(s)
        for field_name, field_value in extracted_fields.items():
            if field_value is None or (
                isinstance(field_value, str) and not field_value.strip()
            ):
                continue

            # Find which sections this field belongs to
            target_sections = self._find_field_sections(field_name)

            for section_name in target_sections:
                section_data[section_name][field_name] = field_value

        return dict(section_data)

    def _find_field_sections(self, field_name: str) -> List[str]:
        """
        Find which sections a field belongs to.

        Args:
            field_name: Name of the field

        Returns:
            List of section names that should include this field
        """
        applicable_sections = []

        for section_name, section_config in self.section_templates.items():
            required_fields = section_config.get("required_fields", [])
            optional_fields = section_config.get("optional_fields", [])

            if field_name in required_fields or field_name in optional_fields:
                applicable_sections.append(section_name)

        # If no specific section found, categorize based on field type
        if not applicable_sections:
            if field_name in ["name", "first_name", "last_name", "display_name"]:
                applicable_sections.append("personal_information")
            elif "contact" in field_name or field_name in ["email", "phone", "address"]:
                applicable_sections.append("contact_information")
            elif "skill" in field_name or field_name in [
                "technical_skills",
                "soft_skills",
            ]:
                applicable_sections.append("skills_expertise")
            elif "education" in field_name or field_name in [
                "degree",
                "college",
                "graduation_year",
            ]:
                applicable_sections.append("education")
            elif "salary" in field_name or "experience" in field_name:
                applicable_sections.append("work_experience")
            else:
                applicable_sections.append("additional_information")

        return applicable_sections

    def _format_section(
        self, section_name: str, section_fields: Dict[str, Any], format_style: str
    ) -> str:
        """
        Format a single resume section.

        Args:
            section_name: Name of the section
            section_fields: Fields belonging to this section
            format_style: Formatting style

        Returns:
            Formatted section text
        """
        if not section_fields:
            return ""

        section_config = self.section_templates.get(section_name, {})
        section_title = section_config.get(
            "title", section_name.upper().replace("_", " ")
        )
        section_format_style = section_config.get("format_style", "list")

        # Format section content based on section type
        if section_name == "personal_information":
            content = self._format_personal_information(section_fields, format_style)
        elif section_name == "contact_information":
            content = self._format_contact_information(section_fields, format_style)
        elif section_name == "professional_summary":
            content = self._format_professional_summary(section_fields, format_style)
        elif section_name == "work_experience":
            content = self._format_work_experience(section_fields, format_style)
        elif section_name == "skills_expertise":
            content = self._format_skills_expertise(section_fields, format_style)
        elif section_name == "education":
            content = self._format_education(section_fields, format_style)
        elif section_name == "salary_preferences":
            content = self._format_salary_preferences(section_fields, format_style)
        else:  # additional_information
            content = self._format_additional_information(section_fields, format_style)

        if not content or not content.strip():
            return ""

        # Combine title and content
        if format_style == "comprehensive":
            return f"{section_title}:\n{content}"
        elif format_style == "compact":
            return f"{section_title}: {content}"
        else:  # minimal
            return content

    def _format_personal_information(
        self, fields: Dict[str, Any], format_style: str
    ) -> str:
        """Format personal information section."""
        lines = []

        # Name (always first if present)
        name = fields.get("name") or fields.get("display_name")
        if name:
            lines.append(self._format_field_value(name, "name"))

        # Contact info in one line
        contact_parts = []
        if "email" in fields:
            email_formatted = self._format_field_value(fields["email"], "email")
            contact_parts.append(email_formatted)

        if "phone" in fields:
            phone_formatted = self._format_field_value(fields["phone"], "phone")
            contact_parts.append(phone_formatted)

        if contact_parts:
            lines.append(" | ".join(contact_parts))

        # Location info
        location_parts = []
        for field in ["city", "state", "country"]:
            if field in fields and fields[field]:
                location_parts.append(str(fields[field]))

        if location_parts:
            location_text = ", ".join(location_parts)
            lines.append(f"Location: {location_text}")

        # Personal details (age, gender, etc.)
        personal_details = []
        for field in ["age", "gender", "marital_status"]:
            if field in fields and fields[field]:
                field_label = field.replace("_", " ").title()
                personal_details.append(f"{field_label}: {fields[field]}")

        if personal_details and format_style == "comprehensive":
            lines.append(" | ".join(personal_details))

        return "\n".join(lines)

    def _format_contact_information(
        self, fields: Dict[str, Any], format_style: str
    ) -> str:
        """Format contact information section."""
        contact_lines = []

        # Prioritized contact fields
        priority_fields = [
            "email",
            "phone",
            "alternate_phone",
            "address",
            "city",
            "state",
            "country",
            "pincode",
        ]

        for field in priority_fields:
            if field in fields and fields[field]:
                formatted_value = self._format_field_value(fields[field], field)

                if field == "email":
                    contact_lines.append(f"Email: {formatted_value}")
                elif field in ["phone", "alternate_phone"]:
                    label = "Phone" if field == "phone" else "Alternate Phone"
                    contact_lines.append(f"{label}: {formatted_value}")
                elif field == "address":
                    contact_lines.append(f"Address: {formatted_value}")
                elif field in ["city", "state", "country", "pincode"]:
                    # These are usually combined in personal_information
                    continue

        return "\n".join(contact_lines)

    def _format_professional_summary(
        self, fields: Dict[str, Any], format_style: str
    ) -> str:
        """Format professional summary section."""
        summary_parts = []

        # Build professional summary
        if "professional_summary" in fields:
            summary_parts.append(fields["professional_summary"])
        else:
            # Generate summary from available fields
            summary_components = []

            if "current_role" in fields:
                summary_components.append(
                    f"Currently working as {fields['current_role']}"
                )

            if "current_company" in fields:
                if summary_components:
                    summary_components[-1] += f" at {fields['current_company']}"
                else:
                    summary_components.append(
                        f"Associated with {fields['current_company']}"
                    )

            if "experience" in fields:
                exp_data = fields["experience"]
                if isinstance(exp_data, dict) and "formatted" in exp_data:
                    exp_text = exp_data["formatted"]
                elif isinstance(exp_data, str):
                    exp_text = exp_data
                else:
                    exp_text = str(exp_data)

                summary_components.append(f"with {exp_text} of professional experience")
            elif "experience_years" in fields:
                summary_components.append(
                    f"with {fields['experience_years']} years of professional experience"
                )

            if summary_components:
                summary_parts.append(". ".join(summary_components) + ".")

        return "\n".join(summary_parts)

    def _format_work_experience(self, fields: Dict[str, Any], format_style: str) -> str:
        """Format work experience section."""
        experience_lines = []

        # Current position
        if "current_role" in fields or "current_company" in fields:
            current_parts = []

            if "current_role" in fields:
                current_parts.append(f"Current Role: {fields['current_role']}")

            if "current_company" in fields:
                current_parts.append(f"Company: {fields['current_company']}")

            if "experience" in fields:
                exp_data = fields["experience"]
                if isinstance(exp_data, dict) and "formatted" in exp_data:
                    current_parts.append(f"Total Experience: {exp_data['formatted']}")
            elif "experience_years" in fields:
                current_parts.append(
                    f"Total Experience: {fields['experience_years']} years"
                )

            if current_parts:
                if format_style == "comprehensive":
                    experience_lines.extend(current_parts)
                else:
                    experience_lines.append(" | ".join(current_parts))

        # Previous companies
        if "previous_company" in fields:
            experience_lines.append(f"Previous Company: {fields['previous_company']}")

        return "\n".join(experience_lines)

    def _format_skills_expertise(
        self, fields: Dict[str, Any], format_style: str
    ) -> str:
        """Format skills and expertise section."""
        skills_sections = []

        # Technical Skills
        if "technical_skills" in fields:
            tech_skills = self._format_skills_list(fields["technical_skills"])
            if tech_skills:
                if format_style == "comprehensive":
                    skills_sections.append(f"Technical Skills: {tech_skills}")
                else:
                    skills_sections.append(tech_skills)

        # General Skills
        if "skills" in fields and "technical_skills" not in fields:
            skills = self._format_skills_list(fields["skills"])
            if skills:
                if format_style == "comprehensive":
                    skills_sections.append(f"Skills: {skills}")
                else:
                    skills_sections.append(skills)

        # Soft Skills
        if "soft_skills" in fields:
            soft_skills = self._format_skills_list(fields["soft_skills"])
            if soft_skills:
                if format_style == "comprehensive":
                    skills_sections.append(f"Soft Skills: {soft_skills}")
                else:
                    skills_sections.append(soft_skills)

        return "\n".join(skills_sections)

    def _format_skills_list(self, skills_data: Any) -> str:
        """Format skills data into a readable string."""
        if isinstance(skills_data, list):
            if skills_data:
                return ", ".join(str(skill) for skill in skills_data if skill)
        elif isinstance(skills_data, str):
            return skills_data
        elif skills_data:
            return str(skills_data)

        return ""

    def _format_education(self, fields: Dict[str, Any], format_style: str) -> str:
        """Format education section."""
        education_lines = []

        # Highest qualification
        if "highest_qualification" in fields:
            education_lines.append(
                f"Highest Qualification: {fields['highest_qualification']}"
            )
        elif "education" in fields:
            education_lines.append(f"Education: {fields['education']}")
        elif "degree" in fields:
            education_lines.append(f"Degree: {fields['degree']}")

        # College/University
        if "college" in fields:
            education_lines.append(f"College/University: {fields['college']}")

        # Year and percentage
        year_percentage = []
        if "graduation_year" in fields:
            year_percentage.append(f"Year: {fields['graduation_year']}")

        if "percentage" in fields:
            percentage_data = fields["percentage"]
            if isinstance(percentage_data, dict) and "formatted" in percentage_data:
                year_percentage.append(f"Grade: {percentage_data['formatted']}")
            else:
                year_percentage.append(f"Grade: {percentage_data}")

        if year_percentage:
            education_lines.append(" | ".join(year_percentage))

        return "\n".join(education_lines)

    def _format_salary_preferences(
        self, fields: Dict[str, Any], format_style: str
    ) -> str:
        """Format salary and preferences section."""
        salary_lines = []

        # Salary information
        salary_info = []
        for field in ["current_salary", "expected_salary", "last_salary"]:
            if field in fields:
                salary_data = fields[field]
                label = field.replace("_", " ").title()

                if isinstance(salary_data, dict) and "formatted" in salary_data:
                    salary_info.append(f"{label}: {salary_data['formatted']}")
                else:
                    salary_info.append(f"{label}: {salary_data}")

        if salary_info:
            if format_style == "comprehensive":
                salary_lines.extend(salary_info)
            else:
                salary_lines.append(" | ".join(salary_info))

        # Preferences
        preferences = []
        if "notice_period" in fields:
            notice_data = fields["notice_period"]
            if isinstance(notice_data, dict) and "formatted" in notice_data:
                preferences.append(f"Notice Period: {notice_data['formatted']}")
            else:
                preferences.append(f"Notice Period: {notice_data}")

        if "preferred_location" in fields:
            preferences.append(f"Preferred Location: {fields['preferred_location']}")

        if "work_mode" in fields:
            preferences.append(f"Work Mode: {fields['work_mode']}")

        if preferences:
            if format_style == "comprehensive":
                salary_lines.extend(preferences)
            else:
                salary_lines.append(" | ".join(preferences))

        return "\n".join(salary_lines)

    def _format_additional_information(
        self, fields: Dict[str, Any], format_style: str
    ) -> str:
        """Format additional information section."""
        additional_lines = []

        # Process additional fields
        additional_fields = [
            "languages",
            "certifications",
            "projects",
            "achievements",
            "hobbies",
            "references",
        ]

        for field in additional_fields:
            if field in fields and fields[field]:
                field_data = fields[field]
                field_label = field.replace("_", " ").title()

                if isinstance(field_data, list):
                    field_text = ", ".join(str(item) for item in field_data if item)
                else:
                    field_text = str(field_data)

                if field_text:
                    additional_lines.append(f"{field_label}: {field_text}")

        return "\n".join(additional_lines)

    def _format_field_value(self, value: Any, field_type: str) -> str:
        """
        Format a field value according to its type and formatting rules.

        Args:
            value: Field value to format
            field_type: Type of field for formatting rules

        Returns:
            Formatted field value
        """
        if value is None:
            return ""

        # Get formatting rules
        formatting_rule = self.formatting_rules.get(
            field_type, self.formatting_rules.get("text", {})
        )
        format_type = formatting_rule.get("format", "as_is")

        # Apply formatting based on type
        if format_type == "title_case":
            return str(value).title()
        elif format_type == "lowercase":
            return str(value).lower()
        elif format_type == "uppercase":
            return str(value).upper()
        elif format_type == "list" and isinstance(value, list):
            separator = formatting_rule.get("separator", ", ")
            return separator.join(str(item) for item in value if item)
        elif format_type == "currency" and isinstance(value, dict):
            if "formatted" in value:
                return value["formatted"]
            elif "amount" in value and "currency" in value:
                return f"{value['currency']} {value['amount']:,.2f}"
        elif format_type == "smart_format" and isinstance(value, dict):
            if "formatted" in value:
                return value["formatted"]
        elif format_type == "date_format" and isinstance(value, dict):
            if "formatted" in value:
                return value["formatted"]

        # Default formatting
        return str(value).strip()

    def _post_process_text(self, text: str, format_style: str) -> str:
        """
        Post-process the formatted text for final cleanup.

        Args:
            text: Raw formatted text
            format_style: Formatting style used

        Returns:
            Post-processed text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)  # Reduce multiple blank lines
        text = re.sub(r"[ \t]+", " ", text)  # Normalize spaces
        text = re.sub(r"\n[ \t]+", "\n", text)  # Remove leading spaces on lines

        # Ensure proper line endings
        text = text.strip()

        # Add professional footer if comprehensive format
        if format_style == "comprehensive" and text:
            text += "\n\n---\nProfile generated from Excel data"

        return text

    def format_resume_json(self, extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format extracted fields into a structured JSON format.

        Args:
            extracted_fields: Dictionary of extracted fields

        Returns:
            Structured resume data in JSON format
        """
        # Organize fields into sections
        section_data = self._organize_fields_into_sections(extracted_fields)

        # Create structured JSON
        resume_json = {
            "personal_information": self._extract_personal_json(
                section_data.get("personal_information", {})
            ),
            "contact_information": self._extract_contact_json(
                section_data.get("contact_information", {})
            ),
            "professional_summary": self._extract_professional_json(
                section_data.get("professional_summary", {})
            ),
            "work_experience": self._extract_work_json(
                section_data.get("work_experience", {})
            ),
            "skills_expertise": self._extract_skills_json(
                section_data.get("skills_expertise", {})
            ),
            "education": self._extract_education_json(
                section_data.get("education", {})
            ),
            "salary_preferences": self._extract_salary_json(
                section_data.get("salary_preferences", {})
            ),
            "additional_information": self._extract_additional_json(
                section_data.get("additional_information", {})
            ),
            "metadata": {
                "total_fields": len(extracted_fields),
                "completeness_score": extracted_fields.get("profile_completeness", 0),
                "generated_at": datetime.now().isoformat(),
                "source": "excel_parser",
            },
        }

        return resume_json

    def _extract_personal_json(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Extract personal information for JSON format."""
        return {
            "name": fields.get("name"),
            "display_name": fields.get("display_name"),
            "first_name": fields.get("first_name"),
            "last_name": fields.get("last_name"),
            "age": fields.get("age"),
            "date_of_birth": fields.get("date_of_birth"),
            "gender": fields.get("gender"),
            "marital_status": fields.get("marital_status"),
        }

    def _extract_contact_json(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contact information for JSON format."""
        return {
            "email": fields.get("email"),
            "phone": fields.get("phone"),
            "alternate_phone": fields.get("alternate_phone"),
            "address": fields.get("address"),
            "city": fields.get("city"),
            "state": fields.get("state"),
            "country": fields.get("country"),
            "pincode": fields.get("pincode"),
            "location": fields.get("location"),
        }

    def _extract_professional_json(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Extract professional information for JSON format."""
        return {
            "summary": fields.get("professional_summary"),
            "current_role": fields.get("current_role"),
            "current_company": fields.get("current_company"),
            "experience": fields.get("experience"),
            "experience_years": fields.get("experience_years"),
        }

    def _extract_work_json(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Extract work experience for JSON format."""
        return {
            "current_role": fields.get("current_role"),
            "current_company": fields.get("current_company"),
            "previous_company": fields.get("previous_company"),
            "total_experience": fields.get("experience"),
            "experience_years": fields.get("experience_years"),
        }

    def _extract_skills_json(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Extract skills information for JSON format."""
        return {
            "technical_skills": fields.get("technical_skills"),
            "skills": fields.get("skills"),
            "soft_skills": fields.get("soft_skills"),
            "skills_count": fields.get("skills_count", 0),
        }

    def _extract_education_json(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Extract education information for JSON format."""
        return {
            "highest_qualification": fields.get("highest_qualification"),
            "education": fields.get("education"),
            "degree": fields.get("degree"),
            "college": fields.get("college"),
            "graduation_year": fields.get("graduation_year"),
            "percentage": fields.get("percentage"),
        }

    def _extract_salary_json(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Extract salary and preferences for JSON format."""
        return {
            "current_salary": fields.get("current_salary"),
            "expected_salary": fields.get("expected_salary"),
            "last_salary": fields.get("last_salary"),
            "notice_period": fields.get("notice_period"),
            "preferred_location": fields.get("preferred_location"),
            "work_mode": fields.get("work_mode"),
        }

    def _extract_additional_json(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Extract additional information for JSON format."""
        return {
            "languages": fields.get("languages"),
            "certifications": fields.get("certifications"),
            "projects": fields.get("projects"),
            "achievements": fields.get("achievements"),
            "hobbies": fields.get("hobbies"),
            "references": fields.get("references"),
        }
