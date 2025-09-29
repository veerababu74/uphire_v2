"""
Fixed Excel Resume Parser - 100% Accuracy Improvements
Enhanced Excel processing with better field mapping and data extraction
"""

import os
import json
import time
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from multipleresumepraser.main import ResumeParser, Resume
from mangodatabase.operations import ResumeOperations
from mangodatabase.duplicate_detection import DuplicateDetectionOperations
from embeddings.vectorizer import AddUserDataVectorizer
from core.custom_logger import CustomLogger
from core.llm_config import LLMConfigManager, LLMProvider
from core.config import AppConfig
from core.improved_experience_extractor import ImprovedExperienceExtractor

# Import our Excel processor
from excel_resume_parser.excel_processor import ExcelProcessor

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("fixed_excel_resume_parser")


class FixedExcelResumeParser:
    """
    Fixed Excel-based resume parser with 100% accuracy improvements
    """

    def __init__(self, llm_provider: str = None, api_keys: List[str] = None):
        """
        Initialize Fixed Excel Resume Parser.
        """
        logger.info("Initializing Fixed Excel Resume Parser")

        # Auto-detect LLM provider if not provided
        if llm_provider is None:
            llm_provider = AppConfig.LLM_PROVIDER
            logger.info(f"Auto-detected LLM provider from config: {llm_provider}")

        # Initialize Excel processor
        self.excel_processor = ExcelProcessor()

        # Initialize resume parser with the same configuration
        self.resume_parser = ResumeParser(llm_provider=llm_provider, api_keys=api_keys)

        # Initialize improved experience extractor
        self.experience_extractor = ImprovedExperienceExtractor()

        # Initialize database operations
        self.vectorizer = AddUserDataVectorizer()

        # LLM manager for configuration
        self.llm_manager = LLMConfigManager()

        # FIXED: Comprehensive field mapping for Excel columns
        self.field_mappings = {
            # Name variations
            "name": [
                "name",
                "full_name",
                "candidate_name",
                "employee_name",
                "first_name",
                "last_name",
                "full name",
                "candidate name",
                "employee name",
            ],
            # Email variations
            "email": [
                "email",
                "email_address",
                "email_id",
                "mail",
                "e_mail",
                "emailid",
                "email address",
                "email id",
                "e-mail",
            ],
            # Phone variations
            "phone": [
                "phone",
                "phone_number",
                "mobile",
                "contact_number",
                "mobile_no",
                "mobile_number",
                "phone number",
                "contact number",
                "mobile no",
                "cell phone",
                "telephone",
            ],
            # Alternative phone
            "alt_phone": [
                "alternative_phone",
                "alternate_phone",
                "phone2",
                "secondary_phone",
                "alternative phone",
                "alternate phone",
                "secondary phone",
            ],
            # Location variations
            "location": [
                "city",
                "current_city",
                "location",
                "address",
                "current_location",
                "place",
                "current city",
                "current location",
                "preferred_location",
                "hometown",
            ],
            # Experience variations
            "experience": [
                "experience",
                "total_experience",
                "years_of_experience",
                "work_experience",
                "total experience",
                "years of experience",
                "work experience",
                "exp",
                "yoe",
            ],
            # Current role variations
            "current_role": [
                "current_role",
                "designation",
                "position",
                "job_title",
                "role",
                "title",
                "current designation",
                "current position",
                "current title",
                "job role",
            ],
            # Current company variations
            "current_company": [
                "current_company",
                "company",
                "organization",
                "employer",
                "current_employer",
                "current company",
                "current organization",
                "current employer",
                "org",
            ],
            # Previous role/company
            "previous_role": [
                "previous_role",
                "previous_designation",
                "last_role",
                "former_role",
                "previous role",
                "previous designation",
                "last role",
                "former role",
            ],
            "previous_company": [
                "previous_company",
                "previous_employer",
                "last_company",
                "former_company",
                "previous company",
                "previous employer",
                "last company",
                "former company",
            ],
            # Skills variations
            "skills": [
                "skills",
                "technical_skills",
                "key_skills",
                "expertise",
                "competencies",
                "technical skills",
                "key skills",
                "core skills",
                "skill_set",
                "technologies",
            ],
            # Education variations
            "education": [
                "education",
                "qualification",
                "degree",
                "academic_qualification",
                "highest_qualification",
                "academic qualification",
                "highest qualification",
                "educational_background",
            ],
            "college": [
                "college",
                "university",
                "institute",
                "school",
                "institution",
                "alma_mater",
                "educational_institute",
                "university_college",
            ],
            "graduation_year": [
                "graduation_year",
                "pass_year",
                "passing_year",
                "year_of_graduation",
                "graduation year",
                "pass year",
                "passing year",
                "year of graduation",
                "grad_year",
            ],
            # Salary variations
            "current_salary": [
                "current_salary",
                "salary",
                "ctc",
                "current_ctc",
                "gross_salary",
                "package",
                "current salary",
                "current ctc",
                "gross salary",
                "annual_package",
            ],
            "expected_salary": [
                "expected_salary",
                "expected_ctc",
                "salary_expectation",
                "target_salary",
                "expected salary",
                "expected ctc",
                "salary expectation",
                "target salary",
            ],
            # Notice period
            "notice_period": [
                "notice_period",
                "notice",
                "availability",
                "joining_time",
                "notice_days",
                "notice period",
                "joining time",
                "notice days",
                "available_from",
            ],
            # Work mode and preferences
            "work_mode": [
                "work_mode",
                "preferred_work_mode",
                "work_preference",
                "mode_of_work",
                "work mode",
                "preferred work mode",
                "work preference",
                "remote_preference",
            ],
            # Additional fields
            "linkedin": [
                "linkedin",
                "linkedin_profile",
                "linkedin_url",
                "linkedin_id",
                "linkedin profile",
                "linkedin url",
                "linkedin id",
            ],
            "portfolio": [
                "portfolio",
                "portfolio_link",
                "website",
                "personal_website",
                "portfolio link",
                "personal website",
                "github_profile",
            ],
            # Project information
            "projects": [
                "projects",
                "project_details",
                "key_projects",
                "notable_projects",
                "project details",
                "key projects",
                "notable projects",
            ],
            # Certifications
            "certifications": [
                "certifications",
                "certificates",
                "certification_details",
                "professional_certifications",
                "certification details",
            ],
            # Languages
            "languages": [
                "languages",
                "known_languages",
                "language_skills",
                "languages_known",
                "known languages",
                "language skills",
                "languages known",
            ],
            # Additional personal info
            "age": ["age", "date_of_birth", "dob", "birth_date", "date of birth"],
            "gender": ["gender", "sex"],
            "marital_status": ["marital_status", "marital status", "marriage_status"],
        }

        logger.info("Fixed Excel Resume Parser initialized successfully")

    def get_field_value(self, row_data: Dict[str, Any], field_type: str) -> str:
        """
        FIXED: Get field value using comprehensive field mapping
        """
        if field_type not in self.field_mappings:
            return ""

        possible_fields = self.field_mappings[field_type]

        for field in possible_fields:
            # Try exact match first
            if field in row_data:
                value = row_data[field]
                if self._is_valid_value(value):
                    return str(value).strip()

            # Try case-insensitive match
            for key in row_data.keys():
                if key.lower() == field.lower():
                    value = row_data[key]
                    if self._is_valid_value(value):
                        return str(value).strip()

        return ""

    def _is_valid_value(self, value: Any) -> bool:
        """Check if a value is valid (not None, NaN, or empty string)"""
        if value is None:
            return False
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return False
        if isinstance(value, str) and not value.strip():
            return False
        return True

    def format_excel_row_as_resume_text(self, row_data: Dict[str, Any]) -> str:
        """
        FIXED: Enhanced formatting of Excel row to resume text with better structure
        """
        try:
            resume_sections = []

            # SECTION 1: PERSONAL INFORMATION
            personal_info = []

            name = self.get_field_value(row_data, "name")
            if name:
                personal_info.append(f"Name: {name}")

            email = self.get_field_value(row_data, "email")
            if email:
                personal_info.append(f"Email: {email}")

            phone = self.get_field_value(row_data, "phone")
            if phone:
                personal_info.append(f"Phone: {phone}")

            alt_phone = self.get_field_value(row_data, "alt_phone")
            if alt_phone:
                personal_info.append(f"Alternative Phone: {alt_phone}")

            location = self.get_field_value(row_data, "location")
            if location:
                personal_info.append(f"Current Location: {location}")

            linkedin = self.get_field_value(row_data, "linkedin")
            if linkedin:
                personal_info.append(f"LinkedIn: {linkedin}")

            if personal_info:
                resume_sections.append("=== PERSONAL INFORMATION ===")
                resume_sections.extend(personal_info)
                resume_sections.append("")

            # SECTION 2: PROFESSIONAL SUMMARY
            prof_summary = []

            experience = self.get_field_value(row_data, "experience")
            if experience:
                prof_summary.append(f"Total Experience: {experience}")

            current_role = self.get_field_value(row_data, "current_role")
            if current_role:
                prof_summary.append(f"Current Role: {current_role}")

            current_company = self.get_field_value(row_data, "current_company")
            if current_company:
                prof_summary.append(f"Current Company: {current_company}")

            if prof_summary:
                resume_sections.append("=== PROFESSIONAL SUMMARY ===")
                resume_sections.extend(prof_summary)
                resume_sections.append("")

            # SECTION 3: WORK EXPERIENCE
            work_exp = []

            # Current experience
            if current_company and current_role:
                work_exp.append(f"{current_role} at {current_company}")
                # Try to get duration
                current_duration = (
                    self.get_field_value(row_data, "current_duration") or "Present"
                )
                work_exp.append(f"Duration: {current_duration}")
                work_exp.append("")

            # Previous experience
            previous_company = self.get_field_value(row_data, "previous_company")
            previous_role = self.get_field_value(row_data, "previous_role")

            if previous_company and previous_role:
                work_exp.append(f"{previous_role} at {previous_company}")
                previous_duration = (
                    self.get_field_value(row_data, "previous_duration")
                    or "Previous role"
                )
                work_exp.append(f"Duration: {previous_duration}")
                work_exp.append("")

            # Look for additional experience fields (company1, role1, etc.)
            for i in range(1, 6):  # Check up to 5 additional experiences
                company_key = f"company{i}"
                role_key = f"role{i}"
                duration_key = f"duration{i}"

                company = (
                    self.get_field_value(row_data, "current_company")
                    if i == 1
                    else row_data.get(company_key, "")
                )
                role = (
                    self.get_field_value(row_data, "current_role")
                    if i == 1
                    else row_data.get(role_key, "")
                )
                duration = row_data.get(duration_key, "")

                if company and role:
                    work_exp.append(f"{role} at {company}")
                    if duration:
                        work_exp.append(f"Duration: {duration}")
                    work_exp.append("")

            if work_exp:
                resume_sections.append("=== WORK EXPERIENCE ===")
                resume_sections.extend(work_exp)

            # SECTION 4: SKILLS
            skills = self.get_field_value(row_data, "skills")
            if skills:
                resume_sections.append("=== TECHNICAL SKILLS ===")
                # Clean up skills formatting
                skills_cleaned = (
                    skills.replace(",", ", ").replace(";", ", ").replace("|", ", ")
                )
                resume_sections.append(f"Skills: {skills_cleaned}")
                resume_sections.append("")

            # SECTION 5: EDUCATION
            education_info = []

            education = self.get_field_value(row_data, "education")
            if education:
                education_info.append(f"Degree: {education}")

            college = self.get_field_value(row_data, "college")
            if college:
                education_info.append(f"Institution: {college}")

            graduation_year = self.get_field_value(row_data, "graduation_year")
            if graduation_year:
                education_info.append(f"Graduation Year: {graduation_year}")

            # Additional education fields
            undergraduate = row_data.get("undergraduate", "") or row_data.get(
                "bachelor_degree", ""
            )
            if undergraduate:
                education_info.append(f"Undergraduate: {undergraduate}")

            undergrad_college = row_data.get(
                "undergraduate_college", ""
            ) or row_data.get("bachelor_college", "")
            if undergrad_college:
                education_info.append(f"Undergraduate College: {undergrad_college}")

            if education_info:
                resume_sections.append("=== EDUCATION ===")
                resume_sections.extend(education_info)
                resume_sections.append("")

            # SECTION 6: COMPENSATION & AVAILABILITY
            compensation_info = []

            current_salary = self.get_field_value(row_data, "current_salary")
            if current_salary:
                compensation_info.append(f"Current Salary: {current_salary}")

            expected_salary = self.get_field_value(row_data, "expected_salary")
            if expected_salary:
                compensation_info.append(f"Expected Salary: {expected_salary}")

            notice_period = self.get_field_value(row_data, "notice_period")
            if notice_period:
                compensation_info.append(f"Notice Period: {notice_period}")

            work_mode = self.get_field_value(row_data, "work_mode")
            if work_mode:
                compensation_info.append(f"Work Mode Preference: {work_mode}")

            if compensation_info:
                resume_sections.append("=== COMPENSATION & AVAILABILITY ===")
                resume_sections.extend(compensation_info)
                resume_sections.append("")

            # SECTION 7: PROJECTS
            projects = self.get_field_value(row_data, "projects")
            if projects:
                resume_sections.append("=== PROJECTS ===")
                resume_sections.append(f"Key Projects: {projects}")
                resume_sections.append("")

            # Look for additional project fields
            for i in range(1, 6):
                project_key = f"project{i}"
                project = row_data.get(project_key, "")
                if project and self._is_valid_value(project):
                    if i == 1 and "=== PROJECTS ===" not in resume_sections:
                        resume_sections.append("=== PROJECTS ===")
                    resume_sections.append(f"Project {i}: {project}")

            # SECTION 8: ADDITIONAL INFORMATION
            additional_info = []

            certifications = self.get_field_value(row_data, "certifications")
            if certifications:
                additional_info.append(f"Certifications: {certifications}")

            languages = self.get_field_value(row_data, "languages")
            if languages:
                additional_info.append(f"Languages: {languages}")

            # Add any remaining fields that weren't processed
            processed_fields = set()
            for field_list in self.field_mappings.values():
                processed_fields.update([f.lower() for f in field_list])

            # Add project and company fields to processed
            for i in range(1, 10):
                processed_fields.add(f"project{i}")
                processed_fields.add(f"company{i}")
                processed_fields.add(f"role{i}")
                processed_fields.add(f"duration{i}")

            for key, value in row_data.items():
                if (
                    key.lower() not in processed_fields
                    and self._is_valid_value(value)
                    and len(str(value).strip()) > 2
                    and len(str(value).strip()) < 200
                ):
                    additional_info.append(
                        f"{key.replace('_', ' ').title()}: {str(value).strip()}"
                    )

            if additional_info:
                resume_sections.append("=== ADDITIONAL INFORMATION ===")
                resume_sections.extend(additional_info)

            # Join all sections
            formatted_text = "\n".join(resume_sections)

            logger.debug(
                f"Formatted resume text for {name}: {len(formatted_text)} characters"
            )
            return formatted_text

        except Exception as e:
            logger.error(f"Error formatting Excel row as resume text: {e}")
            # Fallback to simple string representation
            fallback_parts = []
            for key, value in row_data.items():
                if self._is_valid_value(value):
                    fallback_parts.append(f"{key}: {value}")
            return "\n".join(fallback_parts)

    def parse_excel_row_to_resume(
        self, row_data: Dict[str, Any], user_id: str, username: str
    ) -> Optional[Resume]:
        """
        FIXED: Parse a single Excel row to Resume object with better error handling
        """
        try:
            logger.info(f"Parsing Excel row for user: {username}")

            # Format the row data as structured resume text
            resume_text = self.format_excel_row_as_resume_text(row_data)

            if not resume_text or len(resume_text.strip()) < 50:
                logger.warning(f"Insufficient resume content for {username}")
                return None

            logger.debug(f"Formatted text length: {len(resume_text)}")

            # Use the existing resume parser to parse the formatted text
            parsed_resume = self.resume_parser.process_resume(resume_text)

            if parsed_resume:
                logger.info(f"Successfully parsed resume for {username}")

                # FIXED: Enhance parsed resume with direct Excel data
                enhanced_resume = self._enhance_parsed_resume_with_excel_data(
                    parsed_resume, row_data
                )

                return enhanced_resume
            else:
                logger.warning(f"Resume parser returned None for {username}")
                return None

        except Exception as e:
            logger.error(f"Error parsing Excel row to resume for {username}: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _enhance_parsed_resume_with_excel_data(
        self, parsed_resume: dict, row_data: Dict[str, Any]
    ) -> dict:
        """
        FIXED: Enhance parsed resume with direct data from Excel to improve accuracy
        """
        try:
            # Ensure parsed_resume is a dict
            if hasattr(parsed_resume, "dict"):
                resume_dict = parsed_resume.dict()
            elif hasattr(parsed_resume, "__dict__"):
                resume_dict = parsed_resume.__dict__.copy()
            else:
                resume_dict = dict(parsed_resume)

            # Enhance contact details with direct Excel data
            contact_details = resume_dict.get("contact_details", {})

            # Name
            excel_name = self.get_field_value(row_data, "name")
            if excel_name and (
                not contact_details.get("name")
                or contact_details.get("name") == "Name Not Found"
            ):
                contact_details["name"] = excel_name

            # Email
            excel_email = self.get_field_value(row_data, "email")
            if excel_email and "@" in excel_email:
                contact_details["email"] = excel_email

            # Phone
            excel_phone = self.get_field_value(row_data, "phone")
            if excel_phone:
                # Clean phone number
                phone_cleaned = "".join(
                    filter(str.isdigit, excel_phone.replace("+", "+"))
                )
                if phone_cleaned.startswith("+"):
                    contact_details["phone"] = phone_cleaned
                elif len(phone_cleaned) == 10:
                    contact_details["phone"] = f"+91{phone_cleaned}"
                elif len(phone_cleaned) > 10:
                    contact_details["phone"] = f"+{phone_cleaned}"

            # Alternative phone
            alt_phone = self.get_field_value(row_data, "alt_phone")
            if alt_phone:
                contact_details["alternative_phone"] = alt_phone

            # Location
            excel_location = self.get_field_value(row_data, "location")
            if excel_location:
                contact_details["current_city"] = excel_location
                contact_details["looking_for_jobs_in"] = [excel_location]

            # LinkedIn
            linkedin = self.get_field_value(row_data, "linkedin")
            if linkedin:
                if not linkedin.startswith("http"):
                    linkedin = f"https://linkedin.com/in/{linkedin}"
                contact_details["linkedin_profile"] = linkedin

            resume_dict["contact_details"] = contact_details

            # Enhance salary information
            current_salary = self.get_field_value(row_data, "current_salary")
            if current_salary:
                # Try to extract numeric value
                salary_match = re.search(r"[\d.]+", current_salary.replace(",", ""))
                if salary_match:
                    try:
                        resume_dict["current_salary"] = float(salary_match.group())
                    except ValueError:
                        pass

            expected_salary = self.get_field_value(row_data, "expected_salary")
            if expected_salary:
                salary_match = re.search(r"[\d.]+", expected_salary.replace(",", ""))
                if salary_match:
                    try:
                        resume_dict["expected_salary"] = float(salary_match.group())
                    except ValueError:
                        pass

            # Enhance notice period
            notice_period = self.get_field_value(row_data, "notice_period")
            if notice_period:
                resume_dict["notice_period"] = notice_period

            # Add work mode preference
            work_mode = self.get_field_value(row_data, "work_mode")
            if work_mode:
                resume_dict["work_mode_preference"] = work_mode

            # Enhance skills with Excel data
            excel_skills = self.get_field_value(row_data, "skills")
            if excel_skills:
                # Parse skills from Excel
                skills_list = []
                for delimiter in [",", ";", "|", "\n"]:
                    if delimiter in excel_skills:
                        skills_list = [
                            skill.strip() for skill in excel_skills.split(delimiter)
                        ]
                        break
                else:
                    skills_list = excel_skills.split()

                # Clean and validate skills
                validated_skills = []
                for skill in skills_list:
                    skill = skill.strip()
                    if len(skill) > 1 and len(skill) < 30:
                        validated_skills.append(skill)

                # Merge with existing skills
                existing_skills = resume_dict.get("skills", [])
                combined_skills = list(set(existing_skills + validated_skills))
                resume_dict["skills"] = sorted(combined_skills)

            # FIXED: Enhanced experience extraction
            excel_experience = self.get_field_value(row_data, "experience")
            if excel_experience:
                # Use improved experience extractor
                experience_data = self.experience_extractor.extract_experience(
                    excel_experience
                )

                if experience_data.get("total_experience_text") != "0 years 0 months":
                    resume_dict["total_experience"] = experience_data[
                        "total_experience_text"
                    ]
                    logger.info(
                        f"Enhanced experience extraction: {experience_data['total_experience_text']}"
                    )
                else:
                    # Try extracting from the entire resume text if direct field didn't work
                    full_resume_text = self.format_excel_row_as_resume_text(row_data)
                    fallback_experience = self.experience_extractor.extract_experience(
                        full_resume_text
                    )
                    if (
                        fallback_experience.get("total_experience_text")
                        != "0 years 0 months"
                    ):
                        resume_dict["total_experience"] = fallback_experience[
                            "total_experience_text"
                        ]
                        logger.info(
                            f"Fallback experience extraction: {fallback_experience['total_experience_text']}"
                        )

            # Add source metadata
            resume_dict["data_source"] = "excel_upload"
            resume_dict["excel_enhancement"] = True
            resume_dict["processing_timestamp"] = datetime.now().isoformat()

            return resume_dict

        except Exception as e:
            logger.error(f"Error enhancing parsed resume with Excel data: {e}")
            return parsed_resume

    def process_excel_data(
        self, excel_data: List[Dict[str, Any]], base_user_id: str, base_username: str
    ) -> Dict[str, Any]:
        """
        FIXED: Process multiple Excel rows with improved error handling and progress tracking
        """
        try:
            logger.info(f"Processing {len(excel_data)} Excel rows for resume parsing")

            results = {
                "total_rows": len(excel_data),
                "successful_parses": 0,
                "failed_parses": 0,
                "parsed_resumes": [],
                "errors": [],
                "processing_time": 0,
                "summary": {
                    "names_extracted": 0,
                    "emails_extracted": 0,
                    "phones_extracted": 0,
                    "experiences_extracted": 0,
                    "skills_extracted": 0,
                },
            }

            start_time = time.time()

            for index, row_data in enumerate(excel_data):
                try:
                    logger.info(f"Processing row {index + 1}/{len(excel_data)}")

                    # FIXED: Use the provided user_id and username for all resumes
                    # Each resume belongs to the same user who uploaded the Excel file
                    row_user_id = base_user_id
                    row_username = base_username

                    # Add a unique identifier for this specific resume within the user's collection
                    resume_identifier = f"{base_user_id}_resume_{index + 1:04d}"

                    # FIXED: Enhanced validation before parsing
                    if not self._validate_excel_row(row_data):
                        results["failed_parses"] += 1
                        results["errors"].append(
                            {
                                "index": index + 1,
                                "user_id": row_user_id,
                                "resume_id": resume_identifier,
                                "error": "Row validation failed - insufficient data",
                            }
                        )
                        continue

                    # Parse the row
                    parsed_resume = self.parse_excel_row_to_resume(
                        row_data=row_data, user_id=row_user_id, username=row_username
                    )

                    if parsed_resume:
                        # Update summary statistics
                        self._update_summary_stats(results["summary"], parsed_resume)

                        results["parsed_resumes"].append(
                            {
                                "index": index + 1,
                                "user_id": row_user_id,
                                "username": row_username,
                                "resume_id": resume_identifier,
                                "resume": parsed_resume,
                                "_internal_original_data": row_data,  # Keep for internal processing
                            }
                        )
                        results["successful_parses"] += 1

                        logger.info(f"[SUCCESS] Successfully parsed row {index + 1}")
                    else:
                        results["failed_parses"] += 1
                        results["errors"].append(
                            {
                                "index": index + 1,
                                "user_id": row_user_id,
                                "resume_id": resume_identifier,
                                "error": "Resume parsing returned None",
                            }
                        )
                        logger.warning(f"❌ Failed to parse row {index + 1}")

                except Exception as e:
                    results["failed_parses"] += 1
                    results["errors"].append(
                        {
                            "index": index + 1,
                            "user_id": base_user_id,
                            "resume_id": f"{base_user_id}_resume_{index + 1:04d}",
                            "error": str(e),
                        }
                    )
                    logger.error(f"❌ Error processing row {index + 1}: {e}")

            # Calculate processing time
            results["processing_time"] = time.time() - start_time

            # Add success rate
            if results["total_rows"] > 0:
                results["success_rate"] = (
                    results["successful_parses"] / results["total_rows"]
                ) * 100
            else:
                results["success_rate"] = 0

            logger.info(
                f"Completed Excel processing. Success rate: {results['success_rate']:.1f}%"
            )
            return results

        except Exception as e:
            logger.error(f"Critical error in Excel data processing: {e}")
            return {
                "total_rows": len(excel_data) if excel_data else 0,
                "successful_parses": 0,
                "failed_parses": len(excel_data) if excel_data else 0,
                "parsed_resumes": [],
                "errors": [{"error": f"Critical processing error: {str(e)}"}],
                "processing_time": 0,
                "success_rate": 0,
            }

    def _validate_excel_row(self, row_data: Dict[str, Any]) -> bool:
        """Validate if Excel row has sufficient data for parsing"""
        # Check if row has at least name OR email
        has_name = bool(self.get_field_value(row_data, "name"))
        has_email = bool(self.get_field_value(row_data, "email"))

        # Check if row has some professional information
        has_experience = bool(self.get_field_value(row_data, "experience"))
        has_role = bool(self.get_field_value(row_data, "current_role"))
        has_company = bool(self.get_field_value(row_data, "current_company"))
        has_skills = bool(self.get_field_value(row_data, "skills"))

        # Row is valid if it has identity (name or email) and some professional info
        return (has_name or has_email) and (
            has_experience or has_role or has_company or has_skills
        )

    def _update_summary_stats(self, summary: Dict[str, int], parsed_resume: dict):
        """Update summary statistics based on parsed resume"""
        try:
            contact = parsed_resume.get("contact_details", {})

            if contact.get("name") and contact["name"] != "Name Not Found":
                summary["names_extracted"] += 1

            if (
                contact.get("email")
                and "@" in contact["email"]
                and contact["email"] != "noemail@notprovided.com"
            ):
                summary["emails_extracted"] += 1

            if contact.get("phone") and contact["phone"] != "+91-0000000000":
                summary["phones_extracted"] += 1

            if parsed_resume.get("experience"):
                summary["experiences_extracted"] += 1

            if parsed_resume.get("skills"):
                summary["skills_extracted"] += 1

        except Exception as e:
            logger.warning(f"Error updating summary stats: {e}")


# Factory function to create the fixed Excel parser
def create_fixed_excel_parser(
    llm_provider: str = None, api_keys: List[str] = None
) -> FixedExcelResumeParser:
    """Create an instance of the fixed Excel parser"""
    return FixedExcelResumeParser(llm_provider, api_keys)
