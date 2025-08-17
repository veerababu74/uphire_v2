"""
Excel Resume Parser

This module extends the existing resume parser to handle Excel-based resume data.
It integrates with the existing resume parsing pipeline and duplicate detection.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import the existing resume parser and related modules
from multipleresumepraser.main import ResumeParser, Resume
from mangodatabase.operations import ResumeOperations
from mangodatabase.duplicate_detection import DuplicateDetectionOperations
from embeddings.vectorizer import AddUserDataVectorizer
from core.custom_logger import CustomLogger
from core.llm_config import LLMConfigManager, LLMProvider
from core.config import AppConfig

# Import our Excel processor
from .excel_processor import ExcelProcessor

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("excel_resume_parser")


class ExcelResumeParser:
    """
    Excel-based resume parser that processes Excel files containing resume data.
    """

    def __init__(self, llm_provider: str = None, api_keys: List[str] = None):
        """
        Initialize Excel Resume Parser.

        Args:
            llm_provider: LLM provider to use for parsing (auto-detected if None)
            api_keys: API keys for the LLM provider
        """
        logger.info("Initializing Excel Resume Parser")

        # Auto-detect LLM provider if not provided
        if llm_provider is None:
            llm_provider = AppConfig.LLM_PROVIDER
            logger.info(f"Auto-detected LLM provider from config: {llm_provider}")

        # Initialize Excel processor
        self.excel_processor = ExcelProcessor()

        # Initialize resume parser with the same configuration
        self.resume_parser = ResumeParser(llm_provider=llm_provider, api_keys=api_keys)

        # Initialize database operations
        self.vectorizer = AddUserDataVectorizer()

        # LLM manager for configuration
        self.llm_manager = LLMConfigManager()

        logger.info("Excel Resume Parser initialized successfully")

    def format_excel_row_as_resume_text(self, row_data: Dict[str, Any]) -> str:
        """
        Convert a dictionary row from Excel into a formatted text string for resume parsing.

        Args:
            row_data: Dictionary containing resume data from Excel row

        Returns:
            Formatted text string representing the resume
        """
        try:
            resume_text_parts = []

            # Helper function to safely get values
            def get_value(key_variations: List[str]) -> str:
                for key in key_variations:
                    value = row_data.get(key)
                    if value is not None and str(value).strip():
                        return str(value).strip()
                return ""

            # Personal Information
            name = get_value(["name", "full_name", "candidate_name", "employee_name"])
            if name:
                resume_text_parts.append(f"Name: {name}")

            email = get_value(["email", "email_address", "email_id"])
            if email:
                resume_text_parts.append(f"Email: {email}")

            phone = get_value(["phone", "phone_number", "mobile", "contact_number"])
            if phone:
                resume_text_parts.append(f"Phone: {phone}")

            # Location Information
            city = get_value(["city", "current_city", "location", "address"])
            if city:
                resume_text_parts.append(f"Current Location: {city}")

            # Professional Information
            experience = get_value(
                [
                    "experience",
                    "total_experience",
                    "years_of_experience",
                    "work_experience",
                ]
            )
            if experience:
                resume_text_parts.append(f"Total Experience: {experience}")

            current_role = get_value(
                ["current_role", "designation", "position", "job_title"]
            )
            if current_role:
                resume_text_parts.append(f"Current Role: {current_role}")

            current_company = get_value(
                ["current_company", "company", "organization", "employer"]
            )
            if current_company:
                resume_text_parts.append(f"Current Company: {current_company}")

            # Skills
            skills = get_value(
                ["skills", "technical_skills", "key_skills", "expertise"]
            )
            if skills:
                resume_text_parts.append(f"Skills: {skills}")

            # Education
            education = get_value(
                ["education", "qualification", "degree", "academic_qualification"]
            )
            if education:
                resume_text_parts.append(f"Education: {education}")

            college = get_value(["college", "university", "institute", "school"])
            if college:
                resume_text_parts.append(f"College/University: {college}")

            # Salary Information
            current_salary = get_value(
                ["current_salary", "salary", "ctc", "current_ctc"]
            )
            if current_salary:
                resume_text_parts.append(f"Current Salary: {current_salary}")

            expected_salary = get_value(
                ["expected_salary", "expected_ctc", "salary_expectation"]
            )
            if expected_salary:
                resume_text_parts.append(f"Expected Salary: {expected_salary}")

            # Notice Period
            notice_period = get_value(["notice_period", "notice", "availability"])
            if notice_period:
                resume_text_parts.append(f"Notice Period: {notice_period}")

            # Additional Information
            for key, value in row_data.items():
                if value is not None and str(value).strip():
                    # Skip already processed fields
                    processed_keys = [
                        "name",
                        "full_name",
                        "candidate_name",
                        "employee_name",
                        "email",
                        "email_address",
                        "email_id",
                        "phone",
                        "phone_number",
                        "mobile",
                        "contact_number",
                        "city",
                        "current_city",
                        "location",
                        "address",
                        "experience",
                        "total_experience",
                        "years_of_experience",
                        "work_experience",
                        "current_role",
                        "designation",
                        "position",
                        "job_title",
                        "current_company",
                        "company",
                        "organization",
                        "employer",
                        "skills",
                        "technical_skills",
                        "key_skills",
                        "expertise",
                        "education",
                        "qualification",
                        "degree",
                        "academic_qualification",
                        "college",
                        "university",
                        "institute",
                        "school",
                        "current_salary",
                        "salary",
                        "ctc",
                        "current_ctc",
                        "expected_salary",
                        "expected_ctc",
                        "salary_expectation",
                        "notice_period",
                        "notice",
                        "availability",
                    ]

                    if key.lower() not in processed_keys:
                        resume_text_parts.append(
                            f"{key.replace('_', ' ').title()}: {str(value).strip()}"
                        )

            # Join all parts
            formatted_text = "\n".join(resume_text_parts)

            logger.debug(
                f"Formatted resume text for {name}: {len(formatted_text)} characters"
            )
            return formatted_text

        except Exception as e:
            logger.error(f"Error formatting Excel row as resume text: {e}")
            return str(row_data)  # Fallback to string representation

    def parse_excel_row_to_resume(
        self, row_data: Dict[str, Any], user_id: str, username: str
    ) -> Optional[Resume]:
        """
        Parse a single Excel row to Resume object.

        Args:
            row_data: Dictionary containing resume data from Excel
            user_id: User ID for the resume
            username: Username for the resume

        Returns:
            Parsed Resume object or None if parsing fails
        """
        try:
            # Format the row data as resume text
            resume_text = self.format_excel_row_as_resume_text(row_data)

            # Use the existing resume parser to parse the formatted text
            parsed_resume = self.resume_parser.parse_resume(
                resume_text=resume_text, user_id=user_id, username=username
            )

            if parsed_resume:
                logger.info(f"Successfully parsed resume for {username}")
                return parsed_resume
            else:
                logger.warning(f"Failed to parse resume for {username}")
                return None

        except Exception as e:
            logger.error(f"Error parsing Excel row to resume for {username}: {e}")
            return None

    def process_excel_data(
        self, excel_data: List[Dict[str, Any]], base_user_id: str, base_username: str
    ) -> Dict[str, Any]:
        """
        Process multiple Excel rows and convert them to resumes.

        Args:
            excel_data: List of dictionaries from Excel processing
            base_user_id: Base user ID (will be suffixed with index)
            base_username: Base username (will be suffixed with index)

        Returns:
            Dictionary containing processing results
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
            }

            start_time = time.time()

            for index, row_data in enumerate(excel_data):
                try:
                    # Generate unique user_id and username for each row
                    row_user_id = f"{base_user_id}_{index + 1}"
                    row_username = f"{base_username}_{index + 1}"

                    # Parse the row
                    parsed_resume = self.parse_excel_row_to_resume(
                        row_data=row_data, user_id=row_user_id, username=row_username
                    )

                    if parsed_resume:
                        results["parsed_resumes"].append(
                            {
                                "index": index + 1,
                                "user_id": row_user_id,
                                "username": row_username,
                                "resume": parsed_resume.dict(),
                                "original_data": row_data,
                            }
                        )
                        results["successful_parses"] += 1
                    else:
                        results["failed_parses"] += 1
                        results["errors"].append(
                            {
                                "index": index + 1,
                                "user_id": row_user_id,
                                "error": "Resume parsing returned None",
                            }
                        )

                except Exception as e:
                    results["failed_parses"] += 1
                    results["errors"].append(
                        {
                            "index": index + 1,
                            "user_id": f"{base_user_id}_{index + 1}",
                            "error": str(e),
                        }
                    )
                    logger.error(f"Error processing row {index + 1}: {e}")

            results["processing_time"] = time.time() - start_time

            logger.info(
                f"Excel processing completed: {results['successful_parses']}/{results['total_rows']} successful"
            )
            return results

        except Exception as e:
            logger.error(f"Error processing Excel data: {e}")
            raise

    def save_parsed_resumes(
        self,
        parsed_results: Dict[str, Any],
        collection,
        duplicate_ops: DuplicateDetectionOperations,
    ) -> Dict[str, Any]:
        """
        Save parsed resumes to database with duplicate detection.

        Args:
            parsed_results: Results from process_excel_data
            collection: MongoDB collection for resumes
            duplicate_ops: Duplicate detection operations

        Returns:
            Dictionary containing save results
        """
        try:
            logger.info(
                f"Saving {len(parsed_results['parsed_resumes'])} parsed resumes"
            )

            save_results = {
                "total_resumes": len(parsed_results["parsed_resumes"]),
                "saved_successfully": 0,
                "duplicates_found": 0,
                "save_errors": 0,
                "duplicate_details": [],
                "save_errors_details": [],
            }

            resume_ops = ResumeOperations(collection, self.vectorizer)

            for resume_data in parsed_results["parsed_resumes"]:
                try:
                    resume = resume_data["resume"]
                    user_id = resume_data["user_id"]

                    # Check for duplicates
                    is_duplicate, duplicate_info = duplicate_ops.check_duplicate(
                        resume_text=self.format_excel_row_as_resume_text(
                            resume_data["original_data"]
                        ),
                        user_id=user_id,
                    )

                    if is_duplicate:
                        save_results["duplicates_found"] += 1
                        save_results["duplicate_details"].append(
                            {"user_id": user_id, "duplicate_info": duplicate_info}
                        )
                        logger.warning(f"Duplicate detected for user {user_id}")
                        continue

                    # Save the resume
                    save_response = resume_ops.add_user_data(
                        user_data=resume, user_id=user_id
                    )

                    if save_response.get("status") == "success":
                        save_results["saved_successfully"] += 1
                        logger.info(f"Successfully saved resume for user {user_id}")
                    else:
                        save_results["save_errors"] += 1
                        save_results["save_errors_details"].append(
                            {
                                "user_id": user_id,
                                "error": save_response.get("message", "Unknown error"),
                            }
                        )

                except Exception as e:
                    save_results["save_errors"] += 1
                    save_results["save_errors_details"].append(
                        {
                            "user_id": resume_data.get("user_id", "unknown"),
                            "error": str(e),
                        }
                    )
                    logger.error(
                        f"Error saving resume for user {resume_data.get('user_id')}: {e}"
                    )

            logger.info(
                f"Save completed: {save_results['saved_successfully']} saved, "
                f"{save_results['duplicates_found']} duplicates, "
                f"{save_results['save_errors']} errors"
            )

            return save_results

        except Exception as e:
            logger.error(f"Error saving parsed resumes: {e}")
            raise
