"""
Adapter for Fixed Excel Parser to work with existing API interface
"""

import os
import time
from typing import List, Dict, Any, Optional
from excel_resume_parser.fixed_excel_resume_parser import FixedExcelResumeParser
from excel_resume_parser.excel_processor import ExcelProcessor
from core.custom_logger import CustomLogger

logger_manager = CustomLogger()
logger = logger_manager.get_logger("excel_parser_adapter")


class FixedExcelParserAdapter(FixedExcelResumeParser):
    """
    Adapter to make FixedExcelResumeParser compatible with existing API interface
    """

    def __init__(self, llm_provider: str = None, api_keys: List[str] = None):
        super().__init__(llm_provider, api_keys)
        # Initialize Excel processor for file handling
        self.excel_processor = ExcelProcessor()

    def process_excel_file(
        self,
        file_path: str,
        sheet_name: Optional[str] = None,
        validation_level: str = "standard",
        cleaning_aggressive: bool = True,
        include_quality_scores: bool = True,
        batch_size: int = 50,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process Excel file with the fixed parser interface

        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet to process
            validation_level: Level of validation (not used in fixed parser)
            cleaning_aggressive: Whether to use aggressive cleaning (not used)
            include_quality_scores: Whether to include quality scores (not used)
            batch_size: Batch size for processing (not used)
            user_id: User ID for the person uploading the Excel
            user_name: User name for the person uploading the Excel

        Returns:
            Processing results compatible with existing API
        """
        try:
            logger.info(f"Processing Excel file with fixed parser: {file_path}")

            # Step 1: Process Excel file to get data
            logger.info("Step 1: Reading Excel data")
            excel_data = self.excel_processor.process_excel_file(
                file_path=file_path, sheet_name=sheet_name
            )

            if not excel_data:
                return {
                    "status": "error",
                    "message": "No data found in Excel file",
                    "parsed_resumes": [],
                    "total_rows": 0,
                    "successful_parses": 0,
                    "failed_parses": 0,
                    "processing_time": 0,
                }

            logger.info(f"Found {len(excel_data)} rows in Excel file")

            # Step 2: Use provided user info or generate base user info
            if user_id and user_name:
                base_user_id = user_id
                base_username = user_name
                logger.info(f"Using provided user info: {user_name} (ID: {user_id})")
            else:
                base_user_id = f"excel_user_{int(time.time())}"
                base_username = f"excel_candidate_{int(time.time())}"
                logger.info(
                    f"Generated user info: {base_username} (ID: {base_user_id})"
                )

            # Step 3: Process with fixed parser
            logger.info("Step 2: Processing with fixed Excel parser")
            processing_result = self.process_excel_data(
                excel_data=excel_data,
                base_user_id=base_user_id,
                base_username=base_username,
            )

            # Step 4: Format result to match expected API response
            api_response = {
                "status": (
                    "success"
                    if processing_result.get("successful_parses", 0) > 0
                    else "partial"
                ),
                "message": f"Processed {processing_result.get('successful_parses', 0)} resumes successfully",
                "parsed_resumes": [],
                "total_rows": processing_result.get("total_rows", 0),
                "successful_parses": processing_result.get("successful_parses", 0),
                "failed_parses": processing_result.get("failed_parses", 0),
                "processing_time": processing_result.get("processing_time", 0),
                "errors": processing_result.get("errors", []),
                "summary": processing_result.get("summary", {}),
                "success_rate": processing_result.get("success_rate", 0),
            }

            # Convert parsed resumes to expected format
            for resume_entry in processing_result.get("parsed_resumes", []):
                # Remove internal data before sending to API response
                resume_data = resume_entry.copy()
                resume_data.pop("_internal_original_data", None)

                # Ensure the resume data has expected structure
                if "resume" in resume_data:
                    resume = resume_data["resume"]

                    # Add API-expected fields
                    resume_data["parsing_metadata"] = {
                        "parser_version": "fixed_excel_parser_v1.0",
                        "processing_time": processing_result.get("processing_time", 0),
                        "accuracy_enhanced": True,
                        "field_mapping_used": True,
                        "validation_passed": True,
                    }

                    # Ensure resume has required structure
                    if isinstance(resume, dict):
                        # Add any missing required fields
                        resume.setdefault("user_id", resume_data.get("user_id", ""))
                        resume.setdefault("username", resume_data.get("username", ""))
                        resume.setdefault("source", "excel_upload")

                        # Ensure contact_details exists
                        if "contact_details" not in resume:
                            resume["contact_details"] = {
                                "name": "Name not found",
                                "email": "noemail@notprovided.com",
                                "phone": "+91-0000000000",
                                "current_city": "City not specified",
                            }

                api_response["parsed_resumes"].append(resume_data)

            logger.info(
                f"Fixed Excel parser completed. Success rate: {api_response['success_rate']:.1f}%"
            )
            return api_response

        except Exception as e:
            logger.error(f"Error in fixed Excel parser adapter: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

            return {
                "status": "error",
                "message": f"Failed to process Excel file: {str(e)}",
                "parsed_resumes": [],
                "total_rows": 0,
                "successful_parses": 0,
                "failed_parses": 1,
                "processing_time": 0,
                "errors": [{"error": str(e)}],
            }

    def save_parsed_resumes_to_database(
        self,
        parsed_resumes: List[Dict[str, Any]],
        detect_duplicates: bool = True,
        update_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Save parsed resumes to database with duplicate detection.

        Args:
            parsed_resumes: List of parsed resume dictionaries
            detect_duplicates: Whether to perform duplicate detection
            update_existing: Whether to update existing records

        Returns:
            Dictionary containing save operation results
        """
        logger.info(f"Saving {len(parsed_resumes)} parsed resumes to database")

        save_result = {
            "total_resumes": len(parsed_resumes),
            "successfully_saved": 0,
            "duplicates_found": 0,
            "updated_records": 0,
            "errors": [],
        }

        try:
            from mangodatabase.operations import ResumeOperations
            from mangodatabase.duplicate_detection import DuplicateDetectionOperations
            from GroqcloudLLM.main import Resume
            from mangodatabase.client import get_collection
            from embeddings.vectorizer import AddUserDataVectorizer

            # Get database dependencies
            collection = get_collection()
            add_user_vectorizer = AddUserDataVectorizer()

            resume_ops = ResumeOperations(collection, add_user_vectorizer)
            duplicate_ops = (
                DuplicateDetectionOperations(collection) if detect_duplicates else None
            )

            for resume_data in parsed_resumes:
                try:
                    # Extract resume dict from the parsed data structure
                    if isinstance(resume_data, dict) and "resume" in resume_data:
                        resume_dict = resume_data["resume"]
                    else:
                        resume_dict = resume_data

                    # Convert to Resume object if needed
                    if isinstance(resume_dict, dict):
                        resume = Resume(**resume_dict)
                    else:
                        resume = resume_dict

                    # Check for duplicates if enabled
                    is_duplicate = False
                    if detect_duplicates and duplicate_ops:
                        try:
                            duplicate_result = duplicate_ops.check_duplicate(resume)
                            if duplicate_result.get("is_duplicate", False):
                                is_duplicate = True
                                save_result["duplicates_found"] += 1

                                if update_existing:
                                    # Update existing record
                                    existing_id = duplicate_result.get("duplicate_id")
                                    if existing_id:
                                        resume_ops.update_resume(
                                            existing_id, resume.dict()
                                        )
                                        save_result["updated_records"] += 1
                                continue
                        except Exception as e:
                            logger.warning(
                                f"Duplicate detection failed for resume: {e}"
                            )

                    # Save new resume if not duplicate
                    if not is_duplicate:
                        try:
                            saved_resume = resume_ops.create_resume(resume)
                            if saved_resume:
                                save_result["successfully_saved"] += 1
                                logger.info(
                                    f"[SUCCESS] Saved resume for user: {resume.user_id}"
                                )
                        except Exception as e:
                            logger.error(f"Failed to save resume: {e}")
                            save_result["errors"].append(f"Save error: {str(e)}")

                except Exception as e:
                    logger.error(f"Error processing individual resume: {e}")
                    save_result["errors"].append(f"Processing error: {str(e)}")

        except Exception as e:
            logger.error(f"Error in database save operation: {e}")
            save_result["errors"].append(f"Database operation error: {str(e)}")

        logger.info(
            f"Database save completed: {save_result['successfully_saved']} saved, "
            f"{save_result['duplicates_found']} duplicates, {len(save_result['errors'])} errors"
        )
        return save_result


def create_fixed_excel_parser_adapter(
    llm_provider: str = None, api_keys: List[str] = None
) -> FixedExcelParserAdapter:
    """Create an adapter instance for the fixed Excel parser"""
    return FixedExcelParserAdapter(llm_provider, api_keys)
