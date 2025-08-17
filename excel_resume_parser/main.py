"""
Main Excel Resume Parser Manager

This module provides the main interface for Excel-based resume parsing,
integrating Excel processing, resume parsing, and database operations.
"""

import os
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

from .excel_processor import ExcelProcessor
from .excel_resume_parser import ExcelResumeParser
from mangodatabase.operations import ResumeOperations
from mangodatabase.client import get_collection, get_resume_extracted_text_collection
from mangodatabase.duplicate_detection import DuplicateDetectionOperations
from embeddings.vectorizer import AddUserDataVectorizer
from core.custom_logger import CustomLogger
from core.llm_config import LLMConfigManager, LLMProvider
from core.config import AppConfig

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("excel_resume_parser_manager")


class ExcelResumeParserManager:
    """
    Main manager class for Excel-based resume parsing operations.
    Orchestrates the entire pipeline from Excel processing to database storage.
    """

    def __init__(self, llm_provider: str = None, api_keys: List[str] = None):
        """
        Initialize the Excel Resume Parser Manager.

        Args:
            llm_provider: LLM provider to use for parsing (auto-detected if None)
            api_keys: API keys for the LLM provider
        """
        logger.info("Initializing Excel Resume Parser Manager")

        # Auto-detect LLM provider if not provided
        if llm_provider is None:
            llm_provider = AppConfig.LLM_PROVIDER
            logger.info(f"Auto-detected LLM provider from config: {llm_provider}")

        # Initialize components
        self.excel_processor = ExcelProcessor()
        self.excel_resume_parser = ExcelResumeParser(
            llm_provider=llm_provider, api_keys=api_keys
        )

        # Initialize database connections
        self.collection = get_collection()
        self.extracted_text_collection = get_resume_extracted_text_collection()
        self.vectorizer = AddUserDataVectorizer()

        # Initialize operations
        self.resume_ops = ResumeOperations(self.collection, self.vectorizer)
        self.duplicate_ops = DuplicateDetectionOperations(
            self.extracted_text_collection
        )

        # Initialize LLM manager
        self.llm_manager = LLMConfigManager()

        # Temporary directories
        self.temp_dir = Path("dummy_data_save/temp_excel_files")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Excel Resume Parser Manager initialized successfully")

    def process_excel_file_from_path(
        self,
        file_path: str,
        base_user_id: str,
        base_username: str,
        sheet_name: Optional[str] = None,
        cleanup_file: bool = True,
    ) -> Dict[str, Any]:
        """
        Process an Excel file from file path.

        Args:
            file_path: Path to the Excel file
            base_user_id: Base user ID for generated resumes
            base_username: Base username for generated resumes
            sheet_name: Specific sheet to process (None for first sheet)
            cleanup_file: Whether to delete the file after processing

        Returns:
            Dictionary containing complete processing results
        """
        temp_file_path = None
        try:
            logger.info(f"Processing Excel file from path: {file_path}")
            start_time = time.time()

            # Step 1: Process Excel file
            logger.info("Step 1: Processing Excel data")
            excel_data = self.excel_processor.process_excel_file(
                file_path=file_path, sheet_name=sheet_name
            )

            if not excel_data:
                # Cleanup on empty data
                if cleanup_file and Path(file_path).exists():
                    try:
                        Path(file_path).unlink()
                        logger.info(f"Cleaned up empty Excel file: {file_path}")
                    except Exception as cleanup_error:
                        logger.warning(
                            f"Failed to cleanup file {file_path}: {cleanup_error}"
                        )

                return {
                    "status": "error",
                    "message": "No data found in Excel file",
                    "processing_time": time.time() - start_time,
                }

            # Step 2: Parse resumes from Excel data
            logger.info("Step 2: Parsing resumes from Excel data")
            parsing_results = self.excel_resume_parser.process_excel_data(
                excel_data=excel_data,
                base_user_id=base_user_id,
                base_username=base_username,
            )

            # Step 3: Save parsed resumes with duplicate detection
            logger.info("Step 3: Saving resumes with duplicate detection")
            save_results = self.excel_resume_parser.save_parsed_resumes(
                parsed_results=parsing_results,
                collection=self.collection,
                duplicate_ops=self.duplicate_ops,
            )

            # Compile final results
            total_time = time.time() - start_time

            final_results = {
                "status": "success",
                "file_path": file_path,
                "sheet_name": sheet_name,
                "total_processing_time": total_time,
                "excel_processing": {
                    "rows_found": len(excel_data),
                    "sample_data": (
                        excel_data[:3] if excel_data else []
                    ),  # First 3 rows as sample
                },
                "resume_parsing": parsing_results,
                "database_operations": save_results,
                "summary": {
                    "total_rows_processed": parsing_results["total_rows"],
                    "successful_parses": parsing_results["successful_parses"],
                    "failed_parses": parsing_results["failed_parses"],
                    "successfully_saved": save_results["saved_successfully"],
                    "duplicates_detected": save_results["duplicates_found"],
                    "save_errors": save_results["save_errors"],
                },
            }

            # Cleanup the file after successful processing
            if cleanup_file and Path(file_path).exists():
                try:
                    Path(file_path).unlink()
                    logger.info(f"Successfully cleaned up Excel file: {file_path}")
                    final_results["file_cleanup"] = "success"
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup file {file_path}: {cleanup_error}"
                    )
                    final_results["file_cleanup"] = f"failed: {cleanup_error}"

            logger.info(f"Excel file processing completed in {total_time:.2f} seconds")
            return final_results

        except Exception as e:
            # Cleanup on error
            if cleanup_file and Path(file_path).exists():
                try:
                    Path(file_path).unlink()
                    logger.info(f"Cleaned up Excel file after error: {file_path}")
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup file after error {file_path}: {cleanup_error}"
                    )

            logger.error(f"Error processing Excel file from path {file_path}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "file_path": file_path,
                "processing_time": (
                    time.time() - start_time if "start_time" in locals() else 0
                ),
            }

    def process_excel_file_from_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        base_user_id: str,
        base_username: str,
        sheet_name: Optional[str] = None,
        save_temp_file: bool = False,
    ) -> Dict[str, Any]:
        """
        Process an Excel file from bytes data.

        Args:
            file_bytes: Excel file as bytes
            filename: Original filename
            base_user_id: Base user ID for generated resumes
            base_username: Base username for generated resumes
            sheet_name: Specific sheet to process (None for first sheet)
            save_temp_file: Whether to save bytes to temporary file first

        Returns:
            Dictionary containing complete processing results
        """
        temp_file_path = None
        try:
            logger.info(f"Processing Excel file from bytes: {filename}")
            start_time = time.time()

            # Option 1: Save to temporary file first (if requested)
            if save_temp_file:
                # Create temporary file
                temp_file_path = self.temp_dir / f"temp_{int(time.time())}_{filename}"
                try:
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(file_bytes)
                    logger.info(f"Saved temporary Excel file: {temp_file_path}")

                    # Process from file path with cleanup
                    return self.process_excel_file_from_path(
                        file_path=str(temp_file_path),
                        base_user_id=base_user_id,
                        base_username=base_username,
                        sheet_name=sheet_name,
                        cleanup_file=True,
                    )
                except Exception as temp_error:
                    # Cleanup temp file if creation failed
                    if temp_file_path and temp_file_path.exists():
                        try:
                            temp_file_path.unlink()
                            logger.info(
                                f"Cleaned up failed temp file: {temp_file_path}"
                            )
                        except Exception as cleanup_error:
                            logger.warning(
                                f"Failed to cleanup temp file: {cleanup_error}"
                            )
                    raise temp_error

            # Option 2: Process directly from bytes (default - no file saved)
            # Step 1: Process Excel bytes
            logger.info("Step 1: Processing Excel data from bytes")
            excel_data = self.excel_processor.process_excel_bytes(
                file_bytes=file_bytes, filename=filename, sheet_name=sheet_name
            )

            if not excel_data:
                return {
                    "status": "error",
                    "message": "No data found in Excel file",
                    "filename": filename,
                    "processing_time": time.time() - start_time,
                }

            # Step 2: Parse resumes from Excel data
            logger.info("Step 2: Parsing resumes from Excel data")
            parsing_results = self.excel_resume_parser.process_excel_data(
                excel_data=excel_data,
                base_user_id=base_user_id,
                base_username=base_username,
            )

            # Step 3: Save parsed resumes with duplicate detection
            logger.info("Step 3: Saving resumes with duplicate detection")
            save_results = self.excel_resume_parser.save_parsed_resumes(
                parsed_results=parsing_results,
                collection=self.collection,
                duplicate_ops=self.duplicate_ops,
            )

            # Compile final results
            total_time = time.time() - start_time

            final_results = {
                "status": "success",
                "filename": filename,
                "sheet_name": sheet_name,
                "total_processing_time": total_time,
                "excel_processing": {
                    "rows_found": len(excel_data),
                    "sample_data": (
                        excel_data[:3] if excel_data else []
                    ),  # First 3 rows as sample
                },
                "resume_parsing": parsing_results,
                "database_operations": save_results,
                "summary": {
                    "total_rows_processed": parsing_results["total_rows"],
                    "successful_parses": parsing_results["successful_parses"],
                    "failed_parses": parsing_results["failed_parses"],
                    "successfully_saved": save_results["saved_successfully"],
                    "duplicates_detected": save_results["duplicates_found"],
                    "save_errors": save_results["save_errors"],
                },
            }

            logger.info(f"Excel bytes processing completed in {total_time:.2f} seconds")
            return final_results

        except Exception as e:
            logger.error(f"Error processing Excel bytes for {filename}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "filename": filename,
                "processing_time": (
                    time.time() - start_time if "start_time" in locals() else 0
                ),
            }

    def get_excel_info(
        self, file_path: str = None, file_bytes: bytes = None, filename: str = None
    ) -> Dict[str, Any]:
        """
        Get information about an Excel file (sheet names, structure, etc.).

        Args:
            file_path: Path to Excel file (for file-based processing)
            file_bytes: Excel file bytes (for upload-based processing)
            filename: Original filename (required for bytes processing)

        Returns:
            Dictionary containing Excel file information
        """
        try:
            if file_path:
                # Process from file path
                sheet_names = self.excel_processor.get_sheet_names(file_path)

                # Get sample data from first sheet
                sample_data = self.excel_processor.process_excel_file(file_path)

                return {
                    "status": "success",
                    "file_path": file_path,
                    "sheet_names": sheet_names,
                    "total_sheets": len(sheet_names),
                    "sample_data": (
                        sample_data[:5] if sample_data else []
                    ),  # First 5 rows
                    "total_rows": len(sample_data) if sample_data else 0,
                    "columns": list(sample_data[0].keys()) if sample_data else [],
                }

            elif file_bytes and filename:
                # Process from bytes
                sheet_names = self.excel_processor.get_sheet_names_from_bytes(
                    file_bytes
                )

                # Get sample data from first sheet
                sample_data = self.excel_processor.process_excel_bytes(
                    file_bytes, filename
                )

                return {
                    "status": "success",
                    "filename": filename,
                    "sheet_names": sheet_names,
                    "total_sheets": len(sheet_names),
                    "sample_data": (
                        sample_data[:5] if sample_data else []
                    ),  # First 5 rows
                    "total_rows": len(sample_data) if sample_data else 0,
                    "columns": list(sample_data[0].keys()) if sample_data else [],
                }
            else:
                return {
                    "status": "error",
                    "message": "Either file_path or (file_bytes and filename) must be provided",
                }

        except Exception as e:
            logger.error(f"Error getting Excel info: {e}")
            return {"status": "error", "message": str(e)}

    def cleanup_temp_files(self, age_limit_minutes: int = 60):
        """
        Cleanup temporary files older than specified age.

        Args:
            age_limit_minutes: Age limit in minutes for file cleanup
        """
        try:
            now = datetime.now()
            cleaned_count = 0

            for file_path in self.temp_dir.iterdir():
                if file_path.is_file():
                    file_age = now - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age.total_seconds() > (age_limit_minutes * 60):
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                            logger.debug(f"Deleted old temp file: {file_path}")
                        except Exception as e:
                            logger.error(f"Failed to delete temp file {file_path}: {e}")

            logger.info(f"Cleaned up {cleaned_count} temporary files")

        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about Excel resume processing.

        Returns:
            Dictionary containing processing statistics
        """
        try:
            # This could be extended to track processing history
            return {
                "status": "success",
                "llm_provider": self.llm_manager.provider.value,
                "temp_directory": str(self.temp_dir),
                "temp_files_count": len(list(self.temp_dir.iterdir())),
                "database_collections": {
                    "resumes": self.collection.name,
                    "extracted_texts": self.extracted_text_collection.name,
                },
            }

        except Exception as e:
            logger.error(f"Error getting processing statistics: {e}")
            return {"status": "error", "message": str(e)}
