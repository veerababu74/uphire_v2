"""
Enhanced Excel Resume Parser with Progress Tracking and Error Handling

This module provides enhanced Excel resume parsing with comprehensive tracking,
error handling, and recovery capabilities.
"""

import asyncio
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from excel_resume_parser.main import ExcelResumeParserManager
from core.progress_tracker import progress_tracker, OperationType, ErrorSeverity
from core.batch_processor import batch_processor, BatchConfig
from core.custom_logger import CustomLogger
from mangodatabase.operations import ResumeOperations
from mangodatabase.duplicate_detection import DuplicateDetectionOperations
from embeddings.vectorizer import AddUserDataVectorizer

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("enhanced_excel_parser_with_tracking")


class EnhancedExcelParserWithTracking:
    """
    Enhanced Excel parser with comprehensive tracking and error handling.

    Features:
    - Real-time progress tracking
    - Error handling and recovery
    - Batch processing with checkpoints
    - Detailed reporting
    - Resume capability
    """

    def __init__(self, llm_provider: str = None, api_keys: List[str] = None):
        """
        Initialize enhanced Excel parser.

        Args:
            llm_provider: LLM provider to use
            api_keys: API keys for the provider
        """
        self.base_parser = ExcelResumeParserManager(llm_provider, api_keys)
        self.batch_config = BatchConfig(
            batch_size=25,  # Smaller batches for Excel processing
            max_workers=2,  # Conservative for Excel processing
            timeout_per_item=120,  # 2 minutes per row
            max_retries=2,
            checkpoint_interval=50,
            error_threshold=0.15,  # 15% error threshold
            enable_recovery=True,
        )

        logger.info("Enhanced Excel Parser with Tracking initialized")

    async def process_excel_file_with_tracking(
        self,
        file_path: str,
        base_user_id: str,
        base_username: str,
        sheet_name: Optional[str] = None,
        cleanup_file: bool = True,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process Excel file with comprehensive tracking and error handling.

        Args:
            file_path: Path to Excel file
            base_user_id: Base user ID
            base_username: Base username
            sheet_name: Sheet name to process
            cleanup_file: Whether to cleanup file after processing
            session_id: Existing session ID for resuming

        Returns:
            Comprehensive processing results with tracking data
        """
        start_time = time.time()

        try:
            logger.info(f"Starting enhanced Excel processing: {file_path}")

            # Step 1: Process Excel file to get data
            logger.info("Step 1: Extracting Excel data")
            excel_data = self.base_parser.excel_processor.process_excel_file(
                file_path=file_path, sheet_name=sheet_name
            )

            if not excel_data:
                return {
                    "status": "error",
                    "message": "No data found in Excel file",
                    "file_path": file_path,
                    "processing_time": time.time() - start_time,
                }

            logger.info(f"Extracted {len(excel_data)} rows from Excel file")

            # Step 2: Process rows with tracking
            logger.info("Step 2: Processing Excel rows with tracking")

            # Create processor function for individual rows
            def process_excel_row(
                row_data: Dict[str, Any], row_index: int
            ) -> Dict[str, Any]:
                return self._process_single_excel_row(
                    row_data=row_data,
                    row_index=row_index,
                    base_user_id=base_user_id,
                    base_username=base_username,
                )

            # Process with batch processor
            batch_results = await batch_processor.process_excel_batch(
                file_path=file_path,
                excel_data=excel_data,
                user_id=base_user_id,
                username=base_username,
                parser_function=process_excel_row,
                session_id=session_id,
            )

            # Step 3: Save successful results to database
            logger.info("Step 3: Saving results to database")
            save_results = await self._save_parsed_results(batch_results)

            # Step 4: Cleanup file if requested
            if cleanup_file and Path(file_path).exists():
                try:
                    Path(file_path).unlink()
                    logger.info(f"Cleaned up Excel file: {file_path}")
                    file_cleanup_status = "success"
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup file {file_path}: {cleanup_error}"
                    )
                    file_cleanup_status = f"failed: {cleanup_error}"
            else:
                file_cleanup_status = "not_requested"

            # Compile comprehensive results
            total_time = time.time() - start_time

            comprehensive_results = {
                "status": "success",
                "file_path": file_path,
                "sheet_name": sheet_name,
                "session_id": batch_processor.current_session_id,
                "total_processing_time": total_time,
                "file_cleanup": file_cleanup_status,
                # Excel processing summary
                "excel_processing": {
                    "total_rows_found": len(excel_data),
                    "rows_processed": batch_results["processed_items"],
                    "successful_rows": batch_results["successful_items"],
                    "failed_rows": batch_results["failed_items"],
                    "skipped_rows": batch_results["skipped_items"],
                    "duplicate_rows": batch_results["duplicate_items"],
                    "success_rate": batch_results["success_rate"],
                },
                # Database operations
                "database_operations": save_results,
                # Detailed metrics
                "detailed_metrics": {
                    "processing_rate": (
                        batch_results["successful_items"] / total_time
                        if total_time > 0
                        else 0
                    ),
                    "average_row_processing_time": (
                        total_time / batch_results["processed_items"]
                        if batch_results["processed_items"] > 0
                        else 0
                    ),
                    "total_retries": batch_results["processing_summary"][
                        "total_retries"
                    ],
                    "peak_memory_usage": batch_results["processing_summary"][
                        "peak_memory_usage"
                    ],
                },
                # Error summary
                "error_summary": self._generate_error_summary(batch_results),
                # Performance insights
                "performance_insights": self._generate_performance_insights(
                    batch_results, total_time
                ),
            }

            logger.info(
                f"Enhanced Excel processing completed in {total_time:.2f} seconds"
            )
            return comprehensive_results

        except Exception as e:
            error_msg = f"Enhanced Excel processing failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())

            # Cleanup file on error if requested
            if cleanup_file and Path(file_path).exists():
                try:
                    Path(file_path).unlink()
                    logger.info(f"Cleaned up Excel file after error: {file_path}")
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to cleanup file after error: {cleanup_error}"
                    )

            return {
                "status": "error",
                "message": error_msg,
                "file_path": file_path,
                "session_id": batch_processor.current_session_id,
                "processing_time": time.time() - start_time,
                "stack_trace": traceback.format_exc(),
            }

    def _process_single_excel_row(
        self,
        row_data: Dict[str, Any],
        row_index: int,
        base_user_id: str,
        base_username: str,
    ) -> Dict[str, Any]:
        """
        Process a single Excel row.

        Args:
            row_data: Row data from Excel
            row_index: Index of the row
            base_user_id: Base user ID
            base_username: Base username

        Returns:
            Processing result
        """
        try:
            start_time = time.time()

            # Use base parser's row processing logic
            parsed_resume = self.base_parser.excel_resume_parser.process_single_row(
                row_data=row_data,
                row_index=row_index,
                base_user_id=base_user_id,
                base_username=base_username,
            )

            processing_time = time.time() - start_time

            if parsed_resume and parsed_resume.get("status") == "success":
                return {
                    "status": "success",
                    "row_index": row_index,
                    "processing_time": processing_time,
                    "parsed_data": parsed_resume,
                    "resume_id": parsed_resume.get("user_id"),
                    "candidate_name": parsed_resume.get("contact_details", {}).get(
                        "name", "Unknown"
                    ),
                }
            else:
                error_msg = (
                    parsed_resume.get("error", "Unknown parsing error")
                    if parsed_resume
                    else "No result returned"
                )
                return {
                    "status": "failed",
                    "row_index": row_index,
                    "processing_time": processing_time,
                    "error": error_msg,
                    "raw_data": str(row_data)[:200],  # First 200 chars for debugging
                }

        except Exception as e:
            processing_time = (
                time.time() - start_time if "start_time" in locals() else 0
            )
            error_msg = f"Error processing Excel row {row_index}: {str(e)}"

            # Add error to progress tracker
            if batch_processor.current_session_id:
                progress_tracker.add_error(
                    session_id=batch_processor.current_session_id,
                    error_type="EXCEL_ROW_PARSING_ERROR",
                    error_message=error_msg,
                    severity=ErrorSeverity.MEDIUM,
                    item_index=row_index,
                    stack_trace=traceback.format_exc(),
                    context={"row_data": str(row_data)[:200]},
                )

            return {
                "status": "failed",
                "row_index": row_index,
                "processing_time": processing_time,
                "error": error_msg,
                "stack_trace": traceback.format_exc(),
            }

    async def _save_parsed_results(
        self, batch_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Save parsed results to database with duplicate detection.

        Args:
            batch_results: Results from batch processing

        Returns:
            Save operation results
        """
        try:
            logger.info("Saving parsed results to database")

            successful_results = batch_results.get("successful_results", [])
            saved_count = 0
            duplicate_count = 0
            save_errors = []

            # Initialize database operations
            collection = self.base_parser.collection
            duplicate_ops = self.base_parser.duplicate_ops

            for result in successful_results:
                try:
                    parsed_data = result.get("parsed_data", {})
                    if not parsed_data:
                        continue

                    # Save with duplicate detection
                    save_result = (
                        self.base_parser.excel_resume_parser.save_single_resume(
                            parsed_data=parsed_data,
                            collection=collection,
                            duplicate_ops=duplicate_ops,
                        )
                    )

                    if save_result.get("status") == "success":
                        if save_result.get("is_duplicate", False):
                            duplicate_count += 1
                        else:
                            saved_count += 1
                    else:
                        save_errors.append(
                            {
                                "row_index": result.get("row_index"),
                                "error": save_result.get("error", "Unknown save error"),
                            }
                        )

                except Exception as e:
                    save_errors.append(
                        {
                            "row_index": result.get("row_index"),
                            "error": f"Save error: {str(e)}",
                        }
                    )

            save_results = {
                "total_attempted": len(successful_results),
                "saved_successfully": saved_count,
                "duplicates_detected": duplicate_count,
                "save_errors": len(save_errors),
                "error_details": save_errors[:10],  # First 10 errors for review
                "save_success_rate": (
                    saved_count / len(successful_results) if successful_results else 0
                ),
            }

            logger.info(
                f"Database save completed: {saved_count} saved, {duplicate_count} duplicates, {len(save_errors)} errors"
            )
            return save_results

        except Exception as e:
            error_msg = f"Database save operation failed: {str(e)}"
            logger.error(error_msg)

            return {
                "total_attempted": len(batch_results.get("successful_results", [])),
                "saved_successfully": 0,
                "duplicates_detected": 0,
                "save_errors": 1,
                "error_details": [{"error": error_msg}],
                "save_success_rate": 0.0,
            }

    def _generate_error_summary(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive error summary."""
        failed_results = batch_results.get("failed_results", [])

        # Categorize errors
        error_categories = {}
        for result in failed_results:
            error = result.get("error", "Unknown error")
            error_type = self._categorize_error(error)

            if error_type not in error_categories:
                error_categories[error_type] = {"count": 0, "examples": []}

            error_categories[error_type]["count"] += 1
            if len(error_categories[error_type]["examples"]) < 3:
                error_categories[error_type]["examples"].append(
                    {"row_index": result.get("row_index"), "error": error}
                )

        return {
            "total_errors": len(failed_results),
            "error_categories": error_categories,
            "error_rate": (
                len(failed_results) / batch_results["total_items"]
                if batch_results["total_items"] > 0
                else 0
            ),
            "most_common_error": (
                max(error_categories.items(), key=lambda x: x[1]["count"])[0]
                if error_categories
                else None
            ),
        }

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error message into error type."""
        error_lower = error_message.lower()

        if "parsing" in error_lower or "parse" in error_lower:
            return "PARSING_ERROR"
        elif "llm" in error_lower or "model" in error_lower:
            return "LLM_ERROR"
        elif "validation" in error_lower or "invalid" in error_lower:
            return "VALIDATION_ERROR"
        elif "timeout" in error_lower:
            return "TIMEOUT_ERROR"
        elif "connection" in error_lower or "network" in error_lower:
            return "NETWORK_ERROR"
        elif "memory" in error_lower or "resource" in error_lower:
            return "RESOURCE_ERROR"
        else:
            return "UNKNOWN_ERROR"

    def _generate_performance_insights(
        self, batch_results: Dict[str, Any], total_time: float
    ) -> Dict[str, Any]:
        """Generate performance insights and recommendations."""
        successful_items = batch_results["successful_items"]
        total_items = batch_results["total_items"]

        insights = {
            "overall_performance": (
                "excellent"
                if batch_results["success_rate"] > 0.95
                else (
                    "good"
                    if batch_results["success_rate"] > 0.85
                    else "fair" if batch_results["success_rate"] > 0.7 else "poor"
                )
            ),
            "processing_speed": (
                "fast"
                if successful_items / total_time > 10
                else "moderate" if successful_items / total_time > 5 else "slow"
            ),
            "recommendations": [],
        }

        # Generate recommendations
        if batch_results["success_rate"] < 0.8:
            insights["recommendations"].append(
                "Consider reviewing data quality or LLM configuration"
            )

        if successful_items / total_time < 5:
            insights["recommendations"].append(
                "Consider increasing batch size or worker count for better performance"
            )

        if (
            batch_results["processing_summary"]["total_retries"]
            > successful_items * 0.1
        ):
            insights["recommendations"].append(
                "High retry rate detected - check network stability and timeout settings"
            )

        return insights

    def get_processing_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current processing status for a session.

        Args:
            session_id: Session ID

        Returns:
            Current status information
        """
        return progress_tracker.get_session_status(session_id)

    def get_processing_errors(
        self, session_id: str, limit: int = 50
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get processing errors for a session.

        Args:
            session_id: Session ID
            limit: Maximum number of errors to return

        Returns:
            List of error information
        """
        return progress_tracker.get_session_errors(session_id, limit)

    def resume_processing(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Resume a paused or failed processing session.

        Args:
            session_id: Session ID to resume

        Returns:
            Resume information
        """
        return progress_tracker.resume_session(session_id)

    def stop_processing(self, session_id: str) -> bool:
        """
        Stop current processing.

        Args:
            session_id: Session ID to stop

        Returns:
            True if stopped successfully
        """
        batch_processor.stop_processing()
        return progress_tracker.pause_session(session_id, "Stopped by user request")


# Global instance
enhanced_excel_parser = EnhancedExcelParserWithTracking()
