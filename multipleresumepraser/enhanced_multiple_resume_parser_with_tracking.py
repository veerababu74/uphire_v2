"""
Enhanced Multiple Resume Parser with Progress Tracking and Error Handling

This module provides enhanced multiple resume parsing with comprehensive tracking,
error handling, and recovery capabilities.
"""

import asyncio
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from multipleresumepraser.main import ResumeParser
from multipleresumepraser.text_extraction import extract_and_clean_text
from core.progress_tracker import progress_tracker, OperationType, ErrorSeverity
from core.batch_processor import batch_processor, BatchConfig
from core.custom_logger import CustomLogger
from mangodatabase.operations import ResumeOperations, SkillsTitlesOperations
from mangodatabase.client import get_collection, get_skills_titles_collection
from embeddings.vectorizer import AddUserDataVectorizer

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("enhanced_multiple_resume_parser_with_tracking")


class EnhancedMultipleResumeParserWithTracking:
    """
    Enhanced multiple resume parser with comprehensive tracking and error handling.

    Features:
    - Real-time progress tracking
    - Error handling and recovery
    - Batch processing with checkpoints
    - Detailed reporting
    - Resume capability
    - Memory management for large batches
    """

    def __init__(self, llm_provider: str = None, api_keys: List[str] = None):
        """
        Initialize enhanced multiple resume parser.

        Args:
            llm_provider: LLM provider to use
            api_keys: API keys for the provider
        """
        self.base_parser = ResumeParser(llm_provider, api_keys)
        self.batch_config = BatchConfig(
            batch_size=10,  # Smaller batches for resume processing
            max_workers=3,  # Conservative for resume processing
            timeout_per_item=180,  # 3 minutes per resume
            max_retries=2,
            checkpoint_interval=25,
            error_threshold=0.2,  # 20% error threshold
            enable_recovery=True,
            memory_threshold=800,  # MB
        )

        # Initialize database operations
        self.collection = get_collection()
        self.skills_titles_collection = get_skills_titles_collection()
        self.skills_ops = SkillsTitlesOperations(self.skills_titles_collection)
        self.add_user_vectorizer = AddUserDataVectorizer()
        self.resume_ops = ResumeOperations(self.collection, self.add_user_vectorizer)

        # Supported file types
        self.supported_extensions = {".pdf", ".doc", ".docx", ".txt"}

        logger.info("Enhanced Multiple Resume Parser with Tracking initialized")

    async def process_multiple_resumes_with_tracking(
        self,
        resume_files: List[str],
        base_user_id: str,
        base_username: str,
        cleanup_files: bool = True,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process multiple resume files with comprehensive tracking and error handling.

        Args:
            resume_files: List of resume file paths
            base_user_id: Base user ID
            base_username: Base username
            cleanup_files: Whether to cleanup files after processing
            session_id: Existing session ID for resuming

        Returns:
            Comprehensive processing results with tracking data
        """
        start_time = time.time()

        try:
            logger.info(
                f"Starting enhanced multiple resume processing: {len(resume_files)} files"
            )

            # Step 1: Validate and filter files
            logger.info("Step 1: Validating resume files")
            valid_files, invalid_files = self._validate_resume_files(resume_files)

            if not valid_files:
                return {
                    "status": "error",
                    "message": "No valid resume files found",
                    "total_files": len(resume_files),
                    "invalid_files": invalid_files,
                    "processing_time": time.time() - start_time,
                }

            logger.info(
                f"Validated {len(valid_files)} files, {len(invalid_files)} invalid"
            )

            # Step 2: Process resumes with tracking
            logger.info("Step 2: Processing resume files with tracking")

            # Create processor function for individual resumes
            def process_resume_file(file_path: str, file_index: int) -> Dict[str, Any]:
                return self._process_single_resume_file(
                    file_path=file_path,
                    file_index=file_index,
                    base_user_id=base_user_id,
                    base_username=base_username,
                )

            # Process with batch processor
            batch_results = await batch_processor.process_resume_batch(
                resume_files=valid_files,
                user_id=base_user_id,
                username=base_username,
                parser_function=process_resume_file,
                session_id=session_id,
            )

            # Step 3: Save successful results to database
            logger.info("Step 3: Saving results to database")
            save_results = await self._save_parsed_results(batch_results)

            # Step 4: Cleanup files if requested
            cleanup_results = {
                "status": "not_requested",
                "cleaned_files": 0,
                "cleanup_errors": 0,
            }
            if cleanup_files:
                cleanup_results = self._cleanup_processed_files(valid_files)

            # Compile comprehensive results
            total_time = time.time() - start_time

            comprehensive_results = {
                "status": "success",
                "session_id": batch_processor.current_session_id,
                "total_processing_time": total_time,
                "file_cleanup": cleanup_results,
                # File validation summary
                "file_validation": {
                    "total_files_provided": len(resume_files),
                    "valid_files": len(valid_files),
                    "invalid_files": len(invalid_files),
                    "invalid_file_details": invalid_files[:10],  # First 10 for review
                },
                # Resume processing summary
                "resume_processing": {
                    "files_processed": batch_results["processed_items"],
                    "successful_files": batch_results["successful_items"],
                    "failed_files": batch_results["failed_items"],
                    "skipped_files": batch_results["skipped_items"],
                    "duplicate_files": batch_results["duplicate_items"],
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
                    "average_file_processing_time": (
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
                # File type analysis
                "file_type_analysis": self._analyze_file_types(
                    valid_files, batch_results
                ),
            }

            logger.info(
                f"Enhanced multiple resume processing completed in {total_time:.2f} seconds"
            )
            return comprehensive_results

        except Exception as e:
            error_msg = f"Enhanced multiple resume processing failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())

            return {
                "status": "error",
                "message": error_msg,
                "session_id": batch_processor.current_session_id,
                "processing_time": time.time() - start_time,
                "stack_trace": traceback.format_exc(),
            }

    def _validate_resume_files(
        self, resume_files: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Validate resume files and return valid/invalid lists.

        Args:
            resume_files: List of file paths

        Returns:
            Tuple of (valid_files, invalid_file_details)
        """
        valid_files = []
        invalid_files = []

        for file_path in resume_files:
            try:
                path_obj = Path(file_path)

                # Check if file exists
                if not path_obj.exists():
                    invalid_files.append(
                        {"file_path": file_path, "reason": "File does not exist"}
                    )
                    continue

                # Check file extension
                if path_obj.suffix.lower() not in self.supported_extensions:
                    invalid_files.append(
                        {
                            "file_path": file_path,
                            "reason": f"Unsupported file type: {path_obj.suffix}",
                        }
                    )
                    continue

                # Check file size (skip files larger than 50MB)
                file_size = path_obj.stat().st_size
                if file_size > 50 * 1024 * 1024:  # 50MB
                    invalid_files.append(
                        {
                            "file_path": file_path,
                            "reason": f"File too large: {file_size / (1024*1024):.1f}MB",
                        }
                    )
                    continue

                # Check if file is readable
                try:
                    with open(file_path, "rb") as f:
                        f.read(1024)  # Try to read first 1KB
                    valid_files.append(file_path)
                except Exception as read_error:
                    invalid_files.append(
                        {
                            "file_path": file_path,
                            "reason": f"Cannot read file: {str(read_error)}",
                        }
                    )

            except Exception as e:
                invalid_files.append(
                    {"file_path": file_path, "reason": f"Validation error: {str(e)}"}
                )

        return valid_files, invalid_files

    def _process_single_resume_file(
        self, file_path: str, file_index: int, base_user_id: str, base_username: str
    ) -> Dict[str, Any]:
        """
        Process a single resume file.

        Args:
            file_path: Path to resume file
            file_index: Index of the file
            base_user_id: Base user ID
            base_username: Base username

        Returns:
            Processing result
        """
        try:
            start_time = time.time()
            file_name = Path(file_path).name

            logger.debug(f"Processing resume file {file_index}: {file_name}")

            # Step 1: Extract text from resume
            try:
                extracted_content = extract_and_clean_text(file_path)
                if not extracted_content or not extracted_content.strip():
                    return {
                        "status": "failed",
                        "file_index": file_index,
                        "file_path": file_path,
                        "file_name": file_name,
                        "error": "No text content extracted from resume",
                        "processing_time": time.time() - start_time,
                    }
            except Exception as extraction_error:
                return {
                    "status": "failed",
                    "file_index": file_index,
                    "file_path": file_path,
                    "file_name": file_name,
                    "error": f"Text extraction failed: {str(extraction_error)}",
                    "processing_time": time.time() - start_time,
                }

            # Step 2: Parse resume using LLM
            try:
                # Generate unique user ID for this resume
                unique_user_id = f"{base_user_id}_{file_index}_{int(time.time())}"
                unique_username = f"{base_username}_{file_index}"

                parsed_result = self.base_parser.parse_resume_from_text(
                    text_input=extracted_content,
                    user_id=unique_user_id,
                    username=unique_username,
                )

                if not parsed_result:
                    return {
                        "status": "failed",
                        "file_index": file_index,
                        "file_path": file_path,
                        "file_name": file_name,
                        "error": "LLM parsing returned no result",
                        "processing_time": time.time() - start_time,
                    }

                processing_time = time.time() - start_time

                return {
                    "status": "success",
                    "file_index": file_index,
                    "file_path": file_path,
                    "file_name": file_name,
                    "processing_time": processing_time,
                    "parsed_data": parsed_result,
                    "resume_id": unique_user_id,
                    "candidate_name": parsed_result.get("contact_details", {}).get(
                        "name", "Unknown"
                    ),
                    "extracted_text_length": len(extracted_content),
                    "file_size": Path(file_path).stat().st_size,
                }

            except Exception as parsing_error:
                return {
                    "status": "failed",
                    "file_index": file_index,
                    "file_path": file_path,
                    "file_name": file_name,
                    "error": f"LLM parsing failed: {str(parsing_error)}",
                    "processing_time": time.time() - start_time,
                    "extracted_text_length": len(extracted_content),
                }

        except Exception as e:
            processing_time = (
                time.time() - start_time if "start_time" in locals() else 0
            )
            error_msg = f"Error processing resume file {file_index} ({Path(file_path).name}): {str(e)}"

            # Add error to progress tracker
            if batch_processor.current_session_id:
                progress_tracker.add_error(
                    session_id=batch_processor.current_session_id,
                    error_type="RESUME_FILE_PROCESSING_ERROR",
                    error_message=error_msg,
                    severity=ErrorSeverity.MEDIUM,
                    item_index=file_index,
                    item_identifier=file_path,
                    stack_trace=traceback.format_exc(),
                    context={"file_name": Path(file_path).name},
                )

            return {
                "status": "failed",
                "file_index": file_index,
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "processing_time": processing_time,
                "error": error_msg,
                "stack_trace": traceback.format_exc(),
            }

    async def _save_parsed_results(
        self, batch_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Save parsed results to database with duplicate detection and skills extraction.

        Args:
            batch_results: Results from batch processing

        Returns:
            Save operation results
        """
        try:
            logger.info("Saving parsed resume results to database")

            successful_results = batch_results.get("successful_results", [])
            saved_count = 0
            duplicate_count = 0
            save_errors = []
            skills_extracted = 0

            for result in successful_results:
                try:
                    parsed_data = result.get("parsed_data", {})
                    if not parsed_data:
                        continue

                    # Extract and save skills
                    skills_extraction_result = await self._extract_and_save_skills(
                        parsed_data
                    )
                    if skills_extraction_result.get("status") == "success":
                        skills_extracted += 1

                    # Generate embeddings
                    embedding_result = self._generate_embeddings_for_resume(parsed_data)

                    # Save to database with duplicate detection
                    save_result = await self._save_single_resume_to_db(
                        parsed_data=parsed_data, embeddings=embedding_result
                    )

                    if save_result.get("status") == "success":
                        if save_result.get("is_duplicate", False):
                            duplicate_count += 1
                        else:
                            saved_count += 1
                    else:
                        save_errors.append(
                            {
                                "file_index": result.get("file_index"),
                                "file_name": result.get("file_name"),
                                "error": save_result.get("error", "Unknown save error"),
                            }
                        )

                except Exception as e:
                    save_errors.append(
                        {
                            "file_index": result.get("file_index"),
                            "file_name": result.get("file_name"),
                            "error": f"Save operation error: {str(e)}",
                        }
                    )

            save_results = {
                "total_attempted": len(successful_results),
                "saved_successfully": saved_count,
                "duplicates_detected": duplicate_count,
                "save_errors": len(save_errors),
                "skills_extracted": skills_extracted,
                "error_details": save_errors[:10],  # First 10 errors for review
                "save_success_rate": (
                    saved_count / len(successful_results) if successful_results else 0
                ),
                "skills_extraction_rate": (
                    skills_extracted / len(successful_results)
                    if successful_results
                    else 0
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
                "skills_extracted": 0,
                "error_details": [{"error": error_msg}],
                "save_success_rate": 0.0,
                "skills_extraction_rate": 0.0,
            }

    async def _extract_and_save_skills(
        self, parsed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract and save skills to skills collection."""
        try:
            skills = parsed_data.get("skills", [])
            may_also_known_skills = parsed_data.get("may_also_known_skills", [])

            all_skills = list(set(skills + may_also_known_skills))

            if all_skills:
                save_result = await self.skills_ops.add_skills_batch(all_skills)
                return {"status": "success", "skills_count": len(all_skills)}
            else:
                return {"status": "no_skills", "skills_count": 0}

        except Exception as e:
            logger.error(f"Skills extraction error: {str(e)}")
            return {"status": "error", "error": str(e), "skills_count": 0}

    def _generate_embeddings_for_resume(
        self, parsed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate embeddings for resume data."""
        try:
            embeddings = {}

            # Skills embeddings
            skills = parsed_data.get("skills", [])
            if skills:
                skills_text = " ".join([str(skill) for skill in skills if skill])
                if skills_text.strip():
                    embeddings["skills_embedding"] = (
                        self.add_user_vectorizer.generate_embedding(skills_text)
                    )

            # Experience embeddings
            experience = parsed_data.get("experience", [])
            if experience:
                experience_texts = []
                for exp in experience:
                    if isinstance(exp, dict):
                        title = exp.get("title", "")
                        company = exp.get("company", "")
                        if title or company:
                            experience_texts.append(f"{title} at {company}")

                if experience_texts:
                    experience_text = " ".join(experience_texts)
                    embeddings["experience_embedding"] = (
                        self.add_user_vectorizer.generate_embedding(experience_text)
                    )

            # Education embeddings
            education = parsed_data.get("academic_details", [])
            if education:
                education_texts = []
                for edu in education:
                    if isinstance(edu, dict):
                        degree = edu.get("education", "")
                        college = edu.get("college", "")
                        if degree or college:
                            education_texts.append(f"{degree} from {college}")

                if education_texts:
                    education_text = " ".join(education_texts)
                    embeddings["education_embedding"] = (
                        self.add_user_vectorizer.generate_embedding(education_text)
                    )

            return {"status": "success", "embeddings": embeddings}

        except Exception as e:
            logger.error(f"Embedding generation error: {str(e)}")
            return {"status": "error", "error": str(e), "embeddings": {}}

    async def _save_single_resume_to_db(
        self, parsed_data: Dict[str, Any], embeddings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save a single resume to database."""
        try:
            # Add embeddings to parsed data if successful
            if embeddings.get("status") == "success":
                parsed_data.update(embeddings.get("embeddings", {}))

            # Save to database
            result = await self.resume_ops.add_user_data(parsed_data)

            return {
                "status": "success",
                "resume_id": parsed_data.get("user_id"),
                "is_duplicate": result.get("is_duplicate", False),
            }

        except Exception as e:
            logger.error(f"Database save error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _cleanup_processed_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Cleanup processed resume files."""
        cleaned_files = 0
        cleanup_errors = []

        for file_path in file_paths:
            try:
                Path(file_path).unlink()
                cleaned_files += 1
            except Exception as e:
                cleanup_errors.append({"file_path": file_path, "error": str(e)})

        return {
            "status": "completed",
            "cleaned_files": cleaned_files,
            "cleanup_errors": len(cleanup_errors),
            "error_details": cleanup_errors[:5],  # First 5 errors
        }

    def _generate_error_summary(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive error summary."""
        failed_results = batch_results.get("failed_results", [])

        # Categorize errors
        error_categories = {}
        file_type_errors = {}

        for result in failed_results:
            error = result.get("error", "Unknown error")
            file_name = result.get("file_name", "unknown")
            file_ext = Path(file_name).suffix.lower()

            # Categorize by error type
            error_type = self._categorize_error(error)
            if error_type not in error_categories:
                error_categories[error_type] = {"count": 0, "examples": []}
            error_categories[error_type]["count"] += 1
            if len(error_categories[error_type]["examples"]) < 3:
                error_categories[error_type]["examples"].append(
                    {
                        "file_index": result.get("file_index"),
                        "file_name": file_name,
                        "error": error,
                    }
                )

            # Track errors by file type
            if file_ext not in file_type_errors:
                file_type_errors[file_ext] = 0
            file_type_errors[file_ext] += 1

        return {
            "total_errors": len(failed_results),
            "error_categories": error_categories,
            "file_type_errors": file_type_errors,
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
            "most_problematic_file_type": (
                max(file_type_errors.items(), key=lambda x: x[1])[0]
                if file_type_errors
                else None
            ),
        }

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error message into error type."""
        error_lower = error_message.lower()

        if "text extraction" in error_lower or "extract" in error_lower:
            return "TEXT_EXTRACTION_ERROR"
        elif "parsing" in error_lower or "parse" in error_lower:
            return "LLM_PARSING_ERROR"
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
        elif "file" in error_lower and (
            "corrupt" in error_lower or "cannot read" in error_lower
        ):
            return "FILE_CORRUPTION_ERROR"
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
                if batch_results["success_rate"] > 0.9
                else (
                    "good"
                    if batch_results["success_rate"] > 0.8
                    else "fair" if batch_results["success_rate"] > 0.6 else "poor"
                )
            ),
            "processing_speed": (
                "fast"
                if successful_items / total_time > 2
                else "moderate" if successful_items / total_time > 1 else "slow"
            ),
            "recommendations": [],
        }

        # Generate recommendations
        if batch_results["success_rate"] < 0.7:
            insights["recommendations"].append(
                "Consider reviewing file quality and LLM configuration"
            )

        if successful_items / total_time < 1:
            insights["recommendations"].append(
                "Consider optimizing text extraction or increasing timeout settings"
            )

        if (
            batch_results["processing_summary"]["total_retries"]
            > successful_items * 0.15
        ):
            insights["recommendations"].append(
                "High retry rate detected - check system resources and network stability"
            )

        return insights

    def _analyze_file_types(
        self, valid_files: List[str], batch_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze processing performance by file type."""
        file_type_stats = {}

        # Initialize stats for all file types
        for file_path in valid_files:
            ext = Path(file_path).suffix.lower()
            if ext not in file_type_stats:
                file_type_stats[ext] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "avg_processing_time": 0,
                    "total_processing_time": 0,
                }
            file_type_stats[ext]["total"] += 1

        # Analyze successful results
        for result in batch_results.get("successful_results", []):
            file_name = result.get("file_name", "")
            ext = Path(file_name).suffix.lower()
            if ext in file_type_stats:
                file_type_stats[ext]["successful"] += 1
                file_type_stats[ext]["total_processing_time"] += result.get(
                    "processing_time", 0
                )

        # Analyze failed results
        for result in batch_results.get("failed_results", []):
            file_name = result.get("file_name", "")
            ext = Path(file_name).suffix.lower()
            if ext in file_type_stats:
                file_type_stats[ext]["failed"] += 1
                file_type_stats[ext]["total_processing_time"] += result.get(
                    "processing_time", 0
                )

        # Calculate averages and success rates
        for ext, stats in file_type_stats.items():
            if stats["total"] > 0:
                stats["success_rate"] = stats["successful"] / stats["total"]
                stats["avg_processing_time"] = (
                    stats["total_processing_time"] / stats["total"]
                )

        return file_type_stats

    def get_processing_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current processing status for a session."""
        return progress_tracker.get_session_status(session_id)

    def get_processing_errors(
        self, session_id: str, limit: int = 50
    ) -> Optional[List[Dict[str, Any]]]:
        """Get processing errors for a session."""
        return progress_tracker.get_session_errors(session_id, limit)

    def resume_processing(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Resume a paused or failed processing session."""
        return progress_tracker.resume_session(session_id)

    def stop_processing(self, session_id: str) -> bool:
        """Stop current processing."""
        batch_processor.stop_processing()
        return progress_tracker.pause_session(session_id, "Stopped by user request")


# Global instance
enhanced_multiple_resume_parser = EnhancedMultipleResumeParserWithTracking()
