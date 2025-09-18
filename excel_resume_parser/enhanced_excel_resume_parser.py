"""
Enhanced Excel Resume Parser

This module provides an advanced Excel resume parser that can handle any type of columns
and data formats with intelligent field mapping, validation, and cleaning.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio

# Import the existing resume parser and related modules
from multipleresumepraser.main import ResumeParser, Resume
from mangodatabase.operations import ResumeOperations
from mangodatabase.duplicate_detection import DuplicateDetectionOperations
from embeddings.vectorizer import AddUserDataVectorizer
from core.custom_logger import CustomLogger
from core.llm_config import LLMConfigManager, LLMProvider
from core.config import AppConfig

# Import Excel processor and our enhanced components
from .excel_processor import ExcelProcessor
from .enhanced_column_mapper import EnhancedColumnMapper
from .enhanced_data_detector import EnhancedDataTypeDetector
from .dynamic_field_extractor import DynamicFieldExtractor
from .enhanced_text_formatter import EnhancedTextFormatter
from .data_validation_cleaner import DataValidationCleaner

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("enhanced_excel_resume_parser")


class EnhancedExcelResumeParser:
    """
    Enhanced Excel-based resume parser that can handle any type of columns and data formats
    with intelligent mapping, validation, and cleaning capabilities.
    """

    def __init__(self, llm_provider: str = None, api_keys: List[str] = None):
        """
        Initialize Enhanced Excel Resume Parser.

        Args:
            llm_provider: LLM provider to use for parsing (auto-detected if None)
            api_keys: API keys for the LLM provider
        """
        logger.info("Initializing Enhanced Excel Resume Parser")

        # Auto-detect LLM provider if not provided
        if llm_provider is None:
            llm_provider = AppConfig.LLM_PROVIDER
            logger.info(f"Auto-detected LLM provider from config: {llm_provider}")

        # Initialize all components
        self.excel_processor = ExcelProcessor()
        self.column_mapper = EnhancedColumnMapper()
        self.data_detector = EnhancedDataTypeDetector()
        self.field_extractor = DynamicFieldExtractor()
        self.text_formatter = EnhancedTextFormatter()
        self.data_validator = DataValidationCleaner()

        # Initialize resume parser with the same configuration
        self.resume_parser = ResumeParser(llm_provider=llm_provider, api_keys=api_keys)

        # Initialize database operations
        self.vectorizer = AddUserDataVectorizer()

        # LLM manager for configuration
        self.llm_manager = LLMConfigManager()

        # Processing statistics
        self.processing_stats = {
            "total_processed": 0,
            "successful_parsing": 0,
            "validation_errors": 0,
            "data_quality_scores": [],
            "processing_times": [],
            "column_mappings_learned": {},
        }

        logger.info("Enhanced Excel Resume Parser initialized successfully")

    def process_excel_file(
        self,
        file_path: str,
        validation_level: str = "standard",
        cleaning_aggressive: bool = False,
        include_quality_scores: bool = True,
        batch_size: int = 10,
    ) -> Dict[str, Any]:
        """
        Process an Excel file with enhanced capabilities.

        Args:
            file_path: Path to the Excel file
            validation_level: Level of validation ("basic", "standard", "strict")
            cleaning_aggressive: Whether to apply aggressive data cleaning
            include_quality_scores: Whether to calculate quality scores
            batch_size: Number of rows to process in each batch

        Returns:
            Dictionary containing processing results and statistics
        """
        logger.info(f"Starting enhanced processing of Excel file: {file_path}")
        start_time = time.time()

        processing_result = {
            "file_path": file_path,
            "processing_start_time": datetime.now().isoformat(),
            "total_rows": 0,
            "successfully_processed": 0,
            "failed_rows": [],
            "column_analysis": {},
            "data_quality_summary": {},
            "parsed_resumes": [],
            "processing_statistics": {},
        }

        try:
            # Step 1: Read and analyze Excel file
            logger.info("Step 1: Reading and analyzing Excel file structure")
            excel_data = self.excel_processor.process_excel_file(file_path)

            if not excel_data or "data" not in excel_data:
                raise ValueError("Could not read Excel file or no data found")

            df_data = excel_data["data"]
            processing_result["total_rows"] = len(df_data)
            logger.info(f"Found {len(df_data)} rows to process")

            # Step 2: Intelligent column mapping
            logger.info("Step 2: Performing intelligent column mapping")
            column_mapping_result = self.column_mapper.analyze_and_map_columns(
                df_data.columns.tolist()
            )
            processing_result["column_analysis"] = {
                "original_columns": df_data.columns.tolist(),
                "mapped_fields": column_mapping_result["field_mappings"],
                "mapping_confidence": column_mapping_result["confidence_scores"],
                "unmapped_columns": column_mapping_result["unmapped_columns"],
            }

            # Step 3: Process rows in batches
            logger.info(f"Step 3: Processing rows in batches of {batch_size}")
            total_rows = len(df_data)
            quality_scores = []
            validation_results_all = []

            for batch_start in range(0, total_rows, batch_size):
                batch_end = min(batch_start + batch_size, total_rows)
                batch_df = df_data.iloc[batch_start:batch_end]

                logger.info(
                    f"Processing batch {batch_start//batch_size + 1}: rows {batch_start+1} to {batch_end}"
                )

                batch_results = self._process_batch(
                    batch_df,
                    column_mapping_result["field_mappings"],
                    validation_level,
                    cleaning_aggressive,
                    include_quality_scores,
                )

                # Accumulate results
                processing_result["successfully_processed"] += batch_results[
                    "successful_count"
                ]
                processing_result["failed_rows"].extend(batch_results["failed_rows"])
                processing_result["parsed_resumes"].extend(
                    batch_results["parsed_resumes"]
                )
                quality_scores.extend(batch_results["quality_scores"])
                validation_results_all.extend(batch_results["validation_results"])

            # Step 4: Generate comprehensive quality summary
            if include_quality_scores and quality_scores:
                logger.info("Step 4: Generating data quality summary")
                processing_result["data_quality_summary"] = (
                    self._generate_quality_summary(
                        quality_scores, validation_results_all
                    )
                )

            # Step 5: Update processing statistics
            end_time = time.time()
            processing_time = end_time - start_time

            processing_result["processing_statistics"] = {
                "total_processing_time_seconds": round(processing_time, 2),
                "average_time_per_row": round(processing_time / total_rows, 3),
                "success_rate": round(
                    (processing_result["successfully_processed"] / total_rows) * 100, 2
                ),
                "validation_pass_rate": self._calculate_validation_pass_rate(
                    validation_results_all
                ),
                "average_data_quality_score": (
                    round(sum(quality_scores) / len(quality_scores), 1)
                    if quality_scores
                    else 0
                ),
            }

            # Update global statistics
            self.processing_stats["total_processed"] += total_rows
            self.processing_stats["successful_parsing"] += processing_result[
                "successfully_processed"
            ]
            self.processing_stats["processing_times"].append(processing_time)
            self.processing_stats["data_quality_scores"].extend(quality_scores)

            logger.info(f"Enhanced Excel processing completed successfully")
            logger.info(
                f"Processed {processing_result['successfully_processed']}/{total_rows} rows successfully"
            )
            logger.info(
                f"Average data quality score: {processing_result['processing_statistics']['average_data_quality_score']}"
            )

        except Exception as e:
            logger.error(f"Error in enhanced Excel processing: {e}")
            processing_result["error"] = str(e)
            processing_result["processing_statistics"] = {
                "total_processing_time_seconds": time.time() - start_time,
                "success_rate": 0,
                "error": str(e),
            }

        processing_result["processing_end_time"] = datetime.now().isoformat()
        return processing_result

    def _process_batch(
        self,
        batch_df,
        field_mappings: Dict[str, str],
        validation_level: str,
        cleaning_aggressive: bool,
        include_quality_scores: bool,
    ) -> Dict[str, Any]:
        """
        Process a batch of rows with enhanced capabilities.

        Args:
            batch_df: DataFrame containing the batch of rows
            field_mappings: Dictionary mapping original columns to standard fields
            validation_level: Level of validation to apply
            cleaning_aggressive: Whether to apply aggressive cleaning
            include_quality_scores: Whether to calculate quality scores

        Returns:
            Dictionary containing batch processing results
        """
        batch_result = {
            "successful_count": 0,
            "failed_rows": [],
            "parsed_resumes": [],
            "quality_scores": [],
            "validation_results": [],
        }

        for index, row in batch_df.iterrows():
            try:
                row_start_time = time.time()

                # Step 1: Extract fields from row using dynamic extractor
                extracted_fields = self.field_extractor.extract_fields_from_row(
                    row.to_dict(), field_mappings
                )

                # Step 2: Validate and clean data
                validation_results = []
                cleaned_fields = {}

                for field_name, field_value in extracted_fields.items():
                    # Validate field
                    if validation_level in ["standard", "strict"]:
                        validation_result = self.data_validator.validate_field(
                            field_name, field_value
                        )
                        validation_results.append(validation_result)

                    # Clean field
                    cleaning_result = self.data_validator.clean_field_value(
                        field_name, field_value, cleaning_aggressive
                    )
                    cleaned_fields[field_name] = cleaning_result["cleaned_value"]

                # Step 3: Cross-field validation
                if validation_level == "strict":
                    cross_validation = (
                        self.data_validator.validate_cross_field_consistency(
                            cleaned_fields
                        )
                    )
                    if not cross_validation["is_consistent"]:
                        logger.warning(
                            f"Row {index}: Cross-field validation issues: {cross_validation['issues']}"
                        )

                # Step 4: Calculate data quality score
                quality_score_result = None
                if include_quality_scores:
                    quality_score_result = (
                        self.data_validator.calculate_data_quality_score(
                            cleaned_fields, validation_results
                        )
                    )
                    batch_result["quality_scores"].append(
                        quality_score_result["overall_quality_score"]
                    )

                # Step 5: Format as resume text
                resume_text = self.text_formatter.format_resume_text(
                    cleaned_fields, format_style="professional", include_metadata=True
                )

                # Step 6: Parse with LLM
                if resume_text.strip():
                    parsed_resume = self.resume_parser.parse_resume_text(resume_text)

                    if parsed_resume:
                        # Add processing metadata
                        parsed_resume.processing_metadata = {
                            "source": "enhanced_excel_parser",
                            "original_row_index": int(index),
                            "processing_time_seconds": round(
                                time.time() - row_start_time, 3
                            ),
                            "validation_results": (
                                validation_results
                                if validation_level != "basic"
                                else []
                            ),
                            "data_quality_score": (
                                quality_score_result["overall_quality_score"]
                                if quality_score_result
                                else None
                            ),
                            "data_quality_grade": (
                                quality_score_result["grade"]
                                if quality_score_result
                                else None
                            ),
                            "field_mappings_used": field_mappings,
                            "cleaning_applied": any(
                                cleaning_result.get("cleaning_applied", [])
                                for cleaning_result in [
                                    self.data_validator.clean_field_value(
                                        fn, fv, cleaning_aggressive
                                    )
                                    for fn, fv in extracted_fields.items()
                                ]
                            ),
                        }

                        batch_result["parsed_resumes"].append(parsed_resume)
                        batch_result["successful_count"] += 1
                        batch_result["validation_results"].extend(validation_results)

                    else:
                        batch_result["failed_rows"].append(
                            {
                                "row_index": int(index),
                                "error": "Failed to parse resume with LLM",
                                "extracted_fields": extracted_fields,
                                "resume_text": (
                                    resume_text[:500] + "..."
                                    if len(resume_text) > 500
                                    else resume_text
                                ),
                            }
                        )
                else:
                    batch_result["failed_rows"].append(
                        {
                            "row_index": int(index),
                            "error": "No meaningful resume text generated",
                            "extracted_fields": extracted_fields,
                        }
                    )

            except Exception as e:
                logger.error(f"Error processing row {index}: {e}")
                batch_result["failed_rows"].append(
                    {
                        "row_index": int(index),
                        "error": str(e),
                        "raw_data": row.to_dict(),
                    }
                )

        return batch_result

    def _generate_quality_summary(
        self, quality_scores: List[float], validation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive data quality summary."""
        if not quality_scores:
            return {}

        # Quality score statistics
        avg_score = sum(quality_scores) / len(quality_scores)
        min_score = min(quality_scores)
        max_score = max(quality_scores)

        # Grade distribution
        grade_distribution = {"A+": 0, "A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        for score in quality_scores:
            if score >= 90:
                grade_distribution["A+"] += 1
            elif score >= 80:
                grade_distribution["A"] += 1
            elif score >= 70:
                grade_distribution["B"] += 1
            elif score >= 60:
                grade_distribution["C"] += 1
            elif score >= 50:
                grade_distribution["D"] += 1
            else:
                grade_distribution["F"] += 1

        # Validation statistics
        validation_stats = {}
        if validation_results:
            total_validations = len(validation_results)
            passed_validations = sum(
                1 for result in validation_results if result.get("is_valid", False)
            )

            validation_stats = {
                "total_field_validations": total_validations,
                "passed_validations": passed_validations,
                "validation_pass_rate": round(
                    (passed_validations / total_validations) * 100, 2
                ),
                "common_validation_errors": self._get_common_validation_errors(
                    validation_results
                ),
            }

        return {
            "quality_score_statistics": {
                "average_score": round(avg_score, 1),
                "minimum_score": round(min_score, 1),
                "maximum_score": round(max_score, 1),
                "standard_deviation": round(
                    (
                        sum((score - avg_score) ** 2 for score in quality_scores)
                        / len(quality_scores)
                    )
                    ** 0.5,
                    1,
                ),
            },
            "grade_distribution": grade_distribution,
            "validation_statistics": validation_stats,
            "recommendations": self._generate_overall_recommendations(
                quality_scores, validation_results
            ),
        }

    def _calculate_validation_pass_rate(
        self, validation_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall validation pass rate."""
        if not validation_results:
            return 100.0

        passed = sum(
            1 for result in validation_results if result.get("is_valid", False)
        )
        return round((passed / len(validation_results)) * 100, 2)

    def _get_common_validation_errors(
        self, validation_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get most common validation errors."""
        error_counts = {}

        for result in validation_results:
            for error in result.get("errors", []):
                if error in error_counts:
                    error_counts[error] += 1
                else:
                    error_counts[error] = 1

        # Sort by frequency and return top 5
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]

        return [
            {
                "error": error,
                "frequency": count,
                "percentage": round((count / len(validation_results)) * 100, 1),
            }
            for error, count in sorted_errors
        ]

    def _generate_overall_recommendations(
        self, quality_scores: List[float], validation_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate overall recommendations for data quality improvement."""
        recommendations = []

        avg_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        if avg_score < 70:
            recommendations.append(
                "Overall data quality is below average - consider data source improvement"
            )

        if validation_results:
            error_rate = sum(
                1 for result in validation_results if not result.get("is_valid", True)
            ) / len(validation_results)
            if error_rate > 0.3:
                recommendations.append(
                    "High validation error rate - implement data entry validation"
                )

        # Check for specific patterns
        low_quality_count = sum(1 for score in quality_scores if score < 50)
        if low_quality_count > len(quality_scores) * 0.2:
            recommendations.append(
                "Many records have very low quality - consider manual data review"
            )

        if not recommendations:
            recommendations.append(
                "Data quality is good - continue current data management practices"
            )

        return recommendations

    def save_parsed_resumes_to_database(
        self,
        parsed_resumes: List[Resume],
        detect_duplicates: bool = True,
        update_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Save parsed resumes to database with duplicate detection.

        Args:
            parsed_resumes: List of parsed Resume objects
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
            resume_ops = ResumeOperations()
            duplicate_ops = (
                DuplicateDetectionOperations() if detect_duplicates else None
            )

            for resume in parsed_resumes:
                try:
                    # Check for duplicates if enabled
                    is_duplicate = False
                    if detect_duplicates and duplicate_ops:
                        duplicate_result = duplicate_ops.check_duplicate(resume)
                        if duplicate_result.get("is_duplicate", False):
                            is_duplicate = True
                            save_result["duplicates_found"] += 1

                            if update_existing:
                                # Update existing record
                                existing_id = duplicate_result.get("duplicate_id")
                                if existing_id:
                                    resume_ops.update_resume(existing_id, resume.dict())
                                    save_result["updated_records"] += 1
                            continue

                    # Save new resume
                    if not is_duplicate:
                        saved_resume = resume_ops.create_resume(resume)
                        if saved_resume:
                            save_result["successfully_saved"] += 1

                            # Add to vectorizer for search
                            try:
                                self.vectorizer.add_user_data(saved_resume)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to add resume to vectorizer: {e}"
                                )

                except Exception as e:
                    logger.error(f"Error saving individual resume: {e}")
                    save_result["errors"].append(str(e))

        except Exception as e:
            logger.error(f"Error in database save operation: {e}")
            save_result["errors"].append(f"Database operation error: {str(e)}")

        logger.info(
            f"Database save completed: {save_result['successfully_saved']} saved, {save_result['duplicates_found']} duplicates"
        )
        return save_result

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = self.processing_stats.copy()

        if stats["processing_times"]:
            stats["average_processing_time"] = sum(stats["processing_times"]) / len(
                stats["processing_times"]
            )
            stats["total_processing_time"] = sum(stats["processing_times"])

        if stats["data_quality_scores"]:
            stats["average_quality_score"] = sum(stats["data_quality_scores"]) / len(
                stats["data_quality_scores"]
            )

        if stats["total_processed"] > 0:
            stats["overall_success_rate"] = (
                stats["successful_parsing"] / stats["total_processed"]
            ) * 100

        return stats

    def export_processing_report(
        self, output_path: str, processing_result: Dict[str, Any]
    ) -> str:
        """
        Export a comprehensive processing report.

        Args:
            output_path: Path to save the report
            processing_result: Result from process_excel_file

        Returns:
            Path to the saved report file
        """
        try:
            report = {
                "report_generated": datetime.now().isoformat(),
                "excel_file_processed": processing_result.get("file_path", "Unknown"),
                "processing_summary": processing_result.get(
                    "processing_statistics", {}
                ),
                "data_quality_analysis": processing_result.get(
                    "data_quality_summary", {}
                ),
                "column_mapping_analysis": processing_result.get("column_analysis", {}),
                "failed_rows_analysis": {
                    "total_failed": len(processing_result.get("failed_rows", [])),
                    "failure_reasons": self._analyze_failure_reasons(
                        processing_result.get("failed_rows", [])
                    ),
                },
                "overall_statistics": self.get_processing_statistics(),
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"Processing report exported to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting processing report: {e}")
            raise

    def _analyze_failure_reasons(
        self, failed_rows: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Analyze reasons for row processing failures."""
        reasons = {}

        for failure in failed_rows:
            error = failure.get("error", "Unknown error")
            if error in reasons:
                reasons[error] += 1
            else:
                reasons[error] = 1

        return reasons
