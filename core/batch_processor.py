"""
Enhanced Batch Processor for Resume Parsing

This module provides robust batch processing capabilities with error handling,
recovery mechanisms, and progress tracking for both Excel and multiple resume parsing.
"""

import asyncio
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from core.progress_tracker import (
    progress_tracker,
    OperationType,
    ProcessingStatus,
    ErrorSeverity,
)
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("batch_processor")


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    batch_size: int = 50
    max_workers: int = 4
    timeout_per_item: int = 60  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    checkpoint_interval: int = 100
    error_threshold: float = 0.1  # 10% error rate threshold
    memory_threshold: int = 1000  # MB
    enable_recovery: bool = True
    recovery_delay: int = 5  # seconds


class BatchProcessor:
    """
    Enhanced batch processor with error handling, recovery, and progress tracking.

    Features:
    - Parallel processing with configurable workers
    - Automatic error handling and recovery
    - Progress tracking and checkpointing
    - Memory management
    - Graceful shutdown
    """

    def __init__(self, config: BatchConfig = None):
        """
        Initialize batch processor.

        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        self.is_running = False
        self.should_stop = False
        self.current_session_id: Optional[str] = None

        # Processing state
        self.processed_items = []
        self.failed_items = []
        self.retry_queue = []

        logger.info(f"Batch Processor initialized with config: {self.config}")

    async def process_excel_batch(
        self,
        file_path: str,
        excel_data: List[Dict[str, Any]],
        user_id: str,
        username: str,
        parser_function: Callable,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process Excel data in batches with error handling and recovery.

        Args:
            file_path: Path to Excel file
            excel_data: Excel data to process
            user_id: User ID
            username: Username
            parser_function: Function to parse individual rows
            session_id: Existing session ID for resuming

        Returns:
            Processing results with detailed metrics
        """
        operation_type = OperationType.EXCEL_PARSING

        # Create or resume session
        if session_id:
            resume_data = progress_tracker.resume_session(session_id)
            if not resume_data:
                raise ValueError(f"Cannot resume session {session_id}")
            self.current_session_id = session_id
            start_index = resume_data["last_processed_index"]
        else:
            self.current_session_id = progress_tracker.create_session(
                operation_type=operation_type,
                user_id=user_id,
                username=username,
                file_name=Path(file_path).name,
                total_items=len(excel_data),
                configuration=self.config.__dict__,
            )
            start_index = 0

        # Start session
        progress_tracker.start_session(self.current_session_id)

        try:
            results = await self._process_items_in_batches(
                items=excel_data[start_index:],
                processor_function=parser_function,
                start_index=start_index,
                item_type="excel_row",
            )

            # Complete session
            progress_tracker.complete_session(self.current_session_id, results)

            return results

        except Exception as e:
            error_msg = f"Batch processing failed: {str(e)}"
            logger.error(error_msg)
            progress_tracker.fail_session(self.current_session_id, error_msg)
            raise

    async def process_resume_batch(
        self,
        resume_files: List[str],
        user_id: str,
        username: str,
        parser_function: Callable,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process multiple resume files in batches.

        Args:
            resume_files: List of resume file paths
            user_id: User ID
            username: Username
            parser_function: Function to parse individual resumes
            session_id: Existing session ID for resuming

        Returns:
            Processing results with detailed metrics
        """
        operation_type = OperationType.MULTIPLE_RESUME_PARSING

        # Create or resume session
        if session_id:
            resume_data = progress_tracker.resume_session(session_id)
            if not resume_data:
                raise ValueError(f"Cannot resume session {session_id}")
            self.current_session_id = session_id
            start_index = resume_data["last_processed_index"]
        else:
            self.current_session_id = progress_tracker.create_session(
                operation_type=operation_type,
                user_id=user_id,
                username=username,
                total_items=len(resume_files),
                configuration=self.config.__dict__,
            )
            start_index = 0

        # Start session
        progress_tracker.start_session(self.current_session_id)

        try:
            results = await self._process_items_in_batches(
                items=resume_files[start_index:],
                processor_function=parser_function,
                start_index=start_index,
                item_type="resume_file",
            )

            # Complete session
            progress_tracker.complete_session(self.current_session_id, results)

            return results

        except Exception as e:
            error_msg = f"Resume batch processing failed: {str(e)}"
            logger.error(error_msg)
            progress_tracker.fail_session(self.current_session_id, error_msg)
            raise

    async def _process_items_in_batches(
        self,
        items: List[Any],
        processor_function: Callable,
        start_index: int = 0,
        item_type: str = "item",
    ) -> Dict[str, Any]:
        """
        Process items in batches with error handling and recovery.

        Args:
            items: Items to process
            processor_function: Function to process individual items
            start_index: Starting index for processing
            item_type: Type of items being processed

        Returns:
            Processing results
        """
        self.is_running = True
        self.should_stop = False

        # Initialize results
        successful_results = []
        failed_results = []
        skipped_results = []
        duplicate_results = []

        total_items = len(items)
        processed_count = 0

        try:
            # Process items in batches
            for batch_start in range(0, total_items, self.config.batch_size):
                if self.should_stop:
                    logger.info("Processing stopped by user request")
                    break

                batch_end = min(batch_start + self.config.batch_size, total_items)
                batch_items = items[batch_start:batch_end]

                logger.info(
                    f"Processing batch {batch_start}-{batch_end} of {total_items}"
                )

                # Process batch with parallel workers
                batch_results = await self._process_batch_parallel(
                    batch_items=batch_items,
                    processor_function=processor_function,
                    batch_start_index=start_index + batch_start,
                    item_type=item_type,
                )

                # Collect results
                for result in batch_results:
                    if result["status"] == "success":
                        successful_results.append(result)
                    elif result["status"] == "failed":
                        failed_results.append(result)
                    elif result["status"] == "skipped":
                        skipped_results.append(result)
                    elif result["status"] == "duplicate":
                        duplicate_results.append(result)

                processed_count += len(batch_items)

                # Update progress
                progress_tracker.update_progress(
                    session_id=self.current_session_id,
                    processed_count=len(batch_items),
                    successful_count=len(
                        [r for r in batch_results if r["status"] == "success"]
                    ),
                    failed_count=len(
                        [r for r in batch_results if r["status"] == "failed"]
                    ),
                    skipped_count=len(
                        [r for r in batch_results if r["status"] == "skipped"]
                    ),
                    duplicate_count=len(
                        [r for r in batch_results if r["status"] == "duplicate"]
                    ),
                )

                # Check error threshold
                error_rate = (
                    len(failed_results) / processed_count if processed_count > 0 else 0
                )
                if error_rate > self.config.error_threshold and processed_count > 100:
                    error_msg = f"Error rate {error_rate:.2%} exceeds threshold {self.config.error_threshold:.2%}"
                    logger.warning(error_msg)

                    # Pause session for manual review
                    progress_tracker.pause_session(self.current_session_id, error_msg)

                    if not self.config.enable_recovery:
                        raise Exception(error_msg)

                    # Wait for recovery
                    logger.info(
                        f"Waiting {self.config.recovery_delay} seconds before recovery"
                    )
                    await asyncio.sleep(self.config.recovery_delay)

                # Memory management
                if processed_count % 500 == 0:  # Check memory every 500 items
                    await self._manage_memory()

                # Small delay between batches to prevent overwhelming
                await asyncio.sleep(0.1)

            # Process retry queue
            if self.retry_queue and not self.should_stop:
                logger.info(
                    f"Processing {len(self.retry_queue)} items from retry queue"
                )
                retry_results = await self._process_retry_queue(processor_function)

                # Merge retry results
                for result in retry_results:
                    if result["status"] == "success":
                        successful_results.append(result)
                    else:
                        failed_results.append(result)

            # Compile final results
            results = {
                "status": "completed" if not self.should_stop else "cancelled",
                "total_items": total_items,
                "processed_items": processed_count,
                "successful_items": len(successful_results),
                "failed_items": len(failed_results),
                "skipped_items": len(skipped_results),
                "duplicate_items": len(duplicate_results),
                "success_rate": (
                    len(successful_results) / processed_count
                    if processed_count > 0
                    else 0
                ),
                "successful_results": successful_results,
                "failed_results": failed_results,
                "skipped_results": skipped_results,
                "duplicate_results": duplicate_results,
                "processing_summary": {
                    "total_processing_time": 0,  # Will be calculated by tracker
                    "average_processing_time": 0,
                    "items_per_second": 0,
                    "peak_memory_usage": 0,
                    "total_retries": sum(
                        r.get("retry_count", 0) for r in failed_results
                    ),
                },
            }

            return results

        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            logger.error(traceback.format_exc())

            # Add error to session
            progress_tracker.add_error(
                session_id=self.current_session_id,
                error_type="BATCH_PROCESSING_ERROR",
                error_message=str(e),
                severity=ErrorSeverity.CRITICAL,
                stack_trace=traceback.format_exc(),
            )

            raise

        finally:
            self.is_running = False

    async def _process_batch_parallel(
        self,
        batch_items: List[Any],
        processor_function: Callable,
        batch_start_index: int,
        item_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of items in parallel.

        Args:
            batch_items: Items in the batch
            processor_function: Function to process items
            batch_start_index: Starting index of the batch
            item_type: Type of items

        Returns:
            List of processing results
        """
        results = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_item = {}
            for i, item in enumerate(batch_items):
                item_index = batch_start_index + i
                future = executor.submit(
                    self._process_single_item,
                    item=item,
                    processor_function=processor_function,
                    item_index=item_index,
                    item_type=item_type,
                )
                future_to_item[future] = (item, item_index)

            # Collect results as they complete
            for future in as_completed(
                future_to_item, timeout=self.config.timeout_per_item * len(batch_items)
            ):
                item, item_index = future_to_item[future]

                try:
                    result = future.result()
                    results.append(result)

                except Exception as e:
                    error_msg = (
                        f"Failed to process {item_type} at index {item_index}: {str(e)}"
                    )
                    logger.error(error_msg)

                    # Add to retry queue if retries are enabled
                    if self.config.max_retries > 0:
                        self.retry_queue.append(
                            {
                                "item": item,
                                "item_index": item_index,
                                "processor_function": processor_function,
                                "retry_count": 0,
                                "last_error": str(e),
                            }
                        )

                    # Add error to session
                    progress_tracker.add_error(
                        session_id=self.current_session_id,
                        error_type="ITEM_PROCESSING_ERROR",
                        error_message=error_msg,
                        severity=ErrorSeverity.MEDIUM,
                        item_index=item_index,
                        stack_trace=traceback.format_exc(),
                    )

                    results.append(
                        {
                            "status": "failed",
                            "item_index": item_index,
                            "error": error_msg,
                            "item_data": str(item)[:100] if item else None,
                        }
                    )

        return results

    def _process_single_item(
        self, item: Any, processor_function: Callable, item_index: int, item_type: str
    ) -> Dict[str, Any]:
        """
        Process a single item with error handling.

        Args:
            item: Item to process
            processor_function: Function to process the item
            item_index: Index of the item
            item_type: Type of item

        Returns:
            Processing result
        """
        start_time = time.time()

        try:
            # Process the item
            result = processor_function(item, item_index)

            processing_time = time.time() - start_time

            # Determine status based on result
            if isinstance(result, dict):
                status = result.get("status", "success")
                if status in ["success", "duplicate", "skipped"]:
                    return {
                        "status": status,
                        "item_index": item_index,
                        "processing_time": processing_time,
                        "result": result,
                        "item_type": item_type,
                    }
                else:
                    return {
                        "status": "failed",
                        "item_index": item_index,
                        "processing_time": processing_time,
                        "error": result.get("error", "Unknown error"),
                        "item_type": item_type,
                    }
            else:
                # Assume success if result is not a dict
                return {
                    "status": "success",
                    "item_index": item_index,
                    "processing_time": processing_time,
                    "result": result,
                    "item_type": item_type,
                }

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing {item_type} at index {item_index}: {str(e)}"

            return {
                "status": "failed",
                "item_index": item_index,
                "processing_time": processing_time,
                "error": error_msg,
                "item_type": item_type,
                "stack_trace": traceback.format_exc(),
            }

    async def _process_retry_queue(
        self, processor_function: Callable
    ) -> List[Dict[str, Any]]:
        """
        Process items in the retry queue.

        Args:
            processor_function: Function to process items

        Returns:
            List of retry results
        """
        results = []
        items_to_retry = self.retry_queue.copy()
        self.retry_queue.clear()

        for retry_item in items_to_retry:
            if retry_item["retry_count"] >= self.config.max_retries:
                # Max retries reached
                results.append(
                    {
                        "status": "failed",
                        "item_index": retry_item["item_index"],
                        "error": f"Max retries ({self.config.max_retries}) reached. Last error: {retry_item['last_error']}",
                        "retry_count": retry_item["retry_count"],
                    }
                )
                continue

            # Wait before retry
            await asyncio.sleep(
                self.config.retry_delay * (retry_item["retry_count"] + 1)
            )

            try:
                result = self._process_single_item(
                    item=retry_item["item"],
                    processor_function=processor_function,
                    item_index=retry_item["item_index"],
                    item_type="retry_item",
                )

                if result["status"] == "success":
                    result["retry_count"] = retry_item["retry_count"] + 1
                    results.append(result)
                else:
                    # Retry failed, add back to queue if retries remaining
                    retry_item["retry_count"] += 1
                    retry_item["last_error"] = result.get("error", "Unknown error")

                    if retry_item["retry_count"] < self.config.max_retries:
                        self.retry_queue.append(retry_item)
                    else:
                        result["retry_count"] = retry_item["retry_count"]
                        results.append(result)

            except Exception as e:
                # Retry failed with exception
                retry_item["retry_count"] += 1
                retry_item["last_error"] = str(e)

                if retry_item["retry_count"] < self.config.max_retries:
                    self.retry_queue.append(retry_item)
                else:
                    results.append(
                        {
                            "status": "failed",
                            "item_index": retry_item["item_index"],
                            "error": f"Retry failed: {str(e)}",
                            "retry_count": retry_item["retry_count"],
                        }
                    )

        # Process remaining retries recursively if any
        if self.retry_queue:
            additional_results = await self._process_retry_queue(processor_function)
            results.extend(additional_results)

        return results

    async def _manage_memory(self):
        """Manage memory usage during processing."""
        try:
            import psutil

            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB

            if memory_usage > self.config.memory_threshold:
                logger.warning(f"High memory usage: {memory_usage:.2f} MB")

                # Clear processed results to free memory
                if len(self.processed_items) > 1000:
                    self.processed_items = self.processed_items[-500:]  # Keep last 500

                # Force garbage collection
                import gc

                gc.collect()

                # Brief pause to allow memory cleanup
                await asyncio.sleep(1)

        except ImportError:
            # psutil not available, skip memory management
            pass
        except Exception as e:
            logger.warning(f"Memory management error: {str(e)}")

    def stop_processing(self):
        """Stop the current processing operation."""
        self.should_stop = True
        if self.current_session_id:
            progress_tracker.pause_session(
                self.current_session_id, "Processing stopped by user request"
            )
        logger.info("Processing stop requested")

    def get_current_status(self) -> Optional[Dict[str, Any]]:
        """Get current processing status."""
        if self.current_session_id:
            return progress_tracker.get_session_status(self.current_session_id)
        return None


# Global instance
batch_processor = BatchProcessor()
