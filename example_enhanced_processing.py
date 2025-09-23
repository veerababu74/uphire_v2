"""
Example Implementation of Enhanced Resume Processing System

This script demonstrates how to use the enhanced resume processing system
with comprehensive tracking, error handling, and recovery capabilities.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Import the enhanced processing modules
from excel_resume_parser.enhanced_excel_parser_with_tracking import (
    enhanced_excel_parser,
)
from multipleresumepraser.enhanced_multiple_resume_parser_with_tracking import (
    enhanced_multiple_resume_parser,
)
from core.progress_tracker import progress_tracker
from core.batch_processor import BatchConfig


class EnhancedProcessingExample:
    """Example class demonstrating enhanced resume processing capabilities."""

    def __init__(self):
        """Initialize the example processor."""
        self.results_log = []

    async def example_excel_processing(self):
        """
        Example: Process Excel file with comprehensive tracking.

        This example shows:
        - How to process Excel files with real-time tracking
        - Error handling and recovery
        - Progress monitoring
        - Result analysis
        """
        print("🔵 Starting Excel Processing Example")
        print("-" * 50)

        # Example Excel file path (replace with your actual file)
        excel_file = "example_resumes.xlsx"  # This file should exist in your project

        if not Path(excel_file).exists():
            print(f"❌ Excel file not found: {excel_file}")
            print("Please ensure you have an Excel file to process.")
            return

        try:
            # Start processing with tracking
            print(f"📁 Processing Excel file: {excel_file}")

            result = await enhanced_excel_parser.process_excel_file_with_tracking(
                file_path=excel_file,
                base_user_id="example_user_excel",
                base_username="excel_demo_user",
                sheet_name=None,  # Process first sheet
                cleanup_file=False,  # Keep file for demo
                session_id=None,  # Create new session
            )

            # Display results
            self._display_excel_results(result)

            # Monitor session if still active
            session_id = result.get("session_id")
            if session_id:
                await self._monitor_session_progress(session_id)

            self.results_log.append(
                {"type": "excel_processing", "result": result, "timestamp": time.time()}
            )

        except Exception as e:
            print(f"❌ Excel processing failed: {str(e)}")

    async def example_multiple_resume_processing(self):
        """
        Example: Process multiple resume files with comprehensive tracking.

        This example shows:
        - How to process multiple resume files
        - File validation and error handling
        - Progress monitoring for batch operations
        - Performance analysis
        """
        print("\n🔵 Starting Multiple Resume Processing Example")
        print("-" * 50)

        # Example resume files (replace with your actual files)
        resume_files = [
            "sample_resume_1.pdf",
            "sample_resume_2.docx",
            "sample_resume_3.pdf",
        ]

        # Filter existing files
        existing_files = [f for f in resume_files if Path(f).exists()]

        if not existing_files:
            print("❌ No resume files found. Creating demo scenario...")
            print("In a real scenario, you would have actual resume files.")

            # Create demo scenario with mock processing
            await self._demo_multiple_resume_scenario()
            return

        try:
            print(f"📁 Processing {len(existing_files)} resume files")

            result = await enhanced_multiple_resume_parser.process_multiple_resumes_with_tracking(
                resume_files=existing_files,
                base_user_id="example_user_resumes",
                base_username="resume_demo_user",
                cleanup_files=False,  # Keep files for demo
                session_id=None,  # Create new session
            )

            # Display results
            self._display_resume_results(result)

            # Monitor session if still active
            session_id = result.get("session_id")
            if session_id:
                await self._monitor_session_progress(session_id)

            self.results_log.append(
                {
                    "type": "multiple_resume_processing",
                    "result": result,
                    "timestamp": time.time(),
                }
            )

        except Exception as e:
            print(f"❌ Multiple resume processing failed: {str(e)}")

    async def example_session_recovery(self):
        """
        Example: Demonstrate session recovery capabilities.

        This example shows:
        - How to resume interrupted sessions
        - Error recovery mechanisms
        - Session management
        """
        print("\n🔵 Session Recovery Example")
        print("-" * 50)

        # Get active sessions
        active_sessions = progress_tracker.list_active_sessions()

        if not active_sessions:
            print("ℹ️ No active sessions found.")
            print("Run processing examples first to create sessions.")
            return

        print(f"📋 Found {len(active_sessions)} active sessions:")

        for i, session in enumerate(active_sessions):
            status = session.get("status", "unknown")
            progress = session.get("completion_percentage", 0)

            print(f"  {i+1}. Session: {session['session_id'][:8]}...")
            print(f"     Status: {status}")
            print(f"     Progress: {progress:.1f}%")
            print(f"     Operation: {session.get('operation_type', 'unknown')}")

            # Demonstrate recovery for paused/failed sessions
            if status in ["paused", "failed"]:
                print(f"     🔄 Attempting to resume session...")

                try:
                    if session["operation_type"] == "excel_parsing":
                        resume_data = enhanced_excel_parser.resume_processing(
                            session["session_id"]
                        )
                    else:
                        resume_data = enhanced_multiple_resume_parser.resume_processing(
                            session["session_id"]
                        )

                    if resume_data:
                        print(
                            f"     ✅ Session can be resumed from item {resume_data.get('last_processed_index', 0)}"
                        )
                    else:
                        print(f"     ❌ Session cannot be resumed")

                except Exception as e:
                    print(f"     ❌ Resume attempt failed: {str(e)}")

            print()

    async def example_error_analysis(self):
        """
        Example: Analyze errors from processing sessions.

        This example shows:
        - How to retrieve and analyze errors
        - Error categorization
        - Performance insights
        """
        print("\n🔵 Error Analysis Example")
        print("-" * 50)

        active_sessions = progress_tracker.list_active_sessions()

        if not active_sessions:
            print("ℹ️ No sessions found for error analysis.")
            return

        total_errors = 0
        error_categories = {}

        for session in active_sessions:
            session_id = session["session_id"]
            errors = progress_tracker.get_session_errors(session_id, limit=20)

            if errors:
                print(f"📊 Session {session_id[:8]}... has {len(errors)} errors:")

                for error in errors[:5]:  # Show first 5 errors
                    error_type = error.get("error_type", "Unknown")
                    severity = error.get("severity", "medium")

                    print(
                        f"  - [{severity.upper()}] {error_type}: {error['error_message'][:80]}..."
                    )

                    # Categorize errors
                    if error_type not in error_categories:
                        error_categories[error_type] = 0
                    error_categories[error_type] += 1

                total_errors += len(errors)
                print()

        # Display error summary
        if total_errors > 0:
            print("📈 Error Summary:")
            print(f"  Total Errors: {total_errors}")
            print("  Error Categories:")

            for error_type, count in sorted(
                error_categories.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"    - {error_type}: {count}")
        else:
            print("✅ No errors found in active sessions!")

    async def example_performance_monitoring(self):
        """
        Example: Monitor processing performance.

        This example shows:
        - How to monitor real-time performance
        - Performance metrics analysis
        - System health indicators
        """
        print("\n🔵 Performance Monitoring Example")
        print("-" * 50)

        active_sessions = progress_tracker.list_active_sessions()

        if not active_sessions:
            print("ℹ️ No active sessions to monitor.")
            return

        print("⏱️ Monitoring active sessions for 30 seconds...")
        print("Press Ctrl+C to stop monitoring early.")

        try:
            for i in range(6):  # Monitor for 30 seconds (6 * 5 seconds)
                print(f"\n📊 Update {i+1}/6:")

                for session in active_sessions:
                    session_id = session["session_id"]
                    current_status = progress_tracker.get_session_status(session_id)

                    if current_status:
                        metrics = current_status.get("metrics", {})

                        print(f"  Session {session_id[:8]}...:")
                        print(
                            f"    Progress: {metrics.get('completion_percentage', 0):.1f}%"
                        )
                        print(
                            f"    Rate: {metrics.get('processing_rate', 0):.2f} items/sec"
                        )
                        print(
                            f"    Processed: {metrics.get('processed_items', 0)}/{metrics.get('total_items', 0)}"
                        )
                        print(
                            f"    Success Rate: {metrics.get('success_rate', 0):.1f}%"
                        )

                        status = current_status.get("status", "unknown")
                        if status in ["completed", "failed", "cancelled"]:
                            print(f"    Status: {status.upper()}")

                if i < 5:  # Don't wait after last iteration
                    await asyncio.sleep(5)

        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user.")

    async def _monitor_session_progress(self, session_id: str, max_updates: int = 10):
        """Monitor a specific session's progress."""
        print(f"\n📊 Monitoring session progress: {session_id[:8]}...")

        for i in range(max_updates):
            status = progress_tracker.get_session_status(session_id)

            if not status:
                print("❌ Session not found")
                break

            metrics = status.get("metrics", {})
            current_status = status.get("status", "unknown")

            print(
                f"  Update {i+1}: {metrics.get('completion_percentage', 0):.1f}% - {current_status}"
            )

            if current_status in ["completed", "failed", "cancelled"]:
                print(f"  ✅ Session {current_status}")
                break

            await asyncio.sleep(2)  # Check every 2 seconds

    async def _demo_multiple_resume_scenario(self):
        """Create a demo scenario for multiple resume processing."""
        print("🔄 Creating demo multiple resume processing scenario...")

        # This would be replaced with actual file processing in a real scenario
        from core.progress_tracker import OperationType

        session_id = progress_tracker.create_session(
            operation_type=OperationType.MULTIPLE_RESUME_PARSING,
            user_id="demo_user",
            username="demo_scenario",
            total_items=5,
            configuration={"demo": True},
        )

        progress_tracker.start_session(session_id)

        # Simulate processing progress
        for i in range(5):
            await asyncio.sleep(1)
            progress_tracker.update_progress(
                session_id=session_id,
                processed_count=1,
                successful_count=1 if i < 4 else 0,
                failed_count=0 if i < 4 else 1,
            )

            print(f"  📄 Demo processing item {i+1}/5...")

        progress_tracker.complete_session(
            session_id,
            {
                "demo_result": "Completed demo scenario",
                "items_processed": 5,
                "success_rate": 80.0,
            },
        )

        print("✅ Demo scenario completed!")

    def _display_excel_results(self, result: Dict[str, Any]):
        """Display Excel processing results in a formatted way."""
        print("\n📊 Excel Processing Results:")
        print("=" * 40)

        if result.get("status") == "success":
            excel_processing = result.get("excel_processing", {})
            database_ops = result.get("database_operations", {})
            metrics = result.get("detailed_metrics", {})

            print(f"✅ Status: SUCCESS")
            print(f"📁 File: {result.get('file_name', 'N/A')}")
            print(f"🆔 Session ID: {result.get('session_id', 'N/A')}")
            print(f"⏱️ Total Time: {result.get('total_processing_time', 0):.2f} seconds")
            print()

            print("📋 Processing Summary:")
            print(f"  Total Rows Found: {excel_processing.get('total_rows_found', 0)}")
            print(f"  Rows Processed: {excel_processing.get('rows_processed', 0)}")
            print(f"  Successful Rows: {excel_processing.get('successful_rows', 0)}")
            print(f"  Failed Rows: {excel_processing.get('failed_rows', 0)}")
            print(f"  Success Rate: {excel_processing.get('success_rate', 0):.1%}")
            print()

            print("💾 Database Operations:")
            print(f"  Successfully Saved: {database_ops.get('saved_successfully', 0)}")
            print(
                f"  Duplicates Detected: {database_ops.get('duplicates_detected', 0)}"
            )
            print(f"  Save Errors: {database_ops.get('save_errors', 0)}")
            print()

            print("⚡ Performance Metrics:")
            print(
                f"  Processing Rate: {metrics.get('processing_rate', 0):.2f} rows/sec"
            )
            print(
                f"  Avg Row Time: {metrics.get('average_row_processing_time', 0):.2f} seconds"
            )

        else:
            print(f"❌ Status: FAILED")
            print(f"Error: {result.get('message', 'Unknown error')}")

    def _display_resume_results(self, result: Dict[str, Any]):
        """Display multiple resume processing results in a formatted way."""
        print("\n📊 Multiple Resume Processing Results:")
        print("=" * 45)

        if result.get("status") == "success":
            file_validation = result.get("file_validation", {})
            resume_processing = result.get("resume_processing", {})
            database_ops = result.get("database_operations", {})
            metrics = result.get("detailed_metrics", {})

            print(f"✅ Status: SUCCESS")
            print(f"🆔 Session ID: {result.get('session_id', 'N/A')}")
            print(f"⏱️ Total Time: {result.get('total_processing_time', 0):.2f} seconds")
            print()

            print("📁 File Validation:")
            print(
                f"  Total Files Provided: {file_validation.get('total_files_provided', 0)}"
            )
            print(f"  Valid Files: {file_validation.get('valid_files', 0)}")
            print(f"  Invalid Files: {file_validation.get('invalid_files', 0)}")
            print()

            print("📋 Processing Summary:")
            print(f"  Files Processed: {resume_processing.get('files_processed', 0)}")
            print(f"  Successful Files: {resume_processing.get('successful_files', 0)}")
            print(f"  Failed Files: {resume_processing.get('failed_files', 0)}")
            print(f"  Success Rate: {resume_processing.get('success_rate', 0):.1%}")
            print()

            print("💾 Database Operations:")
            print(f"  Successfully Saved: {database_ops.get('saved_successfully', 0)}")
            print(
                f"  Duplicates Detected: {database_ops.get('duplicates_detected', 0)}"
            )
            print(f"  Skills Extracted: {database_ops.get('skills_extracted', 0)}")
            print()

            print("⚡ Performance Metrics:")
            print(
                f"  Processing Rate: {metrics.get('processing_rate', 0):.2f} files/sec"
            )
            print(
                f"  Avg File Time: {metrics.get('average_file_processing_time', 0):.2f} seconds"
            )

        else:
            print(f"❌ Status: FAILED")
            print(f"Error: {result.get('message', 'Unknown error')}")

    def display_summary(self):
        """Display a summary of all processing results."""
        print("\n🎯 Processing Summary")
        print("=" * 30)

        if not self.results_log:
            print("No processing operations completed.")
            return

        total_operations = len(self.results_log)
        successful_operations = len(
            [r for r in self.results_log if r["result"].get("status") == "success"]
        )

        print(f"Total Operations: {total_operations}")
        print(f"Successful Operations: {successful_operations}")
        print(f"Success Rate: {successful_operations/total_operations:.1%}")
        print()

        for i, log_entry in enumerate(self.results_log, 1):
            operation_type = log_entry["type"]
            result = log_entry["result"]
            timestamp = log_entry["timestamp"]

            print(f"{i}. {operation_type.replace('_', ' ').title()}")
            print(f"   Status: {result.get('status', 'unknown').upper()}")
            print(f"   Session: {result.get('session_id', 'N/A')}")
            print(f"   Time: {time.strftime('%H:%M:%S', time.localtime(timestamp))}")
            print()


async def main():
    """Main function to run all examples."""
    print("🚀 Enhanced Resume Processing System - Examples")
    print("=" * 60)
    print()

    # Initialize example processor
    processor = EnhancedProcessingExample()

    try:
        # Run examples in sequence
        print("Running comprehensive examples...")
        print("Note: Some examples may not run if required files are missing.")
        print()

        # Example 1: Excel Processing
        await processor.example_excel_processing()

        # Example 2: Multiple Resume Processing
        await processor.example_multiple_resume_processing()

        # Example 3: Session Recovery
        await processor.example_session_recovery()

        # Example 4: Error Analysis
        await processor.example_error_analysis()

        # Example 5: Performance Monitoring
        await processor.example_performance_monitoring()

        # Display final summary
        processor.display_summary()

        print("\n✅ All examples completed!")
        print("\nNext Steps:")
        print(
            "1. Check the dashboard at: http://localhost:8000/processing-dashboard/dashboard"
        )
        print(
            "2. Review the comprehensive guide in: ENHANCED_RESUME_PROCESSING_GUIDE.md"
        )
        print("3. Implement the system in your production environment")

    except KeyboardInterrupt:
        print("\n⏹️ Examples interrupted by user.")
    except Exception as e:
        print(f"\n❌ Examples failed with error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
