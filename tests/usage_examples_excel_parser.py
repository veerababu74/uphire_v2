"""
Excel Resume Parser - Usage Examples

This file demonstrates how to use the Excel Resume Parser module
for processing Excel files containing resume data.
"""

import asyncio
import os
import sys
from pathlib import Path
import pandas as pd
import io

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from excel_resume_parser import ExcelProcessor, get_excel_resume_parser_manager


def example_1_basic_excel_processing():
    """Example 1: Basic Excel file processing."""
    print("=== Example 1: Basic Excel Processing ===")

    # Create sample Excel data
    sample_data = {
        "Name": ["John Doe", "Jane Smith", "Mike Johnson"],
        "Email": ["john@example.com", "jane@example.com", "mike@example.com"],
        "Phone": ["1234567890", "9876543210", "5555555555"],
        "Experience": ["5 years", "3 years", "8 years"],
        "Skills": [
            "Python, Django, PostgreSQL",
            "Java, Spring Boot, MySQL",
            "React, Node.js, MongoDB",
        ],
        "Location": ["New York", "San Francisco", "Austin"],
        "Current Salary": ["120000", "95000", "140000"],
        "Expected Salary": ["140000", "110000", "160000"],
        "Notice Period": ["30 days", "Immediate", "45 days"],
        "Education": ["B.Tech CS", "MS CS", "B.Tech IT"],
        "College": ["MIT", "Stanford", "UC Berkeley"],
    }

    df = pd.DataFrame(sample_data)

    # Save to Excel file
    excel_file = "example_resumes.xlsx"
    df.to_excel(excel_file, index=False, engine="openpyxl")
    print(f"✅ Created sample Excel file: {excel_file}")

    # Process the Excel file
    processor = ExcelProcessor()

    # Get Excel information
    sheet_names = processor.get_sheet_names(excel_file)
    print(f"📋 Available sheets: {sheet_names}")

    # Process the Excel data
    excel_data = processor.process_excel_file(excel_file)
    print(f"📊 Processed {len(excel_data)} records")

    # Show sample processed data
    print("\n📝 Sample processed data:")
    for i, record in enumerate(excel_data[:2]):  # Show first 2 records
        print(f"Record {i+1}:")
        for key, value in record.items():
            print(f"  {key}: {value}")
        print()

    # Cleanup
    os.unlink(excel_file)
    print(f"🧹 Cleaned up {excel_file}")


def example_2_duplicate_headers():
    """Example 2: Handling duplicate headers."""
    print("\n=== Example 2: Handling Duplicate Headers ===")

    # Create data with duplicate headers
    sample_data = {
        "Name": ["Alice Johnson", "Bob Wilson"],
        "name": ["Duplicate Name 1", "Duplicate Name 2"],  # Duplicate header
        "Email": ["alice@example.com", "bob@example.com"],
        "email": [
            "duplicate@example.com",
            "duplicate2@example.com",
        ],  # Another duplicate
        "Phone": ["555-1234", "555-5678"],
        "Experience": ["7 years", "4 years"],
    }

    df = pd.DataFrame(sample_data)
    print(f"📋 Original columns: {list(df.columns)}")

    processor = ExcelProcessor()

    # Detect and remove duplicates
    df_cleaned, duplicates = processor.detect_and_remove_duplicate_headers(df)
    print(f"🔍 Duplicate columns detected: {duplicates}")
    print(f"✅ Cleaned columns: {list(df_cleaned.columns)}")

    # Convert to dictionaries
    row_dicts = processor.convert_rows_to_dictionaries(df_cleaned)
    print(f"📊 Final data (first record): {row_dicts[0] if row_dicts else 'None'}")


def example_3_bytes_processing():
    """Example 3: Processing Excel from bytes (file upload simulation)."""
    print("\n=== Example 3: Excel Bytes Processing ===")

    # Create Excel data in memory
    sample_data = {
        "candidate_name": ["Carol Davis", "David Brown"],
        "email_address": ["carol@example.com", "david@example.com"],
        "phone_number": ["555-9012", "555-3456"],
        "total_experience": ["2 years", "6 years"],
        "technical_skills": ["Python, SQL, Pandas", "JavaScript, React, AWS"],
        "current_location": ["Austin", "Seattle"],
        "notice_period": ["15 days", "30 days"],
    }

    df = pd.DataFrame(sample_data)

    # Convert to Excel bytes
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine="openpyxl")
    excel_bytes = excel_buffer.getvalue()

    print(f"📦 Created Excel bytes: {len(excel_bytes)} bytes")

    # Process from bytes
    processor = ExcelProcessor()
    excel_data = processor.process_excel_bytes(
        file_bytes=excel_bytes, filename="uploaded_resumes.xlsx"
    )

    print(f"📊 Processed {len(excel_data)} records from bytes")
    print(f"📝 Sample record: {excel_data[0] if excel_data else 'None'}")


def example_4_column_mapping():
    """Example 4: Flexible column mapping."""
    print("\n=== Example 4: Flexible Column Mapping ===")

    # Test various column name formats
    test_cases = [
        {
            "Full Name": ["Test User 1"],
            "Email ID": ["test1@example.com"],
            "Contact Number": ["9876543210"],
            "Years of Experience": ["5 years"],
            "Technical Skills": ["Python, Java"],
        },
        {
            "Candidate Name": ["Test User 2"],
            "Email Address": ["test2@example.com"],
            "Mobile": ["8765432109"],
            "Work Experience": ["3 years"],
            "Key Skills": ["React, Node.js"],
        },
        {
            "Employee Name": ["Test User 3"],
            "email": ["test3@example.com"],
            "phone": ["7654321098"],
            "experience": ["7 years"],
            "skills": ["Django, PostgreSQL"],
        },
    ]

    processor = ExcelProcessor()

    for i, data in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}:")
        print(f"   Original columns: {list(data.keys())}")

        df = pd.DataFrame(data)
        df_cleaned = processor.clean_column_names(df)
        row_dicts = processor.convert_rows_to_dictionaries(df_cleaned)

        print(f"   Cleaned columns: {list(df_cleaned.columns)}")
        print(f"   Sample data: {row_dicts[0] if row_dicts else 'None'}")


def example_5_complete_pipeline_simulation():
    """Example 5: Complete pipeline simulation (without LLM processing)."""
    print("\n=== Example 5: Complete Pipeline Simulation ===")

    # Create comprehensive resume data
    comprehensive_data = {
        "Full Name": ["Sarah Johnson", "Michael Chen", "Emily Rodriguez"],
        "Email": ["sarah.j@email.com", "michael.c@email.com", "emily.r@email.com"],
        "Phone": ["555-0101", "555-0202", "555-0303"],
        "Current City": ["New York", "San Francisco", "Austin"],
        "Total Experience": ["5 years", "8 years", "3 years"],
        "Current Role": ["Senior Developer", "Lead Engineer", "Full Stack Developer"],
        "Current Company": ["TechCorp", "InnovateAI", "StartupXYZ"],
        "Skills": [
            "Python, Django, PostgreSQL, AWS, Docker",
            "Java, Spring Boot, Microservices, Kubernetes",
            "JavaScript, React, Node.js, MongoDB, Express",
        ],
        "Education": [
            "MS Computer Science",
            "B.Tech Software Engineering",
            "B.S. Computer Science",
        ],
        "College": ["Stanford University", "IIT Delhi", "UC Berkeley"],
        "Current Salary": ["130000", "180000", "95000"],
        "Expected Salary": ["150000", "200000", "110000"],
        "Notice Period": ["30 days", "45 days", "Immediate"],
        "LinkedIn": [
            "https://linkedin.com/in/sarah-johnson",
            "https://linkedin.com/in/michael-chen",
            "https://linkedin.com/in/emily-rodriguez",
        ],
    }

    df = pd.DataFrame(comprehensive_data)

    # Save to Excel
    excel_file = "comprehensive_resumes.xlsx"
    df.to_excel(excel_file, index=False, engine="openpyxl")

    print(f"📋 Created comprehensive Excel file with {len(df)} candidates")
    print(f"📊 Columns: {list(df.columns)}")

    # Process the file
    processor = ExcelProcessor()
    excel_data = processor.process_excel_file(excel_file)

    print(f"\n✅ Processing completed:")
    print(f"   📊 Records processed: {len(excel_data)}")
    print(
        f"   📝 Columns in processed data: {list(excel_data[0].keys()) if excel_data else 'None'}"
    )

    # Simulate what would happen in the complete pipeline
    print(f"\n🔄 Pipeline simulation:")
    for i, record in enumerate(excel_data):
        name = record.get("full_name", "Unknown")
        email = record.get("email", "Unknown")
        experience = record.get("total_experience", "Unknown")
        skills = record.get("skills", "Unknown")

        print(f"   Candidate {i+1}: {name}")
        print(f"     Email: {email}")
        print(f"     Experience: {experience}")
        print(
            f"     Skills: {skills[:50]}..."
            if len(str(skills)) > 50
            else f"     Skills: {skills}"
        )
        print(f"     Status: Would be processed by LLM parser ✓")
        print()

    # Cleanup
    os.unlink(excel_file)
    print(f"🧹 Cleaned up {excel_file}")


def example_6_api_simulation():
    """Example 6: API endpoint simulation."""
    print("\n=== Example 6: API Endpoint Simulation ===")

    # This example shows how the API would work
    print("🌐 API Endpoint: POST /excel-resume-parser/upload")
    print("📋 Parameters:")
    print("   - file: Excel file (.xlsx, .xls, .xlsm)")
    print("   - user_id: Base user ID")
    print("   - username: Base username")
    print("   - sheet_name: Optional sheet name")
    print("   - llm_provider: Optional LLM provider")

    print("\n📤 Simulated API Response:")

    # Create sample API response structure
    api_response = {
        "status": "success",
        "filename": "candidates.xlsx",
        "sheet_name": None,
        "total_processing_time": 45.67,
        "excel_processing": {
            "rows_found": 25,
            "sample_data": [
                {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "experience": "5 years",
                },
                {
                    "name": "Jane Smith",
                    "email": "jane@example.com",
                    "experience": "3 years",
                },
            ],
        },
        "resume_parsing": {
            "total_rows": 25,
            "successful_parses": 23,
            "failed_parses": 2,
            "processing_time": 38.2,
        },
        "database_operations": {
            "saved_successfully": 21,
            "duplicates_found": 2,
            "save_errors": 0,
        },
        "summary": {
            "total_rows_processed": 25,
            "successful_parses": 23,
            "failed_parses": 2,
            "successfully_saved": 21,
            "duplicates_detected": 2,
            "save_errors": 0,
        },
    }

    print(f"✅ Status: {api_response['status']}")
    print(f"📊 Total rows processed: {api_response['summary']['total_rows_processed']}")
    print(f"✅ Successfully parsed: {api_response['summary']['successful_parses']}")
    print(f"💾 Successfully saved: {api_response['summary']['successfully_saved']}")
    print(f"🔍 Duplicates detected: {api_response['summary']['duplicates_detected']}")
    print(f"⏱️  Total processing time: {api_response['total_processing_time']} seconds")

    print("\n🔧 Additional API endpoints available:")
    print("   - POST /excel-resume-parser/analyze - Analyze Excel structure")
    print("   - GET  /excel-resume-parser/queue-status - Get processing queue status")
    print("   - GET  /excel-resume-parser/statistics - Get processing statistics")
    print("   - POST /excel-resume-parser/cleanup-temp - Cleanup temporary files")
    print("   - GET  /excel-resume-parser/supported-formats - Get supported formats")


def main():
    """Run all examples."""
    print("📚 Excel Resume Parser - Usage Examples")
    print("=" * 60)

    examples = [
        ("Basic Excel Processing", example_1_basic_excel_processing),
        ("Duplicate Headers Handling", example_2_duplicate_headers),
        ("Bytes Processing", example_3_bytes_processing),
        ("Column Mapping", example_4_column_mapping),
        ("Complete Pipeline Simulation", example_5_complete_pipeline_simulation),
        ("API Simulation", example_6_api_simulation),
    ]

    for example_name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"❌ {example_name} failed: {e}")

        print("-" * 60)

    print("\n🎉 All examples completed!")
    print("\n💡 Key Benefits of Excel Resume Parser:")
    print("   ✅ Handles duplicate headers automatically")
    print("   ✅ Flexible column name mapping")
    print("   ✅ Processes both files and bytes (uploads)")
    print("   ✅ Integrates with existing resume parsing pipeline")
    print("   ✅ Built-in duplicate detection")
    print("   ✅ Comprehensive error handling")
    print("   ✅ RESTful API endpoints")
    print("   ✅ Supports multiple Excel formats (.xlsx, .xls, .xlsm)")


if __name__ == "__main__":
    main()
