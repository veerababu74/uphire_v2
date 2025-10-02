#!/usr/bin/env python3
"""
Excel to JSON Conversion - Usage Examples

This script demonstrates various ways to use the cleaned JSON data
returned by the Excel to JSON conversion API.
"""

import requests
import json
import pandas as pd
import sqlite3
from datetime import datetime
from typing import List, Dict, Any


class ExcelToJsonProcessor:
    """
    A class to demonstrate processing cleaned JSON data from Excel files.
    """

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.api_endpoint = f"{api_base_url}/resume-parser/excel-to-json"

    def convert_excel_to_json(self, excel_file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Convert Excel file to clean JSON using the API.

        Args:
            excel_file_path: Path to Excel file
            **kwargs: Additional parameters for the API

        Returns:
            API response with cleaned JSON data
        """
        print(f"Converting Excel file: {excel_file_path}")

        # Default parameters
        params = {"skip_empty_rows": True, "normalize_headers": True, "max_rows": None}
        params.update(kwargs)

        try:
            with open(excel_file_path, "rb") as f:
                files = {
                    "file": (
                        excel_file_path,
                        f,
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                }

                response = requests.post(
                    self.api_endpoint, files=files, data=params, timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    print(
                        f"✅ Conversion successful: {result['statistics']['cleaned_rows']} rows"
                    )
                    return result
                else:
                    print(f"❌ Conversion failed: {response.status_code}")
                    print(response.text)
                    return None

        except Exception as e:
            print(f"❌ Error during conversion: {e}")
            return None

    def save_to_json_file(self, data: List[Dict[str, Any]], output_path: str):
        """Save cleaned data to JSON file."""
        print(f"Saving {len(data)} records to {output_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ Data saved to {output_path}")

    def save_to_csv(self, data: List[Dict[str, Any]], output_path: str):
        """Save cleaned data to CSV file."""
        print(f"Converting {len(data)} records to CSV: {output_path}")

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

        print(f"✅ CSV saved to {output_path}")

    def save_to_database(
        self, data: List[Dict[str, Any]], db_path: str, table_name: str
    ):
        """Save cleaned data to SQLite database."""
        print(f"Saving {len(data)} records to database: {db_path}")

        if not data:
            print("⚠️ No data to save")
            return

        # Create DataFrame for easier database insertion
        df = pd.DataFrame(data)

        # Connect to SQLite database
        conn = sqlite3.connect(db_path)

        try:
            # Save to database
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"✅ Data saved to database table: {table_name}")

            # Show some statistics
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"📊 Total records in database: {count}")

        except Exception as e:
            print(f"❌ Database error: {e}")
        finally:
            conn.close()

    def validate_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate cleaned data and provide statistics."""
        print("🔍 Validating cleaned data...")

        if not data:
            return {"status": "empty", "message": "No data to validate"}

        validation_report = {
            "total_records": len(data),
            "fields": {},
            "data_types": {},
            "completeness": {},
            "issues": [],
        }

        # Get all unique fields
        all_fields = set()
        for record in data:
            all_fields.update(record.keys())

        validation_report["fields"]["total_fields"] = len(all_fields)
        validation_report["fields"]["field_names"] = sorted(list(all_fields))

        # Analyze each field
        for field in all_fields:
            values = [record.get(field) for record in data]
            non_null_values = [v for v in values if v is not None]

            # Completeness
            completeness = len(non_null_values) / len(values) * 100
            validation_report["completeness"][field] = round(completeness, 1)

            # Data types
            if non_null_values:
                types = set(type(v).__name__ for v in non_null_values)
                validation_report["data_types"][field] = list(types)
            else:
                validation_report["data_types"][field] = ["null"]

            # Check for common issues
            if completeness < 50:
                validation_report["issues"].append(
                    f"Field '{field}' has low completeness: {completeness}%"
                )

        print(f"📋 Validation completed:")
        print(f"   - Total records: {validation_report['total_records']}")
        print(f"   - Total fields: {validation_report['fields']['total_fields']}")
        print(f"   - Issues found: {len(validation_report['issues'])}")

        return validation_report

    def demonstrate_data_processing(self, data: List[Dict[str, Any]]):
        """Demonstrate various ways to process the cleaned data."""
        print("\n🔄 Demonstrating data processing techniques...")

        if not data:
            print("⚠️ No data to process")
            return

        # 1. Filter data
        print("\n1. Filtering data:")
        if "email" in data[0]:
            valid_emails = [
                record
                for record in data
                if record.get("email") and "@" in record["email"]
            ]
            print(f"   Records with valid emails: {len(valid_emails)}/{len(data)}")

        # 2. Group data
        print("\n2. Grouping data:")
        if "department" in data[0]:
            departments = {}
            for record in data:
                dept = record.get("department", "Unknown")
                if dept not in departments:
                    departments[dept] = 0
                departments[dept] += 1

            print("   Records by department:")
            for dept, count in departments.items():
                print(f"     {dept}: {count}")

        # 3. Calculate statistics
        print("\n3. Calculating statistics:")
        numeric_fields = []
        for field, value in data[0].items():
            if isinstance(value, (int, float)):
                numeric_fields.append(field)

        for field in numeric_fields:
            values = [
                record.get(field) for record in data if record.get(field) is not None
            ]
            if values:
                avg_val = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                print(f"   {field}: avg={avg_val:.2f}, min={min_val}, max={max_val}")

        # 4. Data transformation
        print("\n4. Data transformation example:")
        transformed_data = []
        for record in data:
            transformed_record = record.copy()

            # Add calculated fields
            transformed_record["processed_at"] = datetime.now().isoformat()
            transformed_record["record_id"] = f"REC_{len(transformed_data)+1:04d}"

            # Normalize text fields
            for key, value in transformed_record.items():
                if isinstance(value, str):
                    transformed_record[key] = value.strip().title()

            transformed_data.append(transformed_record)

        print(f"   Transformed {len(transformed_data)} records with additional fields")

        return transformed_data


def main():
    """Demonstrate the complete Excel to JSON processing workflow."""
    print("🚀 Excel to JSON Processing Workflow Demo")
    print("=" * 50)

    # Initialize processor
    processor = ExcelToJsonProcessor()

    # Example workflow - you would replace this with your actual Excel file
    excel_file_path = "sample_data.xlsx"  # Replace with your Excel file

    # Note: For this demo, we'll create a sample response structure
    # In real usage, you would call processor.convert_excel_to_json(excel_file_path)

    # Simulated API response for demonstration
    simulated_response = {
        "status": "success",
        "filename": "sample_data.xlsx",
        "statistics": {
            "original_rows": 10,
            "cleaned_rows": 8,
            "processing_time_seconds": 1.2,
        },
        "data": [
            {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "123-456-7890",
                "age": 25,
                "salary": 50000,
                "department": "Engineering",
                "active": True,
            },
            {
                "name": "Jane Smith",
                "email": "jane@example.com",
                "phone": "987-654-3210",
                "age": 30,
                "salary": 75000,
                "department": "Marketing",
                "active": True,
            },
            {
                "name": "Bob Johnson",
                "email": "bob@example.com",
                "phone": "555-0123",
                "age": 35,
                "salary": 65000,
                "department": "Sales",
                "active": False,
            },
        ],
    }

    print("📥 Using simulated clean data for demonstration...")
    cleaned_data = simulated_response["data"]

    # 1. Validate the data
    validation_report = processor.validate_data(cleaned_data)

    # 2. Save to different formats
    print("\n💾 Saving data to different formats...")
    processor.save_to_json_file(cleaned_data, "output/cleaned_data.json")
    processor.save_to_csv(cleaned_data, "output/cleaned_data.csv")
    processor.save_to_database(cleaned_data, "output/data.db", "employees")

    # 3. Demonstrate data processing
    transformed_data = processor.demonstrate_data_processing(cleaned_data)

    # 4. Save transformed data
    print("\n💾 Saving transformed data...")
    processor.save_to_json_file(transformed_data, "output/transformed_data.json")

    print("\n🎉 Demo completed!")
    print("\nTo use with real Excel files:")
    print("1. Replace 'sample_data.xlsx' with your Excel file path")
    print("2. Uncomment the API call: processor.convert_excel_to_json(excel_file_path)")
    print("3. Remove the simulated_response and use the actual API response")
    print("\nOutput files created in 'output/' directory:")
    print("- cleaned_data.json (original cleaned data)")
    print("- cleaned_data.csv (CSV format)")
    print("- data.db (SQLite database)")
    print("- transformed_data.json (processed data)")


if __name__ == "__main__":
    # Create output directory
    import os

    os.makedirs("output", exist_ok=True)

    main()
