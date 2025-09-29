#!/usr/bin/env python3
"""
Test script to verify the Excel Column Mapper fix

This script tests the EnhancedColumnMapper method name fix to ensure
the Excel parser can properly map columns using the correct method.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from excel_resume_parser.enhanced_column_mapper import EnhancedColumnMapper
import pandas as pd


def test_column_mapper_fix():
    """Test the column mapper fix."""
    print("=== Testing Excel Column Mapper Fix ===\n")

    # Initialize column mapper
    print("1. Initializing EnhancedColumnMapper...")
    mapper = EnhancedColumnMapper()
    print("✅ EnhancedColumnMapper initialized successfully")

    # Test data - typical Excel columns
    test_columns = [
        "Name",
        "Email",
        "Phone",
        "Experience",
        "Skills",
        "Location",
        "Current Salary",
        "Expected Salary",
        "Notice Period",
        "Education",
        "College",
    ]

    print(f"\n2. Testing column mapping with {len(test_columns)} columns:")
    for col in test_columns:
        print(f"   - {col}")

    # Test the map_columns method (the correct method name)
    print("\n3. Calling map_columns method...")
    try:
        mapping_result = mapper.map_columns(test_columns)
        print("✅ map_columns method executed successfully")

        # Analyze results
        mapped_count = sum(
            1 for info in mapping_result.values() if info["mapped_field"]
        )
        unmapped_count = len(test_columns) - mapped_count

        print(f"\n4. Column Mapping Results:")
        print(f"   - Total columns: {len(test_columns)}")
        print(f"   - Successfully mapped: {mapped_count}")
        print(f"   - Unmapped: {unmapped_count}")

        print(f"\n5. Detailed Mappings:")
        for col_name, mapping_info in mapping_result.items():
            if mapping_info["mapped_field"]:
                print(
                    f"   ✅ '{col_name}' → '{mapping_info['mapped_field']}' (confidence: {mapping_info['confidence']:.3f})"
                )
            else:
                print(
                    f"   ❌ '{col_name}' → unmapped (max confidence: {max(mapping_info.get('all_matches', {}).values()) if mapping_info.get('all_matches') else 0:.3f})"
                )

        # Test the structure we expect in enhanced_excel_resume_parser
        print(f"\n6. Testing data structure compatibility:")
        mapped_fields = {}
        confidence_scores = {}
        unmapped_columns = []

        for col_name, mapping_info in mapping_result.items():
            if mapping_info["mapped_field"]:
                mapped_fields[col_name] = mapping_info["mapped_field"]
                confidence_scores[col_name] = mapping_info["confidence"]
            else:
                unmapped_columns.append(col_name)

        print(f"   - mapped_fields: {len(mapped_fields)} items")
        print(f"   - confidence_scores: {len(confidence_scores)} items")
        print(f"   - unmapped_columns: {len(unmapped_columns)} items")
        print("✅ Data structure compatibility verified!")

        return True

    except Exception as e:
        print(f"❌ Error in map_columns method: {str(e)}")
        return False


def test_analyze_and_map_columns_missing():
    """Verify that the old method name doesn't exist."""
    print("\n=== Testing Old Method Name ===")

    mapper = EnhancedColumnMapper()

    # Check if the old method exists
    if hasattr(mapper, "analyze_and_map_columns"):
        print("⚠️  Warning: Old method 'analyze_and_map_columns' still exists")
        return False
    else:
        print("✅ Confirmed: Old method 'analyze_and_map_columns' does not exist")
        return True


if __name__ == "__main__":
    print("Testing Excel Column Mapper Fix\n")

    # Test 1: Column mapper fix
    success1 = test_column_mapper_fix()

    # Test 2: Old method verification
    success2 = test_analyze_and_map_columns_missing()

    print(f"\n=== Test Summary ===")
    if success1 and success2:
        print("✅ All tests passed! Column mapper fix is working correctly.")
        print("✅ Excel parser should now work without the method name error.")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        sys.exit(1)
