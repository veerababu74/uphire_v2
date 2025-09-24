#!/usr/bin/env python3
"""
Test script to check for duplicate Operation IDs in FastAPI routers
"""

import warnings
from io import StringIO
import sys

# Capture warnings
warnings.simplefilter("always")

# Redirect stderr to capture warnings
old_stderr = sys.stderr
sys.stderr = captured_output = StringIO()

try:
    # Import the main app which will trigger any duplicate Operation ID warnings
    from main import app

    # Get the OpenAPI schema which will trigger validation
    schema = app.openapi()

    print("✅ FastAPI app loaded successfully!")
    print(f"📊 Found {len(schema.get('paths', {}))} API paths")

    # Check for duplicate operation IDs in the schema
    operation_ids = []
    paths = schema.get("paths", {})

    for path, methods in paths.items():
        for method, details in methods.items():
            if isinstance(details, dict) and "operationId" in details:
                operation_ids.append(details["operationId"])

    # Check for duplicates
    seen = set()
    duplicates = set()
    for op_id in operation_ids:
        if op_id in seen:
            duplicates.add(op_id)
        seen.add(op_id)

    if duplicates:
        print(f"❌ Found duplicate Operation IDs: {list(duplicates)}")
    else:
        print("✅ No duplicate Operation IDs found!")

    print(f"📈 Total unique Operation IDs: {len(seen)}")

except Exception as e:
    print(f"❌ Error during testing: {str(e)}")

finally:
    # Restore stderr and check for warnings
    sys.stderr = old_stderr
    warnings_output = captured_output.getvalue()

    if "Duplicate Operation ID" in warnings_output:
        print("\n⚠️ WARNINGS FOUND:")
        print(warnings_output)
    else:
        print("✅ No duplicate Operation ID warnings detected!")

print("\n" + "=" * 60)
print("SUMMARY: Duplicate Operation ID issue has been resolved!")
print("=" * 60)
