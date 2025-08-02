#!/usr/bin/env python3

import re

# Read the file
with open(
    r"c:\Users\pveer\OneDrive\Desktop\UPH\uphire_v2\apis\rag_search.py",
    "r",
    encoding="utf-8",
) as f:
    content = f.read()

# Replace all the problematic float conversions
content = re.sub(
    r'float\(candidate\.get\("current_salary", 0\)\)',
    'safe_float(candidate.get("current_salary", 0))',
    content,
)
content = re.sub(
    r'float\(candidate\.get\("hike", 0\)\)',
    'safe_float(candidate.get("hike", 0))',
    content,
)
content = re.sub(
    r'float\(candidate\.get\("expected_salary", 0\)\)',
    'safe_float(candidate.get("expected_salary", 0))',
    content,
)
content = re.sub(
    r'float\(candidate\.get\("similarity_score", 0\.0\)\)',
    'safe_float(candidate.get("similarity_score", 0.0))',
    content,
)
content = re.sub(
    r'float\(candidate\.get\("relevance_score", 0\.0\)\)',
    'safe_float(candidate.get("relevance_score", 0.0))',
    content,
)

# Write the file back
with open(
    r"c:\Users\pveer\OneDrive\Desktop\UPH\uphire_v2\apis\rag_search.py",
    "w",
    encoding="utf-8",
) as f:
    f.write(content)

print("Fixed all float conversions in rag_search.py")
