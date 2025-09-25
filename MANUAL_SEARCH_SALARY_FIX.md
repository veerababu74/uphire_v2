# Manual Search Salary Field Validation Fix

## Problem Description
The `/manualsearch/` endpoint was returning a `422 Unprocessable Entity` error when the frontend sent empty strings (`""`) for the `min_salary` and `max_salary` fields.

### Original Error
```json
{
    "detail": [
        {
            "type": "float_parsing",
            "loc": ["body", "min_salary"],
            "msg": "Input should be a valid number, unable to parse string as a number",
            "input": ""
        },
        {
            "type": "float_parsing",
            "loc": ["body", "max_salary"],
            "msg": "Input should be a valid number, unable to parse string as a number",
            "input": ""
        }
    ]
}
```

### Original Request Data
```json
{
    "userid": "67b075f7fe29fc1b2d36e18b",
    "experience_titles": ["frontend developer", "software developer"],
    "locations": [],
    "max_experience": "",
    "max_salary": "",  // Problem: Empty string instead of null/number
    "min_education": ["10th Pass", "Graduate", "12th Pass"],
    "min_experience": "",
    "min_salary": "",  // Problem: Empty string instead of null/number
    "skills": ["html", "python"]
}
```

## Root Cause
The Pydantic models for manual search endpoints defined `min_salary` and `max_salary` as `Optional[float]`, but did not handle the case where the frontend sends empty strings instead of `null` values.

## Solution
Added custom validators to convert empty strings to `None` for salary fields across all relevant API endpoints.

### Files Fixed
1. `apis/manual_search.py`
2. `apis/manual.py`
3. `apis/manual_recent_search_save.py`

### Validator Implementation
```python
@validator('min_salary', 'max_salary', pre=True)
def parse_salary(cls, v):
    """Convert empty strings to None for salary fields"""
    if v == "" or v is None:
        return None
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            raise ValueError(f"Invalid salary value: {v}")
    return v
```

## Changes Made

### 1. Updated Imports
Added `validator` to the Pydantic imports:
```python
from pydantic import BaseModel, Field, validator
```

### 2. Added Custom Validators
The validator handles these cases:
- `""` (empty string) → `None`
- `None` → `None` (unchanged)
- `"123456"` (string number) → `123456.0` (converted to float)
- `123456.0` (float) → `123456.0` (unchanged)
- `"invalid"` (invalid string) → Raises `ValueError`

## Testing

### Test Results
```
1️⃣ Empty strings → ✅ Converted to None
2️⃣ None values → ✅ Remain None
3️⃣ Valid floats → ✅ Remain unchanged
4️⃣ String numbers → ✅ Converted to floats
5️⃣ Invalid strings → ✅ Proper error raised
6️⃣ Mixed values → ✅ Handled correctly
```

### Original Request Now Works
The request that previously caused the 422 error now processes successfully:
- `min_salary: ""` → `min_salary: None`
- `max_salary: ""` → `max_salary: None`

## Impact
- ✅ Resolves 422 validation errors for manual search endpoints
- ✅ Maintains backward compatibility with existing valid requests
- ✅ Improves error handling for invalid salary inputs
- ✅ No changes required on the frontend
- ✅ Consistent behavior across all manual search-related endpoints

## API Endpoints Affected
- `POST /manualsearch/` (main manual search)
- `POST /manual_saved_recnet_search/save_search` (save search)
- `POST /manual_saved_recnet_search/save_recent_search` (recent search)

## Deployment Notes
- No database migrations required
- No breaking changes to existing functionality
- Frontend can continue sending empty strings or null values
- Server will handle both gracefully