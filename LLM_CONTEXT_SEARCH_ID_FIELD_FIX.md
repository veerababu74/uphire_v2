# LLM Context Search _id Field Fix

## Problem Description
The `/rag/llm-context-search` endpoint was returning results with empty `_id` fields in the response, even though the data was present in the backend.

### Original Response Issue
```json
{
    "results": [
        {
            "_id": "",  // Problem: Empty _id field
            "user_id": "67b075f7fe29fc1b2d36e18b",
            "username": "Test",
            // ... other fields
        }
    ]
}
```

## Root Cause
The issue was caused by improper field mapping between MongoDB documents and Pydantic models:

1. **Pydantic Field Naming Restriction**: Pydantic doesn't allow field names starting with underscores (`_id`)
2. **Incorrect Field Mapping**: The data transformation wasn't properly mapping `_id` to the Pydantic model's `id` field
3. **Missing Data Transformation**: The API wasn't ensuring both `_id` and `id` fields were present in the response data

## Solution
Implemented a comprehensive fix across multiple functions in `apis/rag_search.py`:

### 1. Correct Pydantic Model Definition
```python
class LLMSearchResult(BaseModel):
    model_config = {"populate_by_name": True}

    id: str = Field(default="", serialization_alias="_id")  # Correct way
    # Other fields...
```

### 2. Data Transformation in API Endpoints
Updated three key functions to properly transform `_id` fields:

#### llm_context_search function:
```python
# Handle _id field properly - map _id to id for Pydantic model
candidate_id = res.get("_id")
if candidate_id is None:
    candidate_id = res.get("id", "")

# Set both _id and id fields to ensure compatibility
id_str = str(candidate_id) if candidate_id is not None else ""
res["_id"] = id_str
res["id"] = id_str  # Map _id to id for Pydantic model
```

#### vector_similarity_search function:
```python
formatted_candidate = {
    "_id": candidate_id,
    "id": candidate_id,  # Map _id to id for Pydantic model
    # Other fields...
}
```

#### JD upload function:
```python
candidate_id = safe_object_id(candidate.get("_id", ""))
formatted_candidate = {
    "_id": candidate_id,
    "id": candidate_id,  # Map _id to id for Pydantic model
    # Other fields...
}
```

## Files Modified
1. **`apis/rag_search.py`** - Fixed Pydantic model and data transformation logic

## Testing Results
✅ **Pydantic Model Test**: Field properly serializes as `_id` in JSON output
✅ **Data Transformation Test**: `_id` values are correctly mapped and preserved
✅ **Score Normalization**: Relevance scores properly normalized to 0-100 range

## Key Features
- **Backward Compatibility**: Handles both `_id` and `id` input formats
- **Proper Serialization**: Output always contains `_id` field with correct values
- **MongoDB Integration**: Works seamlessly with MongoDB ObjectId conversion
- **Score Normalization**: Automatically converts 0-1 range scores to 0-100 percentages

## API Endpoints Fixed
- `POST /rag/llm-context-search` - Main LLM context search
- `POST /rag/vector-search` - Vector similarity search  
- `POST /rag/llm-context-search/by-jd` - Job description upload search

## Expected Result
The API response now properly includes populated `_id` fields:

```json
{
    "results": [
        {
            "_id": "67b075f7fe29fc1b2d36e18b",  // ✅ Now properly populated
            "user_id": "67b075f7fe29fc1b2d36e18b",
            "username": "Test",
            "relevance_score": 54.28,
            // ... other fields
        }
    ]
}
```

## Deployment Notes
- No database changes required
- No breaking changes to existing functionality  
- All existing API calls will continue to work
- The fix ensures data integrity between MongoDB and API responses