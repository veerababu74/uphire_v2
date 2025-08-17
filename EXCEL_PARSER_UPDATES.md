# Excel Resume Parser - Auto-Detection Updates

## Changes Made

The Excel Resume Parser has been updated to automatically detect `llm_provider` and `save_temp_file` settings from the system configuration, eliminating the need to pass these as parameters.

### Key Changes

#### 1. LLM Provider Auto-Detection
- **Before**: Required `llm_provider` parameter in API calls
- **After**: Automatically detected from `core.config.AppConfig.LLM_PROVIDER`
- **Configuration**: Set via environment variable `LLM_PROVIDER` (defaults to "ollama")

#### 2. Save Temp File Policy
- **Before**: Required `save_temp_file` parameter in API calls
- **After**: Automatically set to `False` for security reasons
- **Rationale**: Files are processed in memory to avoid security risks

### Updated API Endpoint

```python
@router.post("/excel-resume-parser/upload")
async def upload_excel_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    username: str = Form(...),
    sheet_name: Optional[str] = Form(None),  # Only optional parameter now
):
```

### Removed Parameters
- ❌ `llm_provider: Optional[str] = Form(None)` - Now auto-detected
- ❌ `save_temp_file: bool = Form(False)` - Now always False

### Files Modified

1. **`apis/excel_resume_parser_api.py`**
   - Removed `llm_provider` and `save_temp_file` parameters
   - Added auto-detection logic for LLM provider
   - Simplified file processing (no temp file saving)

2. **`excel_resume_parser/main.py`**
   - Updated `ExcelResumeParserManager.__init__()` to auto-detect LLM provider
   - Added import for `core.config.AppConfig`

3. **`excel_resume_parser/excel_resume_parser.py`**
   - Updated `ExcelResumeParser.__init__()` to auto-detect LLM provider
   - Added import for `core.config.AppConfig`

### Benefits

1. **Simplified API**: Fewer required parameters
2. **Centralized Configuration**: LLM provider managed in one place
3. **Enhanced Security**: No temporary file saving by default
4. **Consistent Behavior**: Same LLM provider used across all components

### Configuration

To change the LLM provider, update your environment configuration:

```bash
# .env file
LLM_PROVIDER=ollama          # Default
# or
LLM_PROVIDER=groq_cloud      # Alternative
# or
LLM_PROVIDER=openai          # Alternative
```

### Migration Guide

If you have existing API calls, simply remove the `llm_provider` and `save_temp_file` parameters:

**Before:**
```python
# Old API call
response = requests.post(
    "/excel-resume-parser/upload",
    files={"file": file_content},
    data={
        "user_id": "test_user",
        "username": "test_username",
        "sheet_name": "Sheet1",
        "llm_provider": "ollama",     # Remove this
        "save_temp_file": False       # Remove this
    }
)
```

**After:**
```python
# New API call
response = requests.post(
    "/excel-resume-parser/upload",
    files={"file": file_content},
    data={
        "user_id": "test_user", 
        "username": "test_username",
        "sheet_name": "Sheet1"         # Only these parameters needed
    }
)
```

### Response Changes

The API response now includes the auto-detected LLM provider in the `file_info` section:

```json
{
  "session_id": "excel_test_user_1234567890",
  "file_info": {
    "filename": "resumes.xlsx",
    "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "file_size": 12345,
    "temp_file_saved": false,
    "temp_file_path": null,
    "llm_provider": "ollama"  // New: shows auto-detected provider
  },
  // ... rest of response
}
```

### Backward Compatibility

This is a **breaking change** for API consumers. The parameters have been removed completely to enforce the new auto-detection behavior. Update your API calls accordingly.
