# Excel Resume Parser

A specialized module for parsing resumes from Excel/XLSX files. This module extends the existing multiple resume parser to handle structured data from Excel spreadsheets.

## Features

- **Excel File Processing**: Supports .xlsx, .xls, and .xlsm formats
- **Duplicate Header Detection**: Automatically detects and removes duplicate column headers
- **Flexible Column Mapping**: Intelligent mapping of Excel columns to resume fields
- **Batch Processing**: Process multiple resumes from a single Excel file
- **Duplicate Detection**: Integrated duplicate resume detection
- **Database Integration**: Seamless integration with existing MongoDB operations
- **LLM Integration**: Uses the same LLM providers as the main resume parser

## Architecture

```
excel_resume_parser/
├── __init__.py                 # Module initialization
├── excel_processor.py         # Excel file processing and data extraction
├── excel_resume_parser.py     # Resume parsing from Excel data
├── main.py                     # Main manager class
├── requirements.txt            # Additional dependencies
└── README.md                   # This file
```

## Components

### 1. ExcelProcessor
Handles Excel file preprocessing:
- Validates Excel file formats
- Removes duplicate headers (keeps first occurrence)
- Cleans and standardizes column names
- Converts rows to dictionaries

### 2. ExcelResumeParser
Converts Excel data to resume objects:
- Maps Excel row data to resume text format
- Uses existing resume parser for LLM processing
- Handles batch processing of multiple rows

### 3. ExcelResumeParserManager
Main orchestrator class:
- Manages the complete processing pipeline
- Handles database operations
- Provides processing statistics
- Manages temporary files

## API Endpoints

### Upload and Process Excel File
```
POST /excel-resume-parser/upload
```
**Parameters:**
- `file`: Excel file (.xlsx, .xls, .xlsm)
- `user_id`: Base user ID for generated resumes
- `username`: Base username for generated resumes
- `sheet_name`: Optional sheet name to process
- `llm_provider`: Optional LLM provider override

### Analyze Excel File Structure
```
POST /excel-resume-parser/analyze
```
**Parameters:**
- `file`: Excel file to analyze

**Returns:** Sheet names, columns, sample data without processing

### Get Queue Status
```
GET /excel-resume-parser/queue-status
```
**Returns:** Current processing queue statistics

### Get Processing Statistics
```
GET /excel-resume-parser/statistics
```
**Returns:** System configuration and processing stats

### Cleanup Temporary Files
```
POST /excel-resume-parser/cleanup-temp
```
**Parameters:**
- `age_limit_minutes`: Age limit for file cleanup (default: 60)

### Get Supported Formats
```
GET /excel-resume-parser/supported-formats
```
**Returns:** Information about supported formats and requirements

## Usage Examples

### Basic Usage
```python
from excel_resume_parser import ExcelResumeParserManager

# Initialize the manager
manager = ExcelResumeParserManager()

# Process Excel file
results = manager.process_excel_file_from_path(
    file_path="resumes.xlsx",
    base_user_id="user",
    base_username="candidate"
)

print(f"Processed {results['summary']['total_rows_processed']} resumes")
print(f"Successfully saved {results['summary']['successfully_saved']} resumes")
```

### With Custom LLM Provider
```python
manager = ExcelResumeParserManager(
    llm_provider="groq",
    api_keys=["your-api-key"]
)
```

### Processing from Bytes (File Upload)
```python
results = manager.process_excel_file_from_bytes(
    file_bytes=uploaded_file_bytes,
    filename="candidates.xlsx",
    base_user_id="batch_001",
    base_username="candidate"
)
```

### Getting Excel Information
```python
info = manager.get_excel_info(file_path="resumes.xlsx")
print(f"Sheets available: {info['sheet_names']}")
print(f"Columns: {info['columns']}")
```

## Excel File Format Requirements

### Supported File Types
- `.xlsx` (Excel 2007+) - Recommended
- `.xls` (Excel 97-2003)
- `.xlsm` (Excel with macros)

### Data Structure
- **Each row** should represent one resume/candidate
- **First row** should contain column headers
- **Duplicate headers** are automatically detected and removed
- **Column names** are flexible - intelligent mapping is performed

### Common Column Names (Auto-detected)
- **Name**: name, full_name, candidate_name, employee_name
- **Email**: email, email_address, email_id
- **Phone**: phone, phone_number, mobile, contact_number
- **Location**: city, current_city, location, address
- **Experience**: experience, total_experience, years_of_experience
- **Skills**: skills, technical_skills, key_skills, expertise
- **Education**: education, qualification, degree
- **Salary**: current_salary, salary, ctc, expected_salary

## Processing Pipeline

1. **Excel Processing**
   - File validation
   - Duplicate header removal
   - Column name cleaning
   - Row-to-dictionary conversion

2. **Resume Parsing**
   - Excel data formatting as resume text
   - LLM-based resume parsing
   - Data validation and cleaning

3. **Database Operations**
   - Duplicate detection
   - Resume storage
   - Vector embedding generation
   - Statistics tracking

## Configuration

### Environment Variables
The module uses the same configuration as the main resume parser:
- `LLM_PROVIDER`: Default LLM provider
- `GROQ_API_KEYS`: API keys for Groq
- `OLLAMA_API_URL`: Ollama server URL
- Database connection settings

### Temporary Files
- Temporary files are stored in `dummy_data_save/temp_excel_files/`
- Automatic cleanup of old files (configurable age limit)
- Manual cleanup via API endpoint

## Error Handling

### Common Errors and Solutions

1. **"Invalid Excel file"**
   - Ensure file is a valid Excel format
   - Check file is not corrupted
   - Verify file extension matches content

2. **"No data found in Excel file"**
   - Check if the sheet contains data
   - Verify correct sheet name
   - Ensure rows have content beyond headers

3. **"Resume parsing returned None"**
   - Check if row data contains sufficient information
   - Verify LLM provider is properly configured
   - Review column mappings

4. **"Duplicate detected"**
   - This is expected behavior for duplicate resumes
   - Check duplicate detection settings if needed

## Integration with Main Application

Add the Excel resume parser API to your main FastAPI application:

```python
from apis.excel_resume_parser_api import router as excel_router

app.include_router(
    excel_router,
    prefix="/api/v1",
    tags=["Excel Resume Parser"]
)
```

## Dependencies

Additional packages required (see requirements.txt):
- `pandas>=2.0.0`: Excel file processing
- `openpyxl>=3.1.0`: .xlsx file support
- `xlrd>=2.0.0`: .xls file support
- `numpy>=1.24.0`: Data processing

## Logging

The module uses the existing CustomLogger system:
- Component-specific loggers for each module
- Detailed processing information
- Error tracking and debugging support

## Performance Considerations

- **Large Files**: Processing time scales with row count
- **Memory Usage**: Large Excel files are processed in memory
- **Concurrent Processing**: Queue management prevents overload
- **Cleanup**: Regular cleanup of temporary files recommended

## Future Enhancements

Potential improvements:
1. **Streaming Processing**: For very large Excel files
2. **Column Mapping UI**: Interactive column mapping interface
3. **Validation Rules**: Custom validation for Excel data
4. **Batch Status Tracking**: Enhanced tracking for large batches
5. **Excel Export**: Export processed results back to Excel
