# Duplicate Content Detection Feature

## Overview

The duplicate content detection feature has been implemented to prevent processing of duplicate resume content. This feature automatically checks if the extracted text from a resume is similar to previously processed resumes for the same user and rejects duplicates based on a 70% similarity threshold.

## Features

### 1. Automatic Duplicate Detection
- **Pre-processing Check**: Before parsing any resume, the system extracts and cleans the text, then checks for similarity against existing content
- **70% Similarity Threshold**: If content is 70% or more similar to existing content, the resume is rejected
- **User-specific**: Duplicate detection is performed per user, so different users can upload similar resumes

### 2. Text Normalization
- Converts text to lowercase
- Removes special characters and numbers
- Normalizes whitespace
- Removes common resume words that don't contribute to meaningful comparison
- Uses SequenceMatcher for accurate similarity calculation

### 3. Database Storage
- **New Collection**: `resume_extracted_text` collection stores extracted text with metadata
- **Metadata Includes**: user_id, username, filename, extracted text, creation timestamp
- **Efficient Storage**: Normalized text is stored separately for faster comparison

### 4. Enhanced API Response
The multiple resume parser API now returns detailed information about duplicate content:

```json
{
  "message": "✅ Successfully processed 8/10 resumes for John Doe in 2.5 seconds",
  "resume_processing_summary": {
    "successfully_parsed_resumes": ["resume1.pdf", "resume2.pdf"],
    "failed_to_parse_resumes": [
      {
        "filename": "corrupted.pdf",
        "error": "Could not extract text from file"
      }
    ],
    "duplicate_content_resumes": [
      {
        "filename": "duplicate_resume.pdf",
        "error": "Duplicate content detected",
        "similar_documents": [
          {
            "document_id": "60f7b8c8e4b0c8d8e4b0c8d8",
            "filename": "original_resume.pdf",
            "similarity_score": 0.85,
            "created_at": "2023-12-01T10:30:00Z"
          }
        ],
        "message": "This resume content is 85.0% similar to existing content"
      }
    ],
    "successfully_saved_to_database": ["resume1.pdf", "resume2.pdf"],
    "failed_to_save_to_database": []
  },
  "processing_statistics": {
    "total_files_uploaded": 10,
    "successfully_parsed": 8,
    "failed_to_parse": 0,
    "duplicate_content_detected": 2,
    "parsing_success_rate": "80.0%"
  }
}
```

## Implementation Details

### 1. Files Modified/Created

#### New Files:
- `mangodatabase/duplicate_detection.py` - Core duplicate detection logic
- `apis/duplicate_detection_api.py` - API endpoints for duplicate detection management
- `test_duplicate_detection.py` - Test script for validation

#### Modified Files:
- `mangodatabase/client.py` - Added new collection getter
- `apis/multiple_resume_parser_api.py` - Integrated duplicate detection
- `main.py` - Added duplicate detection API routes

### 2. Database Schema

**Collection: `resume_extracted_text`**
```json
{
  "_id": "ObjectId",
  "user_id": "string",
  "username": "string", 
  "filename": "string",
  "extracted_text": "string",
  "text_length": "number",
  "created_at": "datetime",
  "normalized_text": "string"
}
```

### 3. Processing Flow

1. **File Upload**: User uploads resume files via API
2. **Text Extraction**: Extract and clean text from each file
3. **Duplicate Check**: Compare extracted text with existing user content
4. **Decision**:
   - If duplicate (≥70% similar): Skip parsing, add to duplicate list
   - If unique (<70% similar): Proceed with parsing and save extracted text
5. **Response**: Return comprehensive statistics including duplicate information

### 4. API Endpoints

#### Duplicate Detection Management:
- `GET /duplicate-detection/statistics/{user_id}` - Get duplicate detection statistics
- `GET /duplicate-detection/extracted-texts/{user_id}` - Get all extracted texts for user
- `DELETE /duplicate-detection/extracted-text/{document_id}` - Delete extracted text
- `POST /duplicate-detection/check-similarity` - Check text similarity manually
- `GET /duplicate-detection/info` - Get system information

#### Resume Processing (Enhanced):
- `POST /resume-parser-multiple` - Process multiple resumes with duplicate detection
- `POST /parse-bulk-resumes` - Bulk process resumes with duplicate detection

## Configuration

### Similarity Threshold
The default similarity threshold is 70% (0.70). This can be modified in the `DuplicateDetectionOperations` class:

```python
class DuplicateDetectionOperations:
    def __init__(self, extracted_text_collection: Collection):
        self.collection = extracted_text_collection
        self.similarity_threshold = 0.70  # 70% similarity threshold
```

### Text Normalization
The normalization process can be customized by modifying the `normalize_text_for_comparison` method:

```python
def normalize_text_for_comparison(self, text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove common resume words
    common_words = [
        'resume', 'cv', 'curriculum', 'vitae', 'email', 'phone', 'address',
        'experience', 'education', 'skills', 'projects', 'objective',
        'summary', 'references', 'available', 'upon', 'request'
    ]
    
    for word in common_words:
        text = text.replace(word, ' ')
    
    return re.sub(r'\s+', ' ', text).strip()
```

## Testing

### Manual Testing
Run the test script to validate functionality:

```bash
python test_duplicate_detection.py
```

### API Testing
Use the duplicate detection endpoints to test:

```bash
# Check system info
curl -X GET "http://localhost:8000/duplicate-detection/info"

# Get user statistics
curl -X GET "http://localhost:8000/duplicate-detection/statistics/user123"

# Check text similarity
curl -X POST "http://localhost:8000/duplicate-detection/check-similarity" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "text": "Sample resume text..."}'
```

## Benefits

1. **Prevents Duplicate Processing**: Saves computational resources by avoiding processing of duplicate content
2. **User-Friendly Feedback**: Clear information about why a resume was rejected
3. **Maintains Data Quality**: Prevents duplicate entries in the database
4. **Detailed Reporting**: Comprehensive statistics about duplicate detection
5. **Configurable Threshold**: Easy to adjust similarity threshold based on requirements
6. **Efficient Comparison**: Fast text comparison using normalized text and SequenceMatcher

## Future Enhancements

1. **ML-based Similarity**: Use advanced NLP models for better similarity detection
2. **Configurable Thresholds**: Per-user or per-organization similarity thresholds
3. **Partial Duplicate Detection**: Detect and highlight specific sections that are duplicated
4. **Duplicate Resolution**: Allow users to review and approve/reject duplicates manually
5. **Cross-user Duplicate Detection**: Optional feature to detect duplicates across all users
6. **Performance Optimization**: Implement indexing and caching for faster comparison

## Error Handling

The system gracefully handles various error scenarios:

- **Database Connection Issues**: Falls back to allowing processing if duplicate check fails
- **Text Extraction Failures**: Continues with parsing if text extraction fails
- **Similarity Calculation Errors**: Logs errors and continues processing
- **Storage Failures**: Warns about storage issues but doesn't fail the parsing process

## Monitoring and Logging

All duplicate detection activities are logged with appropriate log levels:

- **INFO**: Successful duplicate detection and text storage
- **WARNING**: Duplicate content detected, storage failures
- **ERROR**: System errors during duplicate detection process

Log files are stored in the `logs/` directory with the name pattern: `duplicate_detection.log`
