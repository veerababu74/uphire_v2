# Enhanced Multiple Resume Parser with Automatic Embeddings

## Overview

This enhancement adds comprehensive multiple resume parsing functionality with automatic embedding generation to the UPH resume API system. The new implementation replaces the old GroqcloudLLM-only parser with a flexible multi-provider parser that supports various LLM providers and automatically generates vector embeddings for all parsed resume fields.

## Key Features

### 1. Multi-LLM Provider Support
- **Ollama**: Local models (fast, private)
- **Groq Cloud**: Fast cloud inference
- **OpenAI**: GPT models  
- **Google Gemini**: Google's latest models
- **Hugging Face**: Open source models

### 2. Automatic Embedding Generation
When processing resumes, the system now automatically generates embeddings for:
- **Skills**: Vector embeddings for all extracted skills
- **Experience**: Embeddings for job titles and companies
- **Education**: Embeddings for degrees and institutions
- **Profile**: Combined profile information embedding
- **Combined**: Comprehensive text embedding for full-text search

### 3. Enhanced API Endpoints

#### New Multiple Resume Parser API (`/api/multiple_resume_parser/`)

**Single Resume Parsing:**
```http
POST /api/multiple_resume_parser/parse-single
Content-Type: multipart/form-data

- file: Resume file (PDF, DOCX, TXT)
- llm_provider: Optional provider ('ollama', 'groq', 'openai', 'google')
```

**Multiple Resume Parsing:**
```http
POST /api/multiple_resume_parser/parse-multiple
Content-Type: multipart/form-data

- files: List of resume files
- llm_provider: Optional provider
- max_concurrent: Maximum concurrent processing (default: 3)
```

**Provider Management:**
```http
GET /api/multiple_resume_parser/supported-providers
POST /api/multiple_resume_parser/switch-provider
```

**Database Storage:**
```http
POST /api/multiple_resume_parser/save-to-database
```

#### Updated Existing APIs

**Enhanced Resume Parser API (`/api/resume-parser`)**
- Now uses the multiple resume parser instead of GroqcloudLLM
- Automatically generates embeddings for all parsed data
- Returns embedding information in response

**Enhanced Add User Data API (`/api/add_user/`)**
- Updated to use the new multiple resume parser
- Maintains backward compatibility
- Automatic embedding generation

## Usage Examples

### 1. Parse Single Resume with Automatic Embeddings

```python
import requests

# Upload and parse a single resume
files = {"file": open("resume.pdf", "rb")}
data = {"llm_provider": "ollama"}  # Optional

response = requests.post(
    "http://localhost:8000/api/multiple_resume_parser/parse-single",
    files=files,
    data=data
)

result = response.json()
print(f"Embeddings generated: {result['embeddings_generated']}")
print(f"Provider used: {result['provider_used']}")

# Access parsed data with embeddings
parsed_data = result["result"]["parsed_data"]
embeddings = parsed_data.get("embeddings", {})
print(f"Available embeddings: {list(embeddings.keys())}")
```

### 2. Parse Multiple Resumes Concurrently

```python
import requests

# Upload multiple resumes
files = [
    ("files", open("resume1.pdf", "rb")),
    ("files", open("resume2.pdf", "rb")),
    ("files", open("resume3.pdf", "rb"))
]

response = requests.post(
    "http://localhost:8000/api/multiple_resume_parser/parse-multiple",
    files=files,
    data={"llm_provider": "groq", "max_concurrent": 2}
)

result = response.json()
stats = result["statistics"]
print(f"Processed: {stats['total_files']} files")
print(f"Successful: {stats['successful_parses']}")
print(f"Embeddings generated: {stats['embeddings_generated']}")
```

### 3. Switch LLM Provider

```python
import requests

# Switch to Ollama for local processing
response = requests.post(
    "http://localhost:8000/api/multiple_resume_parser/switch-provider",
    json={"provider": "ollama"}
)

print(response.json()["message"])
```

### 4. Save Parsed Data with Embeddings

```python
import requests

# Save parsed resume data to database
parsed_data = {
    "user_id": "user123",
    "username": "john_doe",
    "contact_details": {
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "+1-555-0123",
        "current_city": "New York",
        "looking_for_jobs_in": ["New York", "Boston"],
        "pan_card": "ABCDE1234F"
    },
    "skills": ["Python", "Machine Learning", "AWS"],
    "experience": [...],
    "embeddings": {
        "skills_embedding": [...],
        "experience_embedding": [...],
        "profile_embedding": [...]
    }
}

response = requests.post(
    "http://localhost:8000/api/multiple_resume_parser/save-to-database",
    json=parsed_data
)

print(f"Saved with ID: {response.json()['database_id']}")
```

## Configuration

### Environment Variables

```env
# LLM Provider Configuration
LLM_PROVIDER=ollama  # Default provider
GROQ_API_KEYS=your_groq_key1,your_groq_key2
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key

# Ollama Configuration
OLLAMA_PRIMARY_MODEL=llama3.2:3b
OLLAMA_BACKUP_MODEL=qwen2.5:3b
OLLAMA_API_URL=http://localhost:11434

# Groq Configuration
GROQ_PRIMARY_MODEL=gemma2-9b-it
GROQ_BACKUP_MODEL=llama-3.1-70b-versatile
GROQ_TEMPERATURE=0.1
GROQ_MAX_TOKENS=1024

# Embedding Configuration
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DIMENSION=1024
```

### Provider Selection Priority

1. **Explicit parameter**: `llm_provider` in API request
2. **Environment variable**: `LLM_PROVIDER`
3. **Default fallback**: Ollama (if available) or Groq Cloud

## Data Structure

### Parsed Resume with Embeddings

```json
{
  "user_id": "user123",
  "username": "john_doe",
  "contact_details": {
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "+1-555-0123",
    "current_city": "New York",
    "looking_for_jobs_in": ["New York", "Boston"],
    "linkedin_profile": "https://linkedin.com/in/johndoe",
    "pan_card": "ABCDE1234F"
  },
  "skills": ["Python", "Machine Learning", "AWS", "Docker"],
  "may_also_known_skills": ["Kubernetes", "React"],
  "experience": [
    {
      "company": "TechCorp",
      "title": "Senior Software Engineer",
      "from_date": "2020-01",
      "to": "2023-12"
    }
  ],
  "academic_details": [
    {
      "education": "Bachelor of Computer Science",
      "college": "MIT",
      "pass_year": 2018
    }
  ],
  "total_experience": "3 years 11 months",
  "embeddings": {
    "skills_embedding": [0.1, 0.2, ...],  // 1024-dimensional vector
    "experience_embedding": [0.3, 0.4, ...],
    "education_embedding": [0.5, 0.6, ...],
    "profile_embedding": [0.7, 0.8, ...],
    "combined_embedding": [0.9, 1.0, ...]
  },
  "embeddings_generated_at": "2024-01-15T10:30:00Z",
  "processing_timestamp": "2024-01-15T10:30:00Z",
  "llm_provider_used": "ollama",
  "filename": "john_doe_resume.pdf"
}
```

## Performance Considerations

### Concurrent Processing
- Default: 3 concurrent resume processing tasks
- Configurable via `max_concurrent` parameter
- Automatic rate limiting for API-based providers

### Embedding Generation
- Automatic for all successfully parsed resumes
- Fallback handling if embedding generation fails
- Cached embeddings to avoid regeneration

### Provider Performance
- **Ollama**: Fastest for local deployment, requires model downloads
- **Groq**: Fast cloud inference, requires API key
- **OpenAI**: Good quality, moderate speed, costs per token
- **Google**: Good for multilingual content

## Error Handling

### Robust Fallback System
1. **Primary provider fails**: Automatic fallback to backup model
2. **Embedding generation fails**: Continue without embeddings, log error
3. **File processing fails**: Detailed error reporting per file
4. **Provider unavailable**: Graceful degradation with error messages

### Error Response Format
```json
{
  "filename": "problematic_resume.pdf",
  "status": "error",
  "error": "Text extraction failed",
  "error_type": "EXTRACTION_ERROR",
  "parsed_data": null,
  "suggestion": "Check file format and content"
}
```

## Testing

### Test Script
Run the included test script to verify functionality:

```bash
cd /path/to/uphires_v1
python test_multiple_resume_parser.py
```

The test script will:
1. Check API connectivity
2. Test single resume parsing
3. Test multiple resume parsing
4. Test provider switching
5. Verify embedding generation

### Manual Testing
Use the FastAPI docs interface at `http://localhost:8000/docs` to test endpoints interactively.

## Migration Guide

### From Old GroqcloudLLM Parser

**Before:**
```python
from GroqcloudLLM.main import ResumeParser
parser = ResumeParser()
result = parser.process_resume(text)
```

**After:**
```python
from multipleresumepraser.main import ResumeParser
parser = ResumeParser(llm_provider="ollama")  # or any provider
result = parser.process_resume(text)
# result now includes embeddings automatically
```

### API Migration

**Old Endpoint:**
```http
POST /api/resume-parser
```

**New Enhanced Endpoints:**
```http
POST /api/multiple_resume_parser/parse-single
POST /api/multiple_resume_parser/parse-multiple
```

The old endpoint still works but now uses the new parser internally.

## Benefits

1. **Provider Flexibility**: Switch between local and cloud LLMs
2. **Automatic Embeddings**: No manual embedding generation needed
3. **Better Performance**: Concurrent processing and optimized models
4. **Enhanced Search**: Rich vector embeddings for better matching
5. **Production Ready**: Robust error handling and fallback systems
6. **Backward Compatible**: Existing APIs continue to work

## Troubleshooting

### Common Issues

**Ollama Not Available:**
- Ensure Ollama is installed and running: `ollama serve`
- Pull required model: `ollama pull llama3.2:3b`

**API Key Issues:**
- Check environment variables for API keys
- Verify API key permissions and quotas

**Embedding Generation Fails:**
- Check embedding model availability
- Verify vectorizer configuration

**Performance Issues:**
- Reduce `max_concurrent` parameter
- Use smaller models for faster processing
- Switch to local Ollama for better performance

### Logs and Monitoring

Check application logs for detailed error information:
```bash
tail -f logs/multiple_resume_parser.log
```

## Future Enhancements

1. **Custom Model Support**: Add support for custom fine-tuned models
2. **Batch Processing**: Queue-based processing for large volumes
3. **Real-time Updates**: WebSocket support for real-time progress
4. **Model Comparison**: A/B testing between different models
5. **Advanced Embeddings**: Support for domain-specific embedding models
