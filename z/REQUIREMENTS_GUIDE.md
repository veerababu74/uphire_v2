# Requirements Installation Guide

This project has multiple requirement files for different use cases:

## 1. Main Requirements (requirements.txt)
**For general development and deployment**
```bash
pip install -r requirements.txt
```

## 2. Development Requirements (dev-requirements.txt)
**For local development with testing and debugging tools**
```bash
pip install -r requirements.txt
pip install -r dev-requirements.txt
```

## 3. Production Requirements (prod-requirements.txt)
**For production deployment with pinned versions**
```bash
pip install -r prod-requirements.txt
```

## 4. Docker Requirements (docker-requirements.txt)
**For containerized deployments (smaller footprint)**
```bash
pip install -r docker-requirements.txt
```

## Key Dependencies Explained

### Core Framework
- **FastAPI**: Web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI applications
- **Pydantic**: Data validation and serialization

### Database & Vector Search
- **PyMongo**: MongoDB driver for Python
- **LangChain-MongoDB**: MongoDB integration for LangChain

### AI/ML Components
- **Sentence-Transformers**: For generating text embeddings
- **Transformers**: Hugging Face transformers library
- **PyTorch**: Deep learning framework
- **LangChain**: Framework for building AI applications
- **LangChain-Groq**: Groq LLM integration
- **LangChain-OpenAI**: OpenAI integration
- **LangChain-Google-GenAI**: Google Generative AI integration

### Document Processing
- **PyPDF2**: PDF text extraction
- **python-docx**: Microsoft Word document processing

### System & Utilities
- **psutil**: System and process monitoring
- **requests/httpx**: HTTP client libraries
- **python-dotenv**: Environment variable management
- **aiofiles**: Async file operations

## Installation Order

1. **For new development setup:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install main dependencies
pip install -r requirements.txt

# Install development tools (optional)
pip install -r dev-requirements.txt
```

2. **For production deployment:**
```bash
pip install -r prod-requirements.txt
```

3. **For Docker deployment:**
Use docker-requirements.txt in your Dockerfile

## Environment Variables Required

Create a `.env` file with:
```env
# MongoDB Configuration
MONGODB_URI=your_mongodb_connection_string
MONGODB_DATABASE_NAME=resume_db

# LLM Provider API Keys
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key

# Application Settings
EMBEDDING_DEPLOYMENT=balanced  # Options: minimal, balanced, full, complete
```

## Troubleshooting

### Common Issues:

1. **PyTorch Installation:**
   - For CPU-only: Use `torch==2.1.0+cpu`
   - For GPU: Use `torch==2.1.0+cu118` (adjust CUDA version)

2. **Memory Issues:**
   - Use smaller embedding models for development
   - Consider CPU-only versions for containers

3. **Model Downloads:**
   - Sentence-transformers models are downloaded automatically
   - Ensure sufficient disk space (2-5GB for models)

### Platform-Specific Notes:

**Windows:**
- Some packages may require Visual Studio Build Tools
- Use `pip install --upgrade setuptools wheel` if installation fails

**Linux/Mac:**
- No additional requirements typically needed
- Ensure Python 3.8+ is installed

**Docker:**
- Use multi-stage builds to reduce image size
- Consider using official Python slim images as base
