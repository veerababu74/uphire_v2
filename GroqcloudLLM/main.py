import os
import json
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, EmailStr
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import requests

# Import core configuration
from core.config import config
from core.custom_logger import CustomLogger
from core.exceptions import LLMProviderError
from core.llm_config import LLMConfigManager, LLMProvider
from core.experience_extractor import ExperienceExtractor

# Load environment variables
load_dotenv()

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("groqcloud_llm")


def get_api_keys() -> List[str]:
    """Get API keys from environment variables using core config."""
    api_keys_str = os.getenv("GROQ_API_KEYS", "")
    if not api_keys_str:
        # Try legacy environment variable
        api_keys_str = config.GROQ_API_KEY

    if not api_keys_str:
        logger.warning("No Groq API keys found in environment variables")
        return []

    api_keys = api_keys_str.split(",")
    clean_keys = [key.strip() for key in api_keys if key.strip()]
    logger.info(f"Found {len(clean_keys)} Groq API keys")
    return clean_keys


# ===== Pydantic Models =====
# class Project(BaseModel):
#     name: str = Field(default="", description="Project name")
#     description: str = Field(default="", description="Project description")
#     technologies: List[str] = Field(
#         default_factory=list, description="Technologies used"
#     )
#     role: str = Field(default="", description="Role in the project")
#     start_date: str = Field(default="", description="Start date")
#     end_date: str = Field(default="", description="End date")
#     duration: str = Field(default="", description="Total duration")


# ===== Pydantic Models =====
class Experience(BaseModel):
    company: str  # Required
    title: str  # Required
    from_date: str  # Required, format: 'YYYY-MM'
    to: Optional[str] = None  # Optional, format: 'YYYY-MM'


class Education(BaseModel):
    education: str  # Required
    college: str  # Required
    pass_year: int  # Required


class ContactDetails(BaseModel):
    name: str  # Required
    email: EmailStr  # Required
    phone: str  # Required
    alternative_phone: Optional[str] = None
    current_city: str  # Required
    looking_for_jobs_in: List[str]  # Required
    gender: Optional[str] = None
    age: Optional[int] = None
    naukri_profile: Optional[str] = None
    linkedin_profile: Optional[str] = None
    portfolio_link: Optional[str] = None
    pan_card: str  # Required
    aadhar_card: Optional[str] = None  # Optional


class Resume(BaseModel):
    user_id: str
    username: str
    contact_details: ContactDetails
    total_experience: Optional[str] = None  # ✅ Already changed to string

    notice_period: Optional[str] = None  # e.g., "Immediate", "30 days"
    currency: Optional[str] = None  # e.g., "INR", "USD"
    pay_duration: Optional[str] = None  # e.g., "monthly", "yearly"
    current_salary: Optional[float] = None
    hike: Optional[float] = None
    expected_salary: Optional[float] = None  # Changed from required to optional
    skills: List[str]
    may_also_known_skills: List[str]
    labels: Optional[List[str]] = None  # Added = None for consistency
    experience: Optional[List[Experience]] = None
    academic_details: Optional[List[Education]] = None
    source: Optional[str] = None  # Source of the resume (e.g., "LinkedIn", "Naukri")
    last_working_day: Optional[str] = None  # Should be ISO format date string
    is_tier1_mba: Optional[bool] = None
    is_tier1_engineering: Optional[bool] = None
    comment: Optional[str] = None
    exit_reason: Optional[str] = None


# ===== Resume Parser Class =====
class ResumeParser:
    def __init__(self, llm_provider: str = None, api_keys: List[str] = None):
        """Initialize ResumeParser with configurable LLM provider.

        Args:
            llm_provider (str, optional): LLM provider ('groq', 'ollama', 'openai', 'google', 'huggingface').
                                        If None, uses LLM_PROVIDER from config.
            api_keys (List[str], optional): List of API keys for API-based providers.
        """
        # Initialize LLM configuration manager
        self.llm_manager = LLMConfigManager()

        # Determine which LLM provider to use
        if llm_provider is None:
            # Use config default
            self.provider = self.llm_manager.provider
        else:
            provider_map = {
                "ollama": LLMProvider.OLLAMA,
                "groq": LLMProvider.GROQ_CLOUD,
                "groq_cloud": LLMProvider.GROQ_CLOUD,
                "openai": LLMProvider.OPENAI,
                "google": LLMProvider.GOOGLE_GEMINI,
                "gemini": LLMProvider.GOOGLE_GEMINI,
                "google_gemini": LLMProvider.GOOGLE_GEMINI,
                "huggingface": LLMProvider.HUGGINGFACE,
                "hf": LLMProvider.HUGGINGFACE,
            }

            if llm_provider.lower() not in provider_map:
                raise LLMProviderError(f"Unsupported LLM provider: {llm_provider}")

            self.provider = provider_map[llm_provider.lower()]
            self.llm_manager.provider = self.provider

        logger.info(f"Initializing ResumeParser with {self.provider.value}")

        # Initialize provider-specific configurations
        if self.provider == LLMProvider.OLLAMA:
            self._setup_ollama()
        elif self.provider == LLMProvider.GROQ_CLOUD:
            self._setup_groq(api_keys)
        elif self.provider == LLMProvider.OPENAI:
            self._setup_openai(api_keys)
        elif self.provider == LLMProvider.GOOGLE_GEMINI:
            self._setup_google(api_keys)
        elif self.provider == LLMProvider.HUGGINGFACE:
            self._setup_huggingface()
        else:
            raise LLMProviderError(f"Provider {self.provider.value} not implemented")

        # Initialize experience extractor with the configured LLM
        try:
            # Get the LLM instance for the experience extractor
            llm_instance = self._get_llm_instance()
            self.experience_extractor = ExperienceExtractor(llm_instance)
            logger.info("Experience extractor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize experience extractor: {e}")
            self.experience_extractor = None

        logger.info(f"ResumeParser initialized successfully with {self.provider.value}")

    def _setup_ollama(self):
        """Setup Ollama provider"""
        self.ollama_config = self.llm_manager.ollama_config
        if not self._check_ollama_connection():
            error_msg = (
                "Ollama is not running or accessible. Please start Ollama service."
            )
            logger.error(error_msg)
            raise LLMProviderError(error_msg)

        # Validate model availability
        if not self._validate_ollama_model(self.ollama_config.primary_model):
            logger.warning(f"{self.ollama_config.primary_model} not found.")
            available_models = self._get_available_ollama_models()
            logger.info(f"Available models: {available_models}")

            # Try backup model
            if self.ollama_config.backup_model in available_models:
                logger.info(f"Using backup model: {self.ollama_config.backup_model}")
                self.ollama_config.primary_model = self.ollama_config.backup_model
            elif self.ollama_config.fallback_model in available_models:
                logger.info(
                    f"Using fallback model: {self.ollama_config.fallback_model}"
                )
                self.ollama_config.primary_model = self.ollama_config.fallback_model
            else:
                error_msg = (
                    f"None of the configured models are available. "
                    f"Please pull a compatible model using: ollama pull {self.ollama_config.primary_model}"
                )
                logger.error(error_msg)
                raise LLMProviderError(error_msg)

        self.api_keys = []
        self.api_usage = {}
        self.current_key_index = 0
        self.processing_chain = self._setup_ollama_chain()

    def _setup_groq(self, api_keys: List[str] = None):
        """Setup Groq provider"""
        self.groq_config = self.llm_manager.groq_config

        if api_keys is None:
            self.api_keys = self.groq_config.api_keys
        else:
            self.api_keys = [key.strip() for key in api_keys if key.strip()]

        if not self.api_keys:
            error_msg = "No Groq API keys provided or found in environment variables."
            logger.error(error_msg)
            raise LLMProviderError(error_msg)

        self.api_usage = {key: 0 for key in self.api_keys}
        self.current_key_index = 0
        self.processing_chain = self._setup_groq_chain(
            self.api_keys[self.current_key_index]
        )

    def _setup_openai(self, api_keys: List[str] = None):
        """Setup OpenAI provider"""
        self.openai_config = self.llm_manager.openai_config

        if api_keys is None:
            self.api_keys = self.openai_config.api_keys
        else:
            self.api_keys = [key.strip() for key in api_keys if key.strip()]

        if not self.api_keys:
            error_msg = "No OpenAI API keys provided or found in environment variables."
            logger.error(error_msg)
            raise LLMProviderError(error_msg)

        self.api_usage = {key: 0 for key in self.api_keys}
        self.current_key_index = 0
        self.processing_chain = self._setup_openai_chain(
            self.api_keys[self.current_key_index]
        )

    def _setup_google(self, api_keys: List[str] = None):
        """Setup Google Gemini provider"""
        self.google_config = self.llm_manager.google_config

        if api_keys is None:
            self.api_keys = self.google_config.api_keys
        else:
            self.api_keys = [key.strip() for key in api_keys if key.strip()]

        if not self.api_keys:
            error_msg = "No Google API keys provided or found in environment variables."
            logger.error(error_msg)
            raise LLMProviderError(error_msg)

        self.api_usage = {key: 0 for key in self.api_keys}
        self.current_key_index = 0
        self.processing_chain = self._setup_google_chain(
            self.api_keys[self.current_key_index]
        )

    def _setup_huggingface(self):
        """Setup Hugging Face provider"""
        self.huggingface_config = self.llm_manager.huggingface_config

        # For Hugging Face, no API keys needed (for public models)
        self.api_keys = []
        self.api_usage = {}
        self.current_key_index = 0

        try:
            self.processing_chain = self._setup_huggingface_chain()
        except Exception as e:
            error_msg = f"Failed to initialize Hugging Face model: {str(e)}"
            logger.error(error_msg)
            raise LLMProviderError(error_msg)

    def _check_ollama_connection(self) -> bool:
        """Check if Ollama service is accessible."""
        try:
            response = requests.get(
                f"{self.ollama_config.api_url}/api/tags",
                timeout=self.ollama_config.connection_timeout,
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama connection check failed: {e}")
            return False

    def _get_available_ollama_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            response = requests.get(
                f"{self.ollama_config.api_url}/api/tags",
                timeout=self.ollama_config.connection_timeout,
            )
            if response.status_code == 200:
                models_data = response.json()
                return [model["name"] for model in models_data.get("models", [])]
            return []
        except Exception as e:
            logger.error(f"Failed to get available Ollama models: {e}")
            return []

    def _validate_ollama_model(self, model_name: str) -> bool:
        """Validate if an Ollama model is available."""
        available_models = self._get_available_ollama_models()
        return model_name in available_models

    def _setup_ollama_chain(self):
        """Set up the LangChain processing chain for Ollama."""
        try:
            model = OllamaLLM(**self.ollama_config.to_langchain_params())

            parser = JsonOutputParser(pydantic_object=Resume)

            prompt_template = """You are an expert resume parser and data extraction specialist. Your task is to first determine if the provided text is a valid resume, and if so, extract comprehensive information from it.

STEP 1: RESUME VALIDATION
First, analyze the provided text to determine if it contains resume/CV content. A valid resume should contain:
- Personal/contact information (name, email, phone, etc.)
- Professional experience OR education OR skills
- Career-related content (job titles, companies, educational institutions, etc.)

If the text does NOT appear to be a resume (e.g., random text, stories, articles, product descriptions, etc.), return this exact JSON structure:
{{
    "error": "The provided content does not appear to be a resume or CV. Please upload a document containing professional information such as work experience, education, skills, and contact details.",
    "error_type": "invalid_content",
    "suggestion": "Ensure your document contains typical resume sections like contact information, work experience, education, and skills."
}}

STEP 2: RESUME EXTRACTION (Only if Step 1 confirms it's a resume)
If the content IS a valid resume, extract ALL available information accurately and return structured data.

EXTRACTION GUIDELINES:
- Name: Extract full name from header or contact section
- Email: Find email address (if missing, use: "email_not_provided@example.com")
- Phone: Extract phone number (if missing, use: "+1-000-000-0000")
- Address/City: Extract current location (if missing, use: "Location not specified")
- Experience: Calculate total years from all employment periods
- Skills: Extract all technical and professional skills mentioned
- Education: Include all degrees, institutions, and graduation years
- Work History: Include company names, positions, dates, and responsibilities

DATA QUALITY REQUIREMENTS:
- All dates should be in YYYY-MM format
- Phone numbers should include country code if available
- Skills should be comprehensive and include both technical and soft skills
- Experience calculation should account for overlapping or concurrent roles
- Education should include degree level, field of study, and institution
- Use placeholder values for missing information rather than null/empty values

{format_instructions}

TEXT TO ANALYZE:
{resume_text}

CRITICAL: 
- If the text is NOT a resume, return the error JSON structure shown above
- If the text IS a resume, return the complete structured resume data as per the format instructions
- Return ONLY valid JSON without any additional text or explanations
                """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["resume_text"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            logger.debug(
                f"Ollama processing chain setup complete for model: {self.ollama_config.primary_model}"
            )
            return prompt | model | parser

        except Exception as e:
            logger.error(f"Failed to setup Ollama processing chain: {str(e)}")
            raise LLMProviderError(f"Failed to setup Ollama processing chain: {str(e)}")

    def _setup_groq_chain(self, api_key: str):
        """Set up the LangChain processing chain with Groq configuration."""
        if not api_key:
            raise LLMProviderError("API key cannot be empty.")

        try:
            model = ChatGroq(**self.groq_config.to_langchain_params())

            parser = JsonOutputParser(pydantic_object=Resume)

            prompt_template = """You are an expert resume parser and data extraction specialist. Your task is to first determine if the provided text is a valid resume, and if so, extract comprehensive information from it.

STEP 1: RESUME VALIDATION
First, analyze the provided text to determine if it contains resume/CV content. A valid resume should contain:
- Personal/contact information (name, email, phone, etc.)
- Professional experience OR education OR skills
- Career-related content (job titles, companies, educational institutions, etc.)

If the text does NOT appear to be a resume (e.g., random text, stories, articles, product descriptions, etc.), return this exact JSON structure:
{{
    "error": "The provided content does not appear to be a resume or CV. Please upload a document containing professional information such as work experience, education, skills, and contact details.",
    "error_type": "invalid_content",
    "suggestion": "Ensure your document contains typical resume sections like contact information, work experience, education, and skills."
}}

STEP 2: RESUME EXTRACTION (Only if Step 1 confirms it's a resume)
If the content IS a valid resume, extract ALL available information accurately and return structured data.

EXTRACTION GUIDELINES:
- Name: Extract full name from header or contact section
- Email: Find email address (if missing, use: "email_not_provided@example.com")
- Phone: Extract phone number (if missing, use: "+1-000-000-0000")
- Address/City: Extract current location (if missing, use: "Location not specified")
- Experience: Calculate total years from all employment periods
- Skills: Extract all technical and professional skills mentioned
- Education: Include all degrees, institutions, and graduation years
- Work History: Include company names, positions, dates, and responsibilities

DATA QUALITY REQUIREMENTS:
- All dates should be in YYYY-MM format
- Phone numbers should include country code if available
- Skills should be comprehensive and include both technical and soft skills
- Experience calculation should account for overlapping or concurrent roles
- Education should include degree level, field of study, and institution
- Use placeholder values for missing information rather than null/empty values

{format_instructions}

TEXT TO ANALYZE:
{resume_text}

CRITICAL: 
- If the text is NOT a resume, return the error JSON structure shown above
- If the text IS a resume, return the complete structured resume data as per the format instructions
- Return ONLY valid JSON without any additional text or explanations
                """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["resume_text"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            logger.debug(
                f"Groq processing chain setup complete for model: {self.groq_config.primary_model}"
            )
            return prompt | model | parser

        except Exception as e:
            logger.error(f"Failed to setup Groq processing chain: {str(e)}")
            raise LLMProviderError(f"Failed to setup Groq processing chain: {str(e)}")

    def _setup_openai_chain(self, api_key: str):
        """Set up the LangChain processing chain for OpenAI."""
        if not api_key:
            raise LLMProviderError("API key cannot be empty.")

        try:
            # Update the OpenAI config with current API key
            self.openai_config.api_keys[self.current_key_index] = api_key

            model = ChatOpenAI(**self.openai_config.to_langchain_params())
            parser = JsonOutputParser(pydantic_object=Resume)

            prompt_template = """You are an expert resume parser and data extraction specialist. Your task is to first determine if the provided text is a valid resume, and if so, extract comprehensive information from it.

STEP 1: RESUME VALIDATION
First, analyze the provided text to determine if it contains resume/CV content. A valid resume should contain:
- Personal/contact information (name, email, phone, etc.)
- Professional experience OR education OR skills
- Career-related content (job titles, companies, educational institutions, etc.)

If the text does NOT appear to be a resume (e.g., random text, stories, articles, product descriptions, etc.), return this exact JSON structure:
{{
    "error": "The provided content does not appear to be a resume or CV. Please upload a document containing professional information such as work experience, education, skills, and contact details.",
    "error_type": "invalid_content",
    "suggestion": "Ensure your document contains typical resume sections like contact information, work experience, education, and skills."
}}

STEP 2: RESUME EXTRACTION (Only if Step 1 confirms it's a resume)
If the content IS a valid resume, extract ALL available information accurately and return structured data.

EXTRACTION GUIDELINES:
- Name: Extract full name from header or contact section
- Email: Find email address (if missing, use: "email_not_provided@example.com")
- Phone: Extract phone number (if missing, use: "+1-000-000-0000")
- Address/City: Extract current location (if missing, use: "Location not specified")
- Experience: Calculate total years from all employment periods
- Skills: Extract all technical and professional skills mentioned
- Education: Include all degrees, institutions, and graduation years
- Work History: Include company names, positions, dates, and responsibilities

DATA QUALITY REQUIREMENTS:
- All dates should be in YYYY-MM format
- Phone numbers should include country code if available
- Skills should be comprehensive and include both technical and soft skills
- Experience calculation should account for overlapping or concurrent roles
- Education should include degree level, field of study, and institution
- Use placeholder values for missing information rather than null/empty values

{format_instructions}

TEXT TO ANALYZE:
{resume_text}

CRITICAL: 
- If the text is NOT a resume, return the error JSON structure shown above
- If the text IS a resume, return the complete structured resume data as per the format instructions
- Return ONLY valid JSON without any additional text or explanations
                """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["resume_text"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            logger.debug(
                f"OpenAI processing chain setup complete for model: {self.openai_config.primary_model}"
            )
            return prompt | model | parser

        except Exception as e:
            logger.error(f"Failed to setup OpenAI processing chain: {str(e)}")
            raise LLMProviderError(f"Failed to setup OpenAI processing chain: {str(e)}")

    def _setup_google_chain(self, api_key: str):
        """Set up the LangChain processing chain for Google Gemini."""
        if not api_key:
            raise LLMProviderError("API key cannot be empty.")

        try:
            # Update the Google config with current API key
            self.google_config.api_keys[self.current_key_index] = api_key

            model = ChatGoogleGenerativeAI(**self.google_config.to_langchain_params())
            parser = JsonOutputParser(pydantic_object=Resume)

            prompt_template = """You are an expert resume parser and data extraction specialist. Your task is to first determine if the provided text is a valid resume, and if so, extract comprehensive information from it.

STEP 1: RESUME VALIDATION
First, analyze the provided text to determine if it contains resume/CV content. A valid resume should contain:
- Personal/contact information (name, email, phone, etc.)
- Professional experience OR education OR skills
- Career-related content (job titles, companies, educational institutions, etc.)

If the text does NOT appear to be a resume (e.g., random text, stories, articles, product descriptions, etc.), return this exact JSON structure:
{{
    "error": "The provided content does not appear to be a resume or CV. Please upload a document containing professional information such as work experience, education, skills, and contact details.",
    "error_type": "invalid_content",
    "suggestion": "Ensure your document contains typical resume sections like contact information, work experience, education, and skills."
}}

STEP 2: RESUME EXTRACTION (Only if Step 1 confirms it's a resume)
If the content IS a valid resume, extract ALL available information accurately and return structured data.

EXTRACTION GUIDELINES:
- Name: Extract full name from header or contact section
- Email: Find email address (if missing, use: "email_not_provided@example.com")
- Phone: Extract phone number (if missing, use: "+1-000-000-0000")
- Address/City: Extract current location (if missing, use: "Location not specified")
- Experience: Calculate total years from all employment periods
- Skills: Extract all technical and professional skills mentioned
- Education: Include all degrees, institutions, and graduation years
- Work History: Include company names, positions, dates, and responsibilities

DATA QUALITY REQUIREMENTS:
- All dates should be in YYYY-MM format
- Phone numbers should include country code if available
- Skills should be comprehensive and include both technical and soft skills
- Experience calculation should account for overlapping or concurrent roles
- Education should include degree level, field of study, and institution
- Use placeholder values for missing information rather than null/empty values

{format_instructions}

TEXT TO ANALYZE:
{resume_text}

CRITICAL: 
- If the text is NOT a resume, return the error JSON structure shown above
- If the text IS a resume, return the complete structured resume data as per the format instructions
- Return ONLY valid JSON without any additional text or explanations
                """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["resume_text"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            logger.debug(
                f"Google Gemini processing chain setup complete for model: {self.google_config.primary_model}"
            )
            return prompt | model | parser

        except Exception as e:
            logger.error(f"Failed to setup Google processing chain: {str(e)}")
            raise LLMProviderError(f"Failed to setup Google processing chain: {str(e)}")

    def _setup_huggingface_chain(self):
        """Set up the LangChain processing chain for Hugging Face."""
        try:
            # Import here to avoid dependency issues if not installed
            try:
                from langchain_huggingface import HuggingFacePipeline
            except ImportError:
                raise LLMProviderError(
                    "langchain_huggingface not installed. Please install it with: pip install langchain-huggingface"
                )

            model = HuggingFacePipeline.from_model_id(
                **self.huggingface_config.to_langchain_params()
            )
            parser = JsonOutputParser(pydantic_object=Resume)

            prompt_template = """You are an expert resume parser and data extraction specialist. Your task is to first determine if the provided text is a valid resume, and if so, extract comprehensive information from it.

STEP 1: RESUME VALIDATION
First, analyze the provided text to determine if it contains resume/CV content. A valid resume should contain:
- Personal/contact information (name, email, phone, etc.)
- Professional experience OR education OR skills
- Career-related content (job titles, companies, educational institutions, etc.)

If the text does NOT appear to be a resume (e.g., random text, stories, articles, product descriptions, etc.), return this exact JSON structure:
{{
    "error": "The provided content does not appear to be a resume or CV. Please upload a document containing professional information such as work experience, education, skills, and contact details.",
    "error_type": "invalid_content",
    "suggestion": "Ensure your document contains typical resume sections like contact information, work experience, education, and skills."
}}

STEP 2: RESUME EXTRACTION (Only if Step 1 confirms it's a resume)
If the content IS a valid resume, extract ALL available information accurately and return structured data.

EXTRACTION GUIDELINES:
- Name: Extract full name from header or contact section
- Email: Find email address (if missing, use: "email_not_provided@example.com")
- Phone: Extract phone number (if missing, use: "+1-000-000-0000")
- Address/City: Extract current location (if missing, use: "Location not specified")
- Experience: Calculate total years from all employment periods
- Skills: Extract all technical and professional skills mentioned
- Education: Include all degrees, institutions, and graduation years
- Work History: Include company names, positions, dates, and responsibilities

DATA QUALITY REQUIREMENTS:
- All dates should be in YYYY-MM format
- Phone numbers should include country code if available
- Skills should be comprehensive and include both technical and soft skills
- Experience calculation should account for overlapping or concurrent roles
- Education should include degree level, field of study, and institution
- Use placeholder values for missing information rather than null/empty values

{format_instructions}

TEXT TO ANALYZE:
{resume_text}

CRITICAL: 
- If the text is NOT a resume, return the error JSON structure shown above
- If the text IS a resume, return the complete structured resume data as per the format instructions
- Return ONLY valid JSON without any additional text or explanations
                """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["resume_text"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

            logger.debug(
                f"Hugging Face processing chain setup complete for model: {self.huggingface_config.model_id}"
            )
            return prompt | model | parser

        except Exception as e:
            logger.error(f"Failed to setup Hugging Face processing chain: {str(e)}")
            raise LLMProviderError(
                f"Failed to setup Hugging Face processing chain: {str(e)}"
            )

    def _get_llm_instance(self):
        """Get the LLM instance from the processing chain for the experience extractor."""
        try:
            # The processing chain is typically: prompt | llm | parser
            # We need to extract the LLM (middle component)
            if hasattr(self, "processing_chain") and hasattr(
                self.processing_chain, "steps"
            ):
                # For RunnableSequence, get the middle step (LLM)
                for step in self.processing_chain.steps:
                    if (
                        hasattr(step, "model_name")
                        or hasattr(step, "model")
                        or "llm" in str(type(step)).lower()
                    ):
                        return step

            # Fallback: Create a new instance based on provider
            if self.provider == LLMProvider.OLLAMA:
                return OllamaLLM(**self.ollama_config.to_langchain_params())
            elif self.provider == LLMProvider.GROQ_CLOUD:
                return ChatGroq(
                    temperature=self.groq_config.temperature,
                    model_name=self.groq_config.primary_model,
                    api_key=self.api_keys[0] if self.api_keys else None,
                    max_tokens=self.groq_config.max_tokens,
                )
            elif self.provider == LLMProvider.OPENAI:
                return ChatOpenAI(
                    temperature=self.openai_config.temperature,
                    model_name=self.openai_config.primary_model,
                    api_key=self.api_keys[0] if self.api_keys else None,
                    max_tokens=self.openai_config.max_tokens,
                )
            elif self.provider == LLMProvider.GOOGLE_GEMINI:
                return ChatGoogleGenerativeAI(
                    temperature=self.google_config.temperature,
                    model=self.google_config.primary_model,
                    api_key=self.api_keys[0] if self.api_keys else None,
                    max_tokens=self.google_config.max_tokens,
                )
            else:
                logger.warning(
                    f"No LLM instance available for provider {self.provider.value}"
                )
                return None

        except Exception as e:
            logger.error(f"Failed to get LLM instance: {e}")
            return None

    def switch_provider(self, new_provider: str, api_keys: List[str] = None):
        """Switch between LLM providers dynamically.

        Args:
            new_provider (str): New provider to use ('groq', 'ollama', 'openai', 'google', 'huggingface')
            api_keys (List[str], optional): API keys for API-based providers
        """
        logger.info(
            f"Switching LLM provider from {self.provider.value} to {new_provider}"
        )

        # Reinitialize with new provider
        self.__init__(llm_provider=new_provider, api_keys=api_keys)

    def _clean_json_response(self, response) -> str:
        """Clean and extract JSON from response."""
        try:
            if isinstance(response, dict):
                return json.dumps(response)
            elif isinstance(response, str):
                # Remove code block markers and clean whitespace
                cleaned = re.sub(r"```[^`]*```", "", response, flags=re.DOTALL)
                cleaned = cleaned.strip()
                # Extract JSON object if wrapped in text
                match = re.search(r"({[\s\S]*})", cleaned)
                return match.group(1) if match else cleaned
            else:
                return str(response)
        except Exception as e:
            print(f"Error cleaning JSON response: {str(e)}")
            return str(response)

    def _repair_json(self, malformed_json: str) -> str:
        """Repair common JSON formatting issues."""
        try:
            # Remove trailing commas
            repaired = re.sub(r",(\s*[}\]])", r"\1", malformed_json)
            # Remove control characters
            repaired = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", repaired)
            # Balance braces
            open_braces = repaired.count("{")
            close_braces = repaired.count("}")
            if open_braces > close_braces:
                repaired += "}" * (open_braces - close_braces)
            elif close_braces > open_braces:
                repaired = "{" * (close_braces - open_braces) + repaired
            return repaired
        except Exception as e:
            print(f"Error repairing JSON: {str(e)}")
            return malformed_json

    def _extract_json_object(self, text) -> str:
        matches = re.findall(r"({.*?})", text, re.DOTALL)
        if matches:
            return max(matches, key=len)
        return None

    def _repair_and_load_json(self, raw_json) -> dict:
        cleaned_json = self._clean_json_response(raw_json)
        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            repaired_json = self._repair_json(cleaned_json)
            try:
                return json.loads(repaired_json)
            except Exception:
                extracted_json = self._extract_json_object(cleaned_json)
                if extracted_json:
                    try:
                        return json.loads(extracted_json)
                    except Exception:
                        pass
                return {"error": "Failed to parse JSON", "raw_output": raw_json}

    def _parse_resume(self, resume_text: str) -> dict:
        """Parse resume text and return structured data with enhanced experience extraction."""
        try:
            raw_output = self.processing_chain.invoke({"resume_text": resume_text})

            # Parse the output based on provider type
            result = None
            if self.provider in [LLMProvider.OLLAMA, LLMProvider.HUGGINGFACE]:
                # These providers return string output, needs JSON parsing
                if isinstance(raw_output, str):
                    cleaned_json = self._clean_json_response(raw_output)
                    try:
                        result = json.loads(cleaned_json)
                    except json.JSONDecodeError:
                        repaired_json = self._repair_json(cleaned_json)
                        try:
                            result = json.loads(repaired_json)
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Failed to parse {self.provider.value} JSON response: {str(e)}"
                            )
                            return {
                                "error": "Failed to parse response",
                                "raw_output": str(raw_output),
                            }
                elif isinstance(raw_output, dict):
                    result = raw_output
            else:
                # API-based providers with JsonOutputParser return dict directly
                if isinstance(raw_output, dict):
                    result = raw_output
                else:
                    cleaned_json = self._clean_json_response(raw_output)
                    try:
                        result = json.loads(cleaned_json)
                    except json.JSONDecodeError:
                        repaired_json = self._repair_json(cleaned_json)
                        try:
                            result = json.loads(repaired_json)
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Failed to parse {self.provider.value} JSON response: {str(e)}"
                            )
                            return {
                                "error": "Failed to parse response",
                                "raw_output": str(raw_output),
                            }

            # Enhanced experience processing if result is valid and experience extractor is available
            if (
                result
                and not result.get("error")
                and hasattr(self, "experience_extractor")
                and self.experience_extractor
            ):
                try:
                    logger.info("Running enhanced experience extraction...")
                    experience_data = self.experience_extractor.extract_experience(
                        resume_text
                    )

                    if experience_data and not experience_data.get("error"):
                        # Update result with enhanced experience data
                        if experience_data.get("experiences"):
                            result["experience"] = experience_data["experiences"]

                        # Set total experience using the specialized calculator
                        result["total_experience"] = experience_data.get(
                            "total_experience_text", "Not calculated"
                        )

                        # Add metadata about the calculation
                        result["experience_metadata"] = {
                            "extraction_confidence": experience_data.get(
                                "extraction_confidence", "unknown"
                            ),
                            "calculation_method": experience_data.get(
                                "calculation_method", "unknown"
                            ),
                            "total_years": experience_data.get("total_years", 0),
                            "total_months": experience_data.get("total_months", 0),
                        }

                        logger.info(
                            f"Enhanced experience calculation: {result['total_experience']}"
                        )
                    else:
                        logger.warning(
                            "Enhanced experience extraction failed, using basic calculation"
                        )
                        result = self._basic_experience_calculation(result)

                except Exception as e:
                    logger.error(f"Enhanced experience extraction error: {e}")
                    result = self._basic_experience_calculation(result)
            else:
                # Fallback to basic experience calculation
                if result and not result.get("error"):
                    result = self._basic_experience_calculation(result)

            return result

        except Exception as e:
            logger.error(f"Error parsing resume with {self.provider.value}: {str(e)}")
            return {"error": str(e)}

    def _basic_experience_calculation(self, result: dict) -> dict:
        """Basic experience calculation as fallback"""
        try:
            # Import the existing experience calculator
            from Expericecal.total_exp import calculator, format_experience

            if "experience" in result and isinstance(result["experience"], list):
                # Convert experience format for the calculator
                experience_data = {"experience": result["experience"]}
                total_years, total_months = calculator.calculate_experience(
                    experience_data
                )
                result["total_experience"] = format_experience(
                    total_years, total_months
                )

                # Add basic metadata
                result["experience_metadata"] = {
                    "extraction_confidence": "medium",
                    "calculation_method": "basic_calculator",
                    "total_years": total_years,
                    "total_months": total_months,
                }

                logger.info(
                    f"Basic experience calculation: {result['total_experience']}"
                )
            else:
                result["total_experience"] = "No experience data found"
                result["experience_metadata"] = {
                    "extraction_confidence": "low",
                    "calculation_method": "no_data",
                    "total_years": 0,
                    "total_months": 0,
                }

            return result

        except Exception as e:
            logger.error(f"Basic experience calculation failed: {e}")
            result["total_experience"] = "Experience calculation failed"
            return result

    def process_resume(self, resume_text: str) -> Dict:
        """Process a resume and handle provider-specific logic."""
        if not resume_text or not resume_text.strip():
            error_msg = "Resume text cannot be empty"
            logger.error(error_msg)
            return {"error": error_msg}

        # Get max length based on provider
        max_length = self._get_max_context_length()

        # Truncate if too long
        if len(resume_text) > max_length:
            logger.warning(
                f"Resume text truncated from {len(resume_text)} to {max_length} characters"
            )
            resume_text = resume_text[:max_length]

        if self.provider in [LLMProvider.OLLAMA, LLMProvider.HUGGINGFACE]:
            # Direct processing for providers without API key rotation
            try:
                logger.debug(f"Processing resume with {self.provider.value}")
                parsed_data = self._parse_resume(resume_text)

                if "error" not in parsed_data:
                    logger.info(
                        f"Successfully processed resume using {self.provider.value}"
                    )
                    return parsed_data
                else:
                    logger.error(
                        f"{self.provider.value} processing failed: {parsed_data.get('error')}"
                    )
                    return parsed_data

            except Exception as e:
                error_msg = f"{self.provider.value} processing error: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg}
        else:
            # API key rotation logic for API-based providers
            return self._process_with_api_rotation(resume_text)

    def _get_max_context_length(self) -> int:
        """Get maximum context length for current provider"""
        if self.provider == LLMProvider.OLLAMA:
            return self.ollama_config.max_context_length
        elif self.provider == LLMProvider.GROQ_CLOUD:
            return self.groq_config.max_context_length
        elif self.provider == LLMProvider.OPENAI:
            return self.openai_config.max_context_length
        elif self.provider == LLMProvider.GOOGLE_GEMINI:
            return self.google_config.max_context_length
        elif self.provider == LLMProvider.HUGGINGFACE:
            return self.huggingface_config.max_context_length
        else:
            return 8000  # Default

    def _get_max_retries(self) -> int:
        """Get maximum retries for current provider"""
        if self.provider == LLMProvider.GROQ_CLOUD:
            return self.groq_config.max_retries
        elif self.provider == LLMProvider.OPENAI:
            return self.openai_config.max_retries
        elif self.provider == LLMProvider.GOOGLE_GEMINI:
            return self.google_config.max_retries
        else:
            return 3  # Default

    def _process_with_api_rotation(self, resume_text: str) -> Dict:
        """Process resume with API key rotation for API-based providers"""
        retry_count = 0
        max_retries = self._get_max_retries()

        while retry_count < max_retries:
            try:
                logger.debug(
                    f"Processing resume with {self.provider.value} API key index {self.current_key_index}, attempt {retry_count + 1}"
                )
                parsed_data = self._parse_resume(resume_text)

                if "error" not in parsed_data:
                    self.api_usage[self.api_keys[self.current_key_index]] += 1
                    logger.info(
                        f"Successfully processed resume using {self.provider.value}"
                    )
                    return parsed_data

                # If we got an error in the parsed data, try next key if available
                if not self._rotate_to_next_key():
                    logger.error("All API keys exhausted")
                    return {
                        "error": "All API keys exhausted",
                        "api_usage": self.api_usage,
                        "last_error": parsed_data.get("error"),
                    }

                retry_count += 1

            except Exception as e:
                error_msg = str(e).lower()
                logger.error(
                    f"Error with API key {self.api_keys[self.current_key_index]}: {error_msg}"
                )

                if self._should_rotate_key(error_msg):
                    if not self._rotate_to_next_key():
                        return {
                            "error": "All API keys exhausted",
                            "api_usage": self.api_usage,
                            "last_error": error_msg,
                        }
                    retry_count += 1
                else:
                    return {
                        "error": "Unexpected error",
                        "details": error_msg,
                        "api_usage": self.api_usage,
                    }

        return {
            "error": f"Failed after {max_retries} retries",
            "api_usage": self.api_usage,
        }

    def _should_rotate_key(self, error_msg: str) -> bool:
        """Check if we should rotate to the next API key based on error message."""
        rotation_triggers = [
            "rate limit",
            "quota exceeded",
            "too many requests",
            "organization_restricted",
        ]
        return any(trigger in error_msg for trigger in rotation_triggers)

    def _rotate_to_next_key(self) -> bool:
        """Rotate to the next available API key."""
        self.current_key_index += 1
        if self.current_key_index < len(self.api_keys):
            try:
                if self.provider == LLMProvider.GROQ_CLOUD:
                    self.processing_chain = self._setup_groq_chain(
                        self.api_keys[self.current_key_index]
                    )
                elif self.provider == LLMProvider.OPENAI:
                    self.processing_chain = self._setup_openai_chain(
                        self.api_keys[self.current_key_index]
                    )
                elif self.provider == LLMProvider.GOOGLE_GEMINI:
                    self.processing_chain = self._setup_google_chain(
                        self.api_keys[self.current_key_index]
                    )
                else:
                    logger.error(
                        f"API key rotation not supported for {self.provider.value}"
                    )
                    return False

                logger.info(f"Switched to API key index: {self.current_key_index}")
                return True
            except Exception as e:
                logger.error(
                    f"Failed to setup processing chain with new API key: {str(e)}"
                )
                return False

        logger.warning("No more API keys available for rotation")
        return False


def main():
    """Main function to demonstrate resume parsing."""
    sample_resume = """
  RESUME YADAV PANAKJ INDRESHKUMAR Email: yadavanush1234@gmail.com Phone: 9023891599 C -499, umiyanagar behind taxshila school Vastral road – ahmedabad -382418 CareerObjective Todevelop career with an organization which provides me excellent opportunity and enable me tolearn skill to achive organization's goal Personal Details  Full Name : YADAV PANKAJ INDRESHKUMAR  Date of Birth : 14/05/1993  Gender : male  Marital Status : Married  Nationality : Indian  Languages Known : Hindi, English, Gujarati  Hobbies : Reading Work Experience  I Have Two Years Experience (BHARAT PETROLEUM ) As Oil Department Supervisor  I Have ONE Years Experience ( H D B FINACE SERVICES ) As Sales Executive  I Have One Years Experience (MAY GATE SOFTWARE ) As Sales Executive  I Have One Years Experience ( BY U Me – SHOREA SOFECH PRIVATE LTD ) As Sales Executive Education Details Pass Out 2008 - CGPA/Percentage : 51.00% G.S.E.B Pass Out 2010 - CGPA/Percentage : 55.00% G.H.S.E.B Pass Out 2022 – Running Gujarat.uni Interests/Hobbies Listening, music, traveling Declaration I hereby declare that all the details furnished above are true to the best of my knowledge andbelief. Date://2019Place: odhav
    """

    try:
        parser = ResumeParser()
        result = parser.process_resume(sample_resume)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
