"""
Configuration for AI Candidate Ranking System

This file contains all configurable parameters for the AI candidate ranking feature.
Modify these values to customize the ranking behavior according to your requirements.

Author: Uphire Team
Version: 1.0.0
"""

# ================================
# CORE RANKING CONFIGURATION
# ================================

# Score threshold below which candidates are automatically rejected
REJECTION_THRESHOLD = 40.0

# Maximum number of candidates that can be analyzed in a single request
MAX_CANDIDATES_LIMIT = 100

# Default number of candidates to analyze if not specified
DEFAULT_MAX_CANDIDATES = 50

# ================================
# STATUS MESSAGES
# ================================

# Status assigned to candidates who meet the threshold
ACCEPTED_STATUS = "CV Accepted - Under Review"

# Status assigned to candidates below the threshold
REJECTED_STATUS = "CV Rejected - In Process"

# ================================
# SCORING WEIGHTS
# ================================

# Weights for different scoring components (must sum to 1.0)
SCORING_WEIGHTS = {
    "skills": 0.40,  # 40% - Technical and soft skills matching
    "experience": 0.30,  # 30% - Years and relevance of experience
    "education": 0.15,  # 15% - Educational background relevance
    "location": 0.10,  # 10% - Geographic compatibility
    "salary": 0.05,  # 5% - Salary expectation alignment
}

# ================================
# SKILLS MATCHING CONFIGURATION
# ================================

# Maximum number of additional skills to show in response
MAX_ADDITIONAL_SKILLS_DISPLAY = 10

# Minimum character length for skill matching
MIN_SKILL_LENGTH = 2

# Case sensitivity for skill matching (False = case insensitive)
CASE_SENSITIVE_SKILLS = False

# ================================
# EXPERIENCE CALCULATION
# ================================

# Percentage of total experience considered relevant when job titles match
RELEVANT_EXPERIENCE_RATIO = 0.7

# Minimum years of experience to be considered experienced
MIN_EXPERIENCE_THRESHOLD = 1.0

# Maximum experience boost percentage for senior roles
MAX_EXPERIENCE_BOOST = 150.0

# ================================
# EDUCATION SCORING
# ================================

# Education level hierarchy and scoring
EDUCATION_LEVELS = {
    "phd": 6,
    "doctorate": 6,
    "master": 5,
    "mba": 5,
    "ms": 5,
    "me": 5,
    "mtech": 5,
    "bachelor": 4,
    "btech": 4,
    "be": 4,
    "bsc": 4,
    "ba": 4,
    "bcom": 4,
    "diploma": 3,
    "12th": 2,
    "intermediate": 2,
    "hsc": 2,
    "10th": 1,
    "ssc": 1,
}

# Default education score when no requirements specified
DEFAULT_EDUCATION_SCORE = 100.0

# Default education score when candidate has no education info
NO_EDUCATION_SCORE = 0.0

# ================================
# LOCATION SCORING
# ================================

# Score for candidates with flexible location preferences
FLEXIBLE_LOCATION_SCORE = 100.0

# Score for candidates with current city information
HAS_LOCATION_INFO_SCORE = 80.0

# Default score when no location information available
NO_LOCATION_INFO_SCORE = 50.0

# ================================
# SALARY SCORING
# ================================

# Maximum reasonable salary hike percentage for full score
REASONABLE_HIKE_THRESHOLD = 30.0

# Moderate salary hike percentage for good score
MODERATE_HIKE_THRESHOLD = 50.0

# Score for reasonable salary expectations (≤30% hike)
REASONABLE_SALARY_SCORE = 100.0

# Score for moderate salary expectations (30-50% hike)
MODERATE_SALARY_SCORE = 80.0

# Score for high salary expectations (>50% hike)
HIGH_SALARY_SCORE = 60.0

# Score when candidate has some salary information
PARTIAL_SALARY_INFO_SCORE = 70.0

# Score when no salary constraints (no salary info)
NO_SALARY_CONSTRAINTS_SCORE = 90.0

# Default salary score for calculation errors
DEFAULT_SALARY_SCORE = 80.0

# ================================
# FILE PROCESSING
# ================================

# Supported file extensions for job description upload
SUPPORTED_FILE_EXTENSIONS = [".txt", ".pdf", ".docx"]

# Minimum length for job description text
MIN_JOB_DESCRIPTION_LENGTH = 50

# Maximum length for job description summary in response
MAX_JOB_SUMMARY_LENGTH = 200

# ================================
# AI PROCESSING
# ================================

# Default context size for LLM processing
DEFAULT_LLM_CONTEXT_SIZE = 5

# Timeout for AI processing (seconds)
AI_PROCESSING_TIMEOUT = 60

# Maximum number of retries for failed AI calls
MAX_AI_RETRIES = 3

# ================================
# RESPONSE CONFIGURATION
# ================================

# Maximum number of missing skills to show in ranking reason
MAX_MISSING_SKILLS_IN_REASON = 3

# Maximum length for ranking reason text
MAX_RANKING_REASON_LENGTH = 200

# Include timestamp in all responses
INCLUDE_TIMESTAMPS = True

# ================================
# STATISTICS CONFIGURATION
# ================================

# Default sample size for statistics calculation
STATISTICS_SAMPLE_SIZE = 100

# Number of top skills to show in statistics
TOP_SKILLS_COUNT = 20

# Experience ranges for distribution analysis
EXPERIENCE_RANGES = {
    "0-2": {"min": 0, "max": 2},
    "2-5": {"min": 2, "max": 5},
    "5-10": {"min": 5, "max": 10},
    "10+": {"min": 10, "max": float("inf")},
}

# ================================
# PERFORMANCE TUNING
# ================================

# Enable caching for improved performance
ENABLE_CACHING = True

# Cache timeout in seconds
CACHE_TIMEOUT = 3600  # 1 hour

# Maximum batch size for database operations
MAX_BATCH_SIZE = 1000

# Enable parallel processing where possible
ENABLE_PARALLEL_PROCESSING = True

# ================================
# LOGGING CONFIGURATION
# ================================

# Log level for ranking operations
RANKING_LOG_LEVEL = "INFO"

# Include detailed timing information in logs
LOG_TIMING_INFO = True

# Log candidate analysis details (disable for privacy)
LOG_CANDIDATE_DETAILS = False

# ================================
# VALIDATION RULES
# ================================

# Minimum user ID length
MIN_USER_ID_LENGTH = 3

# Maximum user ID length
MAX_USER_ID_LENGTH = 50

# Allowed characters in user ID (regex pattern)
USER_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"

# ================================
# CUSTOMIZATION EXAMPLES
# ================================

"""
EXAMPLE CONFIGURATIONS FOR DIFFERENT USE CASES:

1. STRICT TECHNICAL ROLES:
   REJECTION_THRESHOLD = 60.0
   SCORING_WEIGHTS = {
       "skills": 0.60, "experience": 0.30, "education": 0.10, "location": 0.0, "salary": 0.0
   }

2. ENTRY-LEVEL POSITIONS:
   REJECTION_THRESHOLD = 25.0
   SCORING_WEIGHTS = {
       "skills": 0.30, "experience": 0.20, "education": 0.40, "location": 0.05, "salary": 0.05
   }

3. SENIOR MANAGEMENT ROLES:
   REJECTION_THRESHOLD = 50.0
   SCORING_WEIGHTS = {
       "skills": 0.25, "experience": 0.50, "education": 0.15, "location": 0.05, "salary": 0.05
   }

4. LOCATION-SENSITIVE ROLES:
   SCORING_WEIGHTS = {
       "skills": 0.35, "experience": 0.30, "education": 0.10, "location": 0.20, "salary": 0.05
   }
"""

# ================================
# RUNTIME VALIDATION
# ================================


def validate_configuration():
    """Validate configuration parameters at runtime"""

    # Validate scoring weights sum to 1.0
    weights_sum = sum(SCORING_WEIGHTS.values())
    if abs(weights_sum - 1.0) > 0.001:
        raise ValueError(f"SCORING_WEIGHTS must sum to 1.0, got {weights_sum}")

    # Validate rejection threshold
    if not 0 <= REJECTION_THRESHOLD <= 100:
        raise ValueError(
            f"REJECTION_THRESHOLD must be between 0 and 100, got {REJECTION_THRESHOLD}"
        )

    # Validate max candidates limit
    if MAX_CANDIDATES_LIMIT <= 0:
        raise ValueError(
            f"MAX_CANDIDATES_LIMIT must be positive, got {MAX_CANDIDATES_LIMIT}"
        )

    # Validate default max candidates
    if not 1 <= DEFAULT_MAX_CANDIDATES <= MAX_CANDIDATES_LIMIT:
        raise ValueError(
            f"DEFAULT_MAX_CANDIDATES must be between 1 and {MAX_CANDIDATES_LIMIT}"
        )

    return True


# Validate configuration on import
if __name__ == "__main__":
    validate_configuration()
    print("✅ Configuration validation passed!")
else:
    # Validate when imported
    try:
        validate_configuration()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        raise
