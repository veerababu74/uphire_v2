"""
Excel Resume Parser Module

This module provides functionality to parse resumes from Excel/XLSX files.
It preprocesses Excel data, handles duplicate headers, and integrates with
the existing resume parsing and duplicate detection pipeline.
"""

__version__ = "1.0.0"
__author__ = "UPHire Team"

# Only import ExcelProcessor by default to avoid LangChain dependencies
from .excel_processor import ExcelProcessor

# Other imports are available but not imported by default to avoid dependency issues
__all__ = ["ExcelProcessor"]


# Use lazy imports for other components to avoid LangChain dependency issues
def get_excel_resume_parser():
    """Lazy import for ExcelResumeParser to avoid dependency issues."""
    from .excel_resume_parser import ExcelResumeParser

    return ExcelResumeParser


def get_excel_resume_parser_manager():
    """Lazy import for ExcelResumeParserManager to avoid dependency issues."""
    from .main import ExcelResumeParserManager

    return ExcelResumeParserManager
