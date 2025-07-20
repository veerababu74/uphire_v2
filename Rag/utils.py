import json
from typing import List, Dict, Any
from core.helpers import JSONEncoder


def safe_int(value, default=0):
    """Safely convert value to int, handling None values"""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value, default=0.0):
    """Safely convert value to float, handling None values"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_str(value, default=""):
    """Safely convert value to string, handling None values"""
    if value is None:
        return default
    return str(value)


class DocumentProcessor:
    """Utility class for document processing and normalization"""

    @staticmethod
    def normalize_field_value(value) -> str:
        """Normalize field values for consistent processing"""
        if value is None:
            return ""
        if isinstance(value, (list, dict)):
            return json.dumps(value, cls=JSONEncoder)
        return str(value).strip()

    @staticmethod
    def normalize_list_field(value) -> List[str]:
        """Normalize list fields for consistent processing"""
        if not value:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [
                DocumentProcessor.normalize_field_value(item) for item in value if item
            ]
        return [str(value)]

    @staticmethod
    def format_complete_document(doc: Dict) -> Dict:
        """Format a complete document according to the specified structure"""
        return {
            "_id": safe_str(doc.get("_id"), ""),
            "user_id": safe_str(doc.get("user_id"), ""),
            "username": safe_str(doc.get("username"), ""),
            "contact_details": DocumentProcessor._format_contact_details(doc),
            "total_experience": safe_str(doc.get("total_experience"), ""),
            "notice_period": safe_str(doc.get("notice_period"), ""),
            "currency": safe_str(doc.get("currency"), ""),
            "pay_duration": safe_str(doc.get("pay_duration"), ""),
            "current_salary": safe_float(doc.get("current_salary"), 0),
            "hike": safe_float(doc.get("hike"), 0),
            "expected_salary": safe_float(doc.get("expected_salary"), 0),
            "skills": doc.get("skills", []),
            "may_also_known_skills": doc.get("may_also_known_skills", []),
            "labels": doc.get("labels", []),
            "experience": doc.get("experience", []),
            "academic_details": doc.get("academic_details", []),
            "source": safe_str(doc.get("source"), ""),
            "last_working_day": safe_str(doc.get("last_working_day"), ""),
            "is_tier1_mba": bool(doc.get("is_tier1_mba", False)),
            "is_tier1_engineering": bool(doc.get("is_tier1_engineering", False)),
            "comment": safe_str(doc.get("comment"), ""),
            "exit_reason": safe_str(doc.get("exit_reason"), ""),
        }

    @staticmethod
    def _format_contact_details(doc: Dict) -> Dict:
        """Format contact details from document"""
        contact_details = doc.get("contact_details", {})
        return {
            "name": safe_str(contact_details.get("name"), ""),
            "email": safe_str(contact_details.get("email"), ""),
            "phone": safe_str(contact_details.get("phone"), ""),
            "alternative_phone": safe_str(contact_details.get("alternative_phone"), ""),
            "current_city": safe_str(contact_details.get("current_city"), ""),
            "looking_for_jobs_in": contact_details.get("looking_for_jobs_in", []),
            "gender": safe_str(contact_details.get("gender"), ""),
            "age": safe_int(contact_details.get("age"), 0),
            "naukri_profile": safe_str(contact_details.get("naukri_profile"), ""),
            "linkedin_profile": safe_str(contact_details.get("linkedin_profile"), ""),
            "portfolio_link": safe_str(contact_details.get("portfolio_link"), ""),
            "pan_card": safe_str(contact_details.get("pan_card"), ""),
            "aadhar_card": safe_str(contact_details.get("aadhar_card"), ""),
        }
