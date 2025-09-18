"""
Enhanced Column Mapper for Excel Resume Parser

This module provides intelligent column mapping capabilities to automatically detect
and map various column names to standard resume fields using fuzzy matching,
semantic analysis, and pattern recognition.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from difflib import SequenceMatcher
from collections import defaultdict
import logging
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("enhanced_column_mapper")


class EnhancedColumnMapper:
    """
    Intelligent column mapper that can detect and map various column names
    to standard resume fields using multiple strategies.
    """

    def __init__(self):
        """Initialize the enhanced column mapper."""
        self.field_mappings = self._initialize_field_mappings()
        self.pattern_mappings = self._initialize_pattern_mappings()
        self.semantic_keywords = self._initialize_semantic_keywords()
        self.data_type_hints = self._initialize_data_type_hints()

    def _initialize_field_mappings(self) -> Dict[str, List[str]]:
        """
        Initialize comprehensive field mappings for common resume fields.

        Returns:
            Dictionary mapping standard fields to possible column variations
        """
        return {
            # Personal Information
            "name": [
                "name",
                "full_name",
                "candidate_name",
                "employee_name",
                "person_name",
                "applicant_name",
                "first_name",
                "fname",
                "firstname",
                "full name",
                "candidate name",
                "employee name",
                "person name",
                "applicant name",
                "nombre",
                "nom",
                "नाम",
                "姓名",
                "اسم",
            ],
            "first_name": [
                "first_name",
                "fname",
                "firstname",
                "given_name",
                "forename",
                "first name",
                "given name",
                "fore name",
                "प्रथम_नाम",
                "名",
            ],
            "last_name": [
                "last_name",
                "lname",
                "lastname",
                "surname",
                "family_name",
                "last name",
                "family name",
                "अंतिम_नाम",
                "姓",
                "الاسم_الأخير",
            ],
            "middle_name": [
                "middle_name",
                "mname",
                "middlename",
                "middle name",
                "मध्य_नाम",
            ],
            # Contact Information
            "email": [
                "email",
                "email_address",
                "email_id",
                "e_mail",
                "mail",
                "mail_id",
                "email address",
                "email id",
                "e mail",
                "mail id",
                "electronic_mail",
                "ईमेल",
                "邮箱",
                "البريد_الإلكتروني",
            ],
            "phone": [
                "phone",
                "phone_number",
                "mobile",
                "mobile_number",
                "contact",
                "contact_number",
                "cell",
                "cell_number",
                "telephone",
                "tel",
                "phone number",
                "mobile number",
                "contact number",
                "cell number",
                "mob",
                "mob_no",
                "phone_no",
                "ph_no",
                "फोन",
                "手机",
                "رقم_الهاتف",
            ],
            "alternate_phone": [
                "alternate_phone",
                "alt_phone",
                "secondary_phone",
                "phone2",
                "mobile2",
                "alternate phone",
                "alt phone",
                "secondary phone",
                "backup_phone",
            ],
            # Location Information
            "address": [
                "address",
                "full_address",
                "home_address",
                "permanent_address",
                "current_address",
                "address_line1",
                "addr",
                "location",
                "residence",
                "full address",
                "home address",
                "permanent address",
                "current address",
                "address line1",
                "पता",
                "地址",
                "العنوان",
            ],
            "city": [
                "city",
                "current_city",
                "location",
                "place",
                "town",
                "current city",
                "city_name",
                "home_city",
                "residence_city",
                "शहर",
                "城市",
                "المدينة",
            ],
            "state": [
                "state",
                "province",
                "region",
                "current_state",
                "home_state",
                "state_name",
                "राज्य",
                "省",
                "الولاية",
            ],
            "country": [
                "country",
                "nation",
                "current_country",
                "home_country",
                "nationality",
                "country_name",
                "देश",
                "国家",
                "البلد",
            ],
            "pincode": [
                "pincode",
                "pin_code",
                "zip",
                "zip_code",
                "postal_code",
                "pin",
                "pin code",
                "zip code",
                "postal code",
                "पिनकोड",
                "邮编",
                "الرمز_البريدي",
            ],
            # Professional Information
            "experience": [
                "experience",
                "total_experience",
                "years_of_experience",
                "work_experience",
                "professional_experience",
                "exp",
                "yoe",
                "years_exp",
                "total_exp",
                "total experience",
                "years of experience",
                "work experience",
                "professional experience",
                "years exp",
                "total exp",
                "अनुभव",
                "经验",
                "الخبرة",
            ],
            "current_role": [
                "current_role",
                "designation",
                "position",
                "job_title",
                "title",
                "role",
                "current_designation",
                "current_position",
                "current_job",
                "present_role",
                "current role",
                "current designation",
                "current position",
                "current job",
                "present role",
                "job title",
                "वर्तमान_पद",
                "当前职位",
                "المنصب_الحالي",
            ],
            "current_company": [
                "current_company",
                "company",
                "organization",
                "employer",
                "workplace",
                "current_employer",
                "current_organization",
                "present_company",
                "firm",
                "current company",
                "current employer",
                "current organization",
                "present company",
                "वर्तमान_कंपनी",
                "当前公司",
                "الشركة_الحالية",
            ],
            "previous_company": [
                "previous_company",
                "last_company",
                "former_company",
                "ex_company",
                "previous company",
                "last company",
                "former company",
                "ex company",
                "पिछली_कंपनी",
                "前公司",
            ],
            # Skills and Expertise
            "skills": [
                "skills",
                "technical_skills",
                "key_skills",
                "expertise",
                "competencies",
                "abilities",
                "skill_set",
                "core_skills",
                "primary_skills",
                "main_skills",
                "technical skills",
                "key skills",
                "skill set",
                "core skills",
                "primary skills",
                "main skills",
                "technologies",
                "tools",
                "कौशल",
                "技能",
                "المهارات",
            ],
            "technical_skills": [
                "technical_skills",
                "tech_skills",
                "programming_skills",
                "it_skills",
                "software_skills",
                "technical skills",
                "tech skills",
                "programming skills",
                "it skills",
                "software skills",
                "तकनीकी_कौशल",
                "技术技能",
            ],
            "soft_skills": [
                "soft_skills",
                "interpersonal_skills",
                "communication_skills",
                "leadership_skills",
                "soft skills",
                "interpersonal skills",
                "communication skills",
                "leadership skills",
                "व्यक्तित्व_कौशल",
                "软技能",
            ],
            # Education
            "education": [
                "education",
                "qualification",
                "degree",
                "academic_qualification",
                "studies",
                "educational_background",
                "academic_background",
                "schooling",
                "learning",
                "academic qualification",
                "educational background",
                "academic background",
                "शिक्षा",
                "教育",
                "التعليم",
            ],
            "highest_qualification": [
                "highest_qualification",
                "highest_degree",
                "top_qualification",
                "max_qualification",
                "highest qualification",
                "highest degree",
                "top qualification",
                "max qualification",
                "उच्चतम_योग्यता",
                "最高学历",
            ],
            "degree": [
                "degree",
                "graduation",
                "diploma",
                "certificate",
                "course",
                "program",
                "academic_degree",
                "educational_degree",
                "academic degree",
                "educational degree",
                "डिग्री",
                "学位",
                "الدرجة",
            ],
            "college": [
                "college",
                "university",
                "institute",
                "school",
                "institution",
                "academy",
                "educational_institution",
                "alma_mater",
                "educational institution",
                "alma mater",
                "कॉलेज",
                "大学",
                "الجامعة",
            ],
            "graduation_year": [
                "graduation_year",
                "passing_year",
                "completion_year",
                "year_of_graduation",
                "graduation year",
                "passing year",
                "completion year",
                "year of graduation",
                "grad_year",
                "passout_year",
                "स्नातक_वर्ष",
                "毕业年份",
            ],
            "percentage": [
                "percentage",
                "marks",
                "grade",
                "score",
                "cgpa",
                "gpa",
                "result",
                "academic_score",
                "academic score",
                "प्रतिशत",
                "百分比",
                "النسبة",
            ],
            # Salary and Compensation
            "current_salary": [
                "current_salary",
                "salary",
                "ctc",
                "current_ctc",
                "compensation",
                "pay",
                "current_compensation",
                "present_salary",
                "current salary",
                "current ctc",
                "current compensation",
                "present salary",
                "वर्तमान_वेतन",
                "当前薪资",
                "الراتب_الحالي",
            ],
            "expected_salary": [
                "expected_salary",
                "expected_ctc",
                "salary_expectation",
                "expected_compensation",
                "desired_salary",
                "target_salary",
                "expected salary",
                "expected ctc",
                "salary expectation",
                "expected compensation",
                "desired salary",
                "target salary",
                "अपेक्षित_वेतन",
                "期望薪资",
                "الراتب_المتوقع",
            ],
            "last_salary": [
                "last_salary",
                "previous_salary",
                "former_salary",
                "ex_salary",
                "last salary",
                "previous salary",
                "former salary",
                "ex salary",
                "पिछला_वेतन",
                "上一份薪资",
            ],
            # Work Preferences
            "notice_period": [
                "notice_period",
                "notice",
                "availability",
                "joining_time",
                "notice_period_days",
                "notice period",
                "joining time",
                "notice period days",
                "serving_notice",
                "serving notice",
                "नोटिस_अवधि",
                "通知期",
                "فترة_الإشعار",
            ],
            "preferred_location": [
                "preferred_location",
                "preferred_city",
                "work_location",
                "job_location",
                "preferred location",
                "preferred city",
                "work location",
                "job location",
                "पसंदीदा_स्थान",
                "首选地点",
            ],
            "work_mode": [
                "work_mode",
                "work_type",
                "job_type",
                "employment_type",
                "working_mode",
                "work mode",
                "work type",
                "job type",
                "employment type",
                "working mode",
                "remote",
                "onsite",
                "hybrid",
                "कार्य_मोड",
                "工作模式",
            ],
            # Additional Information
            "age": ["age", "years", "age_years", "age years", "उम्र", "年龄", "العمر"],
            "date_of_birth": [
                "date_of_birth",
                "dob",
                "birth_date",
                "birthdate",
                "date of birth",
                "birth date",
                "जन्म_तिथि",
                "出生日期",
                "تاريخ_الميلاد",
            ],
            "gender": ["gender", "sex", "लिंग", "性别", "الجنس"],
            "marital_status": [
                "marital_status",
                "marriage_status",
                "married",
                "single",
                "marital status",
                "marriage status",
                "वैवाहिक_स्थिति",
                "婚姻状况",
            ],
            "languages": [
                "languages",
                "language",
                "known_languages",
                "language_skills",
                "linguistics",
                "known languages",
                "language skills",
                "भाषा",
                "语言",
                "اللغات",
            ],
            "certifications": [
                "certifications",
                "certificates",
                "certification",
                "professional_certification",
                "professional certification",
                "प्रमाणपत्र",
                "认证",
                "الشهادات",
            ],
            "projects": [
                "projects",
                "project",
                "work_projects",
                "personal_projects",
                "portfolio",
                "work projects",
                "personal projects",
                "परियोजना",
                "项目",
                "المشاريع",
            ],
            "achievements": [
                "achievements",
                "accomplishments",
                "awards",
                "recognition",
                "honors",
                "उपलब्धि",
                "成就",
                "الإنجازات",
            ],
            "hobbies": [
                "hobbies",
                "interests",
                "hobby",
                "interest",
                "personal_interests",
                "personal interests",
                "शौक",
                "爱好",
                "الهوايات",
            ],
            "references": [
                "references",
                "reference",
                "referral",
                "recommendation",
                "संदर्भ",
                "推荐人",
            ],
            # Resume Metadata
            "resume_source": [
                "resume_source",
                "source",
                "platform",
                "portal",
                "website",
                "resume source",
                "रिज्यूमे_स्रोत",
                "简历来源",
            ],
            "application_date": [
                "application_date",
                "apply_date",
                "submission_date",
                "date_applied",
                "application date",
                "apply date",
                "submission date",
                "date applied",
                "आवेदन_तिथि",
                "申请日期",
            ],
            "job_id": [
                "job_id",
                "position_id",
                "vacancy_id",
                "requisition_id",
                "job_code",
                "job id",
                "position id",
                "vacancy id",
                "requisition id",
                "job code",
                "नौकरी_आईडी",
                "职位ID",
            ],
        }

    def _initialize_pattern_mappings(self) -> Dict[str, str]:
        """
        Initialize regex patterns for field detection.

        Returns:
            Dictionary mapping field types to regex patterns
        """
        return {
            "email": r".*(?:email|mail|e[-_]?mail).*",
            "phone": r".*(?:phone|mobile|contact|tel|cell|mob)(?:_?(?:no|number))?.*",
            "experience": r".*(?:exp|experience|years?)(?:_?(?:of|in))?(?:_?(?:work|job|professional))?.*",
            "salary": r".*(?:salary|ctc|compensation|pay|package).*",
            "education": r".*(?:education|qualification|degree|graduation|study|studies).*",
            "skills": r".*(?:skill|tech|technical|competenc|abilit|expertis).*",
            "name": r".*(?:name|candidate|person|applicant|employee)(?:_?name)?.*",
            "company": r".*(?:company|organization|employer|firm|workplace).*",
            "location": r".*(?:location|city|place|address|town).*",
            "date": r".*(?:date|time|year|month|day).*",
            "notice": r".*(?:notice|availability|joining)(?:_?(?:period|time|days?))?.*",
        }

    def _initialize_semantic_keywords(self) -> Dict[str, List[str]]:
        """
        Initialize semantic keywords for enhanced matching.

        Returns:
            Dictionary mapping categories to related keywords
        """
        return {
            "personal": [
                "personal",
                "individual",
                "candidate",
                "applicant",
                "person",
                "profile",
            ],
            "contact": ["contact", "communication", "reach", "connect", "touch"],
            "professional": [
                "professional",
                "work",
                "job",
                "career",
                "employment",
                "occupation",
            ],
            "educational": [
                "educational",
                "academic",
                "study",
                "learning",
                "school",
                "college",
            ],
            "technical": [
                "technical",
                "technology",
                "IT",
                "software",
                "hardware",
                "digital",
            ],
            "financial": [
                "financial",
                "money",
                "salary",
                "compensation",
                "pay",
                "package",
            ],
            "temporal": ["time", "period", "duration", "date", "year", "month", "day"],
        }

    def _initialize_data_type_hints(self) -> Dict[str, str]:
        """
        Initialize data type hints for different field types.

        Returns:
            Dictionary mapping field names to expected data types
        """
        return {
            "name": "text",
            "first_name": "text",
            "last_name": "text",
            "email": "email",
            "phone": "phone",
            "age": "number",
            "experience": "experience",
            "current_salary": "currency",
            "expected_salary": "currency",
            "date_of_birth": "date",
            "graduation_year": "year",
            "percentage": "percentage",
            "notice_period": "duration",
            "skills": "list",
            "languages": "list",
            "certifications": "list",
            "projects": "list",
            "achievements": "list",
            "hobbies": "list",
        }

    def normalize_column_name(self, column_name: str) -> str:
        """
        Normalize column name for better matching.

        Args:
            column_name: Original column name

        Returns:
            Normalized column name
        """
        if not column_name:
            return ""

        # Convert to lowercase and remove special characters
        normalized = str(column_name).lower().strip()

        # Remove common prefixes/suffixes
        prefixes_to_remove = [
            "candidate_",
            "employee_",
            "person_",
            "user_",
            "applicant_",
        ]
        suffixes_to_remove = ["_name", "_info", "_details", "_data", "_field"]

        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break

        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break

        # Replace common separators with underscores
        normalized = re.sub(r"[-\s\.\(\)\[\]]+", "_", normalized)

        # Remove consecutive underscores
        normalized = re.sub(r"_+", "_", normalized)

        # Remove leading/trailing underscores
        normalized = normalized.strip("_")

        return normalized

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two strings using sequence matching.

        Args:
            text1: First string
            text2: Second string

        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0

        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def fuzzy_match_field(
        self, column_name: str, confidence_threshold: float = 0.6
    ) -> List[Tuple[str, float]]:
        """
        Find potential field matches using fuzzy matching.

        Args:
            column_name: Column name to match
            confidence_threshold: Minimum confidence score for matches

        Returns:
            List of tuples (field_name, confidence_score) sorted by confidence
        """
        normalized_column = self.normalize_column_name(column_name)
        matches = []

        for field_name, variations in self.field_mappings.items():
            max_similarity = 0.0

            # Check direct variations
            for variation in variations:
                similarity = self.calculate_similarity(normalized_column, variation)
                max_similarity = max(max_similarity, similarity)

            # Check against field name itself
            field_similarity = self.calculate_similarity(normalized_column, field_name)
            max_similarity = max(max_similarity, field_similarity)

            if max_similarity >= confidence_threshold:
                matches.append((field_name, max_similarity))

        # Sort by confidence score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches

    def pattern_match_field(self, column_name: str) -> List[Tuple[str, float]]:
        """
        Find potential field matches using regex patterns.

        Args:
            column_name: Column name to match

        Returns:
            List of tuples (field_name, confidence_score)
        """
        normalized_column = self.normalize_column_name(column_name)
        matches = []

        for pattern_type, pattern in self.pattern_mappings.items():
            if re.search(pattern, normalized_column, re.IGNORECASE):
                # Pattern matches get a base confidence of 0.8
                matches.append((pattern_type, 0.8))

        return matches

    def semantic_match_field(self, column_name: str) -> List[Tuple[str, float]]:
        """
        Find potential field matches using semantic keywords.

        Args:
            column_name: Column name to match

        Returns:
            List of tuples (field_name, confidence_score)
        """
        normalized_column = self.normalize_column_name(column_name)
        matches = []

        # Check semantic categories
        for category, keywords in self.semantic_keywords.items():
            category_score = 0.0

            for keyword in keywords:
                if keyword in normalized_column:
                    category_score = max(category_score, 0.7)

            if category_score > 0:
                # Map semantic categories to field types
                category_field_mapping = {
                    "personal": ["name", "first_name", "last_name", "age", "gender"],
                    "contact": ["email", "phone", "address", "city"],
                    "professional": ["current_role", "current_company", "experience"],
                    "educational": [
                        "education",
                        "degree",
                        "college",
                        "graduation_year",
                    ],
                    "technical": ["skills", "technical_skills", "certifications"],
                    "financial": ["current_salary", "expected_salary"],
                    "temporal": ["date_of_birth", "graduation_year", "notice_period"],
                }

                if category in category_field_mapping:
                    for field in category_field_mapping[category]:
                        matches.append(
                            (field, category_score * 0.8)
                        )  # Reduce confidence for semantic matches

        return matches

    def map_columns(
        self, column_names: List[str], confidence_threshold: float = 0.6
    ) -> Dict[str, Dict[str, Any]]:
        """
        Map a list of column names to standard resume fields.

        Args:
            column_names: List of column names to map
            confidence_threshold: Minimum confidence threshold for mapping

        Returns:
            Dictionary mapping original column names to field information
        """
        column_mapping = {}
        field_assignments = {}  # Track which fields have been assigned

        logger.info(f"Mapping {len(column_names)} columns to standard fields")

        for column_name in column_names:
            if not column_name or str(column_name).strip() == "":
                continue

            column_name = str(column_name).strip()

            # Get matches from different strategies
            fuzzy_matches = self.fuzzy_match_field(column_name, confidence_threshold)
            pattern_matches = self.pattern_match_field(column_name)
            semantic_matches = self.semantic_match_field(column_name)

            # Combine and score matches
            all_matches = defaultdict(float)

            # Weight fuzzy matches highest (most reliable)
            for field, score in fuzzy_matches:
                all_matches[field] = max(all_matches[field], score * 1.0)

            # Weight pattern matches medium
            for field, score in pattern_matches:
                all_matches[field] = max(all_matches[field], score * 0.9)

            # Weight semantic matches lowest
            for field, score in semantic_matches:
                all_matches[field] = max(all_matches[field], score * 0.7)

            # Find the best match that hasn't been assigned yet
            best_field = None
            best_score = confidence_threshold

            sorted_matches = sorted(
                all_matches.items(), key=lambda x: x[1], reverse=True
            )

            for field, score in sorted_matches:
                if score >= confidence_threshold:
                    # Prefer unassigned fields, but allow duplicates with lower confidence
                    if field not in field_assignments:
                        best_field = field
                        best_score = score
                        break
                    elif (
                        score > field_assignments[field]["confidence"] * 1.1
                    ):  # Significantly better match
                        # Reassign field to this column
                        best_field = field
                        best_score = score
                        break

            # Create mapping entry
            mapping_info = {
                "original_name": column_name,
                "normalized_name": self.normalize_column_name(column_name),
                "mapped_field": best_field,
                "confidence": best_score,
                "data_type": (
                    self.data_type_hints.get(best_field, "text")
                    if best_field
                    else "text"
                ),
                "all_matches": dict(
                    sorted_matches[:5]
                ),  # Keep top 5 matches for reference
            }

            column_mapping[column_name] = mapping_info

            # Track field assignment
            if best_field:
                field_assignments[best_field] = {
                    "column": column_name,
                    "confidence": best_score,
                }

                logger.info(
                    f"Mapped '{column_name}' -> '{best_field}' (confidence: {best_score:.3f})"
                )
            else:
                logger.warning(
                    f"Could not map column '{column_name}' (max confidence: {max(all_matches.values()) if all_matches else 0:.3f})"
                )

        # Log summary
        mapped_count = sum(
            1 for mapping in column_mapping.values() if mapping["mapped_field"]
        )
        logger.info(f"Successfully mapped {mapped_count}/{len(column_names)} columns")

        return column_mapping

    def get_unmapped_columns(
        self, column_mapping: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Get list of columns that couldn't be mapped to standard fields.

        Args:
            column_mapping: Column mapping dictionary from map_columns()

        Returns:
            List of unmapped column names
        """
        unmapped = []
        for column_name, mapping_info in column_mapping.items():
            if not mapping_info["mapped_field"]:
                unmapped.append(column_name)

        return unmapped

    def get_mapped_fields(
        self, column_mapping: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Get simplified mapping of original column names to mapped field names.

        Args:
            column_mapping: Column mapping dictionary from map_columns()

        Returns:
            Dictionary mapping original column names to field names
        """
        simple_mapping = {}
        for column_name, mapping_info in column_mapping.items():
            if mapping_info["mapped_field"]:
                simple_mapping[column_name] = mapping_info["mapped_field"]

        return simple_mapping

    def suggest_missing_fields(
        self, column_mapping: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Suggest important fields that are missing from the mapped columns.

        Args:
            column_mapping: Column mapping dictionary from map_columns()

        Returns:
            List of suggested missing field names
        """
        # Define important fields that should typically be present
        important_fields = [
            "name",
            "email",
            "phone",
            "experience",
            "current_role",
            "skills",
            "education",
            "current_company",
        ]

        mapped_fields = set(
            mapping_info["mapped_field"]
            for mapping_info in column_mapping.values()
            if mapping_info["mapped_field"]
        )

        missing_fields = [
            field for field in important_fields if field not in mapped_fields
        ]

        if missing_fields:
            logger.warning(f"Missing important fields: {missing_fields}")

        return missing_fields
