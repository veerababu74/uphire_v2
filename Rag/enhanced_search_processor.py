"""
Enhanced Search Processor for improved semantic search accuracy
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from core.custom_logger import CustomLogger

logger = CustomLogger().get_logger("enhanced_search_processor")


@dataclass
class SearchContext:
    """Structured representation of search requirements"""

    role: Optional[str] = None
    domain: Optional[str] = None
    min_experience: Optional[float] = None
    max_experience: Optional[float] = None
    skills: List[str] = None
    location: Optional[str] = None
    min_salary: Optional[float] = None
    max_salary: Optional[float] = None
    education_requirements: List[str] = None
    notice_period: Optional[str] = None

    def __post_init__(self):
        if self.skills is None:
            self.skills = []
        if self.education_requirements is None:
            self.education_requirements = []


class EnhancedSearchProcessor:
    """Enhanced processor for better search query understanding and candidate matching"""

    def __init__(self):
        self.domain_keywords = {
            "fmcg": [
                "fmcg",
                "fast moving consumer goods",
                "consumer goods",
                "retail",
                "fmcg sales",
            ],
            "it": [
                "it",
                "information technology",
                "software",
                "tech",
                "programming",
                "development",
            ],
            "finance": ["finance", "banking", "accounting", "financial", "investment"],
            "healthcare": [
                "healthcare",
                "medical",
                "pharmaceutical",
                "pharma",
                "hospital",
            ],
            "manufacturing": ["manufacturing", "production", "industrial", "factory"],
            "education": [
                "education",
                "teaching",
                "academic",
                "training",
                "university",
            ],
            "sales": [
                "sales",
                "marketing",
                "business development",
                "account management",
            ],
            "hr": ["hr", "human resources", "recruitment", "talent acquisition"],
            "operations": ["operations", "logistics", "supply chain", "procurement"],
        }

        self.role_keywords = {
            "executive": ["executive", "manager", "senior", "lead", "head"],
            "developer": ["developer", "programmer", "engineer", "architect"],
            "analyst": ["analyst", "associate", "consultant"],
            "specialist": ["specialist", "expert", "advisor"],
            "coordinator": ["coordinator", "assistant", "support"],
        }

        self.skill_keywords = {
            "technical": [
                "python",
                "java",
                "javascript",
                "react",
                "angular",
                "node.js",
                "sql",
                "aws",
                "azure",
            ],
            "sales": [
                "b2b sales",
                "b2c sales",
                "lead generation",
                "crm",
                "sales management",
            ],
            "marketing": [
                "digital marketing",
                "seo",
                "content marketing",
                "social media",
                "ppc",
            ],
            "finance": [
                "financial analysis",
                "budgeting",
                "forecasting",
                "excel",
                "tally",
            ],
            "management": [
                "team management",
                "project management",
                "leadership",
                "strategy",
            ],
        }

    def parse_query(self, query: str) -> SearchContext:
        """Parse search query into structured context"""
        query_lower = query.lower()
        context = SearchContext()

        # Extract role
        context.role = self._extract_role(query_lower)

        # Extract domain
        context.domain = self._extract_domain(query_lower)

        # Extract experience
        context.min_experience, context.max_experience = self._extract_experience(
            query_lower
        )

        # Extract skills
        context.skills = self._extract_skills(query_lower)

        # Extract salary
        context.min_salary, context.max_salary = self._extract_salary(query_lower)

        # Extract location
        context.location = self._extract_location(query_lower)

        # Extract education requirements
        context.education_requirements = self._extract_education(query_lower)

        # Extract notice period
        context.notice_period = self._extract_notice_period(query_lower)

        logger.info(f"Parsed search context: {context}")
        return context

    def _extract_role(self, query: str) -> Optional[str]:
        """Extract job role from query"""
        # Look for specific role patterns
        role_patterns = [
            r"(?:looking for|need|want|hire|hiring)\s+(?:a\s+)?([a-zA-Z\s]+?)(?:\s+with|\s+in|\s+having|$)",
            r"([a-zA-Z\s]+?)\s+(?:executive|manager|developer|analyst|specialist)",
            r"(?:position|role|job)\s+(?:of|for)\s+([a-zA-Z\s]+?)(?:\s+with|\s+in|$)",
        ]

        for pattern in role_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                role = match.group(1).strip()
                # Clean up common words
                role = re.sub(
                    r"\b(a|an|the|for|with|in|having)\b", "", role, flags=re.IGNORECASE
                ).strip()
                if len(role) > 2:
                    return role

        return None

    def _extract_domain(self, query: str) -> Optional[str]:
        """Extract domain/industry from query"""
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    return domain
        return None

    def _extract_experience(
        self, query: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Extract experience range from query"""
        # Patterns for experience extraction
        patterns = [
            r"(?:between|from)\s+(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*years?",
            r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*years?\s*(?:of\s+)?(?:experience|exp)",
            r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*years?",
            r"minimum\s+(\d+(?:\.\d+)?)\s*years?",
            r"at least\s+(\d+(?:\.\d+)?)\s*years?",
            r"(\d+(?:\.\d+)?)\+\s*years?",
            r"(\d+(?:\.\d+)?)\s*years?\s*(?:of\s+)?(?:experience|exp)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2:  # Range
                    return float(groups[0]), float(groups[1])
                elif len(groups) == 1:  # Single value
                    exp = float(groups[0])
                    if (
                        "+" in match.group(0)
                        or "minimum" in match.group(0)
                        or "at least" in match.group(0)
                    ):
                        return exp, None  # Minimum experience
                    else:
                        return exp - 0.5, exp + 0.5  # Range around the value

        return None, None

    def _extract_skills(self, query: str) -> List[str]:
        """Extract skills from query"""
        skills = []

        # Check for specific skill mentions
        for category, skill_list in self.skill_keywords.items():
            for skill in skill_list:
                if skill in query:
                    skills.append(skill)

        # Extract skills mentioned with "with" or "having"
        skill_patterns = [
            r"(?:with|having|knows?|experience in|skilled in)\s+([a-zA-Z\s,&+-]+?)(?:\s+(?:experience|skills?)|$|[.\n])",
            r"skills?\s*[:\-]\s*([a-zA-Z\s,&+-]+?)(?:\n|$|[.])",
        ]

        for pattern in skill_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                skill_text = match.group(1).strip()
                # Split by common delimiters
                extracted_skills = re.split(r"[,&+]|\s+and\s+", skill_text)
                skills.extend(
                    [
                        skill.strip()
                        for skill in extracted_skills
                        if len(skill.strip()) > 1
                    ]
                )

        return list(set(skills))  # Remove duplicates

    def _extract_salary(self, query: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract salary range from query"""
        # Patterns for salary extraction (in lakhs)
        patterns = [
            r"(?:salary|budget|package|ctc).*?(?:is|of|around|upto|up to)\s+(\d+(?:\.\d+)?)\s*lakh",
            r"(\d+(?:\.\d+)?)\s*lakh.*?(?:salary|budget|package|ctc)",
            r"(?:between|from)\s+(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*lakh",
            r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*lakh",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2 and groups[1]:  # Range
                    return float(groups[0]), float(groups[1])
                elif len(groups) >= 1:  # Single value
                    salary = float(groups[0])
                    return None, salary  # Max salary

        return None, None

    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location preferences from query"""
        # Common Indian cities
        cities = [
            "mumbai",
            "delhi",
            "bangalore",
            "chennai",
            "kolkata",
            "pune",
            "hyderabad",
            "ahmedabad",
            "surat",
            "jaipur",
            "noida",
            "gurgaon",
            "kochi",
            "trivandrum",
        ]

        for city in cities:
            if city in query.lower():
                return city.title()

        return None

    def _extract_education(self, query: str) -> List[str]:
        """Extract education requirements from query"""
        education = []
        edu_keywords = [
            "mba",
            "bba",
            "engineering",
            "btech",
            "mtech",
            "bcom",
            "mcom",
            "graduate",
            "diploma",
        ]

        for keyword in edu_keywords:
            if keyword in query.lower():
                education.append(keyword.upper())

        return education

    def _extract_notice_period(self, query: str) -> Optional[str]:
        """Extract notice period requirements from query"""
        patterns = [
            r"(?:notice period|joining)\s+(?:of\s+)?(\d+)\s*(days?|weeks?|months?)",
            r"(?:immediate|ready to join|can join)",
            r"(?:serving|current)\s+notice",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if "immediate" in match.group(0) or "ready to join" in match.group(0):
                    return "immediate"
                elif len(match.groups()) >= 2:
                    return f"{match.group(1)} {match.group(2)}"
                else:
                    return "serving notice"

        return None

    def enhance_query_for_vector_search(
        self, query: str, context: SearchContext
    ) -> str:
        """Enhance query for better vector similarity search"""
        enhanced_parts = [query]

        # Add role-specific terms
        if context.role:
            enhanced_parts.append(f"job role: {context.role}")

        # Add domain-specific terms
        if context.domain:
            enhanced_parts.append(f"industry: {context.domain}")
            # Add related keywords
            if context.domain in self.domain_keywords:
                enhanced_parts.extend(self.domain_keywords[context.domain][:3])

        # Add experience context
        if context.min_experience is not None or context.max_experience is not None:
            if context.min_experience and context.max_experience:
                enhanced_parts.append(
                    f"experience between {context.min_experience} and {context.max_experience} years"
                )
            elif context.min_experience:
                enhanced_parts.append(
                    f"minimum {context.min_experience} years experience"
                )
            elif context.max_experience:
                enhanced_parts.append(
                    f"up to {context.max_experience} years experience"
                )

        # Add skills
        if context.skills:
            enhanced_parts.append(
                f"skills: {' '.join(context.skills[:5])}"
            )  # Limit to top 5 skills

        enhanced_query = " ".join(enhanced_parts)
        logger.info(f"Enhanced query: {enhanced_query}")
        return enhanced_query

    def calculate_relevance_score(
        self, candidate: Dict[str, Any], context: SearchContext, base_score: float = 0.0
    ) -> Tuple[float, str]:
        """
        Calculate enhanced relevance score based on prioritized search criteria:
        1st Priority: Designation/Role (40% weight)
        2nd Priority: Location (30% weight)
        3rd Priority: Skills, Experience, and Salary (30% combined weight)
        """
        score_components = []
        reasons = []
        total_weight = 0

        # 1st Priority: Designation/Role matching (weight: 0.4)
        # This includes both role keywords and domain matching
        role_score, role_reason = self._score_designation(candidate, context)
        if role_score > 0:
            score_components.append(role_score * 0.4)
            reasons.append(f"Designation Match: {role_reason}")
        total_weight += 0.4

        # 2nd Priority: Location matching (weight: 0.3)
        location_score, location_reason = self._score_location(candidate, context)
        if location_score > 0:
            score_components.append(location_score * 0.3)
            reasons.append(f"Location Match: {location_reason}")
        total_weight += 0.3

        # 3rd Priority: Skills, Experience, and Salary (30% combined weight)
        # Skills matching (weight: 0.15)
        skills_score, skills_reason = self._score_skills(candidate, context)
        if skills_score > 0:
            score_components.append(skills_score * 0.15)
            reasons.append(f"Skills Match: {skills_reason}")
        total_weight += 0.15

        # Experience matching (weight: 0.1)
        exp_score, exp_reason = self._score_experience(candidate, context)
        if exp_score > 0:
            score_components.append(exp_score * 0.1)
            reasons.append(f"Experience Match: {exp_reason}")
        total_weight += 0.1

        # Salary matching (weight: 0.05)
        salary_score, salary_reason = self._score_salary(candidate, context)
        if salary_score > 0:
            score_components.append(salary_score * 0.05)
            reasons.append(f"Salary Match: {salary_reason}")
        total_weight += 0.05

        # Calculate final score
        if score_components:
            calculated_score = sum(score_components)
            # Combine with base vector similarity score (if provided)
            if base_score > 0:
                final_score = (calculated_score * 0.7) + (base_score * 0.3)
            else:
                final_score = calculated_score
        else:
            final_score = base_score

        # Normalize to 0-1 range
        final_score = min(1.0, max(0.0, final_score))

        match_reason = (
            "; ".join(reasons[:3]) if reasons else "Basic semantic similarity match"
        )

        return final_score, match_reason

    def _score_designation(
        self, candidate: Dict[str, Any], context: SearchContext
    ) -> Tuple[float, str]:
        """
        Score designation/role matching combining both role keywords and domain matching
        This is the highest priority scoring component
        """
        if not (context.role or context.domain):
            return 0.0, ""

        role_score = 0.0
        domain_score = 0.0
        reasons = []

        # Score role matching
        if context.role:
            role_score, role_reason = self._score_role_match(candidate, context.role)
            if role_score > 0:
                reasons.append(role_reason)

        # Score domain matching
        if context.domain:
            domain_score, domain_reason = self._score_domain(candidate, context)
            if domain_score > 0:
                reasons.append(domain_reason)

        # Combine role and domain scores (weighted average)
        final_score = 0.0
        if context.role and context.domain:
            # Both role and domain specified - give equal weight
            final_score = (role_score * 0.6) + (domain_score * 0.4)
        elif context.role:
            # Only role specified
            final_score = role_score
        elif context.domain:
            # Only domain specified
            final_score = domain_score

        reason = "; ".join(reasons) if reasons else ""
        return final_score, reason

    def _score_role_match(
        self, candidate: Dict[str, Any], target_role: str
    ) -> Tuple[float, str]:
        """Score role/designation matching from experience and labels"""
        if not target_role:
            return 0.0, ""

        target_role_lower = target_role.lower()
        score = 0.0
        reasons = []

        # Check current/latest role from experience
        experience = candidate.get("experience", [])
        if experience:
            # Check most recent role (first in list)
            recent_role = experience[0].get("role", "").lower()
            if target_role_lower in recent_role or any(
                keyword in recent_role for keyword in target_role_lower.split()
            ):
                score = max(score, 1.0)
                reasons.append(f"Current role matches ({recent_role})")

            # Check all roles for partial matches
            for exp in experience[:3]:  # Check top 3 recent roles
                role = exp.get("role", "").lower()
                if target_role_lower in role:
                    score = max(score, 0.8)
                    reasons.append(f"Previous role matches ({role})")
                elif any(keyword in role for keyword in target_role_lower.split()):
                    score = max(score, 0.6)
                    reasons.append(f"Role partially matches ({role})")

        # Check labels/tags for role keywords
        labels = candidate.get("labels", [])
        for label in labels:
            label_lower = label.lower()
            if target_role_lower in label_lower:
                score = max(score, 0.7)
                reasons.append(f"Label matches role ({label})")

        # Check skills for role-related keywords
        skills = candidate.get("skills", []) + candidate.get(
            "may_also_known_skills", []
        )
        role_related_skills = []
        for skill in skills:
            skill_lower = skill.lower()
            if any(keyword in skill_lower for keyword in target_role_lower.split()):
                role_related_skills.append(skill)
                score = max(score, 0.4)

        if role_related_skills:
            reasons.append(
                f"Role-related skills ({', '.join(role_related_skills[:3])})"
            )

        reason = "; ".join(reasons[:2]) if reasons else ""
        return score, reason

    def _score_experience(
        self, candidate: Dict[str, Any], context: SearchContext
    ) -> Tuple[float, str]:
        """Score experience matching"""
        if not (context.min_experience or context.max_experience):
            return 0.0, ""

        total_exp_str = candidate.get("total_experience", "0")
        try:
            # Extract numeric value from experience string
            exp_match = re.search(r"(\d+(?:\.\d+)?)", str(total_exp_str))
            if not exp_match:
                return 0.0, ""

            candidate_exp = float(exp_match.group(1))

            score = 0.0
            reason = ""

            if context.min_experience and context.max_experience:
                if context.min_experience <= candidate_exp <= context.max_experience:
                    score = 1.0
                    reason = f"Experience ({candidate_exp}y) matches required range ({context.min_experience}-{context.max_experience}y)"
                elif candidate_exp < context.min_experience:
                    # Penalize less experience
                    diff = context.min_experience - candidate_exp
                    score = max(0.0, 1.0 - (diff / context.min_experience))
                    reason = f"Experience ({candidate_exp}y) below minimum ({context.min_experience}y)"
                else:
                    # Slight bonus for more experience, but cap it
                    excess = candidate_exp - context.max_experience
                    if excess <= 2:  # Within 2 years over max is still good
                        score = 0.9
                        reason = f"Experience ({candidate_exp}y) slightly above range but still relevant"
                    else:
                        score = 0.7
                        reason = (
                            f"Experience ({candidate_exp}y) significantly above range"
                        )
            elif context.min_experience:
                if candidate_exp >= context.min_experience:
                    score = 1.0
                    reason = f"Experience ({candidate_exp}y) meets minimum requirement ({context.min_experience}y)"
                else:
                    diff = context.min_experience - candidate_exp
                    score = max(0.0, 1.0 - (diff / context.min_experience))
                    reason = f"Experience ({candidate_exp}y) below minimum ({context.min_experience}y)"
            elif context.max_experience:
                if candidate_exp <= context.max_experience:
                    score = 1.0
                    reason = f"Experience ({candidate_exp}y) within maximum limit ({context.max_experience}y)"
                else:
                    excess = candidate_exp - context.max_experience
                    score = max(0.3, 1.0 - (excess / context.max_experience))
                    reason = f"Experience ({candidate_exp}y) exceeds maximum ({context.max_experience}y)"

            return score, reason

        except (ValueError, AttributeError):
            return 0.0, ""

    def _score_skills(
        self, candidate: Dict[str, Any], context: SearchContext
    ) -> Tuple[float, str]:
        """Score skills matching"""
        if not context.skills:
            return 0.0, ""

        candidate_skills = []

        # Get skills from multiple fields
        skills_fields = ["skills", "may_also_known_skills", "labels"]
        for field in skills_fields:
            field_skills = candidate.get(field, [])
            if isinstance(field_skills, list):
                candidate_skills.extend([skill.lower() for skill in field_skills])
            elif isinstance(field_skills, str):
                candidate_skills.extend(
                    [skill.strip().lower() for skill in field_skills.split(",")]
                )

        # Also check in experience descriptions
        experiences = candidate.get("experience", [])
        for exp in experiences:
            if isinstance(exp, dict):
                for key, value in exp.items():
                    if isinstance(value, str):
                        candidate_skills.extend(
                            [
                                skill.strip().lower()
                                for skill in value.split()
                                if len(skill) > 2
                            ]
                        )

        candidate_skills = list(set(candidate_skills))  # Remove duplicates

        if not candidate_skills:
            return 0.0, ""

        # Calculate matching score
        required_skills_lower = [skill.lower() for skill in context.skills]
        matched_skills = []

        for req_skill in required_skills_lower:
            for cand_skill in candidate_skills:
                if req_skill in cand_skill or cand_skill in req_skill:
                    matched_skills.append(req_skill)
                    break

        if matched_skills:
            score = len(matched_skills) / len(required_skills_lower)
            reason = f"Matches {len(matched_skills)}/{len(required_skills_lower)} required skills: {', '.join(matched_skills[:3])}"
            return min(1.0, score), reason

        return 0.0, ""

    def _score_domain(
        self, candidate: Dict[str, Any], context: SearchContext
    ) -> Tuple[float, str]:
        """Score domain/industry matching"""
        if not context.domain:
            return 0.0, ""

        # Check in experience, skills, and other relevant fields
        text_to_check = []

        # Get experience descriptions
        experiences = candidate.get("experience", [])
        for exp in experiences:
            if isinstance(exp, dict):
                for key, value in exp.items():
                    if isinstance(value, str):
                        text_to_check.append(value.lower())

        # Get skills and labels
        for field in ["skills", "may_also_known_skills", "labels"]:
            field_value = candidate.get(field, [])
            if isinstance(field_value, list):
                text_to_check.extend([item.lower() for item in field_value])

        # Check domain keywords
        domain_keywords = self.domain_keywords.get(context.domain, [])
        matched_keywords = []

        for text in text_to_check:
            for keyword in domain_keywords:
                if keyword in text:
                    matched_keywords.append(keyword)

        if matched_keywords:
            score = min(1.0, len(matched_keywords) / len(domain_keywords))
            reason = (
                f"Domain match in {context.domain}: {', '.join(matched_keywords[:2])}"
            )
            return score, reason

        return 0.0, ""

    def _score_salary(
        self, candidate: Dict[str, Any], context: SearchContext
    ) -> Tuple[float, str]:
        """Score salary matching"""
        if not (context.min_salary or context.max_salary):
            return 0.0, ""

        # Check expected salary
        expected_salary = candidate.get("expected_salary", 0)
        if not expected_salary:
            return 0.0, ""

        try:
            expected_salary = float(expected_salary)
            # Convert to lakhs if needed (assuming it's in rupees)
            if expected_salary > 1000000:  # If it's in rupees
                expected_salary = expected_salary / 100000  # Convert to lakhs

            score = 0.0
            reason = ""

            if context.max_salary:
                if expected_salary <= context.max_salary:
                    score = 1.0
                    reason = f"Expected salary ({expected_salary:.1f}L) within budget ({context.max_salary:.1f}L)"
                else:
                    excess = expected_salary - context.max_salary
                    score = max(0.0, 1.0 - (excess / context.max_salary))
                    reason = f"Expected salary ({expected_salary:.1f}L) exceeds budget ({context.max_salary:.1f}L)"

            return score, reason

        except (ValueError, TypeError):
            return 0.0, ""

    def _score_location(
        self, candidate: Dict[str, Any], context: SearchContext
    ) -> Tuple[float, str]:
        """Score location matching"""
        if not context.location:
            return 0.0, ""

        # Check current city and looking for jobs in
        contact_details = candidate.get("contact_details", {})
        current_city = contact_details.get("current_city", "").lower()
        looking_for_jobs_in = contact_details.get("looking_for_jobs_in", [])

        if isinstance(looking_for_jobs_in, str):
            looking_for_jobs_in = [looking_for_jobs_in]

        looking_cities = [city.lower() for city in looking_for_jobs_in]

        context_location_lower = context.location.lower()

        if context_location_lower in current_city:
            return 1.0, f"Currently in {context.location}"
        elif any(context_location_lower in city for city in looking_cities):
            return 0.8, f"Looking for jobs in {context.location}"

        return 0.0, ""
