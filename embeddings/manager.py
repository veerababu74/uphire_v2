# embeddings/manager.py
from typing import Dict, List, Any, Optional
import logging

from .base import BaseEmbeddingProvider, BaseVectorizer
from .providers import EmbeddingProviderFactory

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Centralized embedding management"""

    def __init__(self, provider: Optional[BaseEmbeddingProvider] = None):
        self.provider = provider or EmbeddingProviderFactory.create_default_provider()
        logger.info(
            f"EmbeddingManager initialized with provider: {self.provider.get_provider_name()}"
        )

    def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for the given text"""
        return self.provider.generate_embedding(text)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.provider.get_embedding_dimension()

    def get_provider_info(self) -> str:
        """Get information about the current provider"""
        return self.provider.get_provider_name()


class ResumeVectorizer(BaseVectorizer):
    """Vectorizer for regular resume data"""

    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        self.embedding_manager = embedding_manager or EmbeddingManager()

    def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for the given text"""
        return self.embedding_manager.generate_embedding(text)

    def generate_total_resume_text(self, resume_data: Dict) -> str:
        """Generate a comprehensive text representation of the entire resume"""
        sections = []

        # Add name
        if (
            "name" in resume_data
            and resume_data["name"]
            and resume_data["name"] != "N/A"
        ):
            sections.append(f"Name: {resume_data['name']}")

        # Add contact details
        if "contact_details" in resume_data:
            contact = resume_data["contact_details"]
            contact_info = []
            if contact.get("email") and contact.get("email") != "N/A":
                contact_info.append(f"Email: {contact.get('email')}")
            if contact.get("phone") and contact.get("phone") != "N/A":
                contact_info.append(f"Phone: {contact.get('phone')}")
            if contact.get("address") and contact.get("address") != "N/A":
                contact_info.append(f"Address: {contact.get('address')}")

            if contact_info:
                sections.append("Contact Details: " + "; ".join(contact_info))

        # Add education
        if "education" in resume_data and resume_data["education"]:
            edu_texts = []
            for edu in resume_data["education"]:
                parts = []
                if edu.get("degree") and edu.get("degree") != "N/A":
                    parts.append(edu.get("degree"))
                if edu.get("institution") and edu.get("institution") != "N/A":
                    parts.append(f"from {edu.get('institution')}")
                if edu.get("dates") and edu.get("dates") != "N/A":
                    parts.append(f"({edu.get('dates')})")

                if parts:
                    edu_texts.append(" ".join(parts))

            if edu_texts:
                sections.append("Education: " + ". ".join(edu_texts))

        # Add experience
        if "experience" in resume_data and resume_data["experience"]:
            exp_texts = []
            for exp in resume_data["experience"]:
                parts = []
                if exp.get("title") and exp.get("title") != "N/A":
                    parts.append(exp.get("title"))
                if exp.get("company") and exp.get("company") != "N/A":
                    parts.append(f"at {exp.get('company')}")
                if exp.get("duration") and exp.get("duration") != "N/A":
                    parts.append(f"for {exp.get('duration')}")

                if parts:
                    exp_texts.append(" ".join(parts))

            if exp_texts:
                sections.append("Experience: " + ". ".join(exp_texts))

        # Add total experience
        if (
            "total_experience" in resume_data
            and resume_data["total_experience"]
            and resume_data["total_experience"] != "N/A"
        ):
            sections.append(f"Total Experience: {resume_data['total_experience']}")

        # Add skills
        if "skills" in resume_data and resume_data["skills"]:
            skills_text = (
                ", ".join(resume_data["skills"])
                if isinstance(resume_data["skills"], list)
                else resume_data["skills"]
            )
            if skills_text and skills_text != "N/A":
                sections.append(f"Skills: {skills_text}")

        # Add projects
        if "projects" in resume_data and resume_data["projects"]:
            proj_texts = []
            for proj in resume_data["projects"]:
                project_parts = []
                if proj.get("name") and proj.get("name") != "N/A":
                    project_parts.append(proj.get("name"))

                if proj.get("description") and proj.get("description") != "N/A":
                    project_parts.append(proj.get("description"))

                tech_text = ""
                if proj.get("technologies"):
                    tech_text = (
                        ", ".join(proj.get("technologies"))
                        if isinstance(proj.get("technologies"), list)
                        else proj.get("technologies")
                    )
                    if tech_text and tech_text != "N/A":
                        project_parts.append(f"Technologies: {tech_text}")

                if proj.get("role") and proj.get("role") != "N/A":
                    project_parts.append(f"Role: {proj.get('role')}")

                if project_parts:
                    proj_texts.append(". ".join(project_parts))

            if proj_texts:
                sections.append("Projects: " + ". ".join(proj_texts))

        # Add certifications
        if "certifications" in resume_data and resume_data["certifications"]:
            cert_text = (
                ", ".join(resume_data["certifications"])
                if isinstance(resume_data["certifications"], list)
                else resume_data["certifications"]
            )
            if cert_text and cert_text != "N/A":
                sections.append(f"Certifications: {cert_text}")

        # Join all sections with line breaks
        return "\n\n".join(sections)

    def generate_resume_embeddings(self, resume_data: Dict) -> Dict:
        """Generate embeddings for searchable fields in resume and a combined vector for the entire resume"""
        resume_with_vectors = resume_data.copy()

        # Generate vector for skills (join list to string)
        skills_text = ""
        if "skills" in resume_data and resume_data["skills"]:
            skills_text = (
                ", ".join(resume_data["skills"])
                if isinstance(resume_data["skills"], list)
                else resume_data["skills"]
            )
            resume_with_vectors["skills_vector"] = self.generate_embedding(skills_text)

        # Generate vector for experience text
        experience_text = ""
        if "experience" in resume_data and resume_data["experience"]:
            experience_texts = []
            for exp in resume_data["experience"]:
                exp_text = f"{exp.get('title', '')} at {exp.get('company', '')} for {exp.get('duration', '')}"
                experience_texts.append(exp_text)
            experience_text = ". ".join(experience_texts)
            resume_with_vectors["experience_text_vector"] = self.generate_embedding(
                experience_text
            )

        # Generate vector for education text
        education_text = ""
        if "education" in resume_data and resume_data["education"]:
            education_texts = []
            for edu in resume_data["education"]:
                edu_text = f"{edu.get('degree', '')} from {edu.get('institution', '')} ({edu.get('dates', '')})"
                education_texts.append(edu_text)
            education_text = ". ".join(education_texts)
            resume_with_vectors["education_text_vector"] = self.generate_embedding(
                education_text
            )

        # Generate vector for projects text
        projects_text = ""
        if "projects" in resume_data and resume_data["projects"]:
            project_texts = []
            for proj in resume_data["projects"]:
                tech_text = (
                    ", ".join(proj.get("technologies", []))
                    if isinstance(proj.get("technologies"), list)
                    else proj.get("technologies", "")
                )
                proj_text = f"{proj.get('name', '')}: {proj.get('description', '')}. Technologies: {tech_text}. Role: {proj.get('role', '')}"
                project_texts.append(proj_text)
            projects_text = ". ".join(project_texts)
            resume_with_vectors["projects_text_vector"] = self.generate_embedding(
                projects_text
            )

        # Generate vector for the entire resume
        total_resume_text = self.generate_total_resume_text(resume_data)
        resume_with_vectors["total_resume_text"] = total_resume_text
        resume_with_vectors["total_resume_vector"] = self.generate_embedding(
            total_resume_text
        )

        return resume_with_vectors

    def generate_total_resume_vector(self, resume_data: Dict) -> List[float]:
        """Generate a combined vector representation of the entire resume"""
        total_resume_text = self.generate_total_resume_text(resume_data)
        return self.generate_embedding(total_resume_text)


class AddUserDataVectorizer(BaseVectorizer):
    """Vectorizer for user-added resume data with different schema"""

    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        self.embedding_manager = embedding_manager or EmbeddingManager()

    def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for the given text"""
        return self.embedding_manager.generate_embedding(text)

    def generate_total_resume_text(self, resume_data: Dict) -> str:
        """Generate a comprehensive text representation of the entire resume for AddUserData format"""
        sections = []

        # Add name
        if resume_data.get("name") and resume_data["name"] != "N/A":
            sections.append(f"Name: {resume_data['name']}")

        # Add email
        if resume_data.get("email") and resume_data["email"] != "N/A":
            sections.append(f"Email: {resume_data['email']}")

        # Add phone
        if resume_data.get("phone") and resume_data["phone"] != "N/A":
            sections.append(f"Phone: {resume_data['phone']}")

        # Add total experience
        if resume_data.get("total_exp") and resume_data["total_exp"] != "N/A":
            sections.append(f"Total Experience: {resume_data['total_exp']}")

        # Add skills
        all_skills = resume_data.get("skills", []) + resume_data.get(
            "may_also_known_skills", []
        )
        if all_skills:
            skills_text = ", ".join(all_skills)
            sections.append(f"Skills: {skills_text}")

        # Add experience
        if "experience" in resume_data and resume_data["experience"]:
            exp_texts = []
            for exp in resume_data["experience"]:
                exp_parts = []
                if exp.get("title") and exp.get("title") != "N/A":
                    exp_parts.append(exp.get("title"))
                if exp.get("company") and exp.get("company") != "N/A":
                    exp_parts.append(f"at {exp.get('company')}")
                if exp.get("from_date") and exp.get("until"):
                    exp_parts.append(
                        f"from {exp.get('from_date')} to {exp.get('until')}"
                    )

                if exp_parts:
                    exp_texts.append(" ".join(exp_parts))

            if exp_texts:
                sections.append("Experience: " + ". ".join(exp_texts))

        # Add education
        if "academic_details" in resume_data and resume_data["academic_details"]:
            edu_texts = []
            for edu in resume_data["academic_details"]:
                edu_parts = []
                if edu.get("education") and edu.get("education") != "N/A":
                    edu_parts.append(edu.get("education"))
                if edu.get("college") and edu.get("college") != "N/A":
                    edu_parts.append(f"from {edu.get('college')}")
                if edu.get("pass_year") and edu.get("pass_year") != "N/A":
                    edu_parts.append(f"({edu.get('pass_year')})")

                if edu_parts:
                    edu_texts.append(" ".join(edu_parts))

            if edu_texts:
                sections.append("Education: " + ". ".join(edu_texts))

        # Add combined resume if available
        if (
            resume_data.get("combined_resume")
            and resume_data["combined_resume"] != "N/A"
        ):
            sections.append(f"Additional Info: {resume_data['combined_resume']}")

        return "\n\n".join(sections)

    def generate_resume_embeddings(self, resume_data: Dict) -> Dict:
        """Generate embeddings for searchable fields in resume"""
        resume_with_vectors = resume_data.copy()

        # Generate experience text and vector
        experience_text = ""
        if "experience" in resume_data:
            experience_texts = []
            for exp in resume_data["experience"]:
                exp_text = f"{exp.get('title', '')} at {exp.get('company', '')} for {exp.get('from_date', '')} to {exp.get('until', 'Present')}"
                experience_texts.append(exp_text)
            experience_text = ". ".join(experience_texts)
            resume_with_vectors["experience_text_vector"] = self.generate_embedding(
                experience_text
            )

        # Generate education text and vector
        education_text = ""
        if "academic_details" in resume_data:
            education_texts = []
            for edu in resume_data["academic_details"]:
                edu_text = f"{edu.get('education', '')} from {edu.get('college', '')} ({edu.get('pass_year', '')})"
                education_texts.append(edu_text)
            education_text = ". ".join(education_texts)
            resume_with_vectors["education_text_vector"] = self.generate_embedding(
                education_text
            )

        # Generate skills text and vector (combining primary and additional skills)
        all_skills = resume_data.get("skills", []) + resume_data.get(
            "may_also_known_skills", []
        )
        skills_text = ", ".join(all_skills) if all_skills else ""
        resume_with_vectors["skills_vector"] = self.generate_embedding(skills_text)

        # Generate vector for the combined resume text
        combined_resume = resume_data.get("combined_resume", "")
        resume_with_vectors["combined_resume_vector"] = self.generate_embedding(
            combined_resume
        )

        # Generate total resume text and vector
        total_resume_text = self.generate_total_resume_text(resume_data)
        resume_with_vectors["total_resume_text"] = total_resume_text
        resume_with_vectors["total_resume_vector"] = self.generate_embedding(
            total_resume_text
        )

        return resume_with_vectors
