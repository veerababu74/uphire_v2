# resume_api/database/operations.py
from pymongo.collection import Collection
from bson import ObjectId
from typing import Dict, Any, Union
from embeddings.vectorizer import Vectorizer, AddUserDataVectorizer
from core.helpers import format_resume
from fastapi import HTTPException
from typing import List


class ResumeOperations:
    def __init__(
        self,
        collection: Collection,
        vectorizer: Union[Vectorizer, AddUserDataVectorizer],
    ):
        self.collection = collection
        self.vectorizer = vectorizer

    def create_resume(self, resume_data: Dict[str, Any]) -> Dict[str, str]:
        """Create a new resume with vector embeddings"""
        try:
            from datetime import datetime, UTC

            if not resume_data:
                raise ValueError("Resume data cannot be empty")

            resume_with_vectors = self.vectorizer.generate_resume_embeddings(
                resume_data
            )

            # Add creation timestamp
            resume_with_vectors["created_at"] = datetime.now(UTC)

            result = self.collection.insert_one(resume_with_vectors)
            return {
                "id": str(result.inserted_id),
                "message": "Resume created successfully with vector embeddings",
            }
        except Exception as e:
            raise ValueError(f"Failed to create resume: {str(e)}")

    def update_resume(
        self, resume_id: str, resume_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Update a resume by ID with vector embeddings and updated timestamp"""
        try:
            from datetime import datetime, UTC

            if not ObjectId.is_valid(resume_id):
                raise ValueError("Invalid resume ID format")

            existing = self.collection.find_one({"_id": ObjectId(resume_id)})
            if not existing:
                raise HTTPException(status_code=404, detail="Resume not found")

            if not resume_data:
                raise ValueError("Resume data cannot be empty")

            # Generate fresh vector embeddings for all relevant fields
            resume_with_vectors = self.vectorizer.generate_resume_embeddings(
                resume_data
            )

            # Update the timestamp to reflect when the resume was last modified
            resume_with_vectors["created_at"] = datetime.now(UTC)

            # Ensure all vector fields are properly set
            vector_fields_to_update = {
                "experience_text_vector": resume_with_vectors.get(
                    "experience_text_vector"
                ),
                "education_text_vector": resume_with_vectors.get(
                    "education_text_vector"
                ),
                "skills_vector": resume_with_vectors.get("skills_vector"),
                "combined_resume_vector": resume_with_vectors.get(
                    "combined_resume_vector"
                ),
                "total_resume_text": resume_with_vectors.get("total_resume_text"),
                "total_resume_vector": resume_with_vectors.get("total_resume_vector"),
                "created_at": resume_with_vectors["created_at"],
            }

            # Add the updated resume data fields
            update_data = {**resume_data, **vector_fields_to_update}

            result = self.collection.update_one(
                {"_id": ObjectId(resume_id)}, {"$set": update_data}
            )

            if result.modified_count == 1:
                return {
                    "message": "Resume updated successfully with vector embeddings and timestamp"
                }
            return {"message": "No changes made to the resume"}
        except HTTPException:
            raise
        except Exception as e:
            raise ValueError(f"Failed to update resume: {str(e)}")

    def get_resume(self, resume_id: str) -> Dict:
        """Get a resume by ID"""
        try:
            if not ObjectId.is_valid(resume_id):
                raise ValueError("Invalid resume ID format")

            resume = self.collection.find_one({"_id": ObjectId(resume_id)})
            if not resume:
                raise HTTPException(status_code=404, detail="Resume not found")
            return format_resume(resume)
        except HTTPException:
            raise
        except Exception as e:
            raise ValueError(f"Failed to retrieve resume: {str(e)}")

    def delete_resume(self, resume_id: str) -> Dict:
        """Delete a resume by ID"""
        try:
            if not ObjectId.is_valid(resume_id):
                raise ValueError("Invalid resume ID format")

            result = self.collection.delete_one({"_id": ObjectId(resume_id)})
            if result.deleted_count != 1:
                raise HTTPException(status_code=404, detail="Resume not found")
            return {"message": "Resume deleted successfully"}
        except HTTPException:
            raise
        except Exception as e:
            raise ValueError(f"Failed to delete resume: {str(e)}")

    def list_resumes(self, skip: int = 0, limit: int = 10) -> List[Dict]:
        """List all resumes with pagination"""
        try:
            if skip < 0:
                raise ValueError("Skip parameter cannot be negative")

            if limit <= 0 or limit > 100:
                raise ValueError("Limit must be between 1 and 100")

            cursor = self.collection.find().skip(skip).limit(limit)
            return [format_resume(doc) for doc in cursor]
        except Exception as e:
            raise ValueError(f"Failed to list resumes: {str(e)}")

    def update_all_vector_embeddings(self) -> Dict:
        """Update vector embeddings for all resumes"""
        try:
            from datetime import datetime, UTC

            resumes = list(self.collection.find({}))
            updated_count = 0
            failed_count = 0

            for resume in resumes:
                try:
                    resume_with_vectors = self.vectorizer.generate_resume_embeddings(
                        resume
                    )

                    # Update timestamp for embedding regeneration
                    vector_update = {
                        "skills_vector": resume_with_vectors.get("skills_vector"),
                        "experience_text_vector": resume_with_vectors.get(
                            "experience_text_vector"
                        ),
                        "education_text_vector": resume_with_vectors.get(
                            "education_text_vector"
                        ),
                        "combined_resume_vector": resume_with_vectors.get(
                            "combined_resume_vector"
                        ),
                        "total_resume_text": resume_with_vectors.get(
                            "total_resume_text"
                        ),
                        "total_resume_vector": resume_with_vectors.get(
                            "total_resume_vector"
                        ),
                        "created_at": datetime.now(
                            UTC
                        ),  # Update timestamp to show when vectors were regenerated
                    }

                    result = self.collection.update_one(
                        {"_id": resume["_id"]}, {"$set": vector_update}
                    )
                    if result.modified_count > 0:
                        updated_count += 1
                except Exception as e:
                    failed_count += 1
                    print(
                        f"Failed to update embeddings for resume {resume.get('_id')}: {str(e)}"
                    )

            message = f"Updated vector embeddings for {updated_count} resumes"
            if failed_count > 0:
                message += f", {failed_count} failed"

            return {"message": message}
        except Exception as e:
            raise ValueError(f"Failed to update embeddings: {str(e)}")

    def update_resume_fields_only(
        self,
        resume_id: str,
        resume_data: Dict[str, Any],
        regenerate_vectors: bool = True,
    ) -> Dict[str, str]:
        """Update specific resume fields with optional vector regeneration"""
        try:
            from datetime import datetime, UTC

            if not ObjectId.is_valid(resume_id):
                raise ValueError("Invalid resume ID format")

            existing = self.collection.find_one({"_id": ObjectId(resume_id)})
            if not existing:
                raise HTTPException(status_code=404, detail="Resume not found")

            if not resume_data:
                raise ValueError("Resume data cannot be empty")

            update_data = resume_data.copy()

            if regenerate_vectors:
                # Merge existing data with new data for vector generation
                merged_data = {**existing, **resume_data}
                # Remove MongoDB specific fields before vectorization
                merged_data.pop("_id", None)

                resume_with_vectors = self.vectorizer.generate_resume_embeddings(
                    merged_data
                )

                # Add vector fields to update
                vector_fields = {
                    "experience_text_vector": resume_with_vectors.get(
                        "experience_text_vector"
                    ),
                    "education_text_vector": resume_with_vectors.get(
                        "education_text_vector"
                    ),
                    "skills_vector": resume_with_vectors.get("skills_vector"),
                    "combined_resume_vector": resume_with_vectors.get(
                        "combined_resume_vector"
                    ),
                    "total_resume_text": resume_with_vectors.get("total_resume_text"),
                    "total_resume_vector": resume_with_vectors.get(
                        "total_resume_vector"
                    ),
                }
                update_data.update(vector_fields)

            # Always update the timestamp when any field is modified
            update_data["created_at"] = datetime.now(UTC)

            result = self.collection.update_one(
                {"_id": ObjectId(resume_id)}, {"$set": update_data}
            )

            if result.modified_count == 1:
                action = (
                    "with vector regeneration"
                    if regenerate_vectors
                    else "without vector regeneration"
                )
                return {"message": f"Resume updated successfully {action}"}
            return {"message": "No changes made to the resume"}
        except HTTPException:
            raise
        except Exception as e:
            raise ValueError(f"Failed to update resume: {str(e)}")

    def list_resumes_by_user(
        self, user_id: str, skip: int = 0, limit: int = 10
    ) -> List[Dict]:
        """List resumes for a specific user with pagination"""
        try:
            if skip < 0:
                raise ValueError("Skip parameter cannot be negative")

            if limit <= 0 or limit > 100:
                raise ValueError("Limit must be between 1 and 100")

            cursor = self.collection.find({"user_id": user_id}).skip(skip).limit(limit)
            return [format_resume(doc) for doc in cursor]
        except Exception as e:
            raise ValueError(f"Failed to list resumes for user {user_id}: {str(e)}")

    def update_vector_embeddings_by_user(self, user_id: str) -> Dict:
        """Update vector embeddings for resumes belonging to a specific user"""
        try:
            from datetime import datetime, UTC

            resumes = list(self.collection.find({"user_id": user_id}))
            updated_count = 0
            failed_count = 0

            for resume in resumes:
                try:
                    resume_with_vectors = self.vectorizer.generate_resume_embeddings(
                        resume
                    )

                    # Update timestamp for embedding regeneration
                    vector_update = {
                        "skills_vector": resume_with_vectors.get("skills_vector"),
                        "experience_text_vector": resume_with_vectors.get(
                            "experience_text_vector"
                        ),
                        "education_text_vector": resume_with_vectors.get(
                            "education_text_vector"
                        ),
                        "combined_resume_vector": resume_with_vectors.get(
                            "combined_resume_vector"
                        ),
                        "total_resume_text": resume_with_vectors.get(
                            "total_resume_text"
                        ),
                        "total_resume_vector": resume_with_vectors.get(
                            "total_resume_vector"
                        ),
                        "created_at": datetime.now(
                            UTC
                        ),  # Update timestamp to show when vectors were regenerated
                    }

                    result = self.collection.update_one(
                        {"_id": resume["_id"]}, {"$set": vector_update}
                    )
                    if result.modified_count > 0:
                        updated_count += 1
                except Exception as e:
                    failed_count += 1
                    print(
                        f"Failed to update embeddings for resume {resume.get('_id')}: {str(e)}"
                    )

            message = f"Updated vector embeddings for {updated_count} resumes belonging to user {user_id}"
            if failed_count > 0:
                message += f", {failed_count} failed"

            return {"message": message}
        except Exception as e:
            raise ValueError(
                f"Failed to update embeddings for user {user_id}: {str(e)}"
            )


# ...existing code...


class SkillsTitlesOperations:
    def __init__(self, collection: Collection):
        self.collection = collection

    def _preprocess_text(self, text: str) -> str:
        """Preprocess the text by removing extra spaces and converting to lowercase"""
        return text.strip().lower()

    def add_skill(self, skill: str) -> Dict[str, str]:
        """Add a new skill if it doesn't exist"""
        processed_skill = self._preprocess_text(skill)

        # Check if skill already exists
        existing_skill = self.collection.find_one(
            {"type": "skill", "value": processed_skill}
        )
        if existing_skill:
            return {"message": f"Skill '{skill}' already exists"}

        result = self.collection.insert_one(
            {
                "type": "skill",
                "value": processed_skill,
            }
        )

        return {"message": f"Skill '{skill}' added successfully"}

    def add_title(self, title: str) -> Dict[str, str]:
        """Add a new title if it doesn't exist"""
        processed_title = self._preprocess_text(title)

        # Check if title already exists
        existing_title = self.collection.find_one(
            {"type": "title", "value": processed_title}
        )
        if existing_title:
            return {"message": f"Title '{title}' already exists"}

        result = self.collection.insert_one(
            {
                "type": "title",
                "value": processed_title,
            }
        )

        return {"message": f"Title '{title}' added successfully"}

    def add_multiple_skills(self, skills: List[str]) -> Dict[str, Any]:
        """Add multiple skills at once"""
        added_count = 0
        existing_count = 0

        for skill in skills:
            processed_skill = self._preprocess_text(skill)
            if not self.collection.find_one(
                {"type": "skill", "value": processed_skill}
            ):
                self.collection.insert_one(
                    {
                        "type": "skill",
                        "value": processed_skill,
                    }
                )
                added_count += 1
            else:
                existing_count += 1

        return {
            "message": f"Added {added_count} new skills, {existing_count} were already present"
        }

    def add_multiple_titles(self, titles: List[str]) -> Dict[str, Any]:
        """Add multiple titles at once"""
        added_count = 0
        existing_count = 0

        for title in titles:
            processed_title = self._preprocess_text(title)
            if not self.collection.find_one(
                {"type": "title", "value": processed_title}
            ):
                self.collection.insert_one(
                    {
                        "type": "title",
                        "value": processed_title,
                    }
                )
                added_count += 1
            else:
                existing_count += 1

        return {
            "message": f"Added {added_count} new titles, {existing_count} were already present"
        }

    def get_all_skills(self) -> List[str]:
        """Get all skills"""
        skills = self.collection.find({"type": "skill"})
        return [skill["value"] for skill in skills]

    def get_all_titles(self) -> List[str]:
        """Get all titles"""
        titles = self.collection.find({"type": "title"})
        return [title["value"] for title in titles]

    def delete_skill(self, skill: str) -> Dict[str, str]:
        """Delete a skill"""
        processed_skill = self._preprocess_text(skill)
        result = self.collection.delete_one({"type": "skill", "value": processed_skill})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Skill '{skill}' not found")
        return {"message": f"Skill '{skill}' deleted successfully"}

    def delete_title(self, title: str) -> Dict[str, str]:
        """Delete a title"""
        processed_title = self._preprocess_text(title)
        result = self.collection.delete_one({"type": "title", "value": processed_title})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Title '{title}' not found")
        return {"message": f"Title '{title}' deleted successfully"}
