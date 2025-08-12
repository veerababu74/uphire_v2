"""
Duplicate content detection operations for resume processing.
This module handles text similarity detection and storage of extracted resume text.
"""

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
from pymongo.collection import Collection
from core.custom_logger import CustomLogger

# Initialize logger
logger_manager = CustomLogger()
logger = logger_manager.get_logger("duplicate_detection")


class DuplicateDetectionOperations:
    """Handle duplicate detection operations for resume text."""

    def __init__(self, extracted_text_collection: Collection):
        self.collection = extracted_text_collection
        self.similarity_threshold = 0.70  # 70% similarity threshold

    def normalize_text_for_comparison(self, text: str) -> str:
        """
        Normalize text for better comparison by removing special characters,
        extra spaces, and converting to lowercase.
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r"[^a-z\s]", " ", text)

        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)

        # Remove common resume words that might not be meaningful for comparison
        common_words = [
            "resume",
            "cv",
            "curriculum",
            "vitae",
            "email",
            "phone",
            "address",
            "experience",
            "education",
            "skills",
            "projects",
            "objective",
            "summary",
            "references",
            "available",
            "upon",
            "request",
        ]

        for word in common_words:
            text = text.replace(word, " ")

        # Final cleanup
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using SequenceMatcher.
        Returns a value between 0 and 1, where 1 is identical.
        """
        # Normalize both texts
        normalized_text1 = self.normalize_text_for_comparison(text1)
        normalized_text2 = self.normalize_text_for_comparison(text2)

        # Calculate similarity using SequenceMatcher
        similarity = SequenceMatcher(None, normalized_text1, normalized_text2).ratio()

        return similarity

    def check_duplicate_content(
        self, user_id: str, extracted_text: str
    ) -> Tuple[bool, List[Dict]]:
        """
        Check if the extracted text is similar to existing content for the user.

        Args:
            user_id: The user ID to check against
            extracted_text: The extracted text from the resume

        Returns:
            Tuple of (is_duplicate, list_of_similar_documents)
        """
        try:
            # Find all existing texts for this user
            existing_texts = list(self.collection.find({"user_id": user_id}))

            similar_documents = []

            for existing_doc in existing_texts:
                existing_text = existing_doc.get("extracted_text", "")
                similarity = self.calculate_text_similarity(
                    extracted_text, existing_text
                )

                if similarity >= self.similarity_threshold:
                    similar_documents.append(
                        {
                            "document_id": str(existing_doc["_id"]),
                            "filename": existing_doc.get("filename", "Unknown"),
                            "similarity_score": similarity,
                            "created_at": existing_doc.get("created_at"),
                            "extracted_text_preview": (
                                existing_text[:200] + "..."
                                if len(existing_text) > 200
                                else existing_text
                            ),
                        }
                    )

            is_duplicate = len(similar_documents) > 0

            if is_duplicate:
                logger.info(
                    f"Duplicate content detected for user {user_id}. Found {len(similar_documents)} similar documents."
                )

            return is_duplicate, similar_documents

        except Exception as e:
            logger.error(f"Error checking duplicate content: {e}")
            return False, []

    def save_extracted_text(
        self, user_id: str, username: str, filename: str, extracted_text: str
    ) -> Dict:
        """
        Save extracted text to the database for future duplicate detection.

        Args:
            user_id: The user ID
            username: The username
            filename: The original filename
            extracted_text: The extracted and cleaned text

        Returns:
            Dictionary with save result
        """
        try:
            document = {
                "user_id": user_id,
                "username": username,
                "filename": filename,
                "extracted_text": extracted_text,
                "text_length": len(extracted_text),
                "created_at": datetime.now(timezone.utc),
                "normalized_text": self.normalize_text_for_comparison(extracted_text),
            }

            result = self.collection.insert_one(document)

            logger.info(
                f"Saved extracted text for user {user_id}, filename: {filename}"
            )

            return {
                "success": True,
                "document_id": str(result.inserted_id),
                "message": "Extracted text saved successfully",
            }

        except Exception as e:
            logger.error(f"Error saving extracted text: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to save extracted text",
            }

    def get_user_extracted_texts(self, user_id: str, limit: int = 50) -> List[Dict]:
        """
        Get all extracted texts for a specific user.

        Args:
            user_id: The user ID
            limit: Maximum number of documents to return

        Returns:
            List of extracted text documents
        """
        try:
            documents = list(
                self.collection.find({"user_id": user_id})
                .sort("created_at", -1)
                .limit(limit)
            )

            # Convert ObjectId to string for JSON serialization
            for doc in documents:
                doc["_id"] = str(doc["_id"])

            return documents

        except Exception as e:
            logger.error(f"Error retrieving user extracted texts: {e}")
            return []

    def delete_extracted_text(self, document_id: str) -> Dict:
        """
        Delete an extracted text document.

        Args:
            document_id: The document ID to delete

        Returns:
            Dictionary with deletion result
        """
        try:
            from bson import ObjectId

            result = self.collection.delete_one({"_id": ObjectId(document_id)})

            if result.deleted_count == 1:
                logger.info(f"Deleted extracted text document: {document_id}")
                return {
                    "success": True,
                    "message": "Extracted text deleted successfully",
                }
            else:
                return {"success": False, "message": "Document not found"}

        except Exception as e:
            logger.error(f"Error deleting extracted text: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to delete extracted text",
            }

    def get_duplicate_statistics(self, user_id: str) -> Dict:
        """
        Get statistics about duplicate detection for a user.

        Args:
            user_id: The user ID

        Returns:
            Dictionary with statistics
        """
        try:
            total_documents = self.collection.count_documents({"user_id": user_id})

            return {
                "user_id": user_id,
                "total_extracted_texts": total_documents,
                "similarity_threshold": self.similarity_threshold,
                "last_check": datetime.now(timezone.utc),
            }

        except Exception as e:
            logger.error(f"Error getting duplicate statistics: {e}")
            return {
                "user_id": user_id,
                "total_extracted_texts": 0,
                "similarity_threshold": self.similarity_threshold,
                "error": str(e),
            }
