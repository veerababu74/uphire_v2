"""
Enhanced Search Engines with improved accuracy for semantic search
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from bson import ObjectId
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo.collection import Collection

from core.custom_logger import CustomLogger
from core.helpers import JSONEncoder
from .config import RAGConfig
from .utils import DocumentProcessor
from .chains import ChainManager
from .enhanced_search_processor import EnhancedSearchProcessor, SearchContext

logger = CustomLogger().get_logger("enhanced_search_engines")


def safe_log(message: str) -> str:
    """Safely encode log messages to handle Unicode characters"""
    try:
        return message.encode("ascii", errors="replace").decode("ascii")
    except Exception:
        return "".join(char if ord(char) < 128 else "?" for char in message)


class EnhancedVectorSearchEngine:
    """Enhanced vector search engine with better query processing and relevance scoring"""

    def __init__(self, vector_store: MongoDBAtlasVectorSearch, collection: Collection):
        self.vector_store = vector_store
        self.collection = collection
        self.search_processor = EnhancedSearchProcessor()

    def search(self, query: str, limit: int = 50, user_id: str = None) -> Dict:
        """Perform enhanced vector similarity search with better relevance scoring"""
        if not self.vector_store:
            logger.error("Vector store not initialized. Cannot perform search.")
            return {"error": "Vector store not initialized"}

        try:
            logger.info(f"Performing enhanced vector search for: {safe_log(query)}")

            # Parse the query to understand requirements
            search_context = self.search_processor.parse_query(query)

            # Enhance the query for better vector search
            enhanced_query = self.search_processor.enhance_query_for_vector_search(
                query, search_context
            )

            logger.info(f"Enhanced query: {safe_log(enhanced_query)}")
            logger.info(f"Search context: {search_context}")

            # Increase initial retrieval limit to allow for better filtering
            initial_limit = min(limit * 3, 150)  # Get more candidates initially

            # Perform similarity search with scores using enhanced query
            results_with_scores = self.vector_store.similarity_search_with_score(
                query=enhanced_query, k=initial_limit
            )

            if not results_with_scores:
                logger.warning("No documents found matching the query.")
                return {"results": [], "total_found": 0, "statistics": {"retrieved": 0}}

            # Process and enhance results with context-aware scoring
            processed_results = []
            for doc, vector_score in results_with_scores:
                if hasattr(doc, "metadata") and "_id" in doc.metadata:
                    doc_id = doc.metadata["_id"]
                    complete_doc = self.collection.find_one({"_id": ObjectId(doc_id)})

                    if complete_doc:
                        # Apply user_id filter if specified
                        if user_id and str(complete_doc.get("user_id", "")) != user_id:
                            continue

                        formatted_doc = DocumentProcessor.format_complete_document(
                            complete_doc
                        )

                        # Calculate enhanced relevance score
                        enhanced_score, match_reason = (
                            self.search_processor.calculate_relevance_score(
                                formatted_doc, search_context, float(vector_score)
                            )
                        )

                        formatted_doc["similarity_score"] = enhanced_score
                        formatted_doc["vector_score"] = float(
                            vector_score
                        )  # Keep original vector score
                        formatted_doc["match_reason"] = match_reason

                        processed_results.append(formatted_doc)

            # Sort by enhanced similarity score in descending order
            processed_results.sort(key=lambda x: x["similarity_score"], reverse=True)

            # Apply additional filtering based on context
            filtered_results = self._apply_context_filters(
                processed_results, search_context
            )

            # Limit to requested number of results
            final_results = filtered_results[:limit]

            # Log results with enhanced information
            logger.info("=== ENHANCED VECTOR SEARCH RESULTS ===")
            for i, result in enumerate(final_results[:5]):  # Log top 5
                logger.info(
                    f"Rank {i+1}: ID: {result['_id']}, "
                    f"Enhanced Score: {result['similarity_score']:.4f}, "
                    f"Vector Score: {result.get('vector_score', 0):.4f}, "
                    f"Reason: {result.get('match_reason', 'N/A')[:50]}..."
                )
            logger.info("=== END OF RESULTS ===")

            return {
                "results": final_results,
                "total_found": len(final_results),
                "statistics": {
                    "retrieved": len(processed_results),
                    "filtered": len(filtered_results),
                    "returned": len(final_results),
                    "query": safe_log(query),
                    "enhanced_query": safe_log(enhanced_query),
                    "search_context": str(search_context),
                },
            }

        except Exception as e:
            logger.error(f"Error during enhanced vector search: {e}")
            return {"error": str(e)}

    def _apply_context_filters(
        self, results: List[Dict], context: SearchContext
    ) -> List[Dict]:
        """Apply additional filtering based on search context"""
        if not results:
            return results

        filtered_results = []

        for result in results:
            # Apply strict filters for critical requirements
            include_result = True

            # Strict experience filter if specified
            if context.min_experience is not None:
                total_exp_str = result.get("total_experience", "0")
                try:
                    import re

                    exp_match = re.search(r"(\d+(?:\.\d+)?)", str(total_exp_str))
                    if exp_match:
                        candidate_exp = float(exp_match.group(1))
                        # Allow some flexibility (0.5 years below minimum)
                        if candidate_exp < (context.min_experience - 0.5):
                            include_result = False
                            logger.debug(
                                f"Filtered out candidate {result['_id']} - insufficient experience: {candidate_exp} < {context.min_experience}"
                            )
                except (ValueError, AttributeError):
                    pass

            # Strict salary filter if specified
            if context.max_salary is not None:
                expected_salary = result.get("expected_salary", 0)
                try:
                    expected_salary = float(expected_salary)
                    if expected_salary > 1000000:  # Convert from rupees to lakhs
                        expected_salary = expected_salary / 100000

                    # Allow 20% flexibility over budget
                    if expected_salary > (context.max_salary * 1.2):
                        include_result = False
                        logger.debug(
                            f"Filtered out candidate {result['_id']} - salary too high: {expected_salary}L > {context.max_salary}L"
                        )
                except (ValueError, TypeError):
                    pass

            if include_result:
                filtered_results.append(result)

        logger.info(
            f"Filtered {len(results)} -> {len(filtered_results)} results based on context"
        )
        return filtered_results


class EnhancedLLMSearchEngine:
    """Enhanced LLM search engine with better prompt engineering and context understanding"""

    def __init__(
        self,
        vector_store: MongoDBAtlasVectorSearch,
        collection: Collection,
        chain_manager: ChainManager,
    ):
        self.vector_store = vector_store
        self.collection = collection
        self.chain_manager = chain_manager
        self.enhanced_vector_engine = EnhancedVectorSearchEngine(
            vector_store, collection
        )
        self.search_processor = EnhancedSearchProcessor()

    def search(self, query: str, context_size: int = 5, user_id: str = None) -> Dict:
        """Perform enhanced LLM-based search with better context understanding"""
        try:
            logger.info(f"Performing enhanced LLM search for: {safe_log(query)}")

            # Parse the query to understand requirements
            search_context = self.search_processor.parse_query(query)
            logger.info(f"Parsed search context: {search_context}")

            # Get initial candidates using enhanced vector search
            # Use a larger initial pool for LLM analysis
            initial_pool_size = max(context_size * 2, 10)
            vector_results = self.enhanced_vector_engine.search(
                query, limit=initial_pool_size, user_id=user_id
            )

            if "error" in vector_results:
                return vector_results

            if not vector_results["results"]:
                return {
                    "results": [],
                    "total_analyzed": 0,
                    "statistics": {"retrieved": 0, "analyzed": 0},
                }

            # Take top candidates for LLM analysis
            candidates_for_llm = vector_results["results"][:context_size]
            doc_ids = [result["_id"] for result in candidates_for_llm]

            # Prepare enhanced context with search requirements
            context_string = self._prepare_enhanced_context(
                doc_ids, search_context, query
            )

            # Use enhanced prompt for LLM analysis
            enhanced_prompt = self._create_enhanced_prompt(query, search_context)

            logger.info("Invoking LLM for enhanced document analysis...")
            result = self.chain_manager.ranking_chain.invoke(
                {"context": context_string, "question": enhanced_prompt}
            )

            # Format results with enhanced scoring
            formatted_result = self._format_enhanced_llm_results(
                result, query, search_context, len(candidates_for_llm)
            )

            # Apply final post-processing
            final_results = self._post_process_llm_results(
                formatted_result, search_context
            )

            return final_results

        except Exception as e:
            logger.error(f"Error during enhanced LLM search: {e}")
            return {"error": str(e)}

    def _prepare_enhanced_context(
        self, doc_ids: List[str], search_context: SearchContext, original_query: str
    ) -> str:
        """Prepare enhanced context string with search requirements highlighted"""
        projection = {field: 1 for field in RAGConfig.FIELDS_TO_EXTRACT}
        if "_id" not in projection:
            projection["_id"] = 1

        fetched_docs = list(
            self.collection.find(
                {"_id": {"$in": [ObjectId(doc_id) for doc_id in doc_ids]}},
                projection,
            )
        )

        context_parts = []

        # Add search requirements summary at the beginning
        requirements_summary = self._create_requirements_summary(
            search_context, original_query
        )
        context_parts.append("=== SEARCH REQUIREMENTS ===")
        context_parts.append(requirements_summary)
        context_parts.append("=== CANDIDATE PROFILES ===")

        for doc in fetched_docs:
            # Enhanced document formatting with key highlights
            enhanced_doc = self._enhance_document_for_context(doc, search_context)
            context_parts.append(json.dumps(enhanced_doc, indent=2, cls=JSONEncoder))

        return "\n\n---\n\n".join(context_parts)

    def _create_requirements_summary(
        self, search_context: SearchContext, original_query: str
    ) -> str:
        """Create a clear summary of search requirements for the LLM"""
        requirements = []

        requirements.append(f"Original Query: {original_query}")

        if search_context.role:
            requirements.append(f"Role: {search_context.role}")

        if search_context.domain:
            requirements.append(f"Domain/Industry: {search_context.domain}")

        if search_context.min_experience or search_context.max_experience:
            if search_context.min_experience and search_context.max_experience:
                requirements.append(
                    f"Experience: {search_context.min_experience}-{search_context.max_experience} years"
                )
            elif search_context.min_experience:
                requirements.append(
                    f"Minimum Experience: {search_context.min_experience} years"
                )
            elif search_context.max_experience:
                requirements.append(
                    f"Maximum Experience: {search_context.max_experience} years"
                )

        if search_context.skills:
            requirements.append(f"Required Skills: {', '.join(search_context.skills)}")

        if search_context.max_salary:
            requirements.append(
                f"Salary Budget: Up to {search_context.max_salary} lakhs"
            )

        if search_context.location:
            requirements.append(f"Location Preference: {search_context.location}")

        return "\n".join(requirements)

    def _enhance_document_for_context(
        self, doc: Dict, search_context: SearchContext
    ) -> Dict:
        """Enhance document representation for better LLM understanding"""
        enhanced_doc = {}

        for field, value in doc.items():
            if field == "_id":
                enhanced_doc[field] = str(value)
            elif field in ["skills", "may_also_known_skills", "labels"]:
                enhanced_doc[field] = DocumentProcessor.normalize_list_field(value)
            else:
                enhanced_doc[field] = DocumentProcessor.normalize_field_value(value)

        # Add computed fields for better LLM understanding
        enhanced_doc["_computed_highlights"] = self._compute_highlights(
            enhanced_doc, search_context
        )

        return enhanced_doc

    def _compute_highlights(self, doc: Dict, search_context: SearchContext) -> Dict:
        """Compute highlights that match search requirements"""
        highlights = {}

        # Experience highlight
        total_exp_str = doc.get("total_experience", "0")
        try:
            import re

            exp_match = re.search(r"(\d+(?:\.\d+)?)", str(total_exp_str))
            if exp_match:
                exp_years = float(exp_match.group(1))
                highlights["experience_years"] = exp_years

                if search_context.min_experience or search_context.max_experience:
                    if search_context.min_experience and search_context.max_experience:
                        if (
                            search_context.min_experience
                            <= exp_years
                            <= search_context.max_experience
                        ):
                            highlights["experience_match"] = "PERFECT_MATCH"
                        else:
                            highlights["experience_match"] = "OUT_OF_RANGE"
                    elif search_context.min_experience:
                        highlights["experience_match"] = (
                            "MEETS_MINIMUM"
                            if exp_years >= search_context.min_experience
                            else "BELOW_MINIMUM"
                        )
        except:
            pass

        # Skills highlight
        if search_context.skills:
            doc_skills = []
            for field in ["skills", "may_also_known_skills", "labels"]:
                field_skills = doc.get(field, [])
                if isinstance(field_skills, list):
                    doc_skills.extend([skill.lower() for skill in field_skills])

            matched_skills = []
            for req_skill in search_context.skills:
                for doc_skill in doc_skills:
                    if req_skill.lower() in doc_skill or doc_skill in req_skill.lower():
                        matched_skills.append(req_skill)
                        break

            highlights["matched_skills"] = matched_skills
            highlights["skills_match_ratio"] = (
                len(matched_skills) / len(search_context.skills)
                if search_context.skills
                else 0
            )

        # Salary highlight
        if search_context.max_salary:
            expected_salary = doc.get("expected_salary", 0)
            try:
                expected_salary = float(expected_salary)
                if expected_salary > 1000000:
                    expected_salary = expected_salary / 100000

                highlights["expected_salary_lakhs"] = expected_salary
                highlights["salary_within_budget"] = (
                    expected_salary <= search_context.max_salary
                )
            except:
                pass

        return highlights

    def _create_enhanced_prompt(
        self, original_query: str, search_context: SearchContext
    ) -> str:
        """Create enhanced prompt that incorporates parsed search context"""
        prompt_parts = [original_query]

        # Add specific instructions based on parsed context
        if search_context.min_experience or search_context.max_experience:
            if search_context.min_experience and search_context.max_experience:
                prompt_parts.append(
                    f"CRITICAL: Candidate must have {search_context.min_experience}-{search_context.max_experience} years of experience."
                )
            elif search_context.min_experience:
                prompt_parts.append(
                    f"CRITICAL: Candidate must have at least {search_context.min_experience} years of experience."
                )

        if search_context.skills:
            prompt_parts.append(
                f"IMPORTANT: Look for candidates with skills in: {', '.join(search_context.skills)}"
            )

        if search_context.domain:
            prompt_parts.append(
                f"DOMAIN FOCUS: Prioritize candidates with {search_context.domain} industry experience."
            )

        if search_context.max_salary:
            prompt_parts.append(
                f"BUDGET CONSTRAINT: Expected salary should not exceed {search_context.max_salary} lakhs."
            )

        prompt_parts.append(
            "Analyze the _computed_highlights field for quick relevance assessment."
        )

        return " ".join(prompt_parts)

    def _format_enhanced_llm_results(
        self, result, query: str, search_context: SearchContext, retrieved_count: int
    ) -> Dict:
        """Format LLM results with enhanced scoring and explanations"""
        matches = None

        if hasattr(result, "matches"):
            matches = result.matches
        elif isinstance(result, dict) and "matches" in result:
            matches = result["matches"]

        if matches is not None:
            formatted_results = []
            for match in matches:
                match_id = None
                relevance_score = 0.0
                match_reason = "No explanation provided"

                if hasattr(match, "id"):
                    match_id = match.id
                    relevance_score = match.relevance_score
                    match_reason = match.match_reason
                elif isinstance(match, dict):
                    match_id = match.get("_id") or match.get("id")
                    relevance_score = match.get("relevance_score", 0.0)
                    match_reason = match.get("match_reason", "No explanation provided")

                if match_id:
                    complete_doc = self.collection.find_one({"_id": ObjectId(match_id)})

                    if complete_doc:
                        formatted_doc = DocumentProcessor.format_complete_document(
                            complete_doc
                        )

                        # Enhance the relevance score using our context-aware scoring
                        enhanced_score, enhanced_reason = (
                            self.search_processor.calculate_relevance_score(
                                formatted_doc, search_context, float(relevance_score)
                            )
                        )

                        formatted_doc.update(
                            {
                                "relevance_score": enhanced_score,
                                "llm_score": float(
                                    relevance_score
                                ),  # Keep original LLM score
                                "match_reason": f"{match_reason} | {enhanced_reason}",
                            }
                        )
                        formatted_results.append(formatted_doc)

            # Sort by enhanced relevance score
            formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)

            return {
                "results": formatted_results,
                "total_analyzed": len(formatted_results),
                "statistics": {
                    "retrieved": retrieved_count,
                    "analyzed": len(formatted_results),
                    "query": safe_log(query),
                    "search_context": str(search_context),
                },
            }
        else:
            logger.error(f"Unexpected LLM result format: {result}")
            return {
                "error": "Unexpected LLM result format",
                "raw_result": str(result),
            }

    def _post_process_llm_results(
        self, formatted_result: Dict, search_context: SearchContext
    ) -> Dict:
        """Apply final post-processing to LLM results"""
        if "results" not in formatted_result:
            return formatted_result

        results = formatted_result["results"]

        # Apply final filtering and boost scores for perfect matches
        final_results = []
        for result in results:
            # Boost score for candidates that perfectly match critical criteria
            boost_factor = 1.0

            # Experience perfect match boost
            if search_context.min_experience and search_context.max_experience:
                total_exp_str = result.get("total_experience", "0")
                try:
                    import re

                    exp_match = re.search(r"(\d+(?:\.\d+)?)", str(total_exp_str))
                    if exp_match:
                        exp_years = float(exp_match.group(1))
                        if (
                            search_context.min_experience
                            <= exp_years
                            <= search_context.max_experience
                        ):
                            boost_factor *= (
                                1.1  # 10% boost for perfect experience match
                            )
                except:
                    pass

            # Skills match boost
            if search_context.skills:
                doc_skills = []
                for field in ["skills", "may_also_known_skills", "labels"]:
                    field_skills = result.get(field, [])
                    if isinstance(field_skills, list):
                        doc_skills.extend([skill.lower() for skill in field_skills])

                matched_skills = []
                for req_skill in search_context.skills:
                    for doc_skill in doc_skills:
                        if req_skill.lower() in doc_skill:
                            matched_skills.append(req_skill)
                            break

                match_ratio = len(matched_skills) / len(search_context.skills)
                if match_ratio >= 0.8:  # 80% or more skills match
                    boost_factor *= 1.15  # 15% boost for excellent skills match

            # Apply boost
            current_score = result.get("relevance_score", 0.0)
            result["relevance_score"] = min(1.0, current_score * boost_factor)

            final_results.append(result)

        # Re-sort after boosting
        final_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        formatted_result["results"] = final_results
        return formatted_result
