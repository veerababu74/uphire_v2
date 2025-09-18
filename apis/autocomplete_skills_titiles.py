from fastapi import APIRouter, Query, Body, Depends, HTTPException
from mangodatabase.client import get_collection
from embeddings.vectorizer import Vectorizer
from core.helpers import format_resume
from typing import List, Dict, Any
from collections import Counter
import pymongo
import re
import difflib
import time
import logging
from functools import lru_cache

router = APIRouter(
    prefix="/autocomplete",
    tags=["Job Search Autocomplete"],
)

collection = get_collection()
vectorizer = Vectorizer()

# Simple in-memory cache for frequently accessed results
_autocomplete_cache = {}
_cache_expiry = {}
CACHE_DURATION = 300  # 5 minutes cache

# Performance monitoring
_search_stats = {
    "total_requests": 0,
    "cache_hits": 0,
    "semantic_searches": 0,
    "traditional_searches": 0,
    "average_response_time": 0.0,
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_results(results, key):
    """
    Process the results by converting to lowercase, stripping whitespace,
    and removing duplicates while maintaining order.
    Also filters out empty or blank strings.
    """
    processed = []
    seen = set()
    for result in results:
        value = result.get(key)
        if not isinstance(value, str):
            continue  # Skip invalid or None values
        stripped = value.strip()
        if not stripped:  # Skip empty or whitespace-only strings
            continue
        lower_value = stripped.lower()
        if lower_value not in seen:
            seen.add(lower_value)
            processed.append(value)  # Preserve original casing
    return processed


def extract_skills(raw_data: str) -> List[str]:
    if not raw_data or not isinstance(raw_data, str):
        return []

    # Handle different skill formats more carefully
    skills = []

    # Split by common delimiters while preserving compound skills
    delimiters = r"[,;|/&]|\sand\s|\sor\s"
    raw_skills = re.split(delimiters, raw_data, flags=re.IGNORECASE)

    for skill in raw_skills:
        # Clean the skill but preserve important characters
        cleaned = re.sub(
            r"^[:\-\s]+|[:\-\s]+$", "", skill
        )  # Remove leading/trailing separators
        cleaned = re.sub(r"\s+", " ", cleaned)  # Normalize whitespace
        cleaned = cleaned.strip()

        # Filter out non-skills and very short/long entries
        if (
            cleaned
            and len(cleaned) >= 2
            and len(cleaned) <= 50
            and not re.match(r"^\d+$", cleaned)  # Not just numbers
            and cleaned.lower()
            not in {
                "others",
                "and",
                "in",
                "of",
                "the",
                "with",
                "using",
                "etc",
                "various",
            }
        ):
            skills.append(cleaned)

    return skills


@lru_cache(maxsize=1000)
def get_dynamic_skill_relationships() -> Dict[str, Dict[str, float]]:
    """
    Build dynamic skill relationships from database analysis
    Cached for performance
    """
    try:
        # Sample a reasonable subset for relationship analysis
        pipeline = [
            {"$match": {"skills": {"$exists": True, "$not": {"$size": 0}}}},
            {"$sample": {"size": 5000}},  # Sample for performance
            {"$project": {"skills": 1}},
        ]

        results = list(collection.aggregate(pipeline))
        relationships = {}

        for doc in results:
            skills = doc.get("skills", [])
            if not isinstance(skills, list):
                continue

            # Normalize skills
            normalized_skills = [
                skill.lower().strip()
                for skill in skills
                if isinstance(skill, str) and skill.strip()
            ]

            # Build co-occurrence relationships
            for i, skill1 in enumerate(normalized_skills):
                if skill1 not in relationships:
                    relationships[skill1] = {}

                for j, skill2 in enumerate(normalized_skills):
                    if i != j and skill2 not in relationships[skill1]:
                        relationships[skill1][skill2] = 0.0
                    if i != j:
                        relationships[skill1][skill2] += 1.0

        # Normalize relationship scores
        for skill1 in relationships:
            total_cooccurrences = sum(relationships[skill1].values())
            if total_cooccurrences > 0:
                for skill2 in relationships[skill1]:
                    relationships[skill1][skill2] /= total_cooccurrences

        return relationships
    except Exception as e:
        logger.warning(f"Error building dynamic relationships: {e}")
        return {}


@lru_cache(maxsize=500)
def get_skill_frequencies() -> Dict[str, int]:
    """
    Get skill frequency distribution from database
    Cached for performance
    """
    try:
        pipeline = [
            {"$match": {"skills": {"$exists": True, "$not": {"$size": 0}}}},
            {"$unwind": "$skills"},
            {
                "$group": {
                    "_id": {"$toLower": {"$trim": {"input": "$skills"}}},
                    "count": {"$sum": 1},
                }
            },
            {"$match": {"_id": {"$ne": ""}}},
            {"$sort": {"count": -1}},
        ]

        results = list(collection.aggregate(pipeline))
        frequencies = {doc["_id"]: doc["count"] for doc in results}
        return frequencies
    except Exception as e:
        logger.warning(f"Error getting skill frequencies: {e}")
        return {}


def calculate_relevance_score(
    text: str, prefix: str, semantic_score: float = 0.0
) -> float:
    """
    Calculate relevance score with dynamic relationship and frequency-based scoring
    No hardcoded technology lists - fully data-driven approach
    """
    if not text or not prefix:
        return semantic_score

    text_lower = text.lower()
    prefix_lower = prefix.lower()

    # Get dynamic data (cached for performance)
    relationships = get_dynamic_skill_relationships()
    skill_frequencies = get_skill_frequencies()

    base_score = semantic_score

    # Core relevance scoring (unchanged logic)
    if text_lower == prefix_lower:
        base_score = max(1.0, semantic_score + 0.5)
    elif text_lower.startswith(prefix_lower):
        base_score = max(0.9 - (len(text) - len(prefix)) * 0.01, semantic_score + 0.3)
    elif prefix_lower in text_lower:
        position = text_lower.find(prefix_lower)
        position_penalty = position * 0.01
        base_score = max(0.7 - position_penalty, semantic_score + 0.2)
    elif re.search(r"\b" + re.escape(prefix_lower), text_lower):
        base_score = max(0.6, semantic_score + 0.15)
    else:
        similarity = difflib.SequenceMatcher(None, prefix_lower, text_lower).ratio()
        if similarity > 0.7:
            base_score = max(0.5 * similarity, semantic_score + 0.1)
        else:
            words = text_lower.split()
            for word in words:
                if word.startswith(prefix_lower):
                    base_score = max(0.4, semantic_score + 0.1)
                    break
            else:
                if len(prefix) >= 2:
                    prefix_chars = list(prefix_lower)
                    text_chars = list(text_lower.replace(" ", ""))
                    i = 0
                    for char in text_chars:
                        if i < len(prefix_chars) and char == prefix_chars[i]:
                            i += 1
                    if i == len(prefix_chars):
                        base_score = max(0.3, semantic_score)

    # DYNAMIC BONUSES (replacing static lists)

    # 1. Relationship-based bonus
    relationship_bonus = 0.0
    if prefix_lower in relationships and text_lower in relationships[prefix_lower]:
        relationship_strength = relationships[prefix_lower][text_lower]
        relationship_bonus = min(relationship_strength * 2, 0.3)

    # Reverse relationship check
    if text_lower in relationships and prefix_lower in relationships[text_lower]:
        reverse_strength = relationships[text_lower][prefix_lower]
        relationship_bonus = max(relationship_bonus, min(reverse_strength * 2, 0.3))

    # 2. Frequency-based importance (replaces hardcoded important_tech)
    frequency_bonus = 0.0
    if skill_frequencies and text_lower in skill_frequencies:
        # Normalize frequency to bonus (0.0 to 0.2 range)
        max_freq = max(skill_frequencies.values()) if skill_frequencies else 1
        frequency_ratio = skill_frequencies[text_lower] / max_freq
        frequency_bonus = min(frequency_ratio * 0.2, 0.2)

    # 3. Dynamic penalty for generic terms (data-driven)
    generic_penalty = 0.0
    if skill_frequencies and text_lower in skill_frequencies:
        # If a term appears too frequently, it might be generic
        total_skills = len(skill_frequencies)
        if (
            skill_frequencies[text_lower] > total_skills * 0.1
        ):  # Appears in >10% of records
            generic_penalty = -0.1

    # 4. Static generic terms penalty (minimal hardcoded list)
    very_generic_terms = [
        "and",
        "the",
        "with",
        "for",
        "from",
        "this",
        "that",
        "have",
        "will",
        "can",
    ]
    if text_lower in very_generic_terms:
        generic_penalty = -0.3

    # Apply dynamic bonuses
    final_score = base_score + relationship_bonus + frequency_bonus + generic_penalty

    return max(0.0, min(2.0, final_score))


def fuzzy_search_enhancement(
    candidates: List[str], prefix: str, limit: int
) -> List[str]:
    """Enhance search results with fuzzy matching"""
    if not candidates:
        return []

    # Score all candidates
    scored_candidates = []
    for candidate in candidates:
        score = calculate_relevance_score(candidate, prefix)
        if score > 0:
            scored_candidates.append((candidate, score))

    # Sort by score and return top results
    scored_candidates.sort(key=lambda x: (-x[1], len(x[0]), x[0].lower()))
    return [candidate for candidate, _ in scored_candidates[:limit]]


def get_cache_key(prefix: str, limit: int, search_type: str) -> str:
    """Generate cache key for results"""
    return f"{search_type}:{prefix.lower()}:{limit}"


def is_cache_valid(key: str) -> bool:
    """Check if cache entry is still valid"""
    return key in _cache_expiry and time.time() < _cache_expiry[key]


def set_cache(key: str, value: Any):
    """Set cache value with expiry"""
    _autocomplete_cache[key] = value
    _cache_expiry[key] = time.time() + CACHE_DURATION


def get_dynamic_context_terms(prefix: str, limit: int = 10) -> List[str]:
    """Generate dynamic context terms by analyzing database relationships"""
    cache_key = get_cache_key(prefix, limit, "dynamic_context")

    # Check cache first
    if is_cache_valid(cache_key):
        return _autocomplete_cache[cache_key]

    try:
        context_terms = set()
        prefix_lower = prefix.lower()

        # Strategy 1: Find co-occurring skills and titles in same resumes
        cooccurrence_pipeline = [
            {
                "$match": {
                    "$or": [
                        {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}},
                        {
                            "technical_skills": {
                                "$regex": f".*{prefix}.*",
                                "$options": "i",
                            }
                        },
                        {
                            "experience.title": {
                                "$regex": f".*{prefix}.*",
                                "$options": "i",
                            }
                        },
                        {
                            "experience.skills": {
                                "$regex": f".*{prefix}.*",
                                "$options": "i",
                            }
                        },
                    ]
                }
            },
            {"$limit": 200},  # Limit for performance
            {
                "$project": {
                    "skills": 1,
                    "technical_skills": 1,
                    "experience.title": 1,
                    "experience.skills": 1,
                    "_id": 0,
                }
            },
        ]

        cooccurrence_results = list(
            collection.aggregate(cooccurrence_pipeline, maxTimeMS=3000)
        )

        # Extract related terms from co-occurring documents
        term_frequency = {}

        for doc in cooccurrence_results:
            # Process skills fields
            for field_name in ["skills", "technical_skills"]:
                field_data = doc.get(field_name)
                if field_data:
                    if isinstance(field_data, list):
                        for item in field_data:
                            if isinstance(item, str):
                                extracted_skills = extract_skills(item)
                                for skill in extracted_skills:
                                    if (
                                        skill.lower() != prefix_lower
                                        and len(skill) >= 2
                                    ):
                                        term_frequency[skill.lower()] = (
                                            term_frequency.get(skill.lower(), 0) + 1
                                        )
                    elif isinstance(field_data, str):
                        extracted_skills = extract_skills(field_data)
                        for skill in extracted_skills:
                            if skill.lower() != prefix_lower and len(skill) >= 2:
                                term_frequency[skill.lower()] = (
                                    term_frequency.get(skill.lower(), 0) + 1
                                )

            # Process job titles
            experience_data = doc.get("experience")
            if isinstance(experience_data, list):
                for exp in experience_data[:3]:  # Limit for performance
                    if isinstance(exp, dict):
                        title = exp.get("title")
                        if title and isinstance(title, str):
                            # Extract meaningful words from titles
                            title_words = re.findall(r"\b\w{3,}\b", title.lower())
                            for word in title_words:
                                if word != prefix_lower and len(word) >= 3:
                                    term_frequency[word] = (
                                        term_frequency.get(word, 0) + 1
                                    )

                        # Process experience skills
                        exp_skills = exp.get("skills")
                        if exp_skills:
                            if isinstance(exp_skills, list):
                                for skill in exp_skills:
                                    if isinstance(skill, str):
                                        extracted_skills = extract_skills(skill)
                                        for extracted_skill in extracted_skills:
                                            if (
                                                extracted_skill.lower() != prefix_lower
                                                and len(extracted_skill) >= 2
                                            ):
                                                term_frequency[
                                                    extracted_skill.lower()
                                                ] = (
                                                    term_frequency.get(
                                                        extracted_skill.lower(), 0
                                                    )
                                                    + 1
                                                )
                            elif isinstance(exp_skills, str):
                                extracted_skills = extract_skills(exp_skills)
                                for skill in extracted_skills:
                                    if (
                                        skill.lower() != prefix_lower
                                        and len(skill) >= 2
                                    ):
                                        term_frequency[skill.lower()] = (
                                            term_frequency.get(skill.lower(), 0) + 1
                                        )

        # Strategy 2: Use semantic similarity from vectorizer if available
        if hasattr(vectorizer, "generate_embedding"):
            try:
                # Get semantically similar documents
                query_embedding = vectorizer.generate_embedding(prefix)
                semantic_pipeline = [
                    {
                        "$search": {
                            "index": "vector_search_index",
                            "knnBeta": {
                                "vector": query_embedding,
                                "path": "experience_text_vector",
                                "k": 50,
                            },
                        }
                    },
                    {
                        "$project": {
                            "skills": 1,
                            "technical_skills": 1,
                            "experience.title": 1,
                            "score": {"$meta": "searchScore"},
                            "_id": 0,
                        }
                    },
                    {"$limit": 50},
                ]

                semantic_results = list(
                    collection.aggregate(semantic_pipeline, maxTimeMS=2000)
                )

                # Process semantic results with higher weight
                for doc in semantic_results:
                    semantic_weight = min(doc.get("score", 0) / 5, 2.0)  # Cap weight

                    # Process skills with semantic weight
                    for field_name in ["skills", "technical_skills"]:
                        field_data = doc.get(field_name)
                        if field_data:
                            if isinstance(field_data, list):
                                for item in field_data:
                                    if isinstance(item, str):
                                        extracted_skills = extract_skills(item)
                                        for skill in extracted_skills:
                                            if (
                                                skill.lower() != prefix_lower
                                                and len(skill) >= 2
                                            ):
                                                weighted_count = int(
                                                    semantic_weight + 1
                                                )
                                                term_frequency[skill.lower()] = (
                                                    term_frequency.get(skill.lower(), 0)
                                                    + weighted_count
                                                )
                            elif isinstance(field_data, str):
                                extracted_skills = extract_skills(field_data)
                                for skill in extracted_skills:
                                    if (
                                        skill.lower() != prefix_lower
                                        and len(skill) >= 2
                                    ):
                                        weighted_count = int(semantic_weight + 1)
                                        term_frequency[skill.lower()] = (
                                            term_frequency.get(skill.lower(), 0)
                                            + weighted_count
                                        )

                    # Process titles with semantic weight
                    experience_data = doc.get("experience")
                    if isinstance(experience_data, list):
                        for exp in experience_data[:2]:
                            if isinstance(exp, dict):
                                title = exp.get("title")
                                if title and isinstance(title, str):
                                    title_words = re.findall(
                                        r"\b\w{3,}\b", title.lower()
                                    )
                                    for word in title_words:
                                        if word != prefix_lower and len(word) >= 3:
                                            weighted_count = int(semantic_weight + 1)
                                            term_frequency[word] = (
                                                term_frequency.get(word, 0)
                                                + weighted_count
                                            )

            except Exception as e:
                print(f"Semantic context extraction failed: {e}")

        # Strategy 3: Extract word co-occurrences from job descriptions/experience text
        try:
            text_analysis_pipeline = [
                {
                    "$match": {
                        "$or": [
                            {
                                "experience.description": {
                                    "$regex": f".*{prefix}.*",
                                    "$options": "i",
                                }
                            },
                            {
                                "experience.responsibilities": {
                                    "$regex": f".*{prefix}.*",
                                    "$options": "i",
                                }
                            },
                        ]
                    }
                },
                {"$limit": 100},
                {
                    "$project": {
                        "experience.description": 1,
                        "experience.responsibilities": 1,
                        "_id": 0,
                    }
                },
            ]

            text_results = list(
                collection.aggregate(text_analysis_pipeline, maxTimeMS=2000)
            )

            for doc in text_results:
                experience_data = doc.get("experience")
                if isinstance(experience_data, list):
                    for exp in experience_data[:2]:
                        if isinstance(exp, dict):
                            # Process description text
                            for text_field in ["description", "responsibilities"]:
                                text_content = exp.get(text_field)
                                if (
                                    text_content
                                    and isinstance(text_content, str)
                                    and prefix_lower in text_content.lower()
                                ):
                                    # Extract meaningful terms from text
                                    words = re.findall(
                                        r"\b[a-zA-Z]{3,}\b", text_content.lower()
                                    )
                                    for word in words:
                                        if (
                                            word != prefix_lower
                                            and len(word) >= 3
                                            and word
                                            not in {
                                                "and",
                                                "the",
                                                "for",
                                                "with",
                                                "this",
                                                "that",
                                                "have",
                                                "will",
                                                "are",
                                                "was",
                                                "been",
                                                "from",
                                                "they",
                                                "were",
                                                "but",
                                                "not",
                                                "what",
                                                "can",
                                                "had",
                                                "her",
                                                "his",
                                                "she",
                                                "has",
                                                "him",
                                            }
                                        ):
                                            term_frequency[word] = (
                                                term_frequency.get(word, 0) + 1
                                            )

        except Exception as e:
            print(f"Text analysis for context failed: {e}")

        # Filter and rank terms by frequency and relevance
        filtered_terms = []
        for term, freq in term_frequency.items():
            if freq >= 2 and len(term) >= 2:  # Minimum frequency threshold
                # Calculate relevance score
                relevance = calculate_relevance_score(term, prefix)
                if relevance > 0.01:  # Very low threshold for context terms
                    final_score = freq + (
                        relevance * 10
                    )  # Combine frequency and relevance
                    filtered_terms.append((term, final_score))

        # Sort by combined score and return top terms
        filtered_terms.sort(key=lambda x: -x[1])
        context_terms = [term for term, score in filtered_terms[:limit]]

        # Cache the result
        set_cache(cache_key, context_terms)

        return context_terms

    except Exception as e:
        print(f"Dynamic context generation failed: {e}")
        return []


def build_dynamic_skill_relationships() -> Dict[str, List[str]]:
    """Build skill relationship mapping from database analysis"""
    cache_key = "skill_relationships_global"

    # Check cache first (longer cache for this expensive operation)
    if cache_key in _autocomplete_cache and time.time() < _cache_expiry.get(
        cache_key, 0
    ):
        return _autocomplete_cache[cache_key]

    try:
        print("Building dynamic skill relationships from database...")

        # Get all documents with skills
        relationship_pipeline = [
            {
                "$match": {
                    "$or": [
                        {"skills": {"$exists": True, "$ne": [], "$ne": None}},
                        {"technical_skills": {"$exists": True, "$ne": [], "$ne": None}},
                        {
                            "experience.skills": {
                                "$exists": True,
                                "$ne": [],
                                "$ne": None,
                            }
                        },
                    ]
                }
            },
            {"$limit": 1000},  # Limit for performance but get good sample
            {
                "$project": {
                    "skills": 1,
                    "technical_skills": 1,
                    "experience.skills": 1,
                    "experience.title": 1,
                    "_id": 0,
                }
            },
        ]

        documents = list(collection.aggregate(relationship_pipeline, maxTimeMS=10000))

        # Build co-occurrence matrix
        skill_cooccurrence = {}

        for doc in documents:
            # Extract all skills from this document
            doc_skills = set()

            # Process direct skill fields
            for field_name in ["skills", "technical_skills"]:
                field_data = doc.get(field_name)
                if field_data:
                    if isinstance(field_data, list):
                        for item in field_data:
                            if isinstance(item, str):
                                extracted_skills = extract_skills(item)
                                doc_skills.update(
                                    [s.lower() for s in extracted_skills if len(s) >= 2]
                                )
                    elif isinstance(field_data, str):
                        extracted_skills = extract_skills(field_data)
                        doc_skills.update(
                            [s.lower() for s in extracted_skills if len(s) >= 2]
                        )

            # Process experience skills
            experience_data = doc.get("experience")
            if isinstance(experience_data, list):
                for exp in experience_data:
                    if isinstance(exp, dict):
                        exp_skills = exp.get("skills")
                        if exp_skills:
                            if isinstance(exp_skills, list):
                                for skill in exp_skills:
                                    if isinstance(skill, str):
                                        extracted_skills = extract_skills(skill)
                                        doc_skills.update(
                                            [
                                                s.lower()
                                                for s in extracted_skills
                                                if len(s) >= 2
                                            ]
                                        )
                            elif isinstance(exp_skills, str):
                                extracted_skills = extract_skills(exp_skills)
                                doc_skills.update(
                                    [s.lower() for s in extracted_skills if len(s) >= 2]
                                )

                        # Also extract skills from job titles
                        title = exp.get("title")
                        if title and isinstance(title, str):
                            # Extract technology mentions from job titles
                            title_words = re.findall(r"\b\w{2,}\b", title.lower())
                            for word in title_words:
                                if len(word) >= 2 and word not in {
                                    "senior",
                                    "junior",
                                    "lead",
                                    "manager",
                                    "engineer",
                                    "developer",
                                    "analyst",
                                    "specialist",
                                    "consultant",
                                    "architect",
                                    "team",
                                    "software",
                                    "system",
                                    "application",
                                    "web",
                                    "mobile",
                                }:
                                    doc_skills.add(word)

            # Build co-occurrence relationships
            doc_skills_list = list(doc_skills)
            for i, skill1 in enumerate(doc_skills_list):
                if skill1 not in skill_cooccurrence:
                    skill_cooccurrence[skill1] = {}

                for j, skill2 in enumerate(doc_skills_list):
                    if i != j:  # Don't relate skill to itself
                        if skill2 not in skill_cooccurrence[skill1]:
                            skill_cooccurrence[skill1][skill2] = 0
                        skill_cooccurrence[skill1][skill2] += 1

        # Convert to relationship mapping
        relationships = {}
        for skill, related_skills in skill_cooccurrence.items():
            if len(skill) >= 2:
                # Sort by co-occurrence frequency and take top related skills
                sorted_related = sorted(
                    related_skills.items(), key=lambda x: x[1], reverse=True
                )
                relationships[skill] = [
                    related_skill
                    for related_skill, count in sorted_related[:15]
                    if count >= 2
                ]

        # Cache the result for 1 hour (expensive operation)
        _autocomplete_cache[cache_key] = relationships
        _cache_expiry[cache_key] = time.time() + 3600  # 1 hour cache

        print(f"Built relationships for {len(relationships)} skills")
        return relationships

    except Exception as e:
        print(f"Failed to build skill relationships: {e}")
        return {}


def get_enhanced_dynamic_context_terms(prefix: str, limit: int = 10) -> List[str]:
    """Enhanced dynamic context using pre-built relationship mapping"""
    try:
        # Get the relationship mapping
        relationships = build_dynamic_skill_relationships()

        prefix_lower = prefix.lower()
        context_terms = set()

        # Direct lookup in relationships
        if prefix_lower in relationships:
            context_terms.update(relationships[prefix_lower][:limit])

        # Partial matching in relationship keys
        for skill, related_skills in relationships.items():
            if prefix_lower in skill or skill in prefix_lower:
                context_terms.update(
                    related_skills[:5]
                )  # Limit to avoid too many terms

        # Convert back to list and limit
        context_list = list(context_terms)[:limit]

        # If we don't have enough terms, fall back to the co-occurrence method
        if len(context_list) < limit // 2:
            fallback_terms = get_dynamic_context_terms(
                prefix, limit - len(context_list)
            )
            context_list.extend(
                [term for term in fallback_terms if term not in context_list]
            )

        return context_list[:limit]

    except Exception as e:
        print(f"Enhanced dynamic context failed: {e}")
        # Fallback to basic dynamic context
        return get_dynamic_context_terms(prefix, limit)


def get_semantic_context_terms(prefix: str) -> List[str]:
    """Generate semantic context terms using dynamic database analysis"""
    return get_enhanced_dynamic_context_terms(prefix, limit=10)


def get_semantic_titles_search(prefix: str, limit: int) -> List[tuple]:
    """Enhanced semantic search for job titles"""
    try:
        # Generate context-aware search terms
        context_terms = get_semantic_context_terms(prefix)
        search_queries = [prefix] + context_terms[
            :5
        ]  # Limit context terms for performance

        title_candidates = []
        processed_titles = set()

        for search_term in search_queries:
            try:
                # Generate embedding for search term
                query_embedding = vectorizer.generate_embedding(
                    f"{search_term} job title position role"
                )

                # Semantic search pipeline
                semantic_pipeline = [
                    {
                        "$search": {
                            "index": "vector_search_index",
                            "knnBeta": {
                                "vector": query_embedding,
                                "path": "experience_text_vector",
                                "k": limit * 2,
                            },
                        }
                    },
                    {"$unwind": "$experience"},
                    {
                        "$match": {
                            "experience.title": {
                                "$exists": True,
                                "$ne": "",
                                "$ne": None,
                            }
                        }
                    },
                    {
                        "$project": {
                            "title": "$experience.title",
                            "score": {"$meta": "searchScore"},
                            "_id": 0,
                        }
                    },
                    {"$limit": limit * 3},
                ]

                results = list(collection.aggregate(semantic_pipeline, maxTimeMS=5000))

                for result in results:
                    title = result.get("title", "").strip()
                    if title and title.lower() not in processed_titles:
                        processed_titles.add(title.lower())

                        # Calculate semantic score based on search score and relevance
                        semantic_score = min(result.get("score", 0) / 10, 0.8)

                        # Boost score if it's from the original prefix search
                        if search_term == prefix:
                            semantic_score += 0.3

                        # Calculate final relevance score
                        final_score = calculate_relevance_score(
                            title, prefix, semantic_score
                        )

                        if final_score > 0.1:  # Threshold for inclusion
                            title_candidates.append((title, final_score))

            except Exception as e:
                print(f"Error in semantic search for term '{search_term}': {e}")
                continue

        # Sort by score and return top results
        title_candidates.sort(key=lambda x: (-x[1], len(x[0]), x[0].lower()))
        return title_candidates[:limit]

    except Exception as e:
        print(f"Semantic titles search failed: {e}")
        return []


def get_semantic_skills_search(prefix: str, limit: int) -> List[str]:
    """Enhanced semantic search for skills"""
    try:
        # Generate context-aware search terms
        context_terms = get_semantic_context_terms(prefix)
        search_queries = [prefix] + context_terms[:5]

        skill_candidates = set()

        for search_term in search_queries:
            try:
                # Generate embedding for search term
                query_embedding = vectorizer.generate_embedding(
                    f"{search_term} skill technology tool framework"
                )

                # Semantic search pipeline
                semantic_pipeline = [
                    {
                        "$search": {
                            "index": "vector_search_index",
                            "knnBeta": {
                                "vector": query_embedding,
                                "path": "experience_text_vector",
                                "k": limit * 3,
                            },
                        }
                    },
                    {
                        "$project": {
                            "skills": 1,
                            "technical_skills": 1,
                            "experience.skills": 1,
                            "score": {"$meta": "searchScore"},
                            "_id": 0,
                        }
                    },
                    {"$limit": limit * 4},
                ]

                results = list(collection.aggregate(semantic_pipeline, maxTimeMS=5000))

                for doc in results:
                    # Process skills from various fields
                    for field_name in ["skills", "technical_skills"]:
                        field_data = doc.get(field_name)
                        if field_data:
                            if isinstance(field_data, list):
                                for item in field_data:
                                    if isinstance(item, str):
                                        extracted_skills = extract_skills(item)
                                        skill_candidates.update(extracted_skills)
                            elif isinstance(field_data, str):
                                extracted_skills = extract_skills(field_data)
                                skill_candidates.update(extracted_skills)

                    # Process experience skills
                    experience_data = doc.get("experience")
                    if isinstance(experience_data, list):
                        for exp in experience_data[:2]:  # Limit for performance
                            if isinstance(exp, dict):
                                exp_skills = exp.get("skills")
                                if exp_skills:
                                    if isinstance(exp_skills, list):
                                        for skill in exp_skills:
                                            if isinstance(skill, str):
                                                extracted_skills = extract_skills(skill)
                                                skill_candidates.update(
                                                    extracted_skills
                                                )
                                    elif isinstance(exp_skills, str):
                                        extracted_skills = extract_skills(exp_skills)
                                        skill_candidates.update(extracted_skills)

            except Exception as e:
                print(f"Error in semantic skills search for term '{search_term}': {e}")
                continue

        # Filter and score skills with more flexible matching
        valid_skills = []
        for skill in skill_candidates:
            if len(skill) >= 2 and len(skill) <= 50:
                # Check if skill is relevant to the search (more flexible for semantic results)
                relevance_score = calculate_relevance_score(skill, prefix)

                # For semantic search, also consider skills that might be contextually related
                # even if they don't contain the exact prefix
                is_contextually_relevant = False
                if relevance_score <= 0.05:
                    # Check if skill contains any word from the prefix
                    prefix_words = prefix.lower().split()
                    skill_words = skill.lower().split()
                    for p_word in prefix_words:
                        for s_word in skill_words:
                            if len(p_word) >= 3 and (
                                p_word in s_word or s_word in p_word
                            ):
                                is_contextually_relevant = True
                                relevance_score = (
                                    0.1  # Minimum score for contextual relevance
                                )
                                break
                        if is_contextually_relevant:
                            break

                if relevance_score > 0.05 or is_contextually_relevant:
                    valid_skills.append((skill, relevance_score))

        # Sort and return top results
        valid_skills.sort(key=lambda x: (-x[1], len(x[0]), x[0].lower()))
        return [skill for skill, _ in valid_skills[:limit]]

    except Exception as e:
        print(f"Semantic skills search failed: {e}")
        return []


def get_enhanced_skills(prefix: str, limit: int) -> List[str]:
    """Enhanced skill search across multiple fields with caching and improved scoring"""
    cache_key = get_cache_key(prefix, limit, "skills_enhanced")

    # Check cache first
    if is_cache_valid(cache_key):
        return _autocomplete_cache[cache_key]

    # Optimized search with better indexing strategy
    multi_field_pipeline = [
        {
            "$match": {
                "$or": [
                    {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}},
                    {"technical_skills": {"$regex": f".*{prefix}.*", "$options": "i"}},
                    {"experience.skills": {"$regex": f".*{prefix}.*", "$options": "i"}},
                ]
            }
        },
        {"$limit": limit * 8},  # Optimized limit
        {
            "$project": {
                "skills": 1,
                "technical_skills": 1,
                "experience.skills": 1,
                "_id": 0,  # Exclude _id for better performance
            }
        },
    ]

    try:
        results = list(collection.aggregate(multi_field_pipeline, maxTimeMS=4000))
        skill_frequency = {}  # Track skill frequency for better scoring

        # More efficient skill extraction with frequency tracking
        for doc in results:
            # Process all skill fields uniformly
            for field_name in ["skills", "technical_skills"]:
                field_data = doc.get(field_name)
                if field_data:
                    if isinstance(field_data, list):
                        for item in field_data:
                            if isinstance(item, str) and prefix.lower() in item.lower():
                                extracted_skills = extract_skills(item)
                                for skill in extracted_skills:
                                    if prefix.lower() in skill.lower():
                                        skill_frequency[skill] = (
                                            skill_frequency.get(skill, 0) + 1
                                        )
                    elif (
                        isinstance(field_data, str)
                        and prefix.lower() in field_data.lower()
                    ):
                        extracted_skills = extract_skills(field_data)
                        for skill in extracted_skills:
                            if prefix.lower() in skill.lower():
                                skill_frequency[skill] = (
                                    skill_frequency.get(skill, 0) + 1
                                )

            # Handle experience skills efficiently
            experience_data = doc.get("experience")
            if isinstance(experience_data, list):
                for exp in experience_data[
                    :3
                ]:  # Limit experience entries for performance
                    exp_skills = exp.get("skills") if isinstance(exp, dict) else None
                    if exp_skills:
                        if isinstance(exp_skills, list):
                            for skill in exp_skills:
                                if (
                                    isinstance(skill, str)
                                    and prefix.lower() in skill.lower()
                                ):
                                    extracted_skills = extract_skills(skill)
                                    for extracted_skill in extracted_skills:
                                        if prefix.lower() in extracted_skill.lower():
                                            skill_frequency[extracted_skill] = (
                                                skill_frequency.get(extracted_skill, 0)
                                                + 1
                                            )
                        elif (
                            isinstance(exp_skills, str)
                            and prefix.lower() in exp_skills.lower()
                        ):
                            extracted_skills = extract_skills(exp_skills)
                            for skill in extracted_skills:
                                if prefix.lower() in skill.lower():
                                    skill_frequency[skill] = (
                                        skill_frequency.get(skill, 0) + 1
                                    )

        # Score and sort skills with frequency consideration
        scored_skills = []
        for skill, frequency in skill_frequency.items():
            if len(skill) >= 2 and len(skill) <= 50:
                relevance_score = calculate_relevance_score(skill, prefix)
                frequency_bonus = min(frequency / 20, 0.3)  # Cap frequency bonus
                final_score = relevance_score + frequency_bonus
                if final_score > 0.05:  # Lower threshold for inclusion
                    scored_skills.append((skill, final_score))

        scored_skills.sort(key=lambda x: (-x[1], len(x[0]), x[0].lower()))
        result = [skill for skill, score in scored_skills[:limit]]

        # Cache the result
        set_cache(cache_key, result)
        return result

    except Exception as e:
        print(f"Enhanced skills search failed: {e}")
        # Fallback to basic skill extraction
        basic_skills = []
        try:
            basic_pipeline = [
                {"$unwind": "$skills"},
                {"$match": {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}}},
                {"$group": {"_id": "$skills"}},
                {"$limit": limit},
                {"$project": {"skill": "$_id", "_id": 0}},
            ]
            basic_results = list(collection.aggregate(basic_pipeline))
            basic_skills = [
                result.get("skill", "")
                for result in basic_results
                if result.get("skill")
            ]
        except:
            pass

        return basic_skills[:limit]


@router.get(
    "/job_titles/",
    response_model=List[str],
    summary="Autocomplete Job Titles",
    description="""
    Get autocomplete suggestions for job titles based on input prefix using advanced semantic search.
    Provides context-aware recommendations that understand related roles and technologies.
    **Parameters:**
    - prefix: Text to search for in job titles (e.g., "python developer", "data scientist")
    - limit: Maximum number of suggestions to return
    **Returns:**
    List of matching job titles sorted by semantic relevance and popularity
    **Enhanced Examples:**
    - prefix="python" returns: ["Python Developer", "Senior Python Engineer", "Python Data Scientist", "Machine Learning Engineer", "Backend Python Developer"]
    - prefix="data" returns: ["Data Scientist", "Data Engineer", "Data Analyst", "Senior Data Scientist", "ML Data Engineer"]
    - prefix="frontend" returns: ["Frontend Developer", "React Developer", "Angular Developer", "UI/UX Developer", "Senior Frontend Engineer"]
    """,
    responses={
        200: {
            "description": "Successful job title suggestions with semantic relevance",
            "content": {
                "application/json": {
                    "example": [
                        "Python Developer",
                        "Senior Python Engineer",
                        "Python Data Scientist",
                        "Machine Learning Engineer",
                        "Backend Python Developer",
                    ]
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "Search prefix must be at least 2 characters"}
                }
            },
        },
    },
)
async def autocomplete_titles(
    prefix: str = Query(
        ...,
        description="Job title prefix to search for",
        min_length=2,
        example="python developer",
    ),
    limit: int = Query(
        default=10, description="Maximum number of suggestions", ge=1, le=50, example=5
    ),
):
    try:
        # Input validation and sanitization
        prefix = prefix.strip()
        if not prefix or len(prefix) < 2:
            raise HTTPException(
                status_code=400,
                detail="Search prefix must be at least 2 characters long",
            )

        # Sanitize input to prevent injection
        prefix = re.sub(r"[^\w\s\-\+\#\.]", "", prefix)
        if not prefix:
            raise HTTPException(
                status_code=400, detail="Invalid search prefix after sanitization"
            )

        # Check cache first
        cache_key = get_cache_key(prefix, limit, "titles_semantic")
        if is_cache_valid(cache_key):
            return _autocomplete_cache[cache_key]

        # Primary: Semantic search for context-aware results
        semantic_titles = get_semantic_titles_search(prefix, limit * 2)

        # Secondary: Traditional regex search for exact matches
        traditional_titles = []
        try:
            traditional_pipeline = [
                {"$unwind": "$experience"},
                {
                    "$match": {
                        "$and": [
                            {
                                "experience.title": {
                                    "$exists": True,
                                    "$ne": "",
                                    "$ne": None,
                                }
                            },
                            {
                                "experience.title": {
                                    "$regex": f".*{prefix}.*",
                                    "$options": "i",
                                }
                            },
                        ]
                    }
                },
                {"$group": {"_id": "$experience.title", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": limit},
                {"$project": {"title": "$_id", "frequency": "$count", "_id": 0}},
            ]

            results = list(collection.aggregate(traditional_pipeline, maxTimeMS=3000))
            for result in results:
                title = result.get("title", "").strip()
                if title and len(title) >= 2:
                    frequency_bonus = min(result.get("frequency", 1) / 100, 0.2)
                    score = calculate_relevance_score(title, prefix) + frequency_bonus
                    traditional_titles.append((title, score))

        except Exception as e:
            print(f"Traditional title search failed: {e}")

        # Combine and deduplicate results
        all_titles = []
        seen_titles = set()

        # Add semantic results first (they have priority)
        for title, score in semantic_titles:
            title_lower = title.lower().strip()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                all_titles.append((title, score))

        # Add traditional results that weren't found semantically
        for title, score in traditional_titles:
            title_lower = title.lower().strip()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                all_titles.append((title, score))

        # Sort by relevance score and return top results
        all_titles.sort(key=lambda x: (-x[1], len(x[0]), x[0].lower()))
        final_titles = [title for title, _ in all_titles[:limit]]

        # Cache the result
        set_cache(cache_key, final_titles)

        return final_titles

    except HTTPException:
        raise
    except Exception as e:
        print(f"Title autocomplete error: {str(e)}")
        # Return fallback results instead of complete failure
        try:
            # Simple fallback search
            fallback_pipeline = [
                {"$unwind": "$experience"},
                {
                    "$match": {
                        "experience.title": {"$regex": f"^{prefix}.*", "$options": "i"}
                    }
                },
                {"$group": {"_id": "$experience.title"}},
                {"$limit": limit},
                {"$project": {"title": "$_id", "_id": 0}},
            ]
            fallback_results = list(collection.aggregate(fallback_pipeline))
            return [
                result.get("title", "")
                for result in fallback_results
                if result.get("title")
            ]
        except:
            raise HTTPException(
                status_code=500, detail=f"Title autocomplete failed: {str(e)}"
            )


@router.get(
    "/job_skillsv1/",
    response_model=List[str],
    summary="Autocomplete Technical Skills",
    description="""
    Get autocomplete suggestions for technical skills using advanced semantic search.
    Provides context-aware skill recommendations that understand technology ecosystems and related tools.
    **Parameters:**
    - prefix: Text to search for in skills (e.g., "python", "react", "data")
    - limit: Maximum number of suggestions to return
    **Returns:**
    List of matching skills sorted by semantic relevance and industry popularity
    **Enhanced Examples:**
    - prefix="python" returns: ["Python", "Django", "Flask", "FastAPI", "PyTorch", "Pandas", "NumPy", "Machine Learning"]
    - prefix="react" returns: ["React", "JavaScript", "Redux", "Next.js", "TypeScript", "HTML", "CSS", "Node.js"]
    - prefix="data" returns: ["Data Science", "Python", "SQL", "Machine Learning", "Pandas", "NumPy", "Tableau", "Power BI"]
    - prefix="cloud" returns: ["AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "DevOps", "Terraform"]
    """,
    responses={
        200: {
            "description": "Successful skill suggestions with semantic context",
            "content": {
                "application/json": {
                    "example": [
                        "Python",
                        "Django",
                        "Flask",
                        "FastAPI",
                        "Machine Learning",
                        "PyTorch",
                        "Pandas",
                        "NumPy",
                    ]
                }
            },
        },
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "Search prefix must be at least 2 characters"}
                }
            },
        },
    },
)
async def autocomplete_skills(
    prefix: str = Query(
        ..., description="Skill prefix to search for", min_length=2, example="python"
    ),
    limit: int = Query(
        default=10, description="Maximum number of suggestions", ge=1, le=50, example=8
    ),
):
    try:
        # Input validation and sanitization
        prefix = prefix.strip()
        if not prefix or len(prefix) < 2:
            raise HTTPException(
                status_code=400,
                detail="Search prefix must be at least 2 characters long",
            )

        # Sanitize input to prevent injection
        prefix = re.sub(r"[^\w\s\-\+\#\.]", "", prefix)
        if not prefix:
            raise HTTPException(
                status_code=400, detail="Invalid search prefix after sanitization"
            )

        # Check cache first
        cache_key = get_cache_key(prefix, limit, "skills_semantic")
        if is_cache_valid(cache_key):
            return _autocomplete_cache[cache_key]

        # Primary: Semantic search for context-aware skills
        semantic_skills = get_semantic_skills_search(prefix, limit * 2)

        # Secondary: Enhanced traditional search
        traditional_skills = []
        try:
            # Multi-field skill search with better aggregation
            traditional_pipeline = [
                {
                    "$match": {
                        "$or": [
                            {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}},
                            {
                                "technical_skills": {
                                    "$regex": f".*{prefix}.*",
                                    "$options": "i",
                                }
                            },
                            {
                                "experience.skills": {
                                    "$regex": f".*{prefix}.*",
                                    "$options": "i",
                                }
                            },
                        ]
                    }
                },
                {"$limit": limit * 4},
                {
                    "$project": {
                        "skills": 1,
                        "technical_skills": 1,
                        "experience.skills": 1,
                        "_id": 0,
                    }
                },
            ]

            results = list(collection.aggregate(traditional_pipeline, maxTimeMS=3000))
            skill_candidates = set()

            # Process results from multiple fields
            for doc in results:
                # Process direct skill fields
                for field_name in ["skills", "technical_skills"]:
                    field_data = doc.get(field_name)
                    if field_data:
                        if isinstance(field_data, list):
                            for item in field_data:
                                if (
                                    isinstance(item, str)
                                    and prefix.lower() in item.lower()
                                ):
                                    extracted_skills = extract_skills(item)
                                    skill_candidates.update(extracted_skills)
                        elif (
                            isinstance(field_data, str)
                            and prefix.lower() in field_data.lower()
                        ):
                            extracted_skills = extract_skills(field_data)
                            skill_candidates.update(extracted_skills)

                # Process experience skills
                experience_data = doc.get("experience")
                if isinstance(experience_data, list):
                    for exp in experience_data[:2]:  # Limit for performance
                        if isinstance(exp, dict):
                            exp_skills = exp.get("skills")
                            if exp_skills and prefix.lower() in str(exp_skills).lower():
                                if isinstance(exp_skills, list):
                                    for skill in exp_skills:
                                        if isinstance(skill, str):
                                            extracted_skills = extract_skills(skill)
                                            skill_candidates.update(extracted_skills)
                                elif isinstance(exp_skills, str):
                                    extracted_skills = extract_skills(exp_skills)
                                    skill_candidates.update(extracted_skills)

            # Score and filter traditional skills
            for skill in skill_candidates:
                if (
                    len(skill) >= 2
                    and len(skill) <= 50
                    and prefix.lower() in skill.lower()
                ):
                    score = calculate_relevance_score(skill, prefix)
                    if score > 0.1:
                        traditional_skills.append((skill, score))

        except Exception as e:
            print(f"Traditional skills search failed: {e}")

        # Combine and deduplicate results
        all_skills = []
        seen_skills = set()

        # Add semantic results first (they have priority for context)
        for skill in semantic_skills:
            skill_lower = skill.lower().strip()
            if skill_lower not in seen_skills and len(skill.strip()) >= 2:
                seen_skills.add(skill_lower)
                score = calculate_relevance_score(skill, prefix)
                all_skills.append((skill.strip(), score))

        # Add traditional results that weren't found semantically
        for skill, score in traditional_skills:
            skill_lower = skill.lower().strip()
            if skill_lower not in seen_skills:
                seen_skills.add(skill_lower)
                all_skills.append((skill.strip(), score))

        # Sort by relevance score and return top results
        all_skills.sort(key=lambda x: (-x[1], len(x[0]), x[0].lower()))
        final_skills = [skill for skill, _ in all_skills[:limit]]

        # Cache the result
        set_cache(cache_key, final_skills)

        return final_skills

    except HTTPException:
        raise
    except Exception as e:
        print(f"Skills autocomplete error: {str(e)}")
        # Return fallback results instead of complete failure
        try:
            # Simple fallback search using get_enhanced_skills
            fallback_skills = get_enhanced_skills(prefix, limit)
            return fallback_skills[:limit]
        except:
            raise HTTPException(
                status_code=500, detail=f"Skills autocomplete failed: {str(e)}"
            )


# New combined route
@router.get(
    "/jobs_and_skills/",
    response_model=Dict[str, List[str]],
    summary="Autocomplete Job Titles and Skills with Semantic Context",
    description="""
    Get intelligent autocomplete suggestions for both job titles and technical skills using advanced semantic search.
    Provides context-aware recommendations that understand technology ecosystems and related roles.
    
    **Key Features:**
    - Semantic understanding of technology relationships
    - Context-aware job role and skill matching
    - Industry-standard recommendations
    - Real-time relevance scoring
    
    **Parameters:**
    - prefix: Search term for jobs and skills (e.g., "python developer", "data science", "react")
    - limit: Maximum suggestions per category (1-50)
    
    **Enhanced Examples:**
    
    🐍 **Python Ecosystem:**
    - Input: "python" or "python developer"
    - Titles: ["Python Developer", "Senior Python Engineer", "Machine Learning Engineer", "Data Scientist", "Backend Python Developer"]
    - Skills: ["Python", "Django", "Flask", "FastAPI", "PyTorch", "Pandas", "NumPy", "Machine Learning"]
    
    ⚛️ **Frontend Development:**
    - Input: "react" or "frontend"
    - Titles: ["React Developer", "Frontend Engineer", "JavaScript Developer", "UI Developer", "Senior React Engineer"]
    - Skills: ["React", "JavaScript", "Redux", "Next.js", "TypeScript", "HTML", "CSS", "Node.js"]
    
    📊 **Data Science:**
    - Input: "data" or "data scientist"
    - Titles: ["Data Scientist", "Data Engineer", "ML Engineer", "Data Analyst", "Senior Data Scientist"]
    - Skills: ["Python", "SQL", "Machine Learning", "Pandas", "NumPy", "TensorFlow", "Tableau", "R"]
    
    ☁️ **Cloud & DevOps:**
    - Input: "cloud" or "devops"
    - Titles: ["DevOps Engineer", "Cloud Architect", "Site Reliability Engineer", "Platform Engineer"]
    - Skills: ["AWS", "Docker", "Kubernetes", "Terraform", "CI/CD", "Azure", "Google Cloud", "Jenkins"]
    """,
    responses={
        200: {
            "description": "Intelligent job title and skill suggestions with semantic context",
            "content": {
                "application/json": {
                    "example": {
                        "titles": [
                            "Python Developer",
                            "Senior Python Engineer",
                            "Machine Learning Engineer",
                            "Data Scientist",
                            "Backend Python Developer",
                        ],
                        "skills": [
                            "Python",
                            "Django",
                            "Flask",
                            "FastAPI",
                            "Machine Learning",
                            "PyTorch",
                            "Pandas",
                            "NumPy",
                        ],
                    }
                }
            },
        },
        400: {
            "description": "Bad Request - Invalid input parameters",
            "content": {
                "application/json": {
                    "example": {"detail": "Search prefix must be at least 2 characters"}
                }
            },
        },
        500: {
            "description": "Partial results returned due to search service issues",
            "content": {
                "application/json": {
                    "example": {
                        "titles": [],
                        "skills": [],
                        "error": "Search temporarily unavailable",
                    }
                }
            },
        },
    },
)
async def autocomplete_jobs_and_skills(
    prefix: str = Query(
        ...,
        description="Prefix to search in titles and skills",
        min_length=2,
        max_length=100,
        example="py",
    ),
    limit: int = Query(
        default=5, description="Maximum number of suggestions per category", ge=1, le=50
    ),
):
    try:
        # Input validation and sanitization
        prefix = prefix.strip()
        if not prefix or len(prefix) < 2:
            raise HTTPException(
                status_code=400,
                detail="Search prefix must be at least 2 characters long",
            )

        # Sanitize input to prevent regex injection
        prefix = re.sub(r"[^\w\s\-\+\#\.]", "", prefix)
        if not prefix:
            raise HTTPException(
                status_code=400, detail="Invalid search prefix after sanitization"
            )

        # Check cache for complete response
        cache_key = get_cache_key(prefix, limit, "combined")
        if is_cache_valid(cache_key):
            return _autocomplete_cache[cache_key]

        # --- Enhanced Semantic Search for Both Titles and Skills ---

        # Get semantic titles using the new enhanced function
        semantic_titles = get_semantic_titles_search(prefix, limit)
        print(f"Found {len(semantic_titles)} semantic titles for '{prefix}'")

        # Get semantic skills using the new enhanced function
        semantic_skills = get_semantic_skills_search(prefix, limit)
        print(
            f"Found {len(semantic_skills)} semantic skills for '{prefix}'"
        )  # ENHANCEMENT: Also get skills from resumes that have the matching job titles
        title_based_skills = set()
        if semantic_titles:
            try:
                # Extract just the title names from semantic results
                found_titles = [
                    title for title, score in semantic_titles[:5]
                ]  # Use top 5 titles

                # Find resumes with these job titles and get their skills
                title_skills_pipeline = [
                    {"$match": {"experience.title": {"$in": found_titles}}},
                    {"$limit": 100},  # Limit for performance
                    {
                        "$project": {
                            "skills": 1,
                            "technical_skills": 1,
                            "experience.skills": 1,
                            "_id": 0,
                        }
                    },
                ]

                title_skill_results = list(
                    collection.aggregate(title_skills_pipeline, maxTimeMS=3000)
                )

                # Extract skills from these resumes
                for doc in title_skill_results:
                    # Process direct skill fields
                    for field_name in ["skills", "technical_skills"]:
                        field_data = doc.get(field_name)
                        if field_data:
                            if isinstance(field_data, list):
                                for item in field_data:
                                    if isinstance(item, str):
                                        extracted_skills = extract_skills(item)
                                        title_based_skills.update(extracted_skills)
                            elif isinstance(field_data, str):
                                extracted_skills = extract_skills(field_data)
                                title_based_skills.update(extracted_skills)

                    # Process experience skills
                    experience_data = doc.get("experience")
                    if isinstance(experience_data, list):
                        for exp in experience_data[:3]:
                            if isinstance(exp, dict):
                                exp_skills = exp.get("skills")
                                if exp_skills:
                                    if isinstance(exp_skills, list):
                                        for skill in exp_skills:
                                            if isinstance(skill, str):
                                                extracted_skills = extract_skills(skill)
                                                title_based_skills.update(
                                                    extracted_skills
                                                )
                                    elif isinstance(exp_skills, str):
                                        extracted_skills = extract_skills(exp_skills)
                                        title_based_skills.update(extracted_skills)

                print(
                    f"Found {len(title_based_skills)} skills from resumes with matching titles"
                )

                # ADDITIONAL: Get skills that commonly appear with the primary search term
                if prefix.lower().strip() in [
                    "python",
                    "java",
                    "javascript",
                    "react",
                    "node",
                    "data",
                ]:
                    try:
                        # Find resumes that mention the primary technology in skills
                        tech_skills_pipeline = [
                            {
                                "$match": {
                                    "$or": [
                                        {
                                            "skills": {
                                                "$regex": f".*{prefix.split()[0]}.*",
                                                "$options": "i",
                                            }
                                        },
                                        {
                                            "technical_skills": {
                                                "$regex": f".*{prefix.split()[0]}.*",
                                                "$options": "i",
                                            }
                                        },
                                        {
                                            "experience.skills": {
                                                "$regex": f".*{prefix.split()[0]}.*",
                                                "$options": "i",
                                            }
                                        },
                                    ]
                                }
                            },
                            {"$limit": 50},  # Focused search
                            {
                                "$project": {
                                    "skills": 1,
                                    "technical_skills": 1,
                                    "experience.skills": 1,
                                    "_id": 0,
                                }
                            },
                        ]

                        tech_results = list(
                            collection.aggregate(tech_skills_pipeline, maxTimeMS=2000)
                        )

                        for doc in tech_results:
                            # Process all skill fields
                            for field_name in ["skills", "technical_skills"]:
                                field_data = doc.get(field_name)
                                if field_data:
                                    if isinstance(field_data, list):
                                        for item in field_data:
                                            if isinstance(item, str):
                                                extracted_skills = extract_skills(item)
                                                title_based_skills.update(
                                                    extracted_skills
                                                )
                                    elif isinstance(field_data, str):
                                        extracted_skills = extract_skills(field_data)
                                        title_based_skills.update(extracted_skills)

                            # Process experience skills
                            experience_data = doc.get("experience")
                            if isinstance(experience_data, list):
                                for exp in experience_data[:2]:
                                    if isinstance(exp, dict):
                                        exp_skills = exp.get("skills")
                                        if exp_skills:
                                            if isinstance(exp_skills, list):
                                                for skill in exp_skills:
                                                    if isinstance(skill, str):
                                                        extracted_skills = (
                                                            extract_skills(skill)
                                                        )
                                                        title_based_skills.update(
                                                            extracted_skills
                                                        )
                                            elif isinstance(exp_skills, str):
                                                extracted_skills = extract_skills(
                                                    exp_skills
                                                )
                                                title_based_skills.update(
                                                    extracted_skills
                                                )

                        print(
                            f"Enhanced with technology-specific skills, total: {len(title_based_skills)}"
                        )

                    except Exception as e:
                        print(f"Technology-specific skills search failed: {e}")

            except Exception as e:
                print(
                    f"Title-based skills search failed: {e}"
                )  # Secondary: Traditional search as fallback/supplement
        traditional_titles = []
        traditional_skills_set = set()

        try:
            # Combined pipeline for both titles and skills
            combined_pipeline = [
                {
                    "$match": {
                        "$or": [
                            {
                                "experience.title": {
                                    "$regex": f".*{prefix}.*",
                                    "$options": "i",
                                }
                            },
                            {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}},
                            {
                                "technical_skills": {
                                    "$regex": f".*{prefix}.*",
                                    "$options": "i",
                                }
                            },
                            {
                                "experience.skills": {
                                    "$regex": f".*{prefix}.*",
                                    "$options": "i",
                                }
                            },
                        ]
                    }
                },
                {"$limit": limit * 3},
                {
                    "$project": {
                        "experience.title": 1,
                        "skills": 1,
                        "technical_skills": 1,
                        "experience.skills": 1,
                        "_id": 0,
                    }
                },
            ]

            combined_results = list(
                collection.aggregate(combined_pipeline, maxTimeMS=4000)
            )

            # Process titles from traditional search
            title_freq = {}
            for doc in combined_results:
                experience_data = doc.get("experience")
                if isinstance(experience_data, list):
                    for exp in experience_data:
                        if isinstance(exp, dict):
                            title = exp.get("title", "").strip()
                            if title and prefix.lower() in title.lower():
                                title_freq[title] = title_freq.get(title, 0) + 1

            # Score traditional titles
            for title, freq in title_freq.items():
                relevance_score = calculate_relevance_score(title, prefix)
                frequency_bonus = min(freq / 50, 0.2)
                final_score = relevance_score + frequency_bonus
                if final_score > 0.1:
                    traditional_titles.append((title, final_score))

            # Process skills from traditional search
            for doc in combined_results:
                # Process skill fields
                for field_name in ["skills", "technical_skills"]:
                    field_data = doc.get(field_name)
                    if field_data and prefix.lower() in str(field_data).lower():
                        if isinstance(field_data, list):
                            for item in field_data:
                                if isinstance(item, str):
                                    extracted_skills = extract_skills(item)
                                    traditional_skills_set.update(extracted_skills)
                        elif isinstance(field_data, str):
                            extracted_skills = extract_skills(field_data)
                            traditional_skills_set.update(extracted_skills)

                # Process experience skills
                experience_data = doc.get("experience")
                if isinstance(experience_data, list):
                    for exp in experience_data[:2]:
                        if isinstance(exp, dict):
                            exp_skills = exp.get("skills")
                            if exp_skills and prefix.lower() in str(exp_skills).lower():
                                if isinstance(exp_skills, list):
                                    for skill in exp_skills:
                                        if isinstance(skill, str):
                                            extracted_skills = extract_skills(skill)
                                            traditional_skills_set.update(
                                                extracted_skills
                                            )
                                elif isinstance(exp_skills, str):
                                    extracted_skills = extract_skills(exp_skills)
                                    traditional_skills_set.update(extracted_skills)

        except Exception as e:
            print(f"Traditional combined search failed: {e}")

        # Combine and deduplicate titles
        all_titles = []
        seen_titles = set()

        # Add semantic titles first (priority for context awareness)
        for title, score in semantic_titles:
            title_lower = title.lower().strip()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                all_titles.append((title, score))

        # Add traditional titles that weren't found semantically
        for title, score in traditional_titles:
            title_lower = title.lower().strip()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                all_titles.append((title, score))

        # Sort titles by relevance
        all_titles.sort(key=lambda x: (-x[1], len(x[0]), x[0].lower()))
        final_titles = [title for title, _ in all_titles[:limit]]

        # Combine and deduplicate skills
        all_skills = []
        seen_skills = set()

        # Add semantic skills first (priority for context awareness)
        for skill in semantic_skills:
            skill_lower = skill.lower().strip()
            if skill_lower not in seen_skills and len(skill.strip()) >= 2:
                seen_skills.add(skill_lower)
                score = calculate_relevance_score(skill, prefix)
                all_skills.append((skill.strip(), score))

        # Add title-based skills (high priority since they come from matching job titles)
        if "title_based_skills" in locals():
            for skill in title_based_skills:
                skill_lower = skill.lower().strip()
                if skill_lower not in seen_skills and len(skill.strip()) >= 2:
                    seen_skills.add(skill_lower)
                    # Give higher score to skills from matching job titles
                    base_score = calculate_relevance_score(skill, prefix)
                    title_bonus = 0.3  # Bonus for coming from matching job titles
                    score = base_score + title_bonus
                    all_skills.append((skill.strip(), score))

        # Add traditional skills that weren't found semantically
        for skill in traditional_skills_set:
            skill_lower = skill.lower().strip()
            if (
                skill_lower not in seen_skills
                and len(skill.strip()) >= 2
                and prefix.lower() in skill_lower
            ):
                seen_skills.add(skill_lower)
                score = calculate_relevance_score(skill, prefix)
                all_skills.append((skill.strip(), score))

        # Sort skills by relevance
        all_skills.sort(key=lambda x: (-x[1], len(x[0]), x[0].lower()))
        final_skills = [skill for skill, _ in all_skills[:limit]]

        # Prepare final response
        response = {
            "titles": final_titles,
            "skills": final_skills,
        }

        # Cache the response
        set_cache(cache_key, response)

        return response

    except Exception as e:
        print(f"Autocomplete error: {str(e)}")
        # Return partial results with graceful degradation
        try:
            # Attempt simple fallback search
            fallback_titles = []
            fallback_skills = []

            # Simple title fallback
            try:
                simple_title_pipeline = [
                    {"$unwind": "$experience"},
                    {
                        "$match": {
                            "experience.title": {
                                "$regex": f"^{prefix}.*",
                                "$options": "i",
                            }
                        }
                    },
                    {"$group": {"_id": "$experience.title"}},
                    {"$limit": limit},
                    {"$project": {"title": "$_id", "_id": 0}},
                ]
                title_results = list(collection.aggregate(simple_title_pipeline))
                fallback_titles = [
                    result.get("title", "")
                    for result in title_results
                    if result.get("title")
                ]
            except:
                fallback_titles = []

            # Simple skills fallback
            try:
                fallback_skills = get_enhanced_skills(prefix, limit)
            except:
                fallback_skills = []

            return {
                "titles": fallback_titles[:limit],
                "skills": fallback_skills[:limit],
                "status": "partial_results",
                "message": "Using fallback search due to service limitations",
            }

        except:
            return {
                "titles": [],
                "skills": [],
                "error": f"Search temporarily unavailable: {str(e)[:100]}",
                "status": "error",
            }


@router.get(
    "/clear_cache/",
    summary="Clear Autocomplete Cache",
    description="Clear the internal cache for autocomplete results. Use this if you need fresh results.",
)
async def clear_autocomplete_cache():
    """Clear the autocomplete cache"""
    try:
        global _autocomplete_cache, _cache_expiry
        cleared_entries = len(_autocomplete_cache)
        _autocomplete_cache.clear()
        _cache_expiry.clear()
        return {
            "message": f"Cache cleared successfully. Removed {cleared_entries} entries.",
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


@router.get(
    "/cache_stats/",
    summary="Cache Statistics",
    description="Get statistics about the autocomplete cache usage.",
)
async def get_cache_stats():
    """Get cache statistics"""
    try:
        current_time = time.time()
        valid_entries = sum(
            1 for expiry_time in _cache_expiry.values() if current_time < expiry_time
        )
        expired_entries = len(_cache_expiry) - valid_entries

        return {
            "total_entries": len(_autocomplete_cache),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "cache_duration_seconds": CACHE_DURATION,
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache stats failed: {str(e)}")


@router.post(
    "/skills_by_titles/",
    summary="Get Skills by Job Titles",
    description="""
    Get all skills related to a list of job titles. This endpoint analyzes resumes 
    with the specified job titles and returns all skills associated with those roles.
    
    **Features:**
    - Extracts skills from resumes with matching job titles
    - Semantic analysis for skill relevance
    - Deduplication and ranking of skills
    - Contextual skill relationships
    
    **Use Cases:**
    - Get comprehensive skill requirements for multiple job roles
    - Analyze skill overlap between different positions
    - Build skill profiles for career planning
    """,
)
async def get_skills_by_job_titles(
    request: dict = Body(
        ...,
        description="Request body containing list of job titles",
        example={
            "titles": [
                "python developer",
                "data scientist",
                "full stack developer",
                "software engineer",
            ],
            "include_related": True,
        },
    )
):
    """
    Get all skills related to the provided job titles

    Args:
        request: Dictionary containing:
            - titles: List of job titles to analyze
            - include_related: Whether to include semantically related skills (default: True)

    Note: No limit parameter - uses dynamic result sizing based on skill quality and relevance

    Returns:
        Dictionary with skills grouped by categories and relevance scores
    """
    try:
        # Validate request
        if not isinstance(request, dict) or "titles" not in request:
            raise HTTPException(
                status_code=400,
                detail="Request must contain 'titles' field with list of job titles",
            )

        titles = request.get("titles", [])
        include_related = request.get("include_related", True)

        # No limit parameter - using fully dynamic result sizing

        if not isinstance(titles, list) or not titles:
            raise HTTPException(
                status_code=400,
                detail="'titles' must be a non-empty list of job titles",
            )

        if len(titles) > 20:
            raise HTTPException(
                status_code=400, detail="Maximum 20 job titles allowed per request"
            )

        # Sanitize titles
        clean_titles = []
        for title in titles:
            if isinstance(title, str) and title.strip():
                # Clean and normalize title
                clean_title = re.sub(r"[^\w\s\-\+\#\.]", "", title.strip())
                if clean_title:
                    clean_titles.append(clean_title)

        if not clean_titles:
            raise HTTPException(
                status_code=400, detail="No valid job titles found after sanitization"
            )

        print(f"Processing skills for {len(clean_titles)} job titles: {clean_titles}")

        # Check cache (dynamic sizing - no limit in cache key)
        cache_key = f"skills_by_titles_{hash(str(sorted(clean_titles)))}_dynamic_related_{include_related}"
        if is_cache_valid(cache_key):
            print("Returning cached results for skills by titles")
            return _autocomplete_cache[cache_key]

        collection = get_collection()
        all_skills = set()
        skill_scores = {}
        title_matches = {}

        # Find resumes with matching job titles
        for title in clean_titles:
            title_skills = set()
            try:
                # Create regex pattern for flexible matching
                title_pattern = title.replace(" ", ".*")

                # Pipeline to find resumes with matching job titles
                title_pipeline = [
                    {
                        "$match": {
                            "experience.title": {
                                "$regex": f".*{title_pattern}.*",
                                "$options": "i",
                            }
                        }
                    },
                    {"$limit": 200},  # Limit per title for performance
                    {
                        "$project": {
                            "skills": 1,
                            "technical_skills": 1,
                            "experience.skills": 1,
                            "experience.title": 1,
                            "experience.description": 1,
                            "experience.responsibilities": 1,
                            "_id": 0,
                        }
                    },
                ]

                title_results = list(
                    collection.aggregate(title_pipeline, maxTimeMS=5000)
                )
                print(f"Found {len(title_results)} resumes for title: {title}")

                for doc in title_results:
                    # Extract skills from direct skill fields
                    for field_name in ["skills", "technical_skills"]:
                        field_data = doc.get(field_name)
                        if field_data:
                            if isinstance(field_data, list):
                                for item in field_data:
                                    if isinstance(item, str):
                                        extracted_skills = extract_skills(item)
                                        title_skills.update(extracted_skills)
                            elif isinstance(field_data, str):
                                extracted_skills = extract_skills(field_data)
                                title_skills.update(extracted_skills)

                    # Extract skills from experience sections
                    experience_data = doc.get("experience")
                    if isinstance(experience_data, list):
                        for exp in experience_data[:3]:  # Limit to top 3 experiences
                            if isinstance(exp, dict):
                                # Check if this experience matches our target title
                                exp_title = exp.get("title", "")
                                if title.lower() in exp_title.lower():
                                    # Extract from experience skills
                                    exp_skills = exp.get("skills")
                                    if exp_skills:
                                        if isinstance(exp_skills, list):
                                            for skill in exp_skills:
                                                if isinstance(skill, str):
                                                    extracted_skills = extract_skills(
                                                        skill
                                                    )
                                                    title_skills.update(
                                                        extracted_skills
                                                    )
                                        elif isinstance(exp_skills, str):
                                            extracted_skills = extract_skills(
                                                exp_skills
                                            )
                                            title_skills.update(extracted_skills)

                                    # Extract from description and responsibilities
                                    if include_related:
                                        for text_field in [
                                            "description",
                                            "responsibilities",
                                        ]:
                                            text_content = exp.get(text_field, "")
                                            if text_content and isinstance(
                                                text_content, str
                                            ):
                                                # Extract technical terms from text
                                                tech_terms = re.findall(
                                                    r"\b(?:python|java|javascript|react|angular|vue|node|express|django|flask|spring|html|css|sql|mongodb|mysql|postgresql|redis|aws|azure|gcp|docker|kubernetes|git|api|rest|graphql|tensorflow|pytorch|pandas|numpy|typescript|scss|sass|jquery|bootstrap|tailwind|linux|ubuntu|windows|mac|ios|android|swift|kotlin|php|laravel|symfony|ruby|rails|go|rust|c\+\+|c#|\.net|scala|r|matlab|tableau|powerbi|excel|jira|confluence|slack|trello|figma|sketch|photoshop|illustrator|agile|scrum|kanban|devops|cicd|jenkins|travis|circleci|github|gitlab|bitbucket|npm|yarn|pip|composer|maven|gradle|webpack|babel|eslint|prettier|jest|mocha|chai|junit|selenium|cypress|postman|swagger|oauth|jwt|ssl|https|json|xml|yaml|csv|etl|ml|ai|nlp|opencv|scikit|keras|spark|hadoop|kafka|elasticsearch|kibana|grafana|prometheus|nagios|ansible|terraform|vagrant|virtualbox|vmware)\b",
                                                    text_content.lower(),
                                                )
                                                title_skills.update(tech_terms)

                # Store skills for this title
                title_matches[title] = len(title_skills)
                all_skills.update(title_skills)

                # Score skills based on how many titles they appear in
                for skill in title_skills:
                    if skill in skill_scores:
                        skill_scores[skill] += 1
                    else:
                        skill_scores[skill] = 1

            except Exception as e:
                print(f"Error processing title '{title}': {e}")
                continue

        print(f"Total unique skills found: {len(all_skills)}")

        # Enhanced skill filtering and scoring
        filtered_skills = {}

        # Dynamic categorization - no hardcoded categories needed

        for skill in all_skills:
            if len(skill.strip()) < 2:
                continue

            skill_clean = skill.strip().lower()

            # Calculate comprehensive score
            base_score = skill_scores.get(skill, 1)

            # Bonus for appearing in multiple titles
            title_coverage_bonus = (base_score / len(clean_titles)) * 2

            # Dynamic relevance bonus based on skill frequency and relationships
            category_bonus = 0
            skill_frequencies = get_skill_frequencies()
            if skill_frequencies and skill_clean in skill_frequencies:
                # Higher frequency skills get bonus (but not generic ones)
                total_skills = len(skill_frequencies)
                freq_ratio = skill_frequencies[skill_clean] / max(
                    skill_frequencies.values()
                )
                if (
                    freq_ratio > 0.01
                    and skill_frequencies[skill_clean] < total_skills * 0.1
                ):  # Not too generic
                    category_bonus = min(freq_ratio * 1.5, 1.5)

            # Length and quality checks
            length_bonus = 0.5 if 3 <= len(skill_clean) <= 15 else 0

            # Penalty for generic terms
            generic_penalty = 0
            generic_terms = [
                "experience",
                "knowledge",
                "understanding",
                "ability",
                "skill",
                "working",
                "good",
                "excellent",
                "strong",
                "basic",
                "advanced",
                "years",
                "work",
                "team",
                "project",
                "development",
                "application",
                "system",
                "business",
                "management",
                "communication",
                "problem",
                "solving",
                "analytical",
                "technical",
            ]
            if skill_clean in generic_terms:
                generic_penalty = -2

            final_score = (
                base_score
                + title_coverage_bonus
                + category_bonus
                + length_bonus
                + generic_penalty
            )

            if final_score > 0.5:  # Filter out low-scoring skills
                filtered_skills[skill.strip()] = final_score

        # Sort skills by score and apply DYNAMIC sizing (no hardcoded limits)
        sorted_skills = sorted(
            filtered_skills.items(), key=lambda x: x[1], reverse=True
        )

        # DYNAMIC RESULT SIZING - Quality-based adaptive results
        high_quality_skills = [skill for skill, score in sorted_skills if score >= 3.0]
        medium_quality_skills = [
            skill for skill, score in sorted_skills if 1.5 <= score < 3.0
        ]
        low_quality_skills = [
            skill for skill, score in sorted_skills if 0.5 <= score < 1.5
        ]

        # FULLY DYNAMIC result sizing - no hardcoded limits
        if len(high_quality_skills) >= 30:
            # Lots of high-quality results - return up to 80 of the best
            top_skills = high_quality_skills[:80]
        elif len(high_quality_skills) >= 15:
            # Good amount of high-quality - include medium quality skills
            top_skills = high_quality_skills + medium_quality_skills[:50]
        elif len(high_quality_skills) >= 5:
            # Some high-quality - include all medium and some low quality
            top_skills = (
                high_quality_skills + medium_quality_skills + low_quality_skills[:30]
            )
        else:
            # Limited high-quality results - return ALL qualifying skills
            top_skills = [skill for skill, score in sorted_skills if score > 0.5]

        # Dynamic categorization based on patterns and relationships
        def categorize_skill_dynamically(skill_name):
            skill_lower = skill_name.lower()

            # Pattern-based categorization
            if any(
                pattern in skill_lower
                for pattern in [
                    "python",
                    "java",
                    "javascript",
                    "js",
                    "typescript",
                    "php",
                    "c#",
                    "c++",
                    "go",
                    "rust",
                    "ruby",
                ]
            ):
                return "programming_languages"
            elif any(
                pattern in skill_lower
                for pattern in [
                    "html",
                    "css",
                    "react",
                    "angular",
                    "vue",
                    "jquery",
                    "bootstrap",
                ]
            ):
                return "web_technologies"
            elif any(
                pattern in skill_lower
                for pattern in [
                    "django",
                    "flask",
                    "spring",
                    "express",
                    "laravel",
                    "rails",
                ]
            ):
                return "backend_frameworks"
            elif any(
                pattern in skill_lower
                for pattern in [
                    "sql",
                    "mysql",
                    "postgresql",
                    "mongodb",
                    "redis",
                    "elasticsearch",
                    "oracle",
                ]
            ):
                return "databases"
            elif any(
                pattern in skill_lower
                for pattern in [
                    "aws",
                    "azure",
                    "gcp",
                    "docker",
                    "kubernetes",
                    "jenkins",
                    "git",
                    "cicd",
                ]
            ):
                return "cloud_devops"
            elif any(
                pattern in skill_lower
                for pattern in [
                    "pandas",
                    "numpy",
                    "tensorflow",
                    "pytorch",
                    "sklearn",
                    "tableau",
                    "powerbi",
                ]
            ):
                return "data_science"
            elif any(
                pattern in skill_lower
                for pattern in ["android", "ios", "swift", "kotlin", "flutter"]
            ):
                return "mobile_development"
            elif any(
                pattern in skill_lower
                for pattern in [
                    "jira",
                    "confluence",
                    "slack",
                    "figma",
                    "photoshop",
                    "excel",
                    "linux",
                ]
            ):
                return "tools_software"
            else:
                return "other"

        categorized_skills = {}
        for skill in top_skills:
            category = categorize_skill_dynamically(skill)
            if category not in categorized_skills:
                categorized_skills[category] = []
            categorized_skills[category].append(skill)

        result = {
            "request_summary": {
                "titles_processed": clean_titles,
                "titles_count": len(clean_titles),
                "total_skills_found": len(all_skills),
                "filtered_skills_returned": len(top_skills),
            },
            "title_matches": title_matches,
            "skills": top_skills,
            "skills_by_category": categorized_skills,
            "skill_scores": {
                skill: round(score, 2)
                for skill, score in sorted_skills[: len(top_skills)]
            },
            "status": "success",
        }

        # Cache the result for 30 minutes
        _autocomplete_cache[cache_key] = result
        _cache_expiry[cache_key] = time.time() + 1800  # 30 minutes cache

        print(f"Returning {len(top_skills)} skills for {len(clean_titles)} job titles")
        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"Skills by titles error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get skills by titles: {str(e)[:200]}"
        )


@router.get(
    "/performance_stats/",
    summary="Performance Statistics",
    description="Get performance metrics for the autocomplete system including response times and search patterns.",
)
async def get_performance_stats():
    """Get performance statistics for monitoring and optimization"""
    try:
        cache_hit_rate = (
            _search_stats["cache_hits"] / max(_search_stats["total_requests"], 1)
        ) * 100

        return {
            "total_requests": _search_stats["total_requests"],
            "cache_hits": _search_stats["cache_hits"],
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "semantic_searches": _search_stats["semantic_searches"],
            "traditional_searches": _search_stats["traditional_searches"],
            "average_response_time_ms": round(
                _search_stats["average_response_time"] * 1000, 2
            ),
            "current_cache_size": len(_autocomplete_cache),
            "status": "success",
            "dynamic_relationships_built": "skill_relationships_global"
            in _autocomplete_cache,
            "recommendations": {
                "cache_performance": (
                    "excellent"
                    if cache_hit_rate > 70
                    else "good" if cache_hit_rate > 50 else "needs_optimization"
                ),
                "semantic_usage": (
                    "high"
                    if _search_stats["semantic_searches"]
                    > _search_stats["traditional_searches"]
                    else "moderate"
                ),
                "dynamic_context": (
                    "active"
                    if "skill_relationships_global" in _autocomplete_cache
                    else "building_on_demand"
                ),
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Performance stats failed: {str(e)}"
        )


@router.post(
    "/rebuild_relationships/",
    summary="Rebuild Dynamic Skill Relationships",
    description="Manually rebuild the dynamic skill relationship mapping from current database. Use this after significant data updates.",
)
async def rebuild_dynamic_relationships():
    """Manually rebuild dynamic skill relationships"""
    try:
        # Clear existing relationships cache
        relationship_cache_key = "skill_relationships_global"
        if relationship_cache_key in _autocomplete_cache:
            del _autocomplete_cache[relationship_cache_key]
        if relationship_cache_key in _cache_expiry:
            del _cache_expiry[relationship_cache_key]

        # Rebuild relationships
        start_time = time.time()
        relationships = build_dynamic_skill_relationships()
        build_time = time.time() - start_time

        return {
            "status": "success",
            "message": "Dynamic skill relationships rebuilt successfully",
            "skills_analyzed": len(relationships),
            "build_time_seconds": round(build_time, 2),
            "cache_expires_at": time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(_cache_expiry.get(relationship_cache_key, 0)),
            ),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Relationship rebuild failed: {str(e)}"
        )


@router.get(
    "/dynamic_context_preview/",
    summary="Preview Dynamic Context Terms",
    description="Preview what context terms would be generated dynamically for a given search term.",
)
async def preview_dynamic_context(
    prefix: str = Query(
        ...,
        description="Search term to analyze for context",
        min_length=2,
        example="python",
    ),
    method: str = Query(
        default="enhanced",
        description="Context generation method",
        regex="^(basic|enhanced)$",
    ),
):
    """Preview dynamic context generation for debugging and optimization"""
    try:
        start_time = time.time()

        if method == "enhanced":
            context_terms = get_enhanced_dynamic_context_terms(prefix, limit=15)
        else:
            context_terms = get_dynamic_context_terms(prefix, limit=15)

        generation_time = time.time() - start_time

        return {
            "search_term": prefix,
            "method_used": method,
            "context_terms": context_terms,
            "terms_count": len(context_terms),
            "generation_time_ms": round(generation_time * 1000, 2),
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context preview failed: {str(e)}")


# @router.get(
#     "/debug/sample_data",
#     summary="Debug: Sample Data Structure",
#     description="Returns a sample document to understand the data structure",
# )
# async def get_sample_data():
#     try:
#         # Get a sample document to understand the structure
#         sample = collection.find_one({}, {"_id": 0})
#         if sample:
#             # Show keys and sample values
#             structure = {}
#             for key, value in sample.items():
#                 if isinstance(value, list) and value:
#                     structure[key] = (
#                         f"Array with {len(value)} items, sample: {value[0] if value else 'empty'}"
#                     )
#                 elif isinstance(value, dict):
#                     structure[key] = f"Object with keys: {list(value.keys())}"
#                 else:
#                     structure[key] = (
#                         f"Type: {type(value).__name__}, value: {str(value)[:100]}"
#                     )
#             return {"structure": structure, "sample_keys": list(sample.keys())}
#         else:
#             return {"message": "No documents found in collection"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")


# @router.get(
#     "/debug/skills_fields",
#     summary="Debug: Available Skills Fields",
#     description="Check what skill-related fields exist in the database",
# )
# async def check_skills_fields():
#     try:
#         # Check for various skill-related fields
#         pipeline = [
#             {"$limit": 10},
#             {
#                 "$project": {
#                     "has_skills": {"$ifNull": ["$skills", "NOT_FOUND"]},
#                     "has_technical_skills": {
#                         "$ifNull": ["$technical_skills", "NOT_FOUND"]
#                     },
#                     "has_experience_skills": {
#                         "$ifNull": ["$experience.skills", "NOT_FOUND"]
#                     },
#                     "skills_type": {"$type": "$skills"},
#                     "skills_sample": {"$slice": ["$skills", 3]},
#                 }
#             },
#         ]
#         results = list(collection.aggregate(pipeline))
#         return {"sample_documents": results, "total_checked": len(results)}
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Skills fields check failed: {str(e)}"
#         )
