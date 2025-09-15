#!/usr/bin/env python3
"""
Debug script to test LLM search and check _id field issue
"""

import os
import sys

sys.path.append(".")

from Rag.runner import initialize_rag_app
from core.custom_logger import CustomLogger

logger = CustomLogger().get_logger("debug_llm_search")


def test_llm_search():
    """Test LLM search functionality and debug _id issue"""
    try:
        logger.info("Initializing RAG application...")
        rag_app = initialize_rag_app()

        # Test query
        test_query = "Find me a Python developer with 3 years experience"

        logger.info(f"Testing LLM search with query: {test_query}")

        # Perform LLM search
        result = rag_app.llm_context_search(
            query=test_query,
            context_size=3,
            user_id=None,  # Search all users
            use_enhanced=True,
        )

        logger.info(f"LLM search result type: {type(result)}")
        logger.info(
            f"LLM search result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}"
        )

        if "results" in result:
            logger.info(f"Number of results: {len(result['results'])}")

            for i, candidate in enumerate(
                result["results"][:2]
            ):  # Check first 2 results
                logger.info(f"Result {i+1}:")
                logger.info(
                    f"  _id: '{candidate.get('_id', 'MISSING')}' (type: {type(candidate.get('_id'))})"
                )
                logger.info(f"  user_id: '{candidate.get('user_id', 'MISSING')}'")
                logger.info(
                    f"  relevance_score: {candidate.get('relevance_score', 'MISSING')}"
                )
                logger.info(f"  All keys: {list(candidate.keys())}")

                # Check if _id is empty string specifically
                _id_value = candidate.get("_id")
                if _id_value == "":
                    logger.error(f"Found empty string _id in result {i+1}!")
                elif _id_value is None:
                    logger.error(f"Found None _id in result {i+1}!")
                elif not _id_value:
                    logger.error(
                        f"Found falsy _id value: {repr(_id_value)} in result {i+1}!"
                    )
        else:
            logger.error("No 'results' key in response")

        return result

    except Exception as e:
        logger.error(f"Error during LLM search test: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    result = test_llm_search()
    if result:
        print(f"\nTest completed. Check logs for details.")
        if "results" in result:
            print(f"Found {len(result['results'])} results")
            for i, res in enumerate(result["results"][:3]):
                print(
                    f"Result {i+1}: _id='{res.get('_id')}', score={res.get('relevance_score')}"
                )
