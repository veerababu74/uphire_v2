#!/usr/bin/env python3
"""
Test the specific API endpoint to check if _id fields are empty
"""

import asyncio
import json
from apis.rag_search import llm_context_search
from apis.rag_search import LLMContextSearchRequest
from core.custom_logger import CustomLogger

logger = CustomLogger().get_logger("test_api_endpoint")


async def test_api_endpoint():
    """Test the API endpoint directly"""
    try:
        # Create a test request - use a user_id that exists in the users collection
        # "veera123" exists in users collection so it can search all documents
        request = LLMContextSearchRequest(
            user_id="veera123",  # This should allow searching all documents
            query="Find me a Python developer with 3 years experience",
            context_size=3,
            relevant_score=0.0,  # Get all results regardless of score
            use_enhanced_search=True,
        )

        logger.info(f"Testing API endpoint with request: {request}")

        # Call the API endpoint function directly
        result = await llm_context_search(request)

        logger.info(f"API result type: {type(result)}")
        logger.info(
            f"API result keys: {result.keys() if hasattr(result, 'keys') else 'No keys method'}"
        )

        if hasattr(result, "results"):
            logger.info(f"Number of results: {len(result.results)}")

            for i, candidate in enumerate(result.results[:2]):  # Check first 2 results
                logger.info(f"Result {i+1}:")
                logger.info(
                    f"  _id: '{getattr(candidate, '_id', 'MISSING')}' (type: {type(getattr(candidate, '_id', None))})"
                )
                logger.info(f"  user_id: '{getattr(candidate, 'user_id', 'MISSING')}'")
                logger.info(
                    f"  relevance_score: {getattr(candidate, 'relevance_score', 'MISSING')}"
                )

                # Check if _id is empty string specifically
                _id_value = getattr(candidate, "_id", None)
                if _id_value == "":
                    logger.error(f"Found empty string _id in result {i+1}!")
                elif _id_value is None:
                    logger.error(f"Found None _id in result {i+1}!")
                elif not _id_value:
                    logger.error(
                        f"Found falsy _id value: {repr(_id_value)} in result {i+1}!"
                    )
        elif isinstance(result, dict) and "results" in result:
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
            logger.error("No 'results' attribute or key in response")
            logger.info(f"Result content: {result}")

        return result

    except Exception as e:
        logger.error(f"Error during API endpoint test: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


if __name__ == "__main__":
    result = asyncio.run(test_api_endpoint())
    if result:
        print(f"\nAPI test completed. Check logs for details.")
        # Try to access results in different ways
        if hasattr(result, "results"):
            print(f"Found {len(result.results)} results via attribute")
        elif isinstance(result, dict) and "results" in result:
            print(f"Found {len(result['results'])} results via dict key")
        else:
            print(f"Result type: {type(result)}")
            print(f"Result: {result}")
