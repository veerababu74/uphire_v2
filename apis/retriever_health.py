"""
Health check endpoint for retriever services.
"""

from fastapi import APIRouter
from core.custom_logger import CustomLogger

# Initialize logger
logger_instance = CustomLogger()
logger = logger_instance.get_logger("retriever_health")

router = APIRouter(
    prefix="/retriever",
    tags=["retriever-health"],
    responses={404: {"description": "Not found"}},
)


@router.get("/health")
async def health_check():
    """Check the health of retriever services."""
    try:
        # Import here to avoid circular imports and startup blocking
        from Retrivers.retriver import MangoRetriever, LangChainRetriever

        # Try to initialize and check both retrievers
        mango_health = False
        langchain_health = False

        try:
            mango_retriever = MangoRetriever(lazy_init=True)
            mango_retriever._ensure_initialized()
            mango_health = True
        except Exception as e:
            logger.error(f"MangoRetriever health check failed: {e}")

        try:
            langchain_retriever = LangChainRetriever(lazy_init=True)
            langchain_retriever._ensure_initialized()
            langchain_health = True
        except Exception as e:
            logger.error(f"LangChainRetriever health check failed: {e}")

        overall_health = mango_health or langchain_health

        return {
            "status": "healthy" if overall_health else "unhealthy",
            "services": {
                "mango_retriever": "healthy" if mango_health else "unhealthy",
                "langchain_retriever": "healthy" if langchain_health else "unhealthy",
            },
        }
    except Exception as e:
        logger.error(f"Health check endpoint error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "services": {
                "mango_retriever": "unknown",
                "langchain_retriever": "unknown",
            },
        }
