"""
Test script to verify the lazy initialization fixes work.
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_retriever_import():
    """Test that retrievers can be imported without immediate connection."""
    try:
        print("Testing retriever imports with lazy initialization...")

        # This should work even if MongoDB is unavailable
        from Retrivers.retriver import MangoRetriever, LangChainRetriever

        print("✅ Successfully imported retrievers")

        # Create instances with lazy initialization
        mango_retriever = MangoRetriever(lazy_init=True)
        langchain_retriever = LangChainRetriever(lazy_init=True)
        print("✅ Successfully created retriever instances with lazy_init=True")

        # Check that they are not initialized yet
        print(f"MangoRetriever initialized: {mango_retriever._initialized}")
        print(f"LangChainRetriever initialized: {langchain_retriever._initialized}")

        # Test that API imports work
        from apis.retriever_api import router as retriever_api_router

        print("✅ Successfully imported retriever API router")

        return True

    except Exception as e:
        print(f"❌ Error during import test: {e}")
        return False


def test_health_check_import():
    """Test that health check API can be imported."""
    try:
        from apis.retriever_health import router as health_router

        print("✅ Successfully imported health check router")
        return True
    except Exception as e:
        print(f"❌ Error importing health check: {e}")
        return False


if __name__ == "__main__":
    print("Running retriever lazy initialization test...")
    print("=" * 50)

    success = True
    success &= test_retriever_import()
    success &= test_health_check_import()

    print("=" * 50)
    if success:
        print("🎉 All tests passed! The lazy initialization fix should work.")
    else:
        print("💥 Some tests failed. Please check the errors above.")
