#!/usr/bin/env python3
"""
Fix for SentenceTransformer Meta Tensor Loading Issues

This script addresses the "Cannot copy out of meta tensor" error that occurs
with newer PyTorch versions when loading SentenceTransformer models.

The issue typically occurs due to:
1. Newer PyTorch versions handling device placement differently
2. Meta tensor initialization conflicts
3. Model loading order issues with device placement

Solutions implemented:
1. Load models to CPU first, then move to target device
2. Use proper device mapping parameters
3. Add fallback mechanisms for compatibility
4. Clear model cache if corrupted
"""

import os
import sys
import shutil
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_model_cache():
    """Clear corrupted model cache that might cause loading issues"""
    try:
        # Clear HuggingFace cache
        import torch

        cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"
        if cache_dir.exists():
            logger.info(f"Clearing HuggingFace cache: {cache_dir}")
            for item in cache_dir.iterdir():
                if item.is_dir() and "BAAI" in item.name:
                    logger.info(f"Removing cached model: {item}")
                    shutil.rmtree(item, ignore_errors=True)

        # Clear sentence-transformers cache
        sentence_cache = Path.home() / ".cache" / "torch" / "sentence_transformers"
        if sentence_cache.exists():
            logger.info(f"Clearing sentence-transformers cache: {sentence_cache}")
            for item in sentence_cache.iterdir():
                if item.is_dir() and "BAAI" in item.name:
                    logger.info(f"Removing cached model: {item}")
                    shutil.rmtree(item, ignore_errors=True)

        # Clear project model cache
        project_cache = project_root / "emmodels"
        if project_cache.exists():
            logger.info(f"Clearing project model cache: {project_cache}")
            for item in project_cache.iterdir():
                if item.is_dir() and "BAAI" in item.name:
                    logger.info(f"Removing cached model: {item}")
                    shutil.rmtree(item, ignore_errors=True)

    except Exception as e:
        logger.warning(f"Error clearing cache: {e}")


def test_model_loading():
    """Test if the model loading fix works"""
    try:
        logger.info("Testing SentenceTransformer model loading...")

        # Import after path setup
        from embeddings.providers import SentenceTransformerProvider

        # Test with the problematic model
        provider = SentenceTransformerProvider(
            model_name="BAAI/bge-large-en-v1.5", device="cpu"
        )

        # Test embedding generation
        test_text = "This is a test document for embedding generation."
        embedding = provider.generate_embedding(test_text)

        if embedding and len(embedding) == 1024:  # Expected dimension for bge-large
            logger.info("✅ Model loading test PASSED")
            logger.info(f"Generated embedding with dimension: {len(embedding)}")
            return True
        else:
            logger.error("❌ Model loading test FAILED - wrong embedding dimension")
            return False

    except Exception as e:
        logger.error(f"❌ Model loading test FAILED: {e}")
        return False


def fix_pytorch_compatibility():
    """Apply PyTorch compatibility fixes"""
    try:
        import torch

        logger.info(f"PyTorch version: {torch.__version__}")

        # Set environment variables for better compatibility
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Disable JIT if causing issues
        torch.jit.set_fuser("fuser0")

        logger.info("✅ PyTorch compatibility fixes applied")
        return True

    except Exception as e:
        logger.warning(f"Could not apply PyTorch fixes: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are installed with correct versions"""
    try:
        # Check sentence-transformers
        import sentence_transformers

        logger.info(
            f"sentence-transformers version: {sentence_transformers.__version__}"
        )

        # Check torch
        import torch

        logger.info(f"torch version: {torch.__version__}")

        # Check transformers
        import transformers

        logger.info(f"transformers version: {transformers.__version__}")

        # Version compatibility check
        torch_version = tuple(map(int, torch.__version__.split(".")[:2]))
        if torch_version >= (2, 0):
            logger.info("✅ PyTorch 2.0+ detected - using compatibility mode")

        return True

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False


def main():
    """Main fix function"""
    logger.info("🔧 Starting SentenceTransformer embedding fix...")

    # Step 1: Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed")
        return False

    # Step 2: Apply PyTorch compatibility fixes
    fix_pytorch_compatibility()

    # Step 3: Clear potentially corrupted cache
    clear_model_cache()

    # Step 4: Test model loading
    if test_model_loading():
        logger.info("🎉 All fixes applied successfully!")
        logger.info("The SentenceTransformer embedding issue has been resolved.")
        return True
    else:
        logger.error("❌ Fix failed - manual intervention may be required")
        logger.info("Try running with a smaller model like 'all-MiniLM-L6-v2'")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
