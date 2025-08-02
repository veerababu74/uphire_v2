"""
Test main application import.
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_main_import():
    """Test that main application can be imported."""
    try:
        print("Testing main application import...")

        # Import main
        from main import app

        print("✅ Successfully imported main application")
        print(f"App type: {type(app)}")

        return True

    except Exception as e:
        print(f"❌ Error during main import: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running main application import test...")
    print("=" * 50)

    success = test_main_import()

    print("=" * 50)
    if success:
        print("🎉 Main application imported successfully!")
    else:
        print("💥 Failed to import main application.")
