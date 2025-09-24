"""
Startup Script for Real-time Resume Processing System

This script helps you start the system and provides usage instructions.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def check_requirements():
    """Check if required dependencies are installed."""
    print("🔍 Checking system requirements...")

    required_packages = ["fastapi", "uvicorn", "aiofiles", "python-multipart"]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n🚨 Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False

    return True


def start_server():
    """Start the FastAPI server."""
    print("\n🚀 Starting FastAPI server...")
    print("📡 Server will be available at: http://localhost:8000")
    print(
        "📊 Dashboard will be at: http://localhost:8000/static/resume-processing-dashboard.html"
    )
    print("\n⏳ Starting server (this may take a moment)...")

    try:
        # Start the server
        subprocess.run([sys.executable, "main.py"], check=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")


def show_usage_guide():
    """Show usage guide."""
    print("\n" + "=" * 60)
    print("📖 REAL-TIME RESUME PROCESSING SYSTEM")
    print("=" * 60)

    print("\n🎯 What This System Does:")
    print("• Processes Excel files with resume data")
    print("• Handles bulk resume file uploads (PDF, DOC, DOCX, TXT)")
    print("• Provides REAL-TIME progress tracking")
    print("• Shows exactly which items succeeded/failed")
    print("• Handles interruptions gracefully")
    print("• Supports 10,000+ resumes with live updates")

    print("\n🚀 How to Use:")
    print("1. Start the server (this script does that)")
    print("2. Open: http://localhost:8000/static/resume-processing-dashboard.html")
    print("3. Choose Excel Upload or Bulk Resume Upload")
    print("4. Watch real-time progress as files are processed")
    print("5. Get detailed results when complete")

    print("\n📊 Real-time Features:")
    print("• Live progress bars")
    print("• Success/failure counts updating every 2 seconds")
    print("• Current item being processed")
    print("• Estimated completion time")
    print("• Error logs with timestamps")
    print("• Job control (pause/resume/cancel)")

    print("\n🛡️ Error Recovery:")
    print("• If processing stops at item 4,000 out of 10,000")
    print("• You'll know exactly which 4,000 items were processed")
    print("• Which ones succeeded and which failed")
    print("• Detailed error messages for each failure")
    print("• Option to retry or continue from where it stopped")

    print("\n🌐 API Endpoints:")
    print("📊 Excel Processing:")
    print("  POST /enhanced-excel-parser/upload-async")
    print("  GET  /enhanced-excel-parser/status/{job_id}")
    print("  GET  /enhanced-excel-parser/results/{job_id}")
    print("  POST /enhanced-excel-parser/control/{job_id}/{action}")

    print("\n📁 Bulk Processing:")
    print("  POST /enhanced-bulk-parser/upload-async")
    print("  GET  /enhanced-bulk-parser/status/{job_id}")
    print("  GET  /enhanced-bulk-parser/results/{job_id}")
    print("  POST /enhanced-bulk-parser/control/{job_id}/{action}")

    print("\n💡 Pro Tips:")
    print("• Use duplicate detection for bulk uploads")
    print("• Monitor the statistics endpoints for system health")
    print("• Check error logs for detailed failure information")
    print("• Use job control to pause/resume long-running processes")


def main():
    """Main startup function."""
    print("🔧 Real-time Resume Processing System Startup")
    print("=" * 60)

    # Show usage guide first
    show_usage_guide()

    # Check requirements
    if not check_requirements():
        print("\n🛑 Please install missing packages first")
        return

    # Check if main.py exists
    if not Path("main.py").exists():
        print(
            "\n❌ main.py not found. Please run this script from the project root directory."
        )
        return

    print("\n" + "=" * 60)
    print("🚀 READY TO START!")
    print("=" * 60)

    response = input("\nStart the server now? (y/n): ").lower()

    if response in ["y", "yes"]:
        start_server()
    else:
        print("\n📋 To start manually, run: python main.py")
        print(
            "📊 Then open: http://localhost:8000/static/resume-processing-dashboard.html"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Startup error: {e}")
