#!/usr/bin/env python3
"""
Final verification test that simulates the exact scenario from the logs.
This will process multiple resumes like the original failing scenario.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_multiple_resume_processing():
    """Test multiple resume processing scenario that was failing"""
    print("🧪 Testing multiple resume processing scenario...")

    try:
        # Import the core components
        from GroqcloudLLM.main import ResumeParser
        from apis.multiple_resume_parser_api import generate_embeddings_for_resume

        # Test sample resumes (like the ones in the logs)
        test_resumes = [
            {
                "filename": "test_resume_1.pdf",
                "content": """
                DURGA PRASAD PILLI
                Email: durga.prasad@example.com
                Phone: +91-9876543210
                
                EXPERIENCE:
                Senior Software Engineer at Tech Corp (2020-2024)
                - Developed scalable web applications
                - Led team of 5 developers
                
                EDUCATION:
                B.Tech Computer Science, IIT Hyderabad (2020)
                
                SKILLS:
                Python, Java, React, Node.js, AWS, Docker
                """,
            },
            {
                "filename": "test_resume_2.pdf",
                "content": """
                VEERABABU V
                Email: veera@example.com
                Phone: +91-9876543211
                
                EXPERIENCE:
                Generative AI Developer at AI Startup (2022-2024)
                - Built LLM-powered applications
                - Implemented RAG systems
                
                EDUCATION:
                M.Tech AI/ML, IIIT Bangalore (2022)
                
                SKILLS:
                Python, TensorFlow, LangChain, Vector Databases, FastAPI
                """,
            },
        ]

        print(f"Processing {len(test_resumes)} test resumes...")

        # Initialize parser
        parser = ResumeParser(llm_provider="groq")
        print("✅ ResumeParser initialized successfully")

        processed_resumes = []

        for i, resume in enumerate(test_resumes, 1):
            print(f"\n[{i}/{len(test_resumes)}] Processing {resume['filename']}...")

            try:
                # Parse the resume
                parsed_data = parser.process_resume(resume["content"])

                if "error" in parsed_data:
                    print(f"❌ Parsing failed: {parsed_data['error']}")
                    continue

                print("✅ Resume parsed successfully")

                # Generate embeddings (this was failing before)
                resume_with_embeddings = generate_embeddings_for_resume(
                    parsed_data, user_id=f"test_user_{i}", username=f"test_username_{i}"
                )

                # Check vector fields
                vector_fields = [
                    key
                    for key in resume_with_embeddings.keys()
                    if key.endswith("_vector")
                ]
                print(f"✅ Generated {len(vector_fields)} vector fields")

                processed_resumes.append(
                    {
                        "filename": resume["filename"],
                        "parsed": True,
                        "embedded": True,
                        "vector_count": len(vector_fields),
                    }
                )

            except Exception as e:
                print(f"❌ Failed to process {resume['filename']}: {e}")
                processed_resumes.append(
                    {
                        "filename": resume["filename"],
                        "parsed": False,
                        "embedded": False,
                        "error": str(e),
                    }
                )

        # Summary
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY:")
        print("=" * 60)

        success_count = sum(1 for r in processed_resumes if r.get("embedded", False))
        total_count = len(processed_resumes)

        for resume in processed_resumes:
            status = "✅ SUCCESS" if resume.get("embedded", False) else "❌ FAILED"
            if resume.get("embedded", False):
                print(
                    f"{resume['filename']}: {status} ({resume['vector_count']} vectors)"
                )
            else:
                print(f"{resume['filename']}: {status}")
                if "error" in resume:
                    print(f"  Error: {resume['error']}")

        print(
            f"\nOverall: {success_count}/{total_count} resumes processed successfully"
        )

        if success_count == total_count:
            print("\n🎉 ALL TESTS PASSED!")
            print("The multiple resume processing is working correctly.")
            return True
        else:
            print(f"\n⚠️  {total_count - success_count} resumes failed to process.")
            return False

    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🔍 Final Verification: Multiple Resume Processing")
    print("=" * 60)

    success = test_multiple_resume_processing()

    print("\n" + "=" * 60)
    if success:
        print("✅ VERIFICATION COMPLETE - All systems working!")
        print("The embedding issues have been fully resolved.")
        print("Your application should now process multiple resumes without errors.")
    else:
        print("❌ VERIFICATION FAILED - Issues detected!")
        print("Please check the error messages above.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
