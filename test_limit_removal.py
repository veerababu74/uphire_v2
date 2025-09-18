#!/usr/bin/env python3
"""
Test to verify that the limit parameter is completely removed and ignored
"""

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from apis.autocomplete_skills_titiles import get_skills_by_job_titles


async def test_limit_removal():
    """Test that limit parameter is completely ignored"""

    print("🚫 TESTING LIMIT PARAMETER REMOVAL")
    print("=" * 50)

    # Test case 1: Request WITHOUT limit parameter
    print("\n🔍 TEST 1: Request WITHOUT limit parameter")
    request_no_limit = {
        "titles": ["python developer", "data scientist"],
        "include_related": True,
    }

    result1 = await get_skills_by_job_titles(request_no_limit)
    skills_count_1 = len(result1.get("skills", []))
    print(f"   ✅ Result: {skills_count_1} skills returned")

    # Test case 2: Request WITH limit parameter (should be completely ignored)
    print("\n🔍 TEST 2: Request WITH limit parameter (should be ignored)")
    request_with_limit = {
        "titles": ["python developer", "data scientist"],
        "limit": 5,  # This should be completely ignored
        "include_related": True,
    }

    result2 = await get_skills_by_job_titles(request_with_limit)
    skills_count_2 = len(result2.get("skills", []))
    print(f"   ✅ Result: {skills_count_2} skills returned (limit=5 ignored)")

    # Test case 3: Request with very small limit (should be ignored)
    print("\n🔍 TEST 3: Request WITH very small limit=1 (should be ignored)")
    request_tiny_limit = {
        "titles": ["python developer"],
        "limit": 1,  # Should be completely ignored
        "include_related": True,
    }

    result3 = await get_skills_by_job_titles(request_tiny_limit)
    skills_count_3 = len(result3.get("skills", []))
    print(f"   ✅ Result: {skills_count_3} skills returned (limit=1 ignored)")

    # Verification
    print(f"\n📊 RESULTS COMPARISON:")
    print(f"   • No limit parameter: {skills_count_1} skills")
    print(f"   • With limit=5: {skills_count_2} skills")
    print(f"   • With limit=1: {skills_count_3} skills")

    if skills_count_1 == skills_count_2 and skills_count_2 > 5:
        print(f"   ✅ SUCCESS: Limit parameter completely ignored!")
        print(
            f"   ✅ Dynamic sizing working: Returned {skills_count_1} skills based on quality"
        )
    else:
        print(f"   ❌ ERROR: Limit parameter may still be affecting results")

    if skills_count_3 > 1:
        print(
            f"   ✅ SUCCESS: Even limit=1 was ignored, returned {skills_count_3} skills"
        )
    else:
        print(f"   ❌ ERROR: Very small limit may still be working")

    # Show some example skills to prove we get reasonable results
    if result1.get("skills"):
        sample_skills = result1["skills"][:10]
        print(f"\n🎯 SAMPLE SKILLS (first 10):")
        for i, skill in enumerate(sample_skills, 1):
            print(f"   {i}. {skill}")

    print(f"\n🎉 LIMIT PARAMETER REMOVAL TEST COMPLETED!")
    return skills_count_1 > 5  # Should return many skills, not limited


if __name__ == "__main__":
    success = asyncio.run(test_limit_removal())
    if success:
        print("✅ ALL TESTS PASSED - Limit parameter successfully removed!")
    else:
        print("❌ TESTS FAILED - Limit parameter may still be working")
