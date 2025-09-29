#!/usr/bin/env python3
"""
Test the fixed manual search with the actual API endpoint
"""

import requests
import json
import time


def test_fixed_manual_search_api():
    """Test the fixed manual search API with various scenarios"""

    base_url = "http://localhost:8000/manualsearch/"

    print("🧪 TESTING FIXED MANUAL SEARCH API")
    print("=" * 60)

    # Test cases with different scenarios
    test_cases = [
        {
            "name": "Original Problematic Request",
            "payload": {
                "userid": "66c8771a20bd68c725758679",
                "experience_titles": [
                    "frontend developer",
                    "software developer",
                    "python developer",
                ],
                "skills": ["cobol"],
                "min_education": ["10th Pass"],
                "min_experience": "6 Months",
                "max_experience": "1 Year",
                "locations": ["adwani"],
                "min_salary": 1.0,
                "max_salary": 2.0,
                "relevant_score": 40.0,
            },
        },
        {
            "name": "More Lenient Request (Lower Threshold)",
            "payload": {
                "userid": "66c8771a20bd68c725758679",
                "experience_titles": [
                    "frontend developer",
                    "software developer",
                    "python developer",
                ],
                "skills": ["cobol"],
                "min_education": ["10th Pass"],
                "min_experience": "6 Months",
                "max_experience": "1 Year",
                "locations": ["adwani"],
                "min_salary": 1.0,
                "max_salary": 2.0,
                "relevant_score": 20.0,  # Lower threshold
            },
        },
        {
            "name": "Broader Skills Search",
            "payload": {
                "userid": "66c8771a20bd68c725758679",
                "experience_titles": ["developer"],  # More general
                "skills": ["python", "javascript"],  # Common skills
                "relevant_score": 25.0,
            },
        },
        {
            "name": "Location-Based Search",
            "payload": {
                "userid": "66c8771a20bd68c725758679",
                "locations": ["ahmedabad", "mumbai"],  # Common cities
                "relevant_score": 15.0,
            },
        },
        {
            "name": "No Criteria (All Resumes)",
            "payload": {
                "userid": "66c8771a20bd68c725758679"
                # No search criteria
            },
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print("-" * 50)

        try:
            # Make the request
            print(f"📤 Request: {json.dumps(test_case['payload'], indent=2)}")

            response = requests.post(base_url, json=test_case["payload"], timeout=30)

            print(f"📊 Response Status: {response.status_code}")

            if response.status_code == 200:
                results = response.json()

                # Check if it's a single item with message (no results case)
                if (
                    isinstance(results, list)
                    and len(results) == 1
                    and "message" in results[0]
                ):
                    print(f"📄 No Results Message: {results[0]['message']}")
                    search_summary = results[0].get("search_summary", {})
                    print(
                        f"📊 Total candidates available: {search_summary.get('total_candidates_available', 'N/A')}"
                    )
                    print(
                        f"📊 Total candidates searched: {search_summary.get('total_candidates_searched', 'N/A')}"
                    )

                    suggestions = search_summary.get("suggestions", [])
                    if suggestions:
                        print(f"💡 Suggestions:")
                        for suggestion in suggestions[:3]:  # Show top 3 suggestions
                            print(f"   - {suggestion}")
                else:
                    # Regular results
                    print(f"✅ Found {len(results)} results:")
                    for j, result in enumerate(results[:3], 1):  # Show top 3 results
                        name = result.get("contact_details", {}).get("name", "Unknown")
                        score = result.get("match_score", 0)
                        experience = result.get("total_experience", "N/A")
                        salary = (
                            result.get("expected_salary")
                            or result.get("current_salary")
                            or "N/A"
                        )

                        print(f"   {j}. {name}")
                        print(
                            f"      Score: {score}%, Experience: {experience}, Salary: {salary}"
                        )

                        # Show match details if available
                        match_details = result.get("match_details", {})
                        if match_details:
                            matched_titles = match_details.get(
                                "matched_experience_titles", []
                            )
                            matched_skills = match_details.get("matched_skills", [])
                            if matched_titles:
                                print(f"      Matched Titles: {matched_titles}")
                            if matched_skills:
                                print(f"      Matched Skills: {matched_skills}")

                        # Show if threshold was adjusted
                        if result.get("threshold_adjusted"):
                            print(
                                f"      ⚠️ Threshold adjusted from {result.get('original_threshold')}% to {result.get('effective_threshold')}%"
                            )
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")

        except requests.exceptions.ConnectionError:
            print(
                "❌ Connection Error: Make sure the API server is running on localhost:8000"
            )
        except requests.exceptions.Timeout:
            print("❌ Request Timeout: The request took too long")
        except Exception as e:
            print(f"❌ Unexpected Error: {str(e)}")

        time.sleep(1)  # Brief pause between requests

    print(f"\n🎉 Test completed!")
    print(f"\n💡 Key Improvements Made:")
    print(f"   - 🎯 More lenient salary filtering (10% variance)")
    print(f"   - ⏱️ More lenient experience filtering (25% variance)")
    print(f"   - 🔄 Automatic threshold adjustment when no results found")
    print(f"   - 🎨 Enhanced skills matching with partial matches")
    print(f"   - 📊 Better result scoring and ranking")


if __name__ == "__main__":
    test_fixed_manual_search_api()
