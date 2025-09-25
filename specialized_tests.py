#!/usr/bin/env python3
"""
Specialized test cases for specific scenarios and edge cases
"""

import asyncio
import json
import requests
import time
from typing import Dict, List, Any
import concurrent.futures
import threading


class SpecializedTests:
    """Specialized test cases for specific scenarios"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []

    def make_request(self, endpoint: str, data: Dict, timeout: int = 30) -> tuple:
        """Make HTTP request and return response and success status"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.post(url, json=data, timeout=timeout)
            return response, True, ""
        except requests.exceptions.RequestException as e:
            return None, False, f"Request failed: {str(e)}"
        except Exception as e:
            return None, False, f"Unexpected error: {str(e)}"

    def test_concurrent_requests(self, num_threads: int = 5):
        """Test concurrent API requests"""
        print(f"🔄 Testing Concurrent Requests ({num_threads} threads)")
        print("=" * 70)

        def make_concurrent_request(thread_id):
            test_data = {
                "userid": "67b075f7fe29fc1b2d36e18b",
                "experience_titles": [f"developer_{thread_id}"],
                "skills": ["python", "javascript"],
            }

            start_time = time.time()
            response, success, error = self.make_request("/manualsearch/", test_data)
            end_time = time.time()

            return {
                "thread_id": thread_id,
                "success": success and response and response.status_code == 200,
                "response_time": end_time - start_time,
                "status_code": response.status_code if response else None,
                "error": error,
            }

        # Run concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(make_concurrent_request, i) for i in range(num_threads)
            ]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # Analyze results
        successful_requests = sum(1 for r in results if r["success"])
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        max_response_time = max(r["response_time"] for r in results)
        min_response_time = min(r["response_time"] for r in results)

        print(f"✅ Successful requests: {successful_requests}/{num_threads}")
        print(f"⏱️  Average response time: {avg_response_time:.2f}s")
        print(f"⏱️  Max response time: {max_response_time:.2f}s")
        print(f"⏱️  Min response time: {min_response_time:.2f}s")

        # Show any failures
        failures = [r for r in results if not r["success"]]
        if failures:
            print("❌ Failed requests:")
            for failure in failures:
                print(f"   Thread {failure['thread_id']}: {failure['error']}")

        return {
            "total_requests": num_threads,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / num_threads * 100,
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "min_response_time": min_response_time,
        }

    def test_large_data_handling(self):
        """Test handling of large data sets"""
        print("📊 Testing Large Data Handling")
        print("=" * 70)

        # Test 1: Large skills array
        large_skills = [f"skill_{i}" for i in range(100)]
        test_data = {
            "userid": "67b075f7fe29fc1b2d36e18b",
            "experience_titles": ["developer"],
            "skills": large_skills,
        }

        start_time = time.time()
        response, success, error = self.make_request("/manualsearch/", test_data)
        end_time = time.time()

        if success and response and response.status_code == 200:
            print(
                f"✅ Large skills array handled successfully ({end_time - start_time:.2f}s)"
            )
        else:
            print(f"❌ Large skills array failed: {error}")

        # Test 2: Large experience titles array
        large_titles = [f"developer_{i}" for i in range(50)]
        test_data = {
            "userid": "67b075f7fe29fc1b2d36e18b",
            "experience_titles": large_titles,
        }

        start_time = time.time()
        response, success, error = self.make_request("/manualsearch/", test_data)
        end_time = time.time()

        if success and response and response.status_code == 200:
            print(
                f"✅ Large experience titles handled successfully ({end_time - start_time:.2f}s)"
            )
        else:
            print(f"❌ Large experience titles failed: {error}")

        # Test 3: Large locations array
        large_locations = [f"City_{i}" for i in range(200)]
        test_data = {
            "userid": "67b075f7fe29fc1b2d36e18b",
            "experience_titles": ["developer"],
            "locations": large_locations,
        }

        start_time = time.time()
        response, success, error = self.make_request("/manualsearch/", test_data)
        end_time = time.time()

        if success and response and response.status_code == 200:
            print(
                f"✅ Large locations array handled successfully ({end_time - start_time:.2f}s)"
            )
        else:
            print(f"❌ Large locations array failed: {error}")

    def test_special_characters(self):
        """Test handling of special characters and Unicode"""
        print("🌐 Testing Special Characters and Unicode")
        print("=" * 70)

        special_test_cases = [
            {
                "name": "Unicode Characters",
                "data": {
                    "userid": "67b075f7fe29fc1b2d36e18b",
                    "experience_titles": ["डेवलपर", "程序员", "مطور"],
                    "skills": ["пайтон", "जावास्क्रिप्ट"],
                },
            },
            {
                "name": "Special Symbols",
                "data": {
                    "userid": "67b075f7fe29fc1b2d36e18b",
                    "experience_titles": [
                        "C++ developer",
                        "C# programmer",
                        ".NET engineer",
                    ],
                    "skills": ["C++", "C#", ".NET", "SQL*Plus"],
                },
            },
            {
                "name": "SQL Injection Attempt",
                "data": {
                    "userid": "67b075f7fe29fc1b2d36e18b",
                    "experience_titles": ["'; DROP TABLE users; --", "developer"],
                    "skills": ["python"],
                },
            },
            {
                "name": "HTML/XSS Characters",
                "data": {
                    "userid": "67b075f7fe29fc1b2d36e18b",
                    "experience_titles": ["<script>alert('xss')</script>", "developer"],
                    "skills": ["<html>", "&amp;", "javascript"],
                },
            },
        ]

        for test_case in special_test_cases:
            response, success, error = self.make_request(
                "/manualsearch/", test_case["data"]
            )

            if success and response and response.status_code == 200:
                print(f"✅ {test_case['name']}: Handled safely")
            elif success and response and response.status_code == 422:
                print(f"⚠️  {test_case['name']}: Validation error (expected)")
            else:
                print(f"❌ {test_case['name']}: Failed - {error}")

    def test_data_consistency(self):
        """Test data consistency across multiple calls"""
        print("🔄 Testing Data Consistency")
        print("=" * 70)

        test_data = {
            "userid": "67b075f7fe29fc1b2d36e18b",
            "experience_titles": ["python developer"],
            "skills": ["python"],
        }

        # Make multiple requests with same data
        responses = []
        for i in range(3):
            response, success, error = self.make_request("/manualsearch/", test_data)
            if success and response and response.status_code == 200:
                try:
                    result = response.json()
                    responses.append(result)
                except json.JSONDecodeError:
                    print(f"❌ Request {i+1}: Invalid JSON response")
                    return
            else:
                print(f"❌ Request {i+1}: Failed - {error}")
                return

        # Compare responses
        if len(responses) >= 2:
            first_response = responses[0]

            # Check if all responses have same number of results
            result_counts = [len(r) if isinstance(r, list) else 0 for r in responses]

            if len(set(result_counts)) == 1:
                print(f"✅ Consistent result count across calls: {result_counts[0]}")

                # Check if first result is consistent (if results exist)
                if result_counts[0] > 0:
                    first_result_ids = []
                    for response in responses:
                        if isinstance(response, list) and len(response) > 0:
                            first_result_ids.append(response[0].get("user_id", ""))

                    if len(set(first_result_ids)) == 1:
                        print("✅ First result consistent across calls")
                    else:
                        print(
                            "⚠️  First result varies across calls (could be normal due to sorting)"
                        )

            else:
                print(f"⚠️  Result count varies: {result_counts}")
        else:
            print("❌ Not enough successful responses to compare")

    def test_boundary_values(self):
        """Test boundary values for all parameters"""
        print("🔢 Testing Boundary Values")
        print("=" * 70)

        boundary_tests = [
            {
                "name": "Maximum Context Size (RAG)",
                "endpoint": "/rag/llm-context-search",
                "data": {
                    "user_id": "67b075f7fe29fc1b2d36e18b",
                    "query": "python developer",
                    "context_size": 20,  # Maximum allowed
                },
            },
            {
                "name": "Minimum Context Size (RAG)",
                "endpoint": "/rag/llm-context-search",
                "data": {
                    "user_id": "67b075f7fe29fc1b2d36e18b",
                    "query": "python developer",
                    "context_size": 1,  # Minimum allowed
                },
            },
            {
                "name": "Maximum Relevant Score",
                "endpoint": "/manualsearch/",
                "data": {
                    "userid": "67b075f7fe29fc1b2d36e18b",
                    "experience_titles": ["developer"],
                    "relevant_score": 100.0,  # Maximum
                },
            },
            {
                "name": "Minimum Relevant Score",
                "endpoint": "/manualsearch/",
                "data": {
                    "userid": "67b075f7fe29fc1b2d36e18b",
                    "experience_titles": ["developer"],
                    "relevant_score": 0.0,  # Minimum
                },
            },
            {
                "name": "Very High Salary",
                "endpoint": "/manualsearch/",
                "data": {
                    "userid": "67b075f7fe29fc1b2d36e18b",
                    "experience_titles": ["developer"],
                    "min_salary": 10000000,  # 1 crore
                    "max_salary": 50000000,  # 5 crore
                },
            },
            {
                "name": "Very Low Salary",
                "endpoint": "/manualsearch/",
                "data": {
                    "userid": "67b075f7fe29fc1b2d36e18b",
                    "experience_titles": ["developer"],
                    "min_salary": 1,
                    "max_salary": 100,
                },
            },
        ]

        for test in boundary_tests:
            response, success, error = self.make_request(test["endpoint"], test["data"])

            if success and response and response.status_code == 200:
                print(f"✅ {test['name']}: Handled successfully")
            elif success and response and response.status_code == 422:
                print(f"⚠️  {test['name']}: Validation error (might be expected)")
            else:
                print(f"❌ {test['name']}: Failed - {error}")

    def test_error_recovery(self):
        """Test error recovery and graceful handling"""
        print("🛡️ Testing Error Recovery")
        print("=" * 70)

        # Test with malformed JSON (simulated)
        malformed_tests = [
            {"name": "Empty Request Body", "endpoint": "/manualsearch/", "data": {}},
            {
                "name": "Null Values",
                "endpoint": "/manualsearch/",
                "data": {"userid": None, "experience_titles": None, "skills": None},
            },
            {
                "name": "Wrong Data Types",
                "endpoint": "/manualsearch/",
                "data": {
                    "userid": 12345,  # Should be string
                    "experience_titles": "not_a_list",  # Should be list
                    "skills": {"not": "a_list"},  # Should be list
                    "min_salary": "not_a_number",  # Should be number
                },
            },
        ]

        for test in malformed_tests:
            response, success, error = self.make_request(test["endpoint"], test["data"])

            if success and response:
                if response.status_code == 422:
                    print(f"✅ {test['name']}: Proper validation error returned")
                elif response.status_code == 400:
                    print(f"✅ {test['name']}: Bad request handled properly")
                else:
                    print(
                        f"⚠️  {test['name']}: Unexpected status {response.status_code}"
                    )
            else:
                print(f"❌ {test['name']}: Request failed - {error}")


def run_specialized_tests():
    """Run all specialized test scenarios"""
    print("🔬 Starting Specialized Test Scenarios")
    print("=" * 80)
    print(f"⏰ Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    tests = SpecializedTests()

    # Run all specialized tests
    print("\n🔄 CONCURRENT REQUESTS TEST")
    print("=" * 50)
    concurrent_results = tests.test_concurrent_requests(5)

    print("\n📊 LARGE DATA HANDLING TEST")
    print("=" * 50)
    tests.test_large_data_handling()

    print("\n🌐 SPECIAL CHARACTERS TEST")
    print("=" * 50)
    tests.test_special_characters()

    print("\n🔄 DATA CONSISTENCY TEST")
    print("=" * 50)
    tests.test_data_consistency()

    print("\n🔢 BOUNDARY VALUES TEST")
    print("=" * 50)
    tests.test_boundary_values()

    print("\n🛡️ ERROR RECOVERY TEST")
    print("=" * 50)
    tests.test_error_recovery()

    print(f"\n⏰ Specialized tests completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return concurrent_results


if __name__ == "__main__":
    run_specialized_tests()
