#!/usr/bin/env python3
"""
Comprehensive test cases for RAG Search and Manual Search APIs
Tests all scenarios including edge cases, error conditions, and success paths
"""

import asyncio
import json
import requests
from typing import Dict, List, Any
import time


class APITestSuite:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def log_test_result(
        self,
        test_name: str,
        success: bool,
        details: str = "",
        response_data: Any = None,
    ):
        """Log test result for reporting"""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            self.failed_tests += 1
            status = "❌ FAIL"

        result = {
            "test_name": test_name,
            "status": status,
            "success": success,
            "details": details,
            "response_data": response_data,
        }
        self.test_results.append(result)

        print(f"{status}: {test_name}")
        if details:
            print(f"   Details: {details}")
        print()

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


class RAGSearchTests(APITestSuite):
    """Test cases for RAG Search endpoints"""

    def test_llm_context_search_success(self):
        """Test successful LLM context search"""
        print("🔍 Testing RAG Search - LLM Context Search Success Cases")
        print("=" * 70)

        # Test Case 1: Basic search with valid parameters
        test_data = {
            "user_id": "67b075f7fe29fc1b2d36e18b",
            "query": "python developer with 5 years experience",
            "context_size": 10,
            "relevant_score": 40,
            "use_enhanced_search": True,
        }

        response, success, error = self.make_request(
            "/rag/llm-context-search", test_data
        )

        if not success:
            self.log_test_result(
                "LLM Context Search - Basic Valid Request", False, error
            )
            return

        if response.status_code == 200:
            try:
                result = response.json()

                # Validate response structure
                required_fields = ["results", "total_analyzed", "statistics"]
                missing_fields = [
                    field for field in required_fields if field not in result
                ]

                if missing_fields:
                    self.log_test_result(
                        "LLM Context Search - Response Structure",
                        False,
                        f"Missing fields: {missing_fields}",
                        result,
                    )
                else:
                    self.log_test_result(
                        "LLM Context Search - Response Structure",
                        True,
                        f"All required fields present",
                        result,
                    )

                # Check if _id field is populated
                if result.get("results"):
                    first_result = result["results"][0]
                    id_field = first_result.get("_id", "")

                    if id_field and id_field != "":
                        self.log_test_result(
                            "LLM Context Search - _id Field Population",
                            True,
                            f"_id field populated: {id_field}",
                        )
                    else:
                        self.log_test_result(
                            "LLM Context Search - _id Field Population",
                            False,
                            f"_id field empty or missing: '{id_field}'",
                        )

                    # Validate result structure
                    result_fields = [
                        "_id",
                        "user_id",
                        "username",
                        "contact_details",
                        "skills",
                        "relevance_score",
                    ]
                    first_result_fields = list(first_result.keys())

                    missing_result_fields = [
                        field
                        for field in result_fields
                        if field not in first_result_fields
                    ]
                    if missing_result_fields:
                        self.log_test_result(
                            "LLM Context Search - Result Fields",
                            False,
                            f"Missing result fields: {missing_result_fields}",
                        )
                    else:
                        self.log_test_result(
                            "LLM Context Search - Result Fields",
                            True,
                            "All result fields present",
                        )
                else:
                    self.log_test_result(
                        "LLM Context Search - Results Available",
                        False,
                        "No results returned",
                    )

            except json.JSONDecodeError:
                self.log_test_result(
                    "LLM Context Search - JSON Response", False, "Invalid JSON response"
                )
        else:
            self.log_test_result(
                "LLM Context Search - HTTP Status",
                False,
                f"Status: {response.status_code}, Body: {response.text[:500]}",
            )

    def test_llm_context_search_edge_cases(self):
        """Test LLM context search edge cases"""
        print("🧪 Testing RAG Search - Edge Cases")
        print("=" * 70)

        # Test Case 1: Empty query
        test_data = {
            "user_id": "67b075f7fe29fc1b2d36e18b",
            "query": "",
            "context_size": 5,
        }

        response, success, error = self.make_request(
            "/rag/llm-context-search", test_data
        )
        if success and response.status_code == 200:
            self.log_test_result(
                "LLM Context Search - Empty Query",
                True,
                "Empty query handled successfully",
            )
        else:
            self.log_test_result(
                "LLM Context Search - Empty Query",
                False,
                f"Status: {response.status_code if response else 'No response'}",
            )

        # Test Case 2: Very long query
        long_query = "python developer " * 100  # Very long query
        test_data = {
            "user_id": "67b075f7fe29fc1b2d36e18b",
            "query": long_query,
            "context_size": 5,
        }

        response, success, error = self.make_request(
            "/rag/llm-context-search", test_data
        )
        if success and response.status_code in [200, 400]:
            self.log_test_result(
                "LLM Context Search - Long Query",
                True,
                "Long query handled appropriately",
            )
        else:
            self.log_test_result(
                "LLM Context Search - Long Query",
                False,
                f"Status: {response.status_code if response else 'No response'}",
            )

        # Test Case 3: Maximum context size
        test_data = {
            "user_id": "67b075f7fe29fc1b2d36e18b",
            "query": "java developer",
            "context_size": 20,  # Maximum allowed
            "relevant_score": 0,
        }

        response, success, error = self.make_request(
            "/rag/llm-context-search", test_data
        )
        if success and response.status_code == 200:
            self.log_test_result(
                "LLM Context Search - Max Context Size",
                True,
                "Maximum context size handled",
            )
        else:
            self.log_test_result(
                "LLM Context Search - Max Context Size",
                False,
                f"Status: {response.status_code if response else 'No response'}",
            )

        # Test Case 4: Minimum relevant score
        test_data = {
            "user_id": "67b075f7fe29fc1b2d36e18b",
            "query": "react developer",
            "context_size": 5,
            "relevant_score": 100,  # Very high threshold
        }

        response, success, error = self.make_request(
            "/rag/llm-context-search", test_data
        )
        if success and response.status_code == 200:
            result = response.json()
            # Should return fewer or no results due to high threshold
            self.log_test_result(
                "LLM Context Search - High Relevance Threshold",
                True,
                f"Results filtered by high threshold: {len(result.get('results', []))}",
            )
        else:
            self.log_test_result(
                "LLM Context Search - High Relevance Threshold",
                False,
                f"Status: {response.status_code if response else 'No response'}",
            )

    def test_llm_context_search_error_cases(self):
        """Test LLM context search error cases"""
        print("❌ Testing RAG Search - Error Cases")
        print("=" * 70)

        # Test Case 1: Missing required fields
        test_data = {
            "query": "python developer"
            # Missing user_id
        }

        response, success, error = self.make_request(
            "/rag/llm-context-search", test_data
        )
        if success and response.status_code == 422:
            self.log_test_result(
                "LLM Context Search - Missing user_id",
                True,
                "Validation error returned as expected",
            )
        else:
            self.log_test_result(
                "LLM Context Search - Missing user_id",
                False,
                f"Expected 422, got {response.status_code if response else 'No response'}",
            )

        # Test Case 2: Invalid context_size
        test_data = {
            "user_id": "67b075f7fe29fc1b2d36e18b",
            "query": "python developer",
            "context_size": 25,  # Above maximum
        }

        response, success, error = self.make_request(
            "/rag/llm-context-search", test_data
        )
        if success and response.status_code == 422:
            self.log_test_result(
                "LLM Context Search - Invalid Context Size",
                True,
                "Validation error for invalid context size",
            )
        else:
            self.log_test_result(
                "LLM Context Search - Invalid Context Size",
                False,
                f"Expected 422, got {response.status_code if response else 'No response'}",
            )

        # Test Case 3: Invalid relevant_score
        test_data = {
            "user_id": "67b075f7fe29fc1b2d36e18b",
            "query": "python developer",
            "relevant_score": 150,  # Above 100
        }

        response, success, error = self.make_request(
            "/rag/llm-context-search", test_data
        )
        if success and response.status_code == 422:
            self.log_test_result(
                "LLM Context Search - Invalid Relevant Score",
                True,
                "Validation error for invalid relevant score",
            )
        else:
            self.log_test_result(
                "LLM Context Search - Invalid Relevant Score",
                False,
                f"Expected 422, got {response.status_code if response else 'No response'}",
            )

    def test_vector_similarity_search(self):
        """Test vector similarity search"""
        print("📊 Testing RAG Search - Vector Similarity Search")
        print("=" * 70)

        test_data = {
            "user_id": "67b075f7fe29fc1b2d36e18b",
            "query": "experienced python developer with react skills",
            "limit": 10,
            "use_enhanced_search": True,
        }

        response, success, error = self.make_request("/rag/vector-search", test_data)

        if not success:
            self.log_test_result(
                "Vector Similarity Search - Basic Request", False, error
            )
            return

        if response.status_code == 200:
            try:
                result = response.json()

                # Check response structure
                if "results" in result:
                    self.log_test_result(
                        "Vector Similarity Search - Response Structure",
                        True,
                        f"Found {len(result['results'])} results",
                    )

                    # Check _id field
                    if result["results"]:
                        first_result = result["results"][0]
                        id_field = first_result.get("_id", "")

                        if id_field and id_field != "":
                            self.log_test_result(
                                "Vector Similarity Search - _id Field",
                                True,
                                f"_id field populated: {id_field}",
                            )
                        else:
                            self.log_test_result(
                                "Vector Similarity Search - _id Field",
                                False,
                                f"_id field empty: '{id_field}'",
                            )
                else:
                    self.log_test_result(
                        "Vector Similarity Search - Response Structure",
                        False,
                        "Missing results field",
                    )
            except json.JSONDecodeError:
                self.log_test_result(
                    "Vector Similarity Search - JSON Response",
                    False,
                    "Invalid JSON response",
                )
        else:
            self.log_test_result(
                "Vector Similarity Search - HTTP Status",
                False,
                f"Status: {response.status_code}, Body: {response.text[:500]}",
            )


class ManualSearchTests(APITestSuite):
    """Test cases for Manual Search endpoints"""

    def test_manual_search_success(self):
        """Test successful manual search cases"""
        print("🔍 Testing Manual Search - Success Cases")
        print("=" * 70)

        # Test Case 1: Basic search with all parameters
        test_data = {
            "userid": "67b075f7fe29fc1b2d36e18b",
            "experience_titles": ["python developer", "software developer"],
            "skills": ["python", "javascript", "react"],
            "min_education": ["Graduate", "BTech"],
            "min_experience": "2 years",
            "max_experience": "8 years",
            "locations": ["Mumbai", "Pune", "Bangalore"],
            "min_salary": 500000,
            "max_salary": 1500000,
            "relevant_score": 40,
        }

        response, success, error = self.make_request("/manualsearch/", test_data)

        if not success:
            self.log_test_result("Manual Search - Basic Valid Request", False, error)
            return

        if response.status_code == 200:
            try:
                result = response.json()

                if isinstance(result, list):
                    self.log_test_result(
                        "Manual Search - Response Type",
                        True,
                        f"List response with {len(result)} results",
                    )

                    if result:
                        first_result = result[0]
                        # Check for required fields
                        required_fields = [
                            "user_id",
                            "username",
                            "contact_details",
                            "skills",
                        ]
                        missing_fields = [
                            field
                            for field in required_fields
                            if field not in first_result
                        ]

                        if not missing_fields:
                            self.log_test_result(
                                "Manual Search - Result Structure",
                                True,
                                "All required fields present",
                            )
                        else:
                            self.log_test_result(
                                "Manual Search - Result Structure",
                                False,
                                f"Missing fields: {missing_fields}",
                            )

                        # Check match_score
                        if "match_score" in first_result:
                            score = first_result["match_score"]
                            if isinstance(score, (int, float)) and score >= 0:
                                self.log_test_result(
                                    "Manual Search - Match Score",
                                    True,
                                    f"Valid match score: {score}",
                                )
                            else:
                                self.log_test_result(
                                    "Manual Search - Match Score",
                                    False,
                                    f"Invalid match score: {score}",
                                )
                        else:
                            self.log_test_result(
                                "Manual Search - Match Score",
                                False,
                                "Missing match_score field",
                            )
                    else:
                        self.log_test_result(
                            "Manual Search - Results Available",
                            True,
                            "Empty results (no matches found)",
                        )
                else:
                    self.log_test_result(
                        "Manual Search - Response Type",
                        False,
                        f"Expected list, got {type(result)}",
                    )

            except json.JSONDecodeError:
                self.log_test_result(
                    "Manual Search - JSON Response", False, "Invalid JSON response"
                )
        else:
            self.log_test_result(
                "Manual Search - HTTP Status",
                False,
                f"Status: {response.status_code}, Body: {response.text[:500]}",
            )

    def test_manual_search_salary_fix(self):
        """Test manual search with empty salary strings (the fix we implemented)"""
        print("💰 Testing Manual Search - Salary Field Fix")
        print("=" * 70)

        # Test Case 1: Empty salary strings (the original problem)
        test_data = {
            "userid": "67b075f7fe29fc1b2d36e18b",
            "experience_titles": ["frontend developer", "software developer"],
            "locations": [],
            "max_experience": "",
            "max_salary": "",  # Empty string - should not cause 422 error
            "min_education": ["10th Pass", "Graduate", "12th Pass"],
            "min_experience": "",
            "min_salary": "",  # Empty string - should not cause 422 error
            "skills": ["html", "python"],
        }

        response, success, error = self.make_request("/manualsearch/", test_data)

        if not success:
            self.log_test_result("Manual Search - Empty Salary Strings", False, error)
            return

        if response.status_code == 200:
            self.log_test_result(
                "Manual Search - Empty Salary Strings",
                True,
                "Empty salary strings handled correctly (no 422 error)",
            )
        elif response.status_code == 422:
            try:
                error_detail = response.json()
                self.log_test_result(
                    "Manual Search - Empty Salary Strings",
                    False,
                    f"Still getting 422 error: {error_detail}",
                )
            except:
                self.log_test_result(
                    "Manual Search - Empty Salary Strings",
                    False,
                    f"422 error: {response.text}",
                )
        else:
            self.log_test_result(
                "Manual Search - Empty Salary Strings",
                False,
                f"Unexpected status: {response.status_code}",
            )

        # Test Case 2: String numbers for salary
        test_data = {
            "userid": "67b075f7fe29fc1b2d36e18b",
            "experience_titles": ["developer"],
            "min_salary": "500000",  # String number - should be converted
            "max_salary": "1500000",  # String number - should be converted
        }

        response, success, error = self.make_request("/manualsearch/", test_data)

        if success and response.status_code == 200:
            self.log_test_result(
                "Manual Search - String Number Salaries",
                True,
                "String numbers converted successfully",
            )
        else:
            self.log_test_result(
                "Manual Search - String Number Salaries",
                False,
                f"Status: {response.status_code if response else 'No response'}",
            )

        # Test Case 3: Mixed valid and empty
        test_data = {
            "userid": "67b075f7fe29fc1b2d36e18b",
            "experience_titles": ["developer"],
            "min_salary": 500000,  # Valid float
            "max_salary": "",  # Empty string
        }

        response, success, error = self.make_request("/manualsearch/", test_data)

        if success and response.status_code == 200:
            self.log_test_result(
                "Manual Search - Mixed Salary Values",
                True,
                "Mixed valid and empty salary values handled",
            )
        else:
            self.log_test_result(
                "Manual Search - Mixed Salary Values",
                False,
                f"Status: {response.status_code if response else 'No response'}",
            )

    def test_manual_search_edge_cases(self):
        """Test manual search edge cases"""
        print("🧪 Testing Manual Search - Edge Cases")
        print("=" * 70)

        # Test Case 1: Only userid (minimal request)
        test_data = {"userid": "67b075f7fe29fc1b2d36e18b"}

        response, success, error = self.make_request("/manualsearch/", test_data)

        if success and response.status_code == 200:
            self.log_test_result(
                "Manual Search - Minimal Request (userid only)",
                True,
                "Minimal request handled successfully",
            )
        else:
            self.log_test_result(
                "Manual Search - Minimal Request (userid only)",
                False,
                f"Status: {response.status_code if response else 'No response'}",
            )

        # Test Case 2: Very specific search criteria
        test_data = {
            "userid": "67b075f7fe29fc1b2d36e18b",
            "experience_titles": ["Senior Machine Learning Engineer"],
            "skills": ["TensorFlow", "PyTorch", "Kubernetes"],
            "min_education": ["PhD"],
            "min_experience": "10 years",
            "max_experience": "15 years",
            "locations": ["San Francisco"],
            "min_salary": 2000000,
            "max_salary": 3000000,
            "relevant_score": 90,
        }

        response, success, error = self.make_request("/manualsearch/", test_data)

        if success and response.status_code == 200:
            result = response.json()
            self.log_test_result(
                "Manual Search - Very Specific Criteria",
                True,
                f"Specific search handled, results: {len(result) if isinstance(result, list) else 'non-list'}",
            )
        else:
            self.log_test_result(
                "Manual Search - Very Specific Criteria",
                False,
                f"Status: {response.status_code if response else 'No response'}",
            )

        # Test Case 3: Large arrays
        test_data = {
            "userid": "67b075f7fe29fc1b2d36e18b",
            "experience_titles": ["developer", "engineer", "programmer", "analyst"]
            * 5,  # Large array
            "skills": ["python", "java", "javascript", "react", "angular", "vue"] * 3,
            "locations": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"] * 2,
        }

        response, success, error = self.make_request("/manualsearch/", test_data)

        if success and response.status_code == 200:
            self.log_test_result(
                "Manual Search - Large Arrays",
                True,
                "Large arrays handled successfully",
            )
        else:
            self.log_test_result(
                "Manual Search - Large Arrays",
                False,
                f"Status: {response.status_code if response else 'No response'}",
            )

    def test_manual_search_error_cases(self):
        """Test manual search error cases"""
        print("❌ Testing Manual Search - Error Cases")
        print("=" * 70)

        # Test Case 1: Missing userid
        test_data = {
            "experience_titles": ["developer"]
            # Missing userid
        }

        response, success, error = self.make_request("/manualsearch/", test_data)

        if success and response.status_code == 422:
            self.log_test_result(
                "Manual Search - Missing userid",
                True,
                "Validation error for missing userid",
            )
        else:
            self.log_test_result(
                "Manual Search - Missing userid",
                False,
                f"Expected 422, got {response.status_code if response else 'No response'}",
            )

        # Test Case 2: Invalid salary values
        test_data = {
            "userid": "67b075f7fe29fc1b2d36e18b",
            "experience_titles": ["developer"],
            "min_salary": "invalid_number",
            "max_salary": "also_invalid",
        }

        response, success, error = self.make_request("/manualsearch/", test_data)

        if success and response.status_code == 422:
            self.log_test_result(
                "Manual Search - Invalid Salary Values",
                True,
                "Validation error for invalid salary values",
            )
        else:
            self.log_test_result(
                "Manual Search - Invalid Salary Values",
                False,
                f"Expected 422, got {response.status_code if response else 'No response'}",
            )

        # Test Case 3: Invalid relevant_score
        test_data = {
            "userid": "67b075f7fe29fc1b2d36e18b",
            "experience_titles": ["developer"],
            "relevant_score": -10,  # Negative score
        }

        response, success, error = self.make_request("/manualsearch/", test_data)

        if success and response.status_code == 422:
            self.log_test_result(
                "Manual Search - Invalid Relevant Score",
                True,
                "Validation error for invalid relevant score",
            )
        else:
            self.log_test_result(
                "Manual Search - Invalid Relevant Score",
                False,
                f"Expected 422, got {response.status_code if response else 'No response'}",
            )


def run_comprehensive_tests():
    """Run all test suites"""
    print("🚀 Starting Comprehensive API Test Suite")
    print("=" * 80)
    print(f"⏰ Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Initialize test suites
    rag_tests = RAGSearchTests()
    manual_tests = ManualSearchTests()

    # Run RAG Search tests
    print("\n🎯 RAG SEARCH TEST SUITE")
    print("=" * 80)
    rag_tests.test_llm_context_search_success()
    rag_tests.test_llm_context_search_edge_cases()
    rag_tests.test_llm_context_search_error_cases()
    rag_tests.test_vector_similarity_search()

    # Run Manual Search tests
    print("\n🎯 MANUAL SEARCH TEST SUITE")
    print("=" * 80)
    manual_tests.test_manual_search_success()
    manual_tests.test_manual_search_salary_fix()
    manual_tests.test_manual_search_edge_cases()
    manual_tests.test_manual_search_error_cases()

    # Combined results
    total_tests = rag_tests.total_tests + manual_tests.total_tests
    passed_tests = rag_tests.passed_tests + manual_tests.passed_tests
    failed_tests = rag_tests.failed_tests + manual_tests.failed_tests

    # Generate final report
    print("\n📊 COMPREHENSIVE TEST REPORT")
    print("=" * 80)
    print(f"🎯 Total Tests Run: {total_tests}")
    print(f"✅ Tests Passed: {passed_tests}")
    print(f"❌ Tests Failed: {failed_tests}")
    print(f"📈 Success Rate: {(passed_tests/total_tests*100):.1f}%")

    print(f"\n📋 RAG SEARCH TESTS:")
    print(
        f"   Total: {rag_tests.total_tests}, Passed: {rag_tests.passed_tests}, Failed: {rag_tests.failed_tests}"
    )
    print(f"   Success Rate: {(rag_tests.passed_tests/rag_tests.total_tests*100):.1f}%")

    print(f"\n📋 MANUAL SEARCH TESTS:")
    print(
        f"   Total: {manual_tests.total_tests}, Passed: {manual_tests.passed_tests}, Failed: {manual_tests.failed_tests}"
    )
    print(
        f"   Success Rate: {(manual_tests.passed_tests/manual_tests.total_tests*100):.1f}%"
    )

    # Show failed tests
    if failed_tests > 0:
        print(f"\n❌ FAILED TESTS SUMMARY:")
        print("-" * 50)
        all_results = rag_tests.test_results + manual_tests.test_results
        for result in all_results:
            if not result["success"]:
                print(f"• {result['test_name']}")
                if result["details"]:
                    print(f"  └─ {result['details']}")

    print(f"\n⏰ Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "success_rate": passed_tests / total_tests * 100,
        "rag_tests": {
            "total": rag_tests.total_tests,
            "passed": rag_tests.passed_tests,
            "failed": rag_tests.failed_tests,
        },
        "manual_tests": {
            "total": manual_tests.total_tests,
            "passed": manual_tests.passed_tests,
            "failed": manual_tests.failed_tests,
        },
    }


if __name__ == "__main__":
    # Run the comprehensive test suite
    results = run_comprehensive_tests()

    # Exit with appropriate code
    if results["failed_tests"] == 0:
        print("\n🎉 All tests passed!")
        exit(0)
    else:
        print(f"\n⚠️  {results['failed_tests']} tests failed!")
        exit(1)
