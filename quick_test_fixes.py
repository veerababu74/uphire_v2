#!/usr/bin/env python3
"""
Quick API verification script
Tests the most important fixes: salary validation and _id field serialization
"""

import requests
import json
from typing import Dict, Any


class QuickAPITest:
    """Quick test for critical API fixes"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []

    def test_server_connectivity(self) -> bool:
        """Test if server is accessible"""
        try:
            response = self.session.get(f"{self.base_url}/docs", timeout=5)
            return response.status_code == 200
        except:
            return False

    def test_manual_search_salary_fix(self) -> Dict[str, Any]:
        """Test the salary validation fix (empty strings should not cause 422 errors)"""
        test_cases = [
            {
                "name": "Empty salary strings",
                "data": {
                    "skills": ["Python", "FastAPI"],
                    "min_salary": "",  # This used to cause 422 error
                    "max_salary": "",  # This used to cause 422 error
                    "location": "Remote",
                },
            },
            {
                "name": "Valid salary values",
                "data": {
                    "skills": ["Python"],
                    "min_salary": 50000,
                    "max_salary": 100000,
                    "location": "New York",
                },
            },
            {
                "name": "Mixed salary values",
                "data": {
                    "skills": ["Java"],
                    "min_salary": 60000,
                    "max_salary": "",  # Empty max salary
                    "location": "California",
                },
            },
        ]

        results = {
            "test_name": "Manual Search Salary Fix",
            "passed": 0,
            "failed": 0,
            "details": [],
        }

        for test_case in test_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}/manual_search", json=test_case["data"], timeout=10
                )

                if response.status_code == 200:
                    results["passed"] += 1
                    results["details"].append(
                        {
                            "case": test_case["name"],
                            "status": "PASSED",
                            "status_code": response.status_code,
                        }
                    )
                elif response.status_code == 422:
                    results["failed"] += 1
                    results["details"].append(
                        {
                            "case": test_case["name"],
                            "status": "FAILED (422 Error - Salary fix not working)",
                            "status_code": response.status_code,
                            "error": (
                                response.json() if response.content else "No content"
                            ),
                        }
                    )
                else:
                    results["failed"] += 1
                    results["details"].append(
                        {
                            "case": test_case["name"],
                            "status": f"FAILED (Status {response.status_code})",
                            "status_code": response.status_code,
                        }
                    )

            except Exception as e:
                results["failed"] += 1
                results["details"].append(
                    {
                        "case": test_case["name"],
                        "status": f"FAILED (Exception: {str(e)})",
                        "error": str(e),
                    }
                )

        return results

    def test_rag_search_id_field_fix(self) -> Dict[str, Any]:
        """Test the _id field serialization fix"""
        test_cases = [
            {
                "name": "LLM Context Search",
                "endpoint": "/llm_context_search",
                "data": {
                    "query": "Python developer with 3 years experience",
                    "limit": 5,
                },
            },
            {
                "name": "RAG Vector Search",
                "endpoint": "/rag_search",
                "data": {"query": "Software engineer", "top_k": 3},
            },
        ]

        results = {
            "test_name": "RAG Search _id Field Fix",
            "passed": 0,
            "failed": 0,
            "details": [],
        }

        for test_case in test_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}{test_case['endpoint']}",
                    json=test_case["data"],
                    timeout=15,
                )

                if response.status_code == 200:
                    data = response.json()

                    # Check if response has results
                    results_key = "results" if "results" in data else "candidates"
                    if results_key in data and data[results_key]:
                        # Check if _id field is present in results
                        first_result = data[results_key][0]
                        has_id_field = "_id" in first_result or "id" in first_result
                        id_not_empty = (
                            first_result.get("_id") or first_result.get("id", "")
                        ).strip() != ""

                        if has_id_field and id_not_empty:
                            results["passed"] += 1
                            results["details"].append(
                                {
                                    "case": test_case["name"],
                                    "status": "PASSED (_id field present and not empty)",
                                    "status_code": response.status_code,
                                    "id_value": first_result.get("_id")
                                    or first_result.get("id", ""),
                                    "result_count": len(data[results_key]),
                                }
                            )
                        else:
                            results["failed"] += 1
                            results["details"].append(
                                {
                                    "case": test_case["name"],
                                    "status": "FAILED (_id field missing or empty)",
                                    "status_code": response.status_code,
                                    "has_id_field": has_id_field,
                                    "id_value": first_result.get("_id")
                                    or first_result.get("id", ""),
                                    "first_result_keys": list(first_result.keys()),
                                }
                            )
                    else:
                        results["failed"] += 1
                        results["details"].append(
                            {
                                "case": test_case["name"],
                                "status": "FAILED (No results returned)",
                                "status_code": response.status_code,
                                "response_keys": (
                                    list(data.keys())
                                    if isinstance(data, dict)
                                    else "Not dict"
                                ),
                            }
                        )
                else:
                    results["failed"] += 1
                    results["details"].append(
                        {
                            "case": test_case["name"],
                            "status": f"FAILED (Status {response.status_code})",
                            "status_code": response.status_code,
                        }
                    )

            except Exception as e:
                results["failed"] += 1
                results["details"].append(
                    {
                        "case": test_case["name"],
                        "status": f"FAILED (Exception: {str(e)})",
                        "error": str(e),
                    }
                )

        return results

    def run_quick_tests(self):
        """Run all quick tests"""
        print("🏃‍♂️ RUNNING QUICK API VERIFICATION TESTS")
        print("=" * 80)
        print("🎯 Testing critical fixes:")
        print("   • Salary field validation (422 error fix)")
        print("   • _id field serialization (empty _id fix)")
        print("=" * 80)

        # Check server connectivity
        print("🔍 Checking server connectivity...")
        if not self.test_server_connectivity():
            print("❌ Server is not accessible!")
            print("   Make sure the API server is running on http://localhost:8000")
            return
        print("✅ Server is accessible")

        # Test salary validation fix
        print("\n🧪 Testing manual search salary validation fix...")
        salary_results = self.test_manual_search_salary_fix()
        self.test_results.append(salary_results)

        # Test _id field fix
        print("\n🧪 Testing RAG search _id field fix...")
        id_results = self.test_rag_search_id_field_fix()
        self.test_results.append(id_results)

        # Print results
        self.print_results()

    def print_results(self):
        """Print test results summary"""
        print("\n📊 QUICK TEST RESULTS")
        print("=" * 80)

        total_passed = 0
        total_failed = 0

        for test_result in self.test_results:
            print(f"\n🧪 {test_result['test_name']}:")
            print(f"   ✅ Passed: {test_result['passed']}")
            print(f"   ❌ Failed: {test_result['failed']}")

            total_passed += test_result["passed"]
            total_failed += test_result["failed"]

            # Show details for failed tests
            for detail in test_result["details"]:
                if "FAILED" in detail["status"]:
                    print(f"   ⚠️  {detail['case']}: {detail['status']}")
                    if "error" in detail:
                        print(f"      Error: {detail['error']}")
                else:
                    print(f"   ✅ {detail['case']}: {detail['status']}")

        print(f"\n🏆 OVERALL RESULTS:")
        print(f"   Total Passed: {total_passed}")
        print(f"   Total Failed: {total_failed}")

        if total_failed == 0:
            print("   🎉 ALL CRITICAL FIXES ARE WORKING!")
            print("   ✅ Salary validation fix is working")
            print("   ✅ _id field serialization fix is working")
        else:
            print("   ⚠️  Some critical fixes need attention")
            if any(
                "422" in str(detail.get("status", ""))
                for result in self.test_results
                for detail in result["details"]
            ):
                print("   ❌ Salary validation fix may not be working")
            if any(
                "_id field missing or empty" in str(detail.get("status", ""))
                for result in self.test_results
                for detail in result["details"]
            ):
                print("   ❌ _id field serialization fix may not be working")

        print("=" * 80)


def main():
    """Main function"""
    tester = QuickAPITest()
    tester.run_quick_tests()


if __name__ == "__main__":
    main()
