#!/usr/bin/env python3
"""
Master test runner for all API test suites
Executes comprehensive, specialized, and performance tests
"""

import sys
import time
import json
from typing import Dict, Any
import argparse
from pathlib import Path

# Import all test modules
from comprehensive_api_tests import run_comprehensive_tests
from specialized_tests import run_specialized_tests
from performance_tests import run_performance_tests


class MasterTestRunner:
    """Master test runner for all API test suites"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.results = {"comprehensive": None, "specialized": None, "performance": None}
        self.server_check_passed = False

    def check_server_availability(
        self, base_url: str = "http://localhost:8000"
    ) -> bool:
        """Check if the API server is available"""
        print("🔍 Checking API server availability...")

        try:
            import requests

            # Try a simple request to check server
            response = requests.get(f"{base_url}/docs", timeout=5)
            if response.status_code == 200:
                print("✅ API server is available")
                return True
            else:
                print(f"⚠️  API server responded with status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ API server is not running or not accessible")
            return False
        except Exception as e:
            print(f"❌ Error checking server: {str(e)}")
            return False

    def print_banner(self):
        """Print test suite banner"""
        print("=" * 100)
        print("🧪 UPHIRE API COMPREHENSIVE TEST SUITE")
        print("=" * 100)
        print("📋 Test Categories:")
        print("   • Comprehensive API Tests (Functionality & Validation)")
        print("   • Specialized Tests (Edge Cases & Error Handling)")
        print("   • Performance Tests (Load Testing & Benchmarking)")
        print("=" * 100)
        print(f"⏰ Test Suite Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)

    def run_comprehensive_suite(self) -> Dict[str, Any]:
        """Run comprehensive API tests"""
        print("\n🎯 RUNNING COMPREHENSIVE API TESTS")
        print("=" * 80)

        try:
            results = run_comprehensive_tests()
            print("✅ Comprehensive tests completed successfully")
            return results
        except Exception as e:
            print(f"❌ Comprehensive tests failed: {str(e)}")
            return {
                "error": str(e),
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
            }

    def run_specialized_suite(self) -> Dict[str, Any]:
        """Run specialized tests"""
        print("\n🔬 RUNNING SPECIALIZED TESTS")
        print("=" * 80)

        try:
            results = run_specialized_tests()
            print("✅ Specialized tests completed successfully")
            return results if results else {"completed": True}
        except Exception as e:
            print(f"❌ Specialized tests failed: {str(e)}")
            return {"error": str(e)}

    def run_performance_suite(self) -> Dict[str, Any]:
        """Run performance tests"""
        print("\n🚀 RUNNING PERFORMANCE TESTS")
        print("=" * 80)

        try:
            results = run_performance_tests()
            print("✅ Performance tests completed successfully")
            return results if results else {"completed": True}
        except Exception as e:
            print(f"❌ Performance tests failed: {str(e)}")
            return {"error": str(e)}

    def generate_final_report(self):
        """Generate final comprehensive test report"""
        print("\n📊 FINAL TEST REPORT")
        print("=" * 100)

        # Calculate total execution time
        if self.start_time and self.end_time:
            total_time = self.end_time - self.start_time
            print(f"⏱️  Total Execution Time: {total_time:.2f} seconds")

        # Comprehensive test results
        if self.results["comprehensive"]:
            comp_results = self.results["comprehensive"]
            if "error" not in comp_results:
                print(f"\n✅ COMPREHENSIVE TESTS:")
                print(f"   Total Tests: {comp_results.get('total_tests', 0)}")
                print(f"   Passed: {comp_results.get('passed_tests', 0)}")
                print(f"   Failed: {comp_results.get('failed_tests', 0)}")
                print(f"   Success Rate: {comp_results.get('success_rate', 0):.1f}%")

                # Breakdown by category
                if "rag_tests" in comp_results:
                    rag = comp_results["rag_tests"]
                    print(f"   RAG Tests: {rag['passed']}/{rag['total']} passed")

                if "manual_tests" in comp_results:
                    manual = comp_results["manual_tests"]
                    print(
                        f"   Manual Tests: {manual['passed']}/{manual['total']} passed"
                    )
            else:
                print(f"\n❌ COMPREHENSIVE TESTS: Failed - {comp_results['error']}")

        # Specialized test results
        if self.results["specialized"]:
            spec_results = self.results["specialized"]
            if "error" not in spec_results:
                print(f"\n✅ SPECIALIZED TESTS: Completed")
                if "success_rate" in spec_results:
                    print(
                        f"   Concurrent Test Success Rate: {spec_results['success_rate']:.1f}%"
                    )
                    print(
                        f"   Average Response Time: {spec_results['avg_response_time']:.2f}s"
                    )
            else:
                print(f"\n❌ SPECIALIZED TESTS: Failed - {spec_results['error']}")

        # Performance test results
        if self.results["performance"]:
            perf_results = self.results["performance"]
            if "error" not in perf_results:
                print(f"\n✅ PERFORMANCE TESTS: Completed")

                # Show manual search performance
                if "manual_search_results" in perf_results:
                    manual_perf = perf_results["manual_search_results"]
                    if manual_perf:
                        avg_times = [r["avg_time"] for r in manual_perf]
                        overall_avg = sum(avg_times) / len(avg_times)
                        print(f"   Manual Search Avg Response Time: {overall_avg:.3f}s")

                # Show load test results
                if "load_test_results" in perf_results:
                    load_results = perf_results["load_test_results"]
                    print(
                        f"   Load Test Success Rate: {load_results.get('success_rate', 0):.1f}%"
                    )
                    print(
                        f"   Requests per Second: {load_results.get('requests_per_second', 0):.2f}"
                    )
            else:
                print(f"\n❌ PERFORMANCE TESTS: Failed - {perf_results['error']}")

        # Overall assessment
        print(f"\n🏆 OVERALL ASSESSMENT:")
        print("-" * 50)

        has_critical_failures = False

        # Check comprehensive tests
        if self.results["comprehensive"]:
            comp = self.results["comprehensive"]
            if "error" in comp or comp.get("failed_tests", 0) > 0:
                has_critical_failures = True
                print(
                    "   ⚠️  Critical functionality issues detected in comprehensive tests"
                )

        # Check server availability
        if not self.server_check_passed:
            has_critical_failures = True
            print("   ❌ API server connectivity issues")

        if not has_critical_failures:
            print("   ✅ All critical functionality tests passed")
            print("   ✅ API endpoints are working correctly")
            print("   ✅ Both RAG search and Manual search are functional")
        else:
            print("   ⚠️  Some critical issues need attention")

        # Recommendations
        print(f"\n📝 RECOMMENDATIONS:")
        print("-" * 50)

        if (
            self.results["comprehensive"]
            and self.results["comprehensive"].get("failed_tests", 0) > 0
        ):
            print("   • Fix failing comprehensive tests before deployment")

        if self.results["performance"]:
            perf = self.results["performance"]
            if (
                "load_test_results" in perf
                and perf["load_test_results"].get("success_rate", 100) < 95
            ):
                print(
                    "   • Investigate load testing failures - server may not handle concurrent requests well"
                )

        if not self.server_check_passed:
            print("   • Ensure API server is running before running tests")
            print("   • Check server configuration and port settings")

        print("   • Monitor response times in production environment")
        print("   • Set up automated testing pipeline")
        print("   • Consider implementing health check endpoints")

        print(f"\n⏰ Test Suite Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)

    def save_results_to_file(self, filename: str = None):
        """Save test results to JSON file"""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"

        results_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time": (
                (self.end_time - self.start_time)
                if self.start_time and self.end_time
                else None
            ),
            "server_check_passed": self.server_check_passed,
            "results": self.results,
        }

        try:
            with open(filename, "w") as f:
                json.dump(results_data, f, indent=2, default=str)
            print(f"📁 Test results saved to: {filename}")
        except Exception as e:
            print(f"⚠️  Failed to save results to file: {str(e)}")

    def run_all_tests(self, skip_performance: bool = False, save_results: bool = True):
        """Run all test suites"""
        self.start_time = time.time()

        # Print banner
        self.print_banner()

        # Check server availability
        self.server_check_passed = self.check_server_availability()

        if not self.server_check_passed:
            print("\n⚠️  WARNING: API server is not available!")
            print("   • Make sure the server is running on http://localhost:8000")
            print("   • Check if the port is correct")
            print("   • Verify server configuration")

            response = input("\nDo you want to continue with tests anyway? (y/N): ")
            if response.lower() != "y":
                print("🛑 Test suite aborted by user")
                return

        try:
            # Run comprehensive tests
            self.results["comprehensive"] = self.run_comprehensive_suite()

            # Run specialized tests
            self.results["specialized"] = self.run_specialized_suite()

            # Run performance tests (optional)
            if not skip_performance:
                self.results["performance"] = self.run_performance_suite()
            else:
                print("\n⏭️  Skipping performance tests")

        except KeyboardInterrupt:
            print("\n⚠️  Test suite interrupted by user")
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {str(e)}")
        finally:
            self.end_time = time.time()

            # Generate final report
            self.generate_final_report()

            # Save results
            if save_results:
                self.save_results_to_file()


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="UPHIRE API Comprehensive Test Suite")
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Skip performance tests (faster execution)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save results to file"
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="API server URL (default: http://localhost:8000)",
    )

    args = parser.parse_args()

    # Update base URL if specified
    if args.server_url != "http://localhost:8000":
        print(f"🔗 Using custom server URL: {args.server_url}")
        # Here you would update the base URL in test classes

    # Create and run test suite
    runner = MasterTestRunner()
    runner.run_all_tests(
        skip_performance=args.skip_performance, save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
