#!/usr/bin/env python3
"""
Performance benchmark tests for API endpoints
"""

import time
import requests
import json
import statistics
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


class PerformanceBenchmark:
    """Performance testing and benchmarking for API endpoints"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {"manual_search": [], "rag_search": [], "vector_search": []}

    def measure_response_time(
        self, endpoint: str, data: Dict, iterations: int = 10
    ) -> List[float]:
        """Measure response times for multiple iterations"""
        response_times = []

        for i in range(iterations):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}{endpoint}", json=data, timeout=30
                )
                end_time = time.time()

                if response.status_code == 200:
                    response_times.append(end_time - start_time)
                else:
                    print(f"⚠️  Iteration {i+1}: HTTP {response.status_code}")

            except Exception as e:
                print(f"❌ Iteration {i+1}: Error - {str(e)}")

        return response_times

    def benchmark_manual_search(self):
        """Benchmark manual search endpoint"""
        print("📊 Benchmarking Manual Search Performance")
        print("=" * 60)

        test_scenarios = [
            {
                "name": "Basic Search",
                "data": {
                    "userid": "67b075f7fe29fc1b2d36e18b",
                    "experience_titles": ["python developer"],
                    "skills": ["python", "javascript"],
                },
            },
            {
                "name": "Complex Search",
                "data": {
                    "userid": "67b075f7fe29fc1b2d36e18b",
                    "experience_titles": [
                        "senior python developer",
                        "full stack developer",
                    ],
                    "skills": ["python", "javascript", "react", "django", "postgresql"],
                    "min_education": ["BTech", "MTech"],
                    "min_experience": "3 years",
                    "max_experience": "8 years",
                    "locations": ["Mumbai", "Pune", "Bangalore"],
                    "min_salary": 800000,
                    "max_salary": 2000000,
                    "relevant_score": 60,
                },
            },
            {
                "name": "Large Arrays Search",
                "data": {
                    "userid": "67b075f7fe29fc1b2d36e18b",
                    "experience_titles": [
                        "developer",
                        "engineer",
                        "programmer",
                        "analyst",
                    ],
                    "skills": [
                        "python",
                        "java",
                        "javascript",
                        "react",
                        "angular",
                        "vue",
                        "nodejs",
                        "express",
                    ],
                    "locations": [
                        "Mumbai",
                        "Delhi",
                        "Bangalore",
                        "Chennai",
                        "Hyderabad",
                        "Pune",
                        "Kolkata",
                    ],
                },
            },
        ]

        for scenario in test_scenarios:
            print(f"\n🔍 Testing: {scenario['name']}")
            response_times = self.measure_response_time(
                "/manualsearch/", scenario["data"], 5
            )

            if response_times:
                avg_time = statistics.mean(response_times)
                median_time = statistics.median(response_times)
                min_time = min(response_times)
                max_time = max(response_times)

                print(f"   Average: {avg_time:.3f}s")
                print(f"   Median:  {median_time:.3f}s")
                print(f"   Min:     {min_time:.3f}s")
                print(f"   Max:     {max_time:.3f}s")

                self.results["manual_search"].append(
                    {
                        "scenario": scenario["name"],
                        "avg_time": avg_time,
                        "median_time": median_time,
                        "min_time": min_time,
                        "max_time": max_time,
                        "all_times": response_times,
                    }
                )
            else:
                print("   ❌ No successful responses")

    def benchmark_rag_search(self):
        """Benchmark RAG search endpoints"""
        print("\n📊 Benchmarking RAG Search Performance")
        print("=" * 60)

        test_scenarios = [
            {
                "name": "LLM Context Search - Basic",
                "endpoint": "/rag/llm-context-search",
                "data": {
                    "user_id": "67b075f7fe29fc1b2d36e18b",
                    "query": "python developer with machine learning experience",
                    "context_size": 5,
                    "relevant_score": 40,
                },
            },
            {
                "name": "LLM Context Search - Large Context",
                "endpoint": "/rag/llm-context-search",
                "data": {
                    "user_id": "67b075f7fe29fc1b2d36e18b",
                    "query": "senior full stack developer with microservices architecture experience",
                    "context_size": 15,
                    "relevant_score": 50,
                },
            },
            {
                "name": "Vector Search - Basic",
                "endpoint": "/rag/vector-search",
                "data": {
                    "user_id": "67b075f7fe29fc1b2d36e18b",
                    "query": "experienced react developer",
                    "limit": 10,
                },
            },
        ]

        for scenario in test_scenarios:
            print(f"\n🔍 Testing: {scenario['name']}")
            response_times = self.measure_response_time(
                scenario["endpoint"], scenario["data"], 3
            )

            if response_times:
                avg_time = statistics.mean(response_times)
                median_time = statistics.median(response_times)
                min_time = min(response_times)
                max_time = max(response_times)

                print(f"   Average: {avg_time:.3f}s")
                print(f"   Median:  {median_time:.3f}s")
                print(f"   Min:     {min_time:.3f}s")
                print(f"   Max:     {max_time:.3f}s")

                self.results["rag_search"].append(
                    {
                        "scenario": scenario["name"],
                        "avg_time": avg_time,
                        "median_time": median_time,
                        "min_time": min_time,
                        "max_time": max_time,
                        "all_times": response_times,
                    }
                )
            else:
                print("   ❌ No successful responses")

    def load_test(
        self,
        endpoint: str,
        data: Dict,
        concurrent_users: int = 10,
        duration_seconds: int = 30,
    ):
        """Perform load testing"""
        print(f"\n🚀 Load Testing: {endpoint}")
        print(f"   Concurrent Users: {concurrent_users}")
        print(f"   Duration: {duration_seconds}s")
        print("=" * 60)

        import concurrent.futures
        import threading

        results = []
        start_time = time.time()
        end_time = start_time + duration_seconds

        def make_requests():
            thread_results = []
            while time.time() < end_time:
                request_start = time.time()
                try:
                    response = requests.post(
                        f"{self.base_url}{endpoint}", json=data, timeout=30
                    )
                    request_end = time.time()

                    thread_results.append(
                        {
                            "timestamp": request_start,
                            "response_time": request_end - request_start,
                            "status_code": response.status_code,
                            "success": response.status_code == 200,
                        }
                    )
                except Exception as e:
                    thread_results.append(
                        {
                            "timestamp": request_start,
                            "response_time": 0,
                            "status_code": 0,
                            "success": False,
                            "error": str(e),
                        }
                    )

            return thread_results

        # Run concurrent requests
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrent_users
        ) as executor:
            futures = [executor.submit(make_requests) for _ in range(concurrent_users)]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())

        # Analyze results
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r["success"])
        failed_requests = total_requests - successful_requests

        if successful_requests > 0:
            response_times = [r["response_time"] for r in results if r["success"]]
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
        else:
            avg_response_time = median_response_time = p95_response_time = 0

        requests_per_second = total_requests / duration_seconds
        success_rate = (
            (successful_requests / total_requests * 100) if total_requests > 0 else 0
        )

        print(f"📈 Load Test Results:")
        print(f"   Total Requests: {total_requests}")
        print(f"   Successful: {successful_requests}")
        print(f"   Failed: {failed_requests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Requests/Second: {requests_per_second:.2f}")
        print(f"   Avg Response Time: {avg_response_time:.3f}s")
        print(f"   Median Response Time: {median_response_time:.3f}s")
        print(f"   95th Percentile: {p95_response_time:.3f}s")

        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": success_rate,
            "requests_per_second": requests_per_second,
            "avg_response_time": avg_response_time,
            "median_response_time": median_response_time,
            "p95_response_time": p95_response_time,
        }

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n📋 COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 80)

        # Manual Search Summary
        if self.results["manual_search"]:
            print("\n🔍 MANUAL SEARCH PERFORMANCE:")
            print("-" * 50)
            for result in self.results["manual_search"]:
                print(f"   {result['scenario']}:")
                print(f"      Average: {result['avg_time']:.3f}s")
                print(
                    f"      Range: {result['min_time']:.3f}s - {result['max_time']:.3f}s"
                )

        # RAG Search Summary
        if self.results["rag_search"]:
            print("\n🤖 RAG SEARCH PERFORMANCE:")
            print("-" * 50)
            for result in self.results["rag_search"]:
                print(f"   {result['scenario']}:")
                print(f"      Average: {result['avg_time']:.3f}s")
                print(
                    f"      Range: {result['min_time']:.3f}s - {result['max_time']:.3f}s"
                )

        # Performance Insights
        print("\n💡 PERFORMANCE INSIGHTS:")
        print("-" * 50)

        all_manual_times = []
        for result in self.results["manual_search"]:
            all_manual_times.extend(result["all_times"])

        all_rag_times = []
        for result in self.results["rag_search"]:
            all_rag_times.extend(result["all_times"])

        if all_manual_times:
            manual_avg = statistics.mean(all_manual_times)
            print(f"   Manual Search Overall Average: {manual_avg:.3f}s")

        if all_rag_times:
            rag_avg = statistics.mean(all_rag_times)
            print(f"   RAG Search Overall Average: {rag_avg:.3f}s")

        if all_manual_times and all_rag_times:
            if rag_avg > manual_avg:
                diff = ((rag_avg - manual_avg) / manual_avg) * 100
                print(f"   RAG Search is {diff:.1f}% slower than Manual Search")
            else:
                diff = ((manual_avg - rag_avg) / rag_avg) * 100
                print(f"   RAG Search is {diff:.1f}% faster than Manual Search")

        # Recommendations
        print("\n📝 RECOMMENDATIONS:")
        print("-" * 50)

        if all_manual_times and max(all_manual_times) > 5.0:
            print(
                "   • Manual Search: Consider optimizing complex queries (>5s response time detected)"
            )

        if all_rag_times and max(all_rag_times) > 10.0:
            print(
                "   • RAG Search: Consider reducing context size for better performance (>10s detected)"
            )

        if all_manual_times and statistics.stdev(all_manual_times) > 1.0:
            print(
                "   • Manual Search: High response time variance detected - check database indexing"
            )

        print(
            "   • Consider implementing response caching for frequently searched queries"
        )
        print("   • Monitor database connection pool under high load")
        print("   • Consider pagination for large result sets")


def run_performance_tests():
    """Run comprehensive performance tests"""
    print("🚀 Starting Performance Benchmark Suite")
    print("=" * 80)
    print(f"⏰ Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    benchmark = PerformanceBenchmark()

    # Run benchmarks
    benchmark.benchmark_manual_search()
    benchmark.benchmark_rag_search()

    # Run load tests (lighter load for testing)
    print("\n🚀 LOAD TESTING")
    print("=" * 80)

    # Load test manual search
    manual_load_data = {
        "userid": "67b075f7fe29fc1b2d36e18b",
        "experience_titles": ["developer"],
        "skills": ["python"],
    }

    manual_load_results = benchmark.load_test(
        "/manualsearch/", manual_load_data, concurrent_users=3, duration_seconds=15
    )

    # Generate comprehensive report
    benchmark.generate_performance_report()

    print(f"\n⏰ Performance tests completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return {
        "manual_search_results": benchmark.results["manual_search"],
        "rag_search_results": benchmark.results["rag_search"],
        "load_test_results": manual_load_results,
    }


if __name__ == "__main__":
    try:
        results = run_performance_tests()
        print("\n🎉 Performance testing completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Performance testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Performance testing failed: {str(e)}")
