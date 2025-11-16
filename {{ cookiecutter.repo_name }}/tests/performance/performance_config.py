"""Performance testing configuration and utilities for PromptCraft.

This module provides configuration settings, helper functions, and performance
validation utilities for comprehensive load testing.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PerformanceThresholds:
    """Performance threshold definitions for validation."""

    # Response time thresholds (in seconds)
    create_p95_response_time: float = 2.0
    knowledge_search_p95: float = 1.0
    health_check_p95: float = 0.5
    mcp_integration_p95: float = 10.0

    # Throughput thresholds (requests per second)
    min_rps_overall: float = 1.0
    min_rps_create: float = 0.5
    min_rps_knowledge: float = 2.0

    # Error rate thresholds (percentage)
    max_error_rate: float = 5.0
    max_timeout_rate: float = 1.0

    # Resource usage thresholds
    max_memory_usage_mb: float = 2048.0  # 2GB
    max_cpu_usage_percent: float = 80.0


@dataclass
class LoadTestConfig:
    """Load test configuration settings."""

    # User simulation settings
    total_users: int = 20
    spawn_rate: float = 2.0
    run_time: str = "60s"

    # Test scenarios
    create_weight: int = 3
    knowledge_weight: int = 2
    health_weight: int = 1
    mcp_weight: int = 1

    # Target endpoints
    base_url: str = "http://localhost:7860"
    create_endpoint: str = "/api/v1/create/process"
    knowledge_endpoint: str = "/api/v1/knowledge/search"
    health_endpoint: str = "/health"
    mcp_endpoint: str = "/api/v1/mcp/zen/process"


class PerformanceValidator:
    """Utility class for validating performance test results."""

    def __init__(self, thresholds: PerformanceThresholds):
        self.thresholds = thresholds
        self.results: list[dict] = []

    def add_result(self, endpoint: str, response_time: float, success: bool, error_message: str | None = None):
        """Add a performance test result."""
        self.results.append(
            {
                "endpoint": endpoint,
                "response_time": response_time,
                "success": success,
                "error_message": error_message,
                "timestamp": time.time(),
            },
        )

    def calculate_percentile(self, response_times: list[float], percentile: float) -> float:
        """Calculate percentile from response times."""
        if not response_times:
            return 0.0

        sorted_times = sorted(response_times)
        index = int((percentile / 100.0) * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]

    def validate_results(self) -> tuple[bool, list[str]]:
        """Validate performance results against thresholds."""
        issues = []

        if not self.results:
            return False, ["No performance results to validate"]

        # Group results by endpoint
        endpoint_results = {}
        for result in self.results:
            endpoint = result["endpoint"]
            if endpoint not in endpoint_results:
                endpoint_results[endpoint] = []
            endpoint_results[endpoint].append(result)

        # Validate each endpoint
        for endpoint, results in endpoint_results.items():
            response_times = [r["response_time"] for r in results]
            success_count = sum(1 for r in results if r["success"])
            total_count = len(results)
            error_rate = ((total_count - success_count) / total_count) * 100 if total_count > 0 else 0

            # Calculate P95 response time
            p95_time = self.calculate_percentile(response_times, 95.0)

            # Check P95 thresholds
            if "create" in endpoint and p95_time > self.thresholds.create_p95_response_time:
                issues.append(
                    f"CREATE P95 response time ({p95_time:.2f}s) exceeds threshold ({self.thresholds.create_p95_response_time}s)",
                )

            if "knowledge" in endpoint and p95_time > self.thresholds.knowledge_search_p95:
                issues.append(
                    f"Knowledge search P95 response time ({p95_time:.2f}s) exceeds threshold ({self.thresholds.knowledge_search_p95}s)",
                )

            if "health" in endpoint and p95_time > self.thresholds.health_check_p95:
                issues.append(
                    f"Health check P95 response time ({p95_time:.2f}s) exceeds threshold ({self.thresholds.health_check_p95}s)",
                )

            if "mcp" in endpoint and p95_time > self.thresholds.mcp_integration_p95:
                issues.append(
                    f"MCP integration P95 response time ({p95_time:.2f}s) exceeds threshold ({self.thresholds.mcp_integration_p95}s)",
                )

            # Check error rates
            if error_rate > self.thresholds.max_error_rate:
                issues.append(
                    f"Error rate for {endpoint} ({error_rate:.1f}%) exceeds threshold ({self.thresholds.max_error_rate}%)",
                )

        return len(issues) == 0, issues

    def generate_report(self) -> dict:
        """Generate comprehensive performance report."""
        if not self.results:
            return {"status": "no_data", "message": "No performance data available"}

        # Calculate overall metrics
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r["success"])
        failed_requests = total_requests - successful_requests

        response_times = [r["response_time"] for r in self.results]
        avg_response_time = sum(response_times) / len(response_times)
        p95_response_time = self.calculate_percentile(response_times, 95.0)
        p99_response_time = self.calculate_percentile(response_times, 99.0)

        # Calculate RPS (assuming test duration)
        if self.results:
            test_duration = max(r["timestamp"] for r in self.results) - min(r["timestamp"] for r in self.results)
            rps = total_requests / test_duration if test_duration > 0 else 0
        else:
            rps = 0

        # Group by endpoint
        endpoint_stats = {}
        for result in self.results:
            endpoint = result["endpoint"]
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = []
            endpoint_stats[endpoint].append(result)

        # Generate endpoint-specific stats
        endpoint_metrics = {}
        for endpoint, results in endpoint_stats.items():
            times = [r["response_time"] for r in results]
            successes = sum(1 for r in results if r["success"])

            endpoint_metrics[endpoint] = {
                "total_requests": len(results),
                "successful_requests": successes,
                "error_rate_percent": ((len(results) - successes) / len(results)) * 100,
                "avg_response_time": sum(times) / len(times),
                "p95_response_time": self.calculate_percentile(times, 95.0),
                "p99_response_time": self.calculate_percentile(times, 99.0),
            }

        # Validate against thresholds
        passed, issues = self.validate_results()

        return {
            "status": "passed" if passed else "failed",
            "summary": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "error_rate_percent": (failed_requests / total_requests) * 100,
                "avg_response_time": avg_response_time,
                "p95_response_time": p95_response_time,
                "p99_response_time": p99_response_time,
                "requests_per_second": rps,
            },
            "endpoint_metrics": endpoint_metrics,
            "threshold_violations": issues,
            "timestamp": time.time(),
        }

    def save_report(self, file_path: Path) -> None:
        """Save performance report to file."""
        report = self.generate_report()
        with open(file_path, "w") as f:
            json.dump(report, f, indent=2)


class PerformanceTestRunner:
    """Helper class for running and managing performance tests."""

    def __init__(self, config: LoadTestConfig, thresholds: PerformanceThresholds):
        self.config = config
        self.thresholds = thresholds
        self.validator = PerformanceValidator(thresholds)

    def generate_locust_command(self, test_type: str = "standard") -> str:
        """Generate Locust command for different test types."""
        base_cmd = f"locust -f tests/performance/locustfile.py --host={self.config.base_url}"

        if test_type == "standard":
            return f"{base_cmd} --users {self.config.total_users} --spawn-rate {self.config.spawn_rate} --run-time {self.config.run_time}"

        if test_type == "stress":
            return f"{base_cmd} --users {self.config.total_users * 3} --spawn-rate {self.config.spawn_rate * 2} --run-time 5m"

        if test_type == "spike":
            return f"{base_cmd} --users {self.config.total_users * 5} --spawn-rate {self.config.spawn_rate * 5} --run-time 2m"

        if test_type == "endurance":
            return f"{base_cmd} --users {self.config.total_users} --spawn-rate {self.config.spawn_rate} --run-time 30m"

        if test_type == "headless":
            return f"{base_cmd} --headless --users {self.config.total_users} --spawn-rate {self.config.spawn_rate} --run-time {self.config.run_time} --csv=performance_results"

        return base_cmd

    def validate_environment(self) -> tuple[bool, list[str]]:
        """Validate that the testing environment is ready."""
        issues = []

        try:
            import requests

            # Check if application is running
            response = requests.get(f"{self.config.base_url}/health", timeout=5)
            if response.status_code != 200:
                issues.append(f"Application health check failed: {response.status_code}")

        except requests.exceptions.ConnectionError:
            issues.append("Cannot connect to application - ensure it's running")
        except requests.exceptions.Timeout:
            issues.append("Application health check timed out")
        except ImportError:
            issues.append("requests library not available for environment validation")

        # Check if Locust is available
        try:
            import locust
        except ImportError:
            issues.append("Locust not installed - run 'poetry install' to install dependencies")

        return len(issues) == 0, issues


# Default configurations
DEFAULT_THRESHOLDS = PerformanceThresholds()
DEFAULT_CONFIG = LoadTestConfig()


# Example usage and test scenarios
def create_performance_test_suite() -> dict[str, str]:
    """Create a comprehensive performance test suite."""
    runner = PerformanceTestRunner(DEFAULT_CONFIG, DEFAULT_THRESHOLDS)

    return {
        "standard_load": runner.generate_locust_command("standard"),
        "stress_test": runner.generate_locust_command("stress"),
        "spike_test": runner.generate_locust_command("spike"),
        "endurance_test": runner.generate_locust_command("endurance"),
        "headless_ci": runner.generate_locust_command("headless"),
    }


if __name__ == "__main__":
    """Example usage of performance testing utilities."""

    # Create test suite
    test_suite = create_performance_test_suite()

    print("PromptCraft Performance Test Suite:")
    print("=" * 50)

    for test_name, command in test_suite.items():
        print(f"\n{test_name.upper()}:")
        print(f"  {command}")

    print("\nPerformance Thresholds:")
    print(f"  CREATE P95: {DEFAULT_THRESHOLDS.create_p95_response_time}s")
    print(f"  Knowledge Search P95: {DEFAULT_THRESHOLDS.knowledge_search_p95}s")
    print(f"  Health Check P95: {DEFAULT_THRESHOLDS.health_check_p95}s")
    print(f"  MCP Integration P95: {DEFAULT_THRESHOLDS.mcp_integration_p95}s")
    print(f"  Max Error Rate: {DEFAULT_THRESHOLDS.max_error_rate}%")
