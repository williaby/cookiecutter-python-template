"""Comprehensive performance testing with Locust for PromptCraft.

This module implements load testing scenarios for PromptCraft APIs including:
- CREATE framework request processing
- Knowledge retrieval performance
- MCP integration load testing
- Concurrent user simulation
- Performance threshold validation
"""

import random
import time

from locust import HttpUser, between, events, task


class PromptCraftUser(HttpUser):
    """Simulated user for PromptCraft performance testing."""

    # Wait time between tasks (1-3 seconds)
    wait_time = between(1, 3)

    def on_start(self):
        """Initialize user session and perform any setup."""
        self.headers = {"Content-Type": "application/json", "User-Agent": "LocustPerformanceTest/1.0"}

        # Sample data for testing
        self.sample_queries = [
            "Generate a Python function for calculating factorial",
            "Create a REST API endpoint for user authentication",
            "Write a SQL query to find duplicate records",
            "Implement a binary search algorithm",
            "Design a caching strategy for web applications",
        ]

        self.sample_contexts = [
            "Software development task",
            "Database optimization",
            "Algorithm implementation",
            "System architecture",
            "Performance optimization",
        ]

    @task(3)
    def test_create_request_processing(self):
        """Test CREATE framework request processing performance."""
        create_request = {
            "context": f"You are an AI assistant helping with {random.choice(self.sample_contexts)}",
            "request": random.choice(self.sample_queries),
            "examples": [{"input": "sample_input", "output": "sample_output"}],
            "augmentations": ["Include proper error handling", "Add type hints", "Include comprehensive documentation"],
            "tone_format": {"style": "professional", "format": "code_with_explanation"},
            "evaluation": {"criteria": ["correctness", "efficiency", "readability"]},
        }

        with self.client.post(
            "/api/v1/create/process",
            json=create_request,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"CREATE request failed: {response.status_code}")
            elif response.elapsed.total_seconds() > 5.0:
                response.failure("CREATE request took too long (>5s)")
            else:
                response.success()

    @task(2)
    def test_knowledge_retrieval_performance(self):
        """Test knowledge retrieval performance."""
        query_params = {
            "query": random.choice(self.sample_queries),
            "agent_id": "create_agent",
            "limit": random.randint(5, 20),
        }

        with self.client.post(
            "/api/v1/knowledge/search",
            json=query_params,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Knowledge search failed: {response.status_code}")
            elif response.elapsed.total_seconds() > 2.0:
                response.failure("Knowledge search took too long (>2s)")
            else:
                response.success()

    @task(1)
    def test_health_check_performance(self):
        """Test health check endpoint performance."""
        with self.client.get("/health", headers=self.headers, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")
            elif response.elapsed.total_seconds() > 0.5:
                response.failure("Health check took too long (>0.5s)")
            else:
                response.success()

    @task(1)
    def test_zen_mcp_integration_performance(self):
        """Test Zen MCP integration performance."""
        mcp_request = {
            "query": random.choice(self.sample_queries),
            "context": random.choice(self.sample_contexts),
            "agent_id": "create_agent",
        }

        with self.client.post(
            "/api/v1/mcp/zen/process",
            json=mcp_request,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Zen MCP request failed: {response.status_code}")
            elif response.elapsed.total_seconds() > 10.0:
                response.failure("Zen MCP request took too long (>10s)")
            else:
                response.success()


class HighLoadUser(HttpUser):
    """High-load user simulation for stress testing."""

    wait_time = between(0.1, 0.5)  # Faster requests for stress testing

    def on_start(self):
        """Initialize high-load user session."""
        self.headers = {"Content-Type": "application/json", "User-Agent": "LocustStressTest/1.0"}

    @task(1)
    def stress_test_create_endpoint(self):
        """Stress test CREATE endpoint with minimal payload."""
        minimal_request = {
            "context": "Test context",
            "request": "Generate test code",
            "examples": [],
            "augmentations": [],
            "tone_format": {"style": "simple"},
            "evaluation": {"criteria": []},
        }

        with self.client.post(
            "/api/v1/create/process",
            json=minimal_request,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 429:  # Rate limited
                response.success()  # Expected under high load
            elif response.status_code != 200:
                response.failure(f"Unexpected error: {response.status_code}")
            else:
                response.success()


class PerformanceTestUser(HttpUser):
    """Specialized user for performance threshold validation."""

    wait_time = between(2, 5)

    def on_start(self):
        """Initialize performance test user."""
        self.headers = {"Content-Type": "application/json", "User-Agent": "LocustPerformanceValidation/1.0"}
        self.performance_metrics = []

    @task(1)
    def validate_p95_response_time(self):
        """Validate that p95 response time meets requirements (<2s for CREATE)."""
        start_time = time.time()

        request_data = {
            "context": "Performance validation test",
            "request": "Generate optimized code with performance considerations",
            "examples": [{"input": "test", "output": "test"}],
            "augmentations": ["Performance optimization"],
            "tone_format": {"style": "professional"},
            "evaluation": {"criteria": ["performance"]},
        }

        with self.client.post(
            "/api/v1/create/process",
            json=request_data,
            headers=self.headers,
            catch_response=True,
        ) as response:
            response_time = time.time() - start_time
            self.performance_metrics.append(response_time)

            if response.status_code != 200:
                response.failure(f"Request failed: {response.status_code}")
            elif response_time > 2.0:
                response.failure(f"P95 requirement violation: {response_time:.2f}s > 2.0s")
            else:
                response.success()

    def on_stop(self):
        """Analyze performance metrics when user stops."""
        if self.performance_metrics:
            avg_time = sum(self.performance_metrics) / len(self.performance_metrics)
            max_time = max(self.performance_metrics)
            min_time = min(self.performance_metrics)

            print(f"Performance Summary for {self.__class__.__name__}:")
            print(f"  Requests: {len(self.performance_metrics)}")
            print(f"  Average: {avg_time:.2f}s")
            print(f"  Min: {min_time:.2f}s")
            print(f"  Max: {max_time:.2f}s")


# Performance monitoring event handlers
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    """Monitor request performance and log critical issues."""
    if exception:
        print(f"Request failed: {name} - {exception}")
    elif response_time > 5000:  # 5 seconds
        print(f"Slow request detected: {name} took {response_time}ms")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize performance testing."""
    print("=== PromptCraft Performance Testing Started ===")
    print(f"Target host: {environment.host}")
    print(f"Users: {environment.runner.target_user_count}")
    print("=" * 50)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Finalize performance testing and generate report."""
    print("=== PromptCraft Performance Testing Completed ===")

    stats = environment.runner.stats

    print("\nPerformance Summary:")
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Failed requests: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Min response time: {stats.total.min_response_time}ms")
    print(f"Max response time: {stats.total.max_response_time}ms")
    print(f"Requests per second: {stats.total.current_rps:.2f}")

    # Performance threshold validation
    performance_issues = []

    # Check average response time
    if stats.total.avg_response_time > 2000:  # 2 seconds
        performance_issues.append(f"Average response time too high: {stats.total.avg_response_time:.2f}ms")

    # Check failure rate
    failure_rate = (stats.total.num_failures / stats.total.num_requests) * 100 if stats.total.num_requests > 0 else 0
    if failure_rate > 5:  # 5% failure rate threshold
        performance_issues.append(f"Failure rate too high: {failure_rate:.2f}%")

    # Check RPS capability
    if stats.total.current_rps < 1:  # Minimum 1 RPS
        performance_issues.append(f"RPS too low: {stats.total.current_rps:.2f}")

    if performance_issues:
        print("\n⚠️ Performance Issues Detected:")
        for issue in performance_issues:
            print(f"  - {issue}")
    else:
        print("\n✅ All performance thresholds met!")

    print("=" * 50)


# User class weights for load distribution
user_classes = [
    {"user_class": PromptCraftUser, "weight": 70},
    {"user_class": HighLoadUser, "weight": 20},
    {"user_class": PerformanceTestUser, "weight": 10},
]


class CustomLoadTestShape:
    """Custom load testing shape for gradual ramp-up."""

    def tick(self):
        """Define load testing progression."""
        run_time = self.get_run_time()

        if run_time < 60:
            # Ramp up to 5 users in first minute
            return (5, 1)
        if run_time < 120:
            # Ramp up to 15 users in second minute
            return (15, 2)
        if run_time < 180:
            # Peak load: 25 users for third minute
            return (25, 3)
        if run_time < 240:
            # Sustained load: 20 users for fourth minute
            return (20, 2)
        # End test after 4 minutes
        return None


# Example usage and configuration
if __name__ == "__main__":
    """
    Run performance tests with different configurations:

    # Basic load test (10 users, 2/sec spawn rate, 30 seconds)
    locust -f locustfile.py --host=http://localhost:7860 --users 10 --spawn-rate 2 --run-time 30s

    # Stress test (50 users, 5/sec spawn rate, 2 minutes)
    locust -f locustfile.py --host=http://localhost:7860 --users 50 --spawn-rate 5 --run-time 2m

    # Custom load shape
    locust -f locustfile.py --host=http://localhost:7860 --shape=CustomLoadTestShape

    # Headless mode with CSV output
    locust -f locustfile.py --host=http://localhost:7860 --headless --users 20 --spawn-rate 2 --run-time 60s --csv=results
    """
    print("PromptCraft Performance Testing Framework")
    print("Use with Locust CLI commands for various testing scenarios")
