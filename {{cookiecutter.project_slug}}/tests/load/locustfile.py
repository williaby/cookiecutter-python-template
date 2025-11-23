"""Load testing with Locust for {{ cookiecutter.project_name }}.

Locust is a Python-based load testing tool that's easy to use and highly scalable.
It simulates user behavior and measures application performance under load.

Setup:
    1. Install Locust:
       uv add --dev locust

    2. Run load test:
       # Web UI (recommended)
       locust -f tests/load/locustfile.py --host=http://localhost:8000

       # Headless mode
       locust -f tests/load/locustfile.py --host=http://localhost:8000 \\
              --users 100 --spawn-rate 10 --run-time 5m --headless

    3. View results:
       Open http://localhost:8089 in browser

Features:
    - Realistic user behavior simulation
    - Distributed load testing support
    - Real-time web UI dashboard
    - Detailed performance metrics
    - CSV export for analysis
"""

from __future__ import annotations

import json
import random  # nosec B311 - random is used for load testing simulation, not security
from typing import TYPE_CHECKING

from locust import HttpUser, between, task

if TYPE_CHECKING:
    pass


class APIUser(HttpUser):
    """Simulated API user for load testing.

    This class defines user behavior patterns and task weights.
    """

    # Wait time between tasks (in seconds)
    wait_time = between(1, 3)  # Random wait between 1-3 seconds

    # Authentication token (set on startup)
    auth_token: str | None = None

    def on_start(self) -> None:
        """Called when a simulated user starts.

        Use this for:
        - Authentication
        - Setting up user state
        - Logging in
        """
        # Example: Login to get auth token
        # response = self.client.post("/api/auth/login", json={
        #     "username": f"user_{self.environment.runner.user_count}",
        #     "password": "test123"
        # })
        # self.auth_token = response.json().get("token")
        pass

    def on_stop(self) -> None:
        """Called when a simulated user stops.

        Use this for:
        - Logging out
        - Cleaning up resources
        """
        pass

    # ==========================================================================
    # Tasks (weighted by decorator number)
    # ==========================================================================

    @task(5)  # Weight: 5 (runs 5x more often than weight 1)
    def get_health(self) -> None:
        """Health check endpoint.

        Most common request in production.
        """
        with self.client.get("/health/live", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(3)
    def list_items(self) -> None:
        """List items endpoint.

        Simulates browsing/searching.
        """
        # Add query parameters for realism
        page = random.randint(1, 10)  # nosec B311
        limit = random.choice([10, 20, 50])  # nosec B311

        with self.client.get(
            f"/api/items?page={page}&limit={limit}",
            name="/api/items?page=[page]&limit=[limit]",  # Group similar requests
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    item_count = len(data.get("items", []))

                    if item_count > 0:
                        response.success()
                    else:
                        response.failure("No items returned")

                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Request failed: {response.status_code}")

    @task(2)
    def get_item_detail(self) -> None:
        """Get single item endpoint.

        Simulates viewing item details.
        """
        # Simulate random item IDs
        item_id = random.randint(1, 1000)  # nosec B311

        with self.client.get(
            f"/api/items/{item_id}",
            name="/api/items/[id]",  # Group all detail requests
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # 404 is acceptable for random IDs
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(1)
    def create_item(self) -> None:
        """Create item endpoint.

        Simulates write operations (less frequent than reads).
        """
        payload = {
            "name": f"Test Item {random.randint(1, 10000)}",  # nosec B311
            "description": "Load test item",
            "price": round(random.uniform(10, 1000), 2),  # nosec B311
            "quantity": random.randint(1, 100),  # nosec B311
        }

        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        with self.client.post(
            "/api/items",
            json=payload,
            headers=headers,
            catch_response=True,
        ) as response:
            if response.status_code in (200, 201):
                response.success()
            elif response.status_code == 401:
                # Expected if not authenticated
                response.success()
            else:
                response.failure(f"Create failed: {response.status_code}")


class AdminUser(HttpUser):
    """Simulated admin user with heavier workloads.

    Separate user class for admin operations.
    """

    wait_time = between(5, 10)  # Slower cadence than regular users
    weight = 1  # Only 1 admin per 10 regular users (10% of traffic)

    @task
    def get_analytics(self) -> None:
        """Heavy analytics query.

        Tests database performance under load.
        """
        with self.client.get("/api/admin/analytics", catch_response=True) as response:
            if response.status_code == 200:
                # Check response time
                if response.elapsed.total_seconds() > 5:
                    response.failure("Analytics too slow (>5s)")
                else:
                    response.success()
            else:
                response.failure(f"Analytics failed: {response.status_code}")


# =============================================================================
# Load Test Scenarios
# =============================================================================

"""
# Scenario 1: Gradual Ramp-Up (Recommended for production testing)
locust -f tests/load/locustfile.py --host=http://localhost:8000 \\
       --users 100 --spawn-rate 5 --run-time 10m --headless

# Scenario 2: Spike Test (sudden traffic surge)
locust -f tests/load/locustfile.py --host=http://localhost:8000 \\
       --users 500 --spawn-rate 100 --run-time 2m --headless

# Scenario 3: Stress Test (find breaking point)
locust -f tests/load/locustfile.py --host=http://localhost:8000 \\
       --users 1000 --spawn-rate 50 --run-time 30m --headless

# Scenario 4: Endurance Test (sustained load)
locust -f tests/load/locustfile.py --host=http://localhost:8000 \\
       --users 200 --spawn-rate 10 --run-time 4h --headless

# Distributed Testing (multiple machines)
# Master:
locust -f tests/load/locustfile.py --master --expect-workers=3

# Workers (run on separate machines):
locust -f tests/load/locustfile.py --worker --master-host=<master-ip>
"""


# =============================================================================
# Custom Locust Events for Monitoring
# =============================================================================

"""
# Add custom event handlers for monitoring:

from locust import events

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Load test starting...")
    # Send notification to Slack, etc.

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Load test completed")
    # Generate report, send summary

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, **kwargs):
    # Log slow requests
    if response_time > 2000:  # 2 seconds
        print(f"Slow request: {name} took {response_time}ms")
"""
