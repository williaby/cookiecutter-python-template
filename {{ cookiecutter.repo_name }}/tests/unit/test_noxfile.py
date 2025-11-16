"""Comprehensive tests for noxfile.py sessions and functions."""

# Import the noxfile module
import contextlib
import sys
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest
from nox import Session

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import noxfile


class TestConstants:
    """Test noxfile constants and module-level variables."""

    def test_python_versions_defined(self):
        """Test that PYTHON_VERSIONS is properly defined."""
        assert hasattr(noxfile, "PYTHON_VERSIONS")
        assert isinstance(noxfile.PYTHON_VERSIONS, list)
        assert "3.11" in noxfile.PYTHON_VERSIONS
        assert "3.12" in noxfile.PYTHON_VERSIONS

    def test_src_locations_defined(self):
        """Test that SRC_LOCATIONS is properly defined."""
        assert hasattr(noxfile, "SRC_LOCATIONS")
        assert isinstance(noxfile.SRC_LOCATIONS, list)
        assert "src" in noxfile.SRC_LOCATIONS
        assert "tests" in noxfile.SRC_LOCATIONS
        assert "noxfile.py" in noxfile.SRC_LOCATIONS
        assert "scripts" in noxfile.SRC_LOCATIONS


class TestBasicSessions:
    """Test basic nox sessions."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock nox session."""
        session = Mock(spec=Session)
        session.posargs = []
        session.env = {}
        return session

    def test_tests_session_default_args(self, mock_session):
        """Test tests session with default arguments."""
        noxfile.tests(mock_session)

        # Verify poetry install was called
        mock_session.run.assert_any_call("poetry", "install", "--with", "dev", external=True)

        # Verify pytest was called with default coverage args (updated implementation)
        mock_session.run.assert_any_call(
            "pytest",
            "--cov",
            "--cov-branch",
            "--cov-report=term-missing",
            "--cov-fail-under=80",
        )

    def test_tests_session_custom_args(self, mock_session):
        """Test tests session with custom arguments."""
        mock_session.posargs = ["tests/unit", "-v"]

        noxfile.tests(mock_session)

        # Verify pytest was called with custom args
        mock_session.run.assert_any_call("pytest", "tests/unit", "-v")

    def test_tests_unit_session(self, mock_session):
        """Test tests_unit session functionality."""
        noxfile.tests_unit(mock_session)

        # Verify poetry install
        mock_session.run.assert_any_call("poetry", "install", "--with", "dev", external=True)

        # Verify pytest with unit test args (updated implementation)
        mock_session.run.assert_any_call(
            "pytest",
            "-m",
            "unit",
            "--cov=src",
            "--cov-branch",
            "--cov-report=xml:coverage-unit.xml",
            "--cov-report=json:coverage-unit.json",
            "--cov-report=term-missing",
            "-v",
        )

    def test_tests_unit_with_codecov_token(self, mock_session):
        """Test tests_unit session with CODECOV_TOKEN present."""
        mock_session.env = {"CODECOV_TOKEN": "test-token"}

        noxfile.tests_unit(mock_session)

        # Verify codecov upload was called (updated implementation)
        mock_session.run.assert_any_call(
            "codecov",
            "-f",
            "coverage-unit.xml",
            "-F",
            "unit",
            "-n",
            "unit-tests",
            external=True,
        )

    def test_tests_unit_without_codecov_token(self, mock_session):
        """Test tests_unit session without CODECOV_TOKEN."""
        mock_session.env = {}

        noxfile.tests_unit(mock_session)

        # Verify codecov upload was NOT called
        codecov_calls = [call for call in mock_session.run.call_args_list if "codecov" in str(call)]
        assert len(codecov_calls) == 0

    def test_tests_integration_session(self, mock_session):
        """Test tests_integration session functionality."""
        noxfile.tests_integration(mock_session)

        # Verify poetry install
        mock_session.run.assert_any_call("poetry", "install", "--with", "dev", external=True)

        # Verify pytest with integration test args (updated implementation)
        mock_session.run.assert_any_call(
            "pytest",
            "-m",
            "integration",
            "--cov=src",
            "--cov-branch",
            "--cov-report=xml:coverage-integration.xml",
            "--cov-report=json:coverage-integration.json",
            "--cov-report=term-missing",
            "-v",
        )

    def test_tests_integration_with_codecov_token(self, mock_session):
        """Test tests_integration session with CODECOV_TOKEN present."""
        mock_session.env = {"CODECOV_TOKEN": "test-token"}

        noxfile.tests_integration(mock_session)

        # Verify codecov upload was called (updated implementation)
        mock_session.run.assert_any_call(
            "codecov",
            "-f",
            "coverage-integration.xml",
            "-F",
            "integration",
            "-n",
            "integration-tests",
            external=True,
        )


class TestLintingAndFormatting:
    """Test linting and formatting sessions."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock nox session."""
        session = Mock(spec=Session)
        session.posargs = []
        return session

    def test_lint_session_default_args(self, mock_session):
        """Test lint session with default arguments."""
        noxfile.lint(mock_session)

        # Verify poetry install
        mock_session.run.assert_any_call("poetry", "install", "--with", "dev", external=True)

        # Verify all linting tools are called
        mock_session.run.assert_any_call("black", "--check", *noxfile.SRC_LOCATIONS)
        mock_session.run.assert_any_call("ruff", "check", *noxfile.SRC_LOCATIONS)
        mock_session.run.assert_any_call("markdownlint", "**/*.md", external=True)
        mock_session.run.assert_any_call("yamllint", ".", external=True)

    def test_lint_session_custom_args(self, mock_session):
        """Test lint session with custom arguments."""
        custom_args = ["src/specific_module.py"]
        mock_session.posargs = custom_args

        noxfile.lint(mock_session)

        # Verify linting tools are called with custom args
        mock_session.run.assert_any_call("black", "--check", *custom_args)
        mock_session.run.assert_any_call("ruff", "check", *custom_args)

    def test_type_check_session(self, mock_session):
        """Test type_check session functionality."""
        noxfile.type_check(mock_session)

        # Verify poetry install and mypy execution
        mock_session.run.assert_any_call("poetry", "install", "--with", "dev", external=True)
        mock_session.run.assert_any_call("mypy", "src")

    def test_format_code_session_default_args(self, mock_session):
        """Test format_code session with default arguments."""
        noxfile.format_code(mock_session)

        # Verify poetry install
        mock_session.run.assert_any_call("poetry", "install", "--with", "dev", external=True)

        # Verify formatting tools are called
        mock_session.run.assert_any_call("black", *noxfile.SRC_LOCATIONS)
        mock_session.run.assert_any_call("ruff", "check", "--fix", *noxfile.SRC_LOCATIONS)

    def test_format_code_session_custom_args(self, mock_session):
        """Test format_code session with custom arguments."""
        custom_args = ["src/specific_module.py"]
        mock_session.posargs = custom_args

        noxfile.format_code(mock_session)

        # Verify formatting tools are called with custom args
        mock_session.run.assert_any_call("black", *custom_args)
        mock_session.run.assert_any_call("ruff", "check", "--fix", *custom_args)


class TestSecuritySessions:
    """Test security-related sessions."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock nox session."""
        return Mock(spec=Session)

    def test_security_session(self, mock_session):
        """Test security session functionality."""
        noxfile.security(mock_session)

        # Verify poetry install
        mock_session.run.assert_any_call("poetry", "install", "--with", "dev", external=True)

        # Verify all security tools are called
        mock_session.run.assert_any_call("safety", "check", "--json")
        mock_session.run.assert_any_call("bandit", "-r", "src", "-ll")
        mock_session.run.assert_any_call("detect-secrets", "scan", "--baseline", ".secrets.baseline")


class TestDocumentationSessions:
    """Test documentation-related sessions."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock nox session."""
        return Mock(spec=Session)

    def test_docs_session(self, mock_session):
        """Test docs session functionality."""
        noxfile.docs(mock_session)

        # Verify poetry install
        mock_session.run.assert_any_call("poetry", "install", "--with", "dev", external=True)

        # Verify directory change and mkdocs build
        mock_session.cd.assert_called_once_with("docs")
        mock_session.run.assert_any_call("mkdocs", "build")


class TestDependencyManagement:
    """Test dependency management sessions."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock nox session."""
        return Mock(spec=Session)

    @patch("noxfile.Path")
    def test_deps_session(self, mock_path, mock_session):
        """Test deps session functionality."""
        # Mock temporary directory creation with context manager
        temp_dir = "/tmp/test"  # noqa: S108
        mock_session.create_tmp.return_value = temp_dir

        # Create a proper context manager mock
        class MockContextManager:
            def __enter__(self):
                return temp_dir

            def __exit__(self, *args):
                return None

        mock_session.chdir.return_value = MockContextManager()

        # Mock Path.cwd() to return a mock path
        mock_cwd = Mock()
        mock_cwd.parent = Path("/test/parent")
        mock_path.cwd.return_value = mock_cwd

        noxfile.deps(mock_session)

        # Verify poetry commands
        mock_session.run.assert_any_call("poetry", "install", external=True)
        mock_session.run.assert_any_call("poetry", "show", "--outdated")
        mock_session.run.assert_any_call("./scripts/generate_requirements.sh", external=True)

        # Verify temporary environment creation and testing
        mock_session.create_tmp.assert_called_once()
        mock_session.run.assert_any_call("python", "-m", "venv", "test-env")


class TestPreCommitSession:
    """Test pre-commit session."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock nox session."""
        return Mock(spec=Session)

    def test_pre_commit_session(self, mock_session):
        """Test pre_commit session functionality."""
        noxfile.pre_commit(mock_session)

        # Verify poetry install and pre-commit execution
        mock_session.run.assert_any_call("poetry", "install", "--with", "dev", external=True)
        mock_session.run.assert_any_call("pre-commit", "run", "--all-files")


class TestAdvancedTestingSessions:
    """Test advanced testing sessions."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock nox session."""
        session = Mock(spec=Session)
        session.log = Mock()
        session.error = Mock()
        return session

    def test_mutation_testing_session_success(self, mock_session):
        """Test mutation_testing session successful execution."""
        noxfile.mutation_testing(mock_session)

        # Verify poetry install
        mock_session.run.assert_any_call("poetry", "install", "--with", "dev", external=True)

        # Verify cache cleanup
        mock_session.run.assert_any_call("rm", "-rf", ".mutmut-cache", external=True, success_codes=[0, 1])

        # Verify mutmut execution
        expected_mutmut_call = call(
            "mutmut",
            "run",
            "--paths-to-mutate",
            "src/core/,src/agents/,src/config/",
            "--test-time-multiplier",
            "2.0",
            "--runner",
            "python -m pytest tests/unit/ -x --disable-warnings",
            external=True,
        )
        assert expected_mutmut_call in mock_session.run.call_args_list

        # Verify report generation
        mock_session.run.assert_any_call("mutmut", "html", external=True)
        mock_session.run.assert_any_call("mutmut", "show", external=True)

    def test_mutation_testing_session_with_exception(self, mock_session):
        """Test mutation_testing session handles exceptions gracefully."""

        # Make mutmut run raise an exception
        def side_effect(*args, **kwargs):
            if "mutmut" in args and "run" in args:
                raise Exception("Mutation testing failed")

        mock_session.run.side_effect = side_effect

        noxfile.mutation_testing(mock_session)

        # Verify error handling logs
        mock_session.log.assert_any_call("⚠️ Mutation testing encountered issues: Mutation testing failed")
        mock_session.log.assert_any_call("✅ Mutation testing completed with warnings")

    def test_contract_testing_session(self, mock_session):
        """Test contract_testing session functionality."""
        noxfile.contract_testing(mock_session)

        # Verify poetry install and pytest execution
        mock_session.run.assert_any_call("poetry", "install", "--with", "dev", external=True)
        mock_session.run.assert_any_call("pytest", "tests/contract/", "-v")


class TestDASTScanning:
    """Test DAST scanning session."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock nox session."""
        session = Mock(spec=Session)
        session.log = Mock()
        session.error = Mock()
        session.env = {"PWD": "/test/pwd"}
        return session

    def test_dast_scanning_session_success(self, mock_session):
        """Test dast_scanning session successful execution."""
        noxfile.dast_scanning(mock_session)

        # Verify poetry install
        mock_session.run.assert_any_call("poetry", "install", "--with", "dev", external=True)

        # Verify Docker version check
        mock_session.run.assert_any_call("docker", "--version", external=True, silent=True)

        # Verify application health check
        mock_session.run.assert_any_call("curl", "-f", "http://localhost:7860/health", external=True, silent=True)

        # Verify reports directory creation
        mock_session.run.assert_any_call("mkdir", "-p", "dast-reports", external=True, success_codes=[0, 1])

        # Verify OWASP ZAP execution (check for key arguments)
        zap_calls = [call for call in mock_session.run.call_args_list if "owasp/zap2docker-stable" in str(call)]
        assert len(zap_calls) > 0

    def test_dast_scanning_docker_not_available(self, mock_session):
        """Test dast_scanning session when Docker is not available."""

        # Make Docker check fail
        def side_effect(*args, **kwargs):
            if "docker" in args and "--version" in args:
                raise Exception("Docker not found")

        mock_session.run.side_effect = side_effect

        noxfile.dast_scanning(mock_session)

        # Verify error handling
        mock_session.error.assert_called_once_with("Docker is not available. DAST scanning requires Docker.")

    def test_dast_scanning_application_not_running(self, mock_session):
        """Test dast_scanning session when application is not running."""

        # Make health check fail but docker check succeed, and handle other calls normally
        def side_effect(*args, **kwargs):
            if "curl" in args and "health" in str(args):
                raise Exception("Connection refused")
            if "docker" in args and "--version" in args:
                return  # Docker check succeeds
            if "mkdir" in args:
                return  # Directory creation succeeds
            return  # Other calls succeed

        mock_session.run.side_effect = side_effect

        noxfile.dast_scanning(mock_session)

        # Verify warning message was logged
        warning_calls = [
            call
            for call in mock_session.log.call_args_list
            if "⚠️ Warning: Could not verify application is running" in str(call)
        ]
        assert len(warning_calls) >= 1

    def test_dast_scanning_with_exception(self, mock_session):
        """Test dast_scanning session handles Docker execution exceptions."""

        # Make ZAP execution fail
        def side_effect(*args, **kwargs):
            if "owasp/zap2docker-stable" in str(args):
                raise Exception("ZAP execution failed")

        mock_session.run.side_effect = side_effect

        noxfile.dast_scanning(mock_session)

        # Verify error handling
        mock_session.log.assert_any_call("⚠️ DAST scanning encountered issues: ZAP execution failed")
        mock_session.log.assert_any_call("✅ DAST scanning completed with warnings")


class TestPerformanceTesting:
    """Test performance testing session."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock nox session."""
        session = Mock(spec=Session)
        session.log = Mock()
        return session

    def test_performance_testing_session(self, mock_session):
        """Test performance_testing session functionality."""
        noxfile.performance_testing(mock_session)

        # Verify poetry install
        mock_session.run.assert_any_call("poetry", "install", "--with", "dev", external=True)

        # Verify locust execution with correct parameters
        mock_session.run.assert_any_call(
            "locust",
            "-f",
            "tests/performance/locustfile.py",
            "--host=http://localhost:7860",
            "--headless",
            "--users",
            "10",
            "--spawn-rate",
            "2",
            "--run-time",
            "30s",
        )

        # Verify log message
        mock_session.log.assert_called_with("Starting performance testing - ensure application is running")


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock nox session."""
        session = Mock(spec=Session)
        session.log = Mock()
        session.error = Mock()
        session.env = {}
        return session

    def test_session_with_run_exceptions(self, mock_session):
        """Test sessions handle run() exceptions gracefully."""
        # Make run() raise an exception
        mock_session.run.side_effect = Exception("Command failed")

        # Test that sessions don't crash on run failures
        with pytest.raises(Exception, match="Command failed"):
            noxfile.tests(mock_session)

    def test_empty_posargs_handling(self, mock_session):
        """Test sessions handle empty posargs correctly."""
        mock_session.posargs = []

        # Should not raise any exceptions
        noxfile.tests(mock_session)
        noxfile.lint(mock_session)
        noxfile.format_code(mock_session)

    def test_none_posargs_handling(self, mock_session):
        """Test sessions handle None posargs correctly."""
        mock_session.posargs = None

        # Should not raise any exceptions and use defaults
        noxfile.tests(mock_session)

    @patch("noxfile.Path")
    def test_deps_session_path_errors(self, mock_path, mock_session):
        """Test deps session handles Path errors gracefully."""
        # Mock Path.cwd() to raise an exception
        mock_path.cwd.side_effect = Exception("Path error")
        mock_session.create_tmp.return_value = "/tmp/test"  # noqa: S108

        # Create a proper context manager mock for chdir
        class MockContextManager:
            def __enter__(self):
                return "/tmp/test"  # noqa: S108

            def __exit__(self, *args):
                return None

        mock_session.chdir.return_value = MockContextManager()

        # Should handle the path error without crashing
        with pytest.raises(Exception, match="Path error"):
            noxfile.deps(mock_session)


class TestSessionIntegration:
    """Test session integration and workflow scenarios."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock nox session."""
        session = Mock(spec=Session)
        session.posargs = []
        session.env = {"CODECOV_TOKEN": "test-token", "PWD": "/test/pwd"}
        session.log = Mock()
        return session

    def test_full_testing_workflow(self, mock_session):
        """Test a complete testing workflow across multiple sessions."""
        # Run tests session
        noxfile.tests(mock_session)

        # Run unit tests
        noxfile.tests_unit(mock_session)

        # Run integration tests
        noxfile.tests_integration(mock_session)

        # Verify poetry install was called multiple times
        install_calls = [
            call for call in mock_session.run.call_args_list if "poetry" in call[0] and "install" in call[0]
        ]
        assert len(install_calls) >= 3

    def test_full_quality_workflow(self, mock_session):
        """Test a complete quality check workflow."""
        # Run linting
        noxfile.lint(mock_session)

        # Run type checking
        noxfile.type_check(mock_session)

        # Run security checks
        noxfile.security(mock_session)

        # Verify all quality tools were executed
        run_args = [str(call) for call in mock_session.run.call_args_list]
        assert any("black" in arg for arg in run_args)
        assert any("ruff" in arg for arg in run_args)
        assert any("mypy" in arg for arg in run_args)
        assert any("safety" in arg for arg in run_args)
        assert any("bandit" in arg for arg in run_args)


@pytest.mark.parametrize(
    ("session_function", "expected_installs"),
    [
        (noxfile.tests, 1),
        (noxfile.tests_unit, 1),
        (noxfile.tests_integration, 1),
        (noxfile.lint, 1),
        (noxfile.type_check, 1),
        (noxfile.security, 1),
        (noxfile.format_code, 1),
        (noxfile.docs, 1),
        (noxfile.pre_commit, 1),
        (noxfile.mutation_testing, 1),
        (noxfile.contract_testing, 1),
        (noxfile.dast_scanning, 1),
        (noxfile.performance_testing, 1),
    ],
)
def test_all_sessions_install_dependencies(session_function, expected_installs):
    """Test that all sessions properly install dependencies."""
    mock_session = Mock(spec=Session)
    mock_session.posargs = []
    mock_session.env = {}
    mock_session.log = Mock()
    mock_session.error = Mock()
    mock_session.create_tmp = Mock(return_value="/tmp/test")  # noqa: S108

    # Handle exceptions for sessions that might fail
    with contextlib.suppress(Exception):
        session_function(mock_session)

    # Verify poetry install was called
    install_calls = [call for call in mock_session.run.call_args_list if "poetry" in call[0] and "install" in call[0]]
    assert len(install_calls) >= expected_installs


class TestDASTSecurityReportGeneration:
    """Test DAST security report generation functionality."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock nox session."""
        session = Mock(spec=Session)
        session.log = Mock()
        session.error = Mock()
        session.env = {"PWD": "/test/pwd"}
        return session

    def test_dast_security_report_generation(self, mock_session):
        """Test DAST security report generation python code execution."""
        # Mock successful execution up to report generation
        mock_session.run.side_effect = None

        noxfile.dast_scanning(mock_session)

        # Verify that the Python code for security summary generation was called
        python_calls = [call for call in mock_session.run.call_args_list if call[0][0] == "python" and "-c" in call[0]]
        assert len(python_calls) >= 1

        # Verify the Python code contains the expected security summary logic
        python_code_call = next((call for call in python_calls if "generate_security_summary" in str(call)), None)
        assert python_code_call is not None


class TestDASTReportDirectoryCreation:
    """Test DAST reports directory creation."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock nox session."""
        session = Mock(spec=Session)
        session.log = Mock()
        session.error = Mock()
        session.env = {"PWD": "/test/pwd"}
        return session

    def test_dast_reports_directory_creation(self, mock_session):
        """Test that DAST reports directory is created."""
        noxfile.dast_scanning(mock_session)

        # Verify reports directory creation
        mock_session.run.assert_any_call("mkdir", "-p", "dast-reports", external=True, success_codes=[0, 1])


class TestAdvancedMutationTestingErrorHandling:
    """Test advanced mutation testing error handling scenarios."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock nox session."""
        session = Mock(spec=Session)
        session.log = Mock()
        session.error = Mock()
        return session

    def test_mutation_testing_partial_report_generation(self, mock_session):
        """Test mutation testing generates partial reports on failure."""

        # Make mutmut run fail but allow report generation
        def side_effect(*args, **kwargs):
            if "mutmut" in args and "run" in args:
                raise Exception("Mutation testing failed")
            if "mutmut" in args and "html" in args:
                return  # HTML report generation succeeds
            return

        mock_session.run.side_effect = side_effect

        noxfile.mutation_testing(mock_session)

        # Verify partial report generation with success codes
        mock_session.run.assert_any_call("mutmut", "html", external=True, success_codes=[0, 1])


class TestComplexScenarios:
    """Test complex scenarios and real-world usage patterns."""

    @pytest.fixture
    def mock_session_with_env(self):
        """Create a mock session with realistic environment variables."""
        session = Mock(spec=Session)
        session.env = {
            "CODECOV_TOKEN": "test-codecov-token",
            "PWD": "/home/user/promptcraft",
            "CI": "true",
            "GITHUB_ACTIONS": "true",
        }
        session.posargs = []
        session.log = Mock()
        session.error = Mock()
        return session

    def test_ci_environment_behavior(self, mock_session_with_env):
        """Test behavior in CI environment with full environment variables."""
        # Test unit tests with codecov upload
        noxfile.tests_unit(mock_session_with_env)

        # Verify codecov upload happened
        codecov_calls = [call for call in mock_session_with_env.run.call_args_list if "codecov" in str(call)]
        assert len(codecov_calls) == 1

    def test_local_development_behavior(self):
        """Test behavior in local development environment."""
        mock_session = Mock(spec=Session)
        mock_session.env = {}  # No CI environment variables
        mock_session.posargs = []
        mock_session.log = Mock()

        # Test unit tests without codecov upload
        noxfile.tests_unit(mock_session)

        # Verify no codecov upload happened
        codecov_calls = [call for call in mock_session.run.call_args_list if "codecov" in str(call)]
        assert len(codecov_calls) == 0

    def test_custom_arguments_propagation(self):
        """Test that custom arguments are properly propagated."""
        mock_session = Mock(spec=Session)
        mock_session.posargs = ["--no-cov", "tests/specific/", "-k", "test_pattern"]

        noxfile.tests(mock_session)

        # Verify custom args were passed to pytest
        mock_session.run.assert_any_call("pytest", "--no-cov", "tests/specific/", "-k", "test_pattern")

    @patch("tempfile.mkdtemp")
    def test_temporary_directory_usage(self, mock_mkdtemp):
        """Test sessions that use temporary directories."""
        mock_session = Mock(spec=Session)
        temp_dir = "/tmp/nox-test-123"  # noqa: S108
        mock_session.create_tmp.return_value = temp_dir

        # Create a proper context manager mock for chdir
        class MockContextManager:
            def __enter__(self):
                return temp_dir

            def __exit__(self, *args):
                return None

        mock_session.chdir.return_value = MockContextManager()
        mock_session.env = {"PWD": "/test/project"}

        # Mock Path for deps session
        with patch("noxfile.Path") as mock_path:
            mock_cwd = Mock()
            mock_cwd.parent = Path("/test/project")
            mock_path.cwd.return_value = mock_cwd

            noxfile.deps(mock_session)

            # Verify temporary directory was created and used
            mock_session.create_tmp.assert_called_once()
            mock_session.chdir.assert_called_once()
