#!/usr/bin/env python3
"""Post-generation hook for cookiecutter Python template.

Performs cleanup and setup tasks after project generation.
Runs after all files have been created.
"""

import shutil
import subprocess
import sys
from pathlib import Path


def remove_file(filepath: Path) -> None:
    """Remove a file if it exists.

    Args:
        filepath: Path to the file to remove
    """
    if filepath.exists():
        filepath.unlink()
        print(f"  ✓ Removed: {filepath}")


def remove_dir(dirpath: Path) -> None:
    """Remove a directory if it exists.

    Args:
        dirpath: Path to the directory to remove
    """
    if dirpath.exists():
        shutil.rmtree(dirpath)
        print(f"  ✓ Removed: {dirpath}/")


def make_executable(filepath: Path) -> None:
    """Make a file executable.

    Args:
        filepath: Path to the file to make executable
    """
    if filepath.exists():
        filepath.chmod(filepath.stat().st_mode | 0o111)


def run_command(cmd: list[str], check: bool = True) -> bool:
    """Run a shell command.

    Args:
        cmd: Command and arguments as list
        check: Whether to check return code

    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.run(cmd, check=check, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False


def cleanup_conditional_files() -> None:
    """Remove files based on cookiecutter choices."""
    print("\n🧹 Cleaning up conditional files...")

    # Remove CLI if not needed
    if "{{ cookiecutter.include_cli }}" == "no":
        remove_file(Path("src/{{ cookiecutter.project_slug }}/cli.py"))

    # Remove MkDocs if not needed
    if "{{ cookiecutter.use_mkdocs }}" == "no":
        remove_file(Path("mkdocs.yml"))
        remove_dir(Path("docs"))
        remove_file(Path("tools/validate_front_matter.py"))
        remove_dir(Path("tools/frontmatter_contract"))

    # Remove Nox if not needed
    if "{{ cookiecutter.include_nox }}" == "no":
        remove_file(Path("noxfile.py"))

    # Remove pre-commit if not needed
    if "{{ cookiecutter.use_pre_commit }}" == "no":
        remove_file(Path(".pre-commit-config.yaml"))

    # Remove CODE_OF_CONDUCT if not needed
    if "{{ cookiecutter.include_code_of_conduct }}" == "no":
        remove_file(Path("CODE_OF_CONDUCT.md"))

    # Remove SECURITY if not needed
    if "{{ cookiecutter.include_security_policy }}" == "no":
        remove_file(Path("SECURITY.md"))

    # Remove CONTRIBUTING if not needed
    if "{{ cookiecutter.include_contributing_guide }}" == "no":
        remove_file(Path("CONTRIBUTING.md"))

    # Remove codecov if not needed
    if "{{ cookiecutter.include_codecov }}" == "no":
        remove_file(Path("codecov.yml"))
        remove_file(Path(".github/workflows/codecov.yml"))

    # Remove SonarCloud if not needed
    if "{{ cookiecutter.include_sonarcloud }}" == "no":
        remove_file(Path("sonar-project.properties"))
        remove_file(Path(".github/workflows/sonarcloud.yml"))

    # Remove renovate if not needed
    if "{{ cookiecutter.include_renovate }}" == "no":
        remove_file(Path("renovate.json"))

    # Remove REUSE if not needed
    if "{{ cookiecutter.use_reuse_licensing }}" == "no":
        remove_file(Path("REUSE.toml"))
        remove_dir(Path("LICENSES"))

    # Remove Docker files if not needed
    if "{{ cookiecutter.include_docker }}" == "no":
        remove_file(Path("Dockerfile"))
        remove_file(Path("docker-compose.yml"))
        remove_file(Path("docker-compose.prod.yml"))
        remove_file(Path(".dockerignore"))

    # Remove health check endpoints if not needed or if no API framework
    if (
        "{{ cookiecutter.include_health_checks }}" == "no"
        or "{{ cookiecutter.include_api_framework }}" == "no"
    ):
        remove_file(Path("src/{{ cookiecutter.project_slug }}/api/health.py"))
        # Remove api directory if empty
        api_dir = Path("src/{{ cookiecutter.project_slug }}/api")
        if api_dir.exists() and not any(
            f.name != "__pycache__" and f.name != "__init__.py" for f in api_dir.iterdir()
        ):
            remove_dir(api_dir)

    # Remove Sentry monitoring if not needed
    if "{{ cookiecutter.include_sentry }}" == "no":
        remove_file(Path("src/{{ cookiecutter.project_slug }}/core/sentry.py"))

    # Remove API security middleware if API framework not included
    if "{{ cookiecutter.include_api_framework }}" == "no":
        remove_file(Path("src/{{ cookiecutter.project_slug }}/middleware/security.py"))
        # Remove middleware directory if empty
        middleware_dir = Path("src/{{ cookiecutter.project_slug }}/middleware")
        if middleware_dir.exists() and not any(middleware_dir.iterdir()):
            remove_dir(middleware_dir)

    # Remove background job files if not needed
    if "{{ cookiecutter.include_background_jobs }}" == "no":
        remove_dir(Path("src/{{ cookiecutter.project_slug }}/jobs"))

    # Remove caching utilities if not needed
    if "{{ cookiecutter.include_caching }}" == "no":
        remove_file(Path("src/{{ cookiecutter.project_slug }}/core/cache.py"))

    # Remove load testing files if not needed
    if "{{ cookiecutter.include_load_testing }}" == "no":
        remove_dir(Path("tests/load"))

    # Remove fuzzing workflow if not needed
    if "{{ cookiecutter.include_fuzzing }}" == "no":
        remove_file(Path(".github/workflows/cifuzzy.yml"))

    # Remove GitHub Actions workflows if not needed
    if "{{ cookiecutter.include_github_actions }}" == "no":
        remove_dir(Path(".github/workflows"))
        remove_dir(Path(".github"))

    # Remove MkDocs workflow if MkDocs not used
    if "{{ cookiecutter.use_mkdocs }}" == "no":
        remove_file(Path(".github/workflows/docs.yml"))

    # Remove security scanning workflow if not needed
    if "{{ cookiecutter.include_security_scanning }}" == "no":
        remove_file(Path(".github/workflows/security-analysis.yml"))


def initialize_git() -> None:
    """Initialize git repository."""
    print("\n🔧 Initializing Git repository...")

    if run_command(["git", "init"], check=False):
        print("  ✓ Git repository initialized")

        # Create initial commit
        if run_command(["git", "add", "."], check=False):
            if run_command(
                ["git", "commit", "-m", "Initial commit from cookiecutter template"],
                check=False,
            ):
                print("  ✓ Initial commit created")
    else:
        print("  ⚠ Git not found - skipping git initialization")


def setup_pre_commit() -> None:
    """Install pre-commit hooks if pre-commit is available."""
    if "{{ cookiecutter.use_pre_commit }}" == "no":
        return

    print("\n🔧 Setting up pre-commit hooks...")

    # Check if pre-commit is installed
    if run_command(["pre-commit", "--version"], check=False):
        if run_command(["pre-commit", "install"], check=False):
            print("  ✓ Pre-commit hooks installed")
        else:
            print("  ⚠ Failed to install pre-commit hooks")
    else:
        print(
            "  ⚠ pre-commit not found - run 'uv sync' and 'uv run pre-commit install'"
        )


def create_initial_directories() -> None:
    """Create additional directories that may be needed."""
    print("\n📁 Creating additional directories...")

    directories = [
        "logs",
        "data",
        "scripts",
        "configs",
    ]

    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        # Create .gitkeep to track empty directories
        (dir_path / ".gitkeep").touch()

    print(f"  ✓ Created {len(directories)} directories")


def setup_claude_user_settings() -> None:
    """Interactively set up user-level Claude Code settings."""
    print("\n🤖 Claude Code User-Level Settings Setup")
    print("=" * 60)

    # Check common locations for existing settings
    possible_locations = [
        Path.home() / ".claude",
        Path.home() / ".config" / "claude",
    ]

    existing_location = None
    for location in possible_locations:
        if location.exists() and (location / "CLAUDE.md").exists():
            existing_location = location
            break

    if existing_location:
        print(f"\n  ℹ User-level settings already exist at: {existing_location}")
        print("    Skipping setup.")
        return

    print("\n  User-level Claude settings provide:")
    print("    • Global CLAUDE.md configuration (best practices, workflows)")
    print("    • Skills (reusable capabilities)")
    print("    • Agents (specialized task handlers)")
    print("    • Custom slash commands and hooks")

    print("\n  These settings enhance Claude Code's capabilities across all projects.")

    # Get user input
    try:
        response = (
            input("\n  Would you like to set up user-level Claude settings? (Y/n): ")
            .strip()
            .lower()
        )
    except (EOFError, KeyboardInterrupt):
        print("\n  Skipping user-level settings setup.")
        return

    # Default to yes if empty response
    if response in ["", "y", "yes"]:
        default_repo = "https://github.com/williaby/.claude"
        default_location = str(Path.home() / ".claude")

        try:
            repo_url = input(
                f"\n  Settings repo URL (press Enter for default: {default_repo}): "
            ).strip()
            if not repo_url:
                repo_url = default_repo

            install_location = input(
                f"  Install location (press Enter for default: {default_location}): "
            ).strip()
            if not install_location:
                install_location = default_location

            install_path = Path(install_location).expanduser()

            # Clone the repo
            print(f"\n  📥 Cloning settings from {repo_url}...")

            if run_command(["git", "clone", repo_url, str(install_path)], check=False):
                print(f"  ✓ User-level settings installed at: {install_path}")

                # Check what was installed
                installed_items = []
                if (install_path / "CLAUDE.md").exists():
                    installed_items.append("CLAUDE.md")
                if (install_path / "skills").exists():
                    installed_items.append("skills/")
                if (install_path / "agents").exists():
                    installed_items.append("agents/")
                if (install_path / ".claude" / "commands").exists() or (
                    install_path / "commands"
                ).exists():
                    installed_items.append("slash commands")

                if installed_items:
                    print(f"  ✓ Installed: {', '.join(installed_items)}")

                print(
                    "\n  ✅ User-level settings are now available to all Claude Code sessions!"
                )
            else:
                print(
                    "  ⚠ Failed to clone settings repo. You can manually set up later:"
                )
                print(f"     git clone {repo_url} {install_path}")

        except (EOFError, KeyboardInterrupt):
            print("\n  Setup cancelled.")
    else:
        print("\n  ℹ Skipping setup. You can set up user-level settings later by:")
        print("     git clone https://github.com/williaby/.claude ~/.claude")


def print_success_message() -> None:
    """Print success message with next steps."""
    project_name = "{{ cookiecutter.project_name }}"
    project_slug = "{{ cookiecutter.project_slug }}"
    use_pre_commit = "{{ cookiecutter.use_pre_commit }}" == "yes"
    use_mkdocs = "{{ cookiecutter.use_mkdocs }}" == "yes"
    include_docker = "{{ cookiecutter.include_docker }}" == "yes"
    include_sentry = "{{ cookiecutter.include_sentry }}" == "yes"
    include_health_checks = "{{ cookiecutter.include_health_checks }}" == "yes"
    include_background_jobs = "{{ cookiecutter.include_background_jobs }}"
    include_caching = "{{ cookiecutter.include_caching }}" == "yes"
    include_load_testing = "{{ cookiecutter.include_load_testing }}" == "yes"

    print("\n" + "=" * 60)
    print(f"🎉 SUCCESS! {project_name} has been created!")
    print("=" * 60)

    # Print optional features that were included
    optional_features = []
    if include_docker:
        optional_features.append("Docker containerization")
    if include_sentry:
        optional_features.append("Sentry monitoring")
    if include_health_checks:
        optional_features.append("Health check endpoints")
    if include_background_jobs != "no":
        optional_features.append(f"Background jobs ({include_background_jobs.upper()})")
    if include_caching:
        optional_features.append("Redis caching")
    if include_load_testing:
        optional_features.append("Load testing (Locust & k6)")

    if optional_features:
        print("\n✨ Optional features included:")
        for feature in optional_features:
            print(f"  • {feature}")

    print("\n📦 Next steps:")
    print("\n  1. Navigate to your project:")
    print(f"     cd {project_slug}")

    print("\n  2. Install dependencies:")
    print("     uv sync --with dev")

    if use_pre_commit:
        print("\n  3. Install pre-commit hooks:")
        print("     uv run pre-commit install")

    print("\n  4. Run tests:")
    print("     uv run pytest -v")

    if use_mkdocs:
        print("\n  5. Build documentation:")
        print("     uv run mkdocs build")

    print("\n  6. Initialize git (if not done automatically):")
    print("     git init")
    print("     git add .")
    print("     git commit -m 'Initial commit'")

    print("\n  7. Create GitHub repository:")
    print(f"     gh repo create {project_slug} --public --source=.")

    # Add next steps for optional features
    if include_docker:
        print("\n  📦 Docker:")
        print("     docker-compose up -d    # Start development environment")
        print("     docker build -t app .   # Build production image")

    if include_background_jobs != "no":
        if include_background_jobs == "arq":
            print("\n  ⚙️  ARQ Worker:")
            print(f"     uv run arq {project_slug}.jobs.worker.WorkerSettings")
        else:
            print("\n  ⚙️  Celery Worker:")
            print(f"     uv run celery -A {project_slug}.jobs worker -l info")

    if include_load_testing:
        print("\n  🚀 Load Testing:")
        print("     uv run locust -f tests/load/locustfile.py  # Start Locust")
        print("     k6 run tests/load/k6-script.js              # Run k6")

    if include_sentry:
        print("\n  🔍 Sentry:")
        print("     Set SENTRY_DSN in .env to enable error tracking")

    print("\n" + "=" * 60)
    print("📚 Documentation:")
    print("  - README.md: Project overview and quick start")
    print("  - CONTRIBUTING.md: Contribution guidelines")
    print("  - CLAUDE.md: Claude Code development guidance")
    if include_load_testing:
        print("  - tests/load/README.md: Load testing guide")
    print("=" * 60 + "\n")


def main() -> None:
    """Run post-generation tasks."""
    print("\n🚀 Running post-generation setup...")

    try:
        cleanup_conditional_files()
        create_initial_directories()
        initialize_git()
        setup_pre_commit()
        setup_claude_user_settings()
        print_success_message()
    except Exception as e:
        print(f"\n❌ Error during post-generation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
