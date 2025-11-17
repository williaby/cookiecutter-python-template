#!/usr/bin/env python
"""---
title: "Setup GitHub Branch Protection"
name: "setup_github_protection.py"
description: "Configure branch protection rules for main branch"
category: script
usage: "GITHUB_TOKEN=ghp_xxx python scripts/setup_github_protection.py"
behavior: "Sets up recommended branch protection rules via GitHub API"
inputs: "GITHUB_TOKEN, GITHUB_REPOSITORY environment variables"
outputs: "Branch protection configuration status"
dependencies: "requests"
author: "{{cookiecutter.author_name}}"
last_modified: "{% now 'utc', '%Y-%m-%d' %}"
changelog: "Initial branch protection setup script"
tags: [scripts, tools, security, branch-protection]
---

Module: setup_github_protection

This script configures GitHub branch protection rules for the main branch
according to OpenSSF security best practices.

Functions:
    setup_branch_protection(): Configure branch protection rules.
    check_existing_protection(): Verify current protection status.
"""

import os
import sys

try:
    import requests
except ImportError:
    print("❌ Error: requests library not found")
    print("Install it with: uv pip install requests")
    sys.exit(1)


def check_existing_protection(
    owner: str, repo: str, branch: str, headers: dict
) -> dict | None:
    """Check if branch protection already exists.

    Args:
        owner: Repository owner (user or organization)
        repo: Repository name
        branch: Branch name (e.g., 'main')
        headers: HTTP headers with authentication

    Returns:
        Current protection rules or None if not protected
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/branches/{branch}/protection"

    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        return None
    else:
        print(f"⚠️  Warning: Could not check existing protection: {response.status_code}")
        return None


def setup_branch_protection(
    owner: str, repo: str, branch: str = "main", token: str | None = None
) -> bool:
    """Configure branch protection rules for a repository.

    Sets up OpenSSF-recommended branch protection including:
    - Required status checks (CI Gate, Security Analysis)
    - Required pull request reviews
    - Admin enforcement
    - Linear history requirement
    - Force push prevention

    Args:
        owner: Repository owner (user or organization)
        repo: Repository name
        branch: Branch to protect (default: main)
        token: GitHub API token with repo scope

    Returns:
        True if successful, False otherwise

    Raises:
        ValueError: If required environment variables are missing
    """
    if not token:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError(
                "GITHUB_TOKEN not found. Set it with: export GITHUB_TOKEN=ghp_xxx"
            )

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Check existing protection
    print(f"\n🔍 Checking current protection for {owner}/{repo}:{branch}...")
    existing = check_existing_protection(owner, repo, branch, headers)

    if existing:
        print("⚠️  Branch protection already exists:")
        print(f"  - Required reviews: {existing.get('required_pull_request_reviews', {}).get('required_approving_review_count', 0)}")
        print(f"  - Dismiss stale reviews: {existing.get('required_pull_request_reviews', {}).get('dismiss_stale_reviews', False)}")
        print(f"  - Enforce for admins: {existing.get('enforce_admins', {}).get('enabled', False)}")
        print("\n❓ Overwrite existing protection? [y/N]: ", end="")
        if input().lower() != "y":
            print("✅ Keeping existing protection")
            return True

    # Branch protection configuration
    # See: https://docs.github.com/en/rest/branches/branch-protection
    protection = {
        "required_status_checks": {
            "strict": True,
            "contexts": [
                "CI Gate",  # From ci.yml workflow
                "Security Analysis",  # From security-analysis.yml
            ],
        },
        "enforce_admins": True,
        "required_pull_request_reviews": {
            "dismissal_restrictions": {},
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": True,
            "required_approving_review_count": 1,
            "require_last_push_approval": False,
        },
        "restrictions": None,  # No push restrictions
        "required_linear_history": True,
        "allow_force_pushes": False,
        "allow_deletions": False,
        "block_creations": False,
        "required_conversation_resolution": True,
        "lock_branch": False,
        "allow_fork_syncing": False,
    }

    url = f"https://api.github.com/repos/{owner}/{repo}/branches/{branch}/protection"

    print(f"\n🔧 Configuring branch protection for {owner}/{repo}:{branch}...")
    print("  Protection Rules:")
    print(f"    ✅ Required status checks: {', '.join(protection['required_status_checks']['contexts'])}")
    print("    ✅ Required pull request reviews: 1")
    print("    ✅ Code owner reviews required")
    print("    ✅ Dismiss stale reviews")
    print("    ✅ Enforce for admins")
    print("    ✅ Linear history required")
    print("    ✅ Force pushes blocked")
    print("    ✅ Branch deletions blocked")
    print("    ✅ Conversation resolution required")

    response = requests.put(
        url, headers=headers, json=protection, timeout=10
    )

    if response.status_code in (200, 201):
        print("\n✅ Branch protection configured successfully!")
        return True
    elif response.status_code == 403:
        print(f"\n❌ Error: Permission denied")
        print("  Make sure your token has 'repo' scope")
        print("  Admins: You may need to temporarily enable 'Include administrators'")
        return False
    elif response.status_code == 404:
        print(f"\n❌ Error: Repository or branch not found")
        print(f"  Check that {owner}/{repo}:{branch} exists")
        return False
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(f"  Response: {response.text}")
        return False


def main() -> None:
    """Main entry point for branch protection setup."""
    print("=" * 70)
    print("🔐 GitHub Branch Protection Setup")
    print("=" * 70)

    # Get repository info from environment or user input
    repo_full = os.environ.get("GITHUB_REPOSITORY")

    if not repo_full:
        print("\n📝 Enter repository information:")
        owner = input("  Owner (user/org): ").strip()
        repo = input("  Repository name: ").strip()
    else:
        owner, repo = repo_full.split("/")
        print(f"\n📦 Repository: {owner}/{repo}")

    branch = input("  Branch to protect [main]: ").strip() or "main"

    try:
        success = setup_branch_protection(owner, repo, branch)

        if success:
            print("\n" + "=" * 70)
            print("✅ Setup Complete!")
            print("=" * 70)
            print("\n📚 Next Steps:")
            print("  1. Verify protection at:")
            print(f"     https://github.com/{owner}/{repo}/settings/branches")
            print("  2. Update status check names if your workflows differ")
            print("  3. Consider adding rulesets for additional protection")
            print("\n💡 To modify protection rules:")
            print("  1. Edit this script's 'protection' dictionary")
            print("  2. Re-run the script")
            print("\n🔗 Documentation:")
            print("  https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches")
            sys.exit(0)
        else:
            print("\n❌ Setup failed. See errors above.")
            sys.exit(1)

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Setup your GitHub token:")
        print("  1. Go to https://github.com/settings/tokens")
        print("  2. Generate new token (classic)")
        print("  3. Select 'repo' scope")
        print("  4. Export token: export GITHUB_TOKEN=ghp_xxx")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
