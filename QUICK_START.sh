#!/bin/bash
# Quick Start Script for Pushing Cookiecutter Template

set -e

# Get remote URL if configured
REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "Not configured")

echo "========================================="
echo "Cookiecutter Template Push Script"
echo "========================================="
echo ""
echo "Local Path: $(pwd)"
echo "Branch: $(git branch --show-current)"
echo "Remote: $REMOTE_URL"
echo ""

# Show current status
echo "=== Current Git Status ==="
git status --short
echo ""

# Show commits to be pushed
echo "=== Recent Commits ==="
git log --oneline -5 2>/dev/null || echo "No commits yet"
echo ""

# Show file count
echo "=== File Summary ==="
echo "Total files in git: $(git ls-files | wc -l)"
echo "Analysis documents: $(ls -1 docs-reference/analysis/*.md 2>/dev/null | wc -l || echo "0")"
# shellcheck disable=SC1083
echo "Template files: $(find {{cookiecutter.project_slug}} -type f 2>/dev/null | wc -l || echo "0")"
echo ""

# Check if remote is configured
if [ "$REMOTE_URL" = "Not configured" ]; then
    echo "⚠️  Warning: No remote repository configured!"
    echo "Please run: git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
    echo ""
    read -p "Do you want to continue anyway? (yes/no): " CONFIRM
else
    # Ask for confirmation
    read -p "Ready to push to GitHub? (yes/no): " CONFIRM
fi

if [ "$CONFIRM" = "yes" ]; then
    echo ""
    echo "Pushing to origin/main..."
    if git push -u origin main; then
        echo ""
        echo "========================================="
        echo "✅ Successfully pushed to GitHub!"
        echo "========================================="
        echo ""
        if [ "$REMOTE_URL" != "Not configured" ]; then
            echo "Verify at: $REMOTE_URL"
            echo ""
        fi
        echo "Next steps:"
        echo "1. Enable as template repository (Settings → General → Template repository)"
        echo "2. Add repository topics (cookiecutter, python, template, etc.)"
        echo "3. Test with: cookiecutter gh:YOUR_USERNAME/YOUR_REPO_NAME"
    else
        echo ""
        echo "❌ Push failed. Please check the error message above."
        exit 1
    fi
else
    echo ""
    echo "Push cancelled. Run this script again when ready."
fi
