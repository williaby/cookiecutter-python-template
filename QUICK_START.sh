#!/bin/bash
# Quick Start Script for Pushing Cookiecutter Template

set -e

echo "========================================="
echo "Cookiecutter Template Push Script"
echo "========================================="
echo ""
echo "Repository: https://github.com/williaby/cookiecutter-python-template-2"
echo "Local Path: $(pwd)"
echo "Branch: $(git branch --show-current)"
echo ""

# Show current status
echo "=== Current Git Status ==="
git status --short
echo ""

# Show commits to be pushed
echo "=== Commits to be Pushed ==="
git log --oneline -5
echo ""

# Show file count
echo "=== File Summary ==="
echo "Total files in git: $(git ls-files | wc -l)"
echo "Analysis documents: $(ls -1 docs/analysis/*.md 2>/dev/null | wc -l)"
echo "Template files: $(find {{cookiecutter.project_slug}} -type f 2>/dev/null | wc -l)"
echo ""

# Ask for confirmation
read -p "Ready to push to GitHub? (yes/no): " CONFIRM

if [ "$CONFIRM" = "yes" ]; then
    echo ""
    echo "Pushing to origin/main..."
    git push -u origin main
    
    echo ""
    echo "========================================="
    echo "✅ Successfully pushed to GitHub!"
    echo "========================================="
    echo ""
    echo "Verify at: https://github.com/williaby/cookiecutter-python-template-2"
    echo ""
    echo "Next steps:"
    echo "1. Enable as template repository (Settings → General → Template repository)"
    echo "2. Add repository topics (cookiecutter, python, template, etc.)"
    echo "3. Test with: cookiecutter gh:williaby/cookiecutter-python-template-2"
else
    echo ""
    echo "Push cancelled. Run this script again when ready."
fi
