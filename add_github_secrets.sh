#!/bin/bash
# add_github_secrets.sh
# Adds Notion secrets to GitHub repository using GitHub CLI

set -e

echo "=" | head -c 70 && echo ""
echo "Adding Notion Secrets to GitHub Repository"
echo "=" | head -c 70 && echo ""
echo ""

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) not installed."
    echo ""
    echo "Install with:"
    echo "  brew install gh  # macOS"
    echo "  or visit: https://cli.github.com/"
    echo ""
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "⚠ Not authenticated with GitHub CLI"
    echo "Run: gh auth login"
    exit 1
fi

# Get repository info (format: owner/repo, no .git)
REPO=$(git remote get-url origin 2>/dev/null | sed -E 's/.*github.com[:/]([^/]+\/[^/]+)(\.git)?$/\1/' | sed 's/\.git$//' || echo "")

if [ -z "$REPO" ]; then
    echo "❌ No GitHub remote found."
    echo "Add remote first: git remote add origin <url>"
    exit 1
fi

echo "Repository: $REPO"
echo ""

# Check for .env file
if [ ! -f .env ]; then
    echo "❌ .env file not found"
    echo "Create .env file with NOTION_API_TOKEN and NOTION_VALIDATION_DB_ID"
    exit 1
fi

# Read secrets from .env
source .env

if [ -z "$NOTION_API_TOKEN" ]; then
    echo "❌ NOTION_API_TOKEN not found in .env"
    exit 1
fi

if [ -z "$NOTION_VALIDATION_DB_ID" ]; then
    echo "❌ NOTION_VALIDATION_DB_ID not found in .env"
    exit 1
fi

echo "Found secrets in .env file"
echo ""

# Add secrets
echo "Adding NOTION_API_TOKEN..."
gh secret set NOTION_API_TOKEN --repo "$REPO" --body "$NOTION_API_TOKEN"

echo "Adding NOTION_VALIDATION_DB_ID..."
gh secret set NOTION_VALIDATION_DB_ID --repo "$REPO" --body "$NOTION_VALIDATION_DB_ID"

echo ""
echo "✅ Secrets added successfully!"
echo ""
echo "Verify with: gh secret list --repo $REPO"

