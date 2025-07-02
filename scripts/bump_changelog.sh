#!/bin/bash

set -e

FILE="CHANGELOG.md"

# If the changelog file doesn't exist, create it with the standard header
# and an initial "Unreleased" section for the first commit.
if [ ! -f "$FILE" ]; then
  echo "✅ CHANGELOG.md not found. Creating a new one."
  cat <<EOF > "$FILE"
# Changelog

All notable changes to this project will be documented in this file.

## Versioning semantic
MAJOR – Breaking changes
MINOR – New features, but backward-compatible
PATCH – Bug fixes, small improvements

## [Unreleased] - patch
### Added
- Initial setup of CHANGELOG.md
EOF
fi

# Extract the type (major/minor/patch) from [Unreleased] line (case-insensitive)
TYPE=$(grep -i "^\#\# \[Unreleased\]" "$FILE" | grep -oEi "(major|minor|patch)" | head -n1 | tr '[:upper:]' '[:lower:]')

if [ -z "$TYPE" ]; then
  echo "❌ No release type (major/minor/patch) found in [Unreleased] header"
  exit 1
fi

# Get the latest version (e.g., 1.2.3)
LATEST=$(grep -E "^## \[[0-9]+\.[0-9]+\.[0-9]+\]" "$FILE" | head -n1 | grep -oE "[0-9]+\.[0-9]+\.[0-9]+")

# If no previous version is found, default to 0.0.0 for the first bump.
if [ -z "$LATEST" ]; then
  echo "⚠️ No existing version found. Defaulting to 0.0.0 for the first bump."
  LATEST="0.0.0"
fi

IFS='.'
set -- $LATEST
MAJOR=$1; MINOR=$2; PATCH=$3

case "$TYPE" in
  major)
    MAJOR=$((MAJOR + 1))
    MINOR=0
    PATCH=0
    ;;
  minor)
    MINOR=$((MINOR + 1))
    PATCH=0
    ;;
  patch)
    PATCH=$((PATCH + 1))
    ;;
  *)
    echo "❌ Unknown type: $TYPE"
    exit 1
    ;;
esac

NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"
TODAY=$(date +%Y-%m-%d)

# Replace the existing [Unreleased] line with the new version and date
sed -i -E "s/^## \[Unreleased\].*/## [${NEW_VERSION}] - ${TODAY}/I" "$FILE"

# Add a new [Unreleased] section after the semantic versioning header
sed -i '/^PATCH – Bug fixes, small improvements/a \\n## [Unreleased] - TYPE' "$FILE"

echo "✅ Bumped version to $NEW_VERSION"