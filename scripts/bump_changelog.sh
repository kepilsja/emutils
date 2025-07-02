set -e

FILE="CHANGELOG.md"

# Extract the type (major/minor/patch) from [Unreleased] line (case-insensitive)
TYPE=$(grep -i "^\#\# \[Unreleased\]" "$FILE" | grep -oEi "(major|minor|patch)" | head -n1 | tr '[:upper:]' '[:lower:]')

if [ -z "$TYPE" ]; then
  echo "❌ No release type (major/minor/patch) found in [Unreleased] header"
  exit 1
fi

# Get the latest version (e.g., 1.2.3)
LATEST=$(grep -E "^## \[[0-9]+\.[0-9]+\.[0-9]+\]" "$FILE" | head -n1 | grep -oE "[0-9]+\.[0-9]+\.[0-9]+")

if [ -z "$LATEST" ]; then
  echo "❌ No existing version found in changelog"
  exit 1
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

# Replace line like "## [Unreleased] PATCH" with new version
sed -i -E "s/^## \[Unreleased\].*/## [${NEW_VERSION}] - ${TODAY}/I" "$FILE"
sed -i "2i## [Unreleased] - TYPE" "$FILE"

echo "✅ Bumped version to $NEW_VERSION"
