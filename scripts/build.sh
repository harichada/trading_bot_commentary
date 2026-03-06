#!/usr/bin/env bash
# =============================================================================
# Rudra Trading Engine — Build Script
#
# Called by infra bootstrap after clone/fetch. Handles:
#   1. Version resolution (explicit arg > VERSION file > git tag)
#   2. Auto-generate CHANGELOG.md entry from git commits
#   3. Docker image build with embedded build_info.json
#
# Expected env vars (set by infra bootstrap):
#   INFRA_DIR    Path to infra directory (for Dockerfile, lockfile, src/)
#   BUILD_DIR    Path to cloned trading bot repo
#
# Usage (via infra):
#   make build                          # Build from default branch
#   make build-version V=13.0           # Build with explicit version
# =============================================================================
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
INFRA_DIR="${INFRA_DIR:?INFRA_DIR must be set by bootstrap}"
VERSION="${1:-}"
IMAGE_NAME="rudra-trading-engine"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${GREEN}[BUILD]${NC} $*"; }
warn() { echo -e "${YELLOW}[BUILD]${NC} $*"; }
err()  { echo -e "${RED}[BUILD]${NC} $*"; }

cd "$BUILD_DIR"

# ---------------------------------------------------------------------------
# Resolve version & git metadata
# ---------------------------------------------------------------------------
GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_SHORT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Version: explicit arg > VERSION file > git tag (normalized) > fallback
# Tags like "v12-postgres-risk-fixes" are normalized to "v12.0"
if [ -z "$VERSION" ]; then
    if [ -f VERSION ]; then
        VERSION=$(cat VERSION)
    else
        RAW_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
        if [ -n "$RAW_TAG" ]; then
            NUM=$(echo "$RAW_TAG" | sed 's/^v//' | cut -d'-' -f1)
            case "$NUM" in
                *.*) VERSION="v${NUM}" ;;
                *)   VERSION="v${NUM}.0" ;;
            esac
        else
            VERSION="0.0.0"
        fi
    fi
fi

BUILD_TAG="${GIT_SHORT}"

log "============================================="
log "  Rudra Trading Engine — Build"
log "============================================="
log "  Version:  ${VERSION}"
log "  Git SHA:  ${GIT_SHA}"
log "  Tag:      ${BUILD_TAG}"
log "  Time:     ${BUILD_TIME}"
log "============================================="

# ---------------------------------------------------------------------------
# Auto-update CHANGELOG.md from git commits
# ---------------------------------------------------------------------------
CHANGELOG="${BUILD_DIR}/CHANGELOG.md"
if [ -f "$CHANGELOG" ] && [ "$VERSION" != "0.0.0" ]; then
    if ! grep -q "^## ${VERSION}" "$CHANGELOG" 2>/dev/null; then
        log "Auto-generating CHANGELOG entry for ${VERSION}..."

        PREV_TAG=$(git describe --tags --abbrev=0 HEAD^ 2>/dev/null || echo "")

        if [ -n "$PREV_TAG" ]; then
            COMMITS=$(git log "${PREV_TAG}..HEAD" --pretty=format:'- %s' --no-merges 2>/dev/null || echo "")
        else
            COMMITS=$(git log --pretty=format:'- %s' --no-merges -20 2>/dev/null || echo "")
        fi

        if [ -n "$COMMITS" ]; then
            BUILD_DATE=$(date -u +%Y-%m-%d)
            ENTRY="## ${VERSION} — ${BUILD_DATE}"$'\n'"${COMMITS}"$'\n'

            {
                head -3 "$CHANGELOG"
                echo ""
                echo "$ENTRY"
                tail -n +4 "$CHANGELOG"
            } > "${CHANGELOG}.tmp"
            mv "${CHANGELOG}.tmp" "$CHANGELOG"

            COMMIT_COUNT=$(echo "$COMMITS" | wc -l)
            log "Added ${COMMIT_COUNT} commit(s) to CHANGELOG.md under ${VERSION}"
        else
            warn "No commits found for changelog entry"
        fi
    else
        log "CHANGELOG.md already has entry for ${VERSION}"
    fi
fi

# ---------------------------------------------------------------------------
# Pre-build checks
# ---------------------------------------------------------------------------
if [ ! -f "$BUILD_DIR/gap_fade_app.py" ]; then
    err "gap_fade_app.py not found in ${BUILD_DIR}"
    exit 1
fi

if [ ! -f "$BUILD_DIR/requirements-gap-fade.txt" ]; then
    err "requirements-gap-fade.txt not found"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    err "Docker is not running"
    exit 1
fi

# ---------------------------------------------------------------------------
# Prepare build context
# ---------------------------------------------------------------------------
log "Preparing build context..."

INFRA_DEST="${BUILD_DIR}/infra_src"
rm -rf "$INFRA_DEST"
mkdir -p "$INFRA_DEST"
cp -r "$INFRA_DIR/src/"*.py "$INFRA_DEST/" 2>/dev/null || true

cp "$INFRA_DIR/Dockerfile" "$BUILD_DIR/Dockerfile.rudra"

# ---------------------------------------------------------------------------
# Build the Docker image
# ---------------------------------------------------------------------------
log "Building Docker image..."

docker build \
    -f Dockerfile.rudra \
    --build-arg GIT_SHA="$GIT_SHA" \
    --build-arg BUILD_TIME="$BUILD_TIME" \
    --build-arg VERSION="$VERSION" \
    -t "${IMAGE_NAME}:${BUILD_TAG}" \
    -t "${IMAGE_NAME}:latest" \
    .

if [ "$VERSION" != "0.0.0" ]; then
    docker tag "${IMAGE_NAME}:${BUILD_TAG}" "${IMAGE_NAME}:${VERSION}"
    log "Tagged: ${IMAGE_NAME}:${VERSION}"
fi

# ---------------------------------------------------------------------------
# Extract and save the lockfile from the image
# ---------------------------------------------------------------------------
log "Extracting requirements.lock from image..."
CONTAINER_ID=$(docker create "${IMAGE_NAME}:${BUILD_TAG}")
docker cp "${CONTAINER_ID}:/app/requirements.lock" "$INFRA_DIR/requirements.lock" 2>/dev/null || true
docker rm "$CONTAINER_ID" > /dev/null

if [ -f "$INFRA_DIR/requirements.lock" ]; then
    LOCK_LINES=$(wc -l < "$INFRA_DIR/requirements.lock")
    log "Lockfile saved: ${INFRA_DIR}/requirements.lock (${LOCK_LINES} packages)"
fi

# ---------------------------------------------------------------------------
# Cleanup build artifacts from build directory
# ---------------------------------------------------------------------------
rm -rf "$INFRA_DEST"
rm -f "$BUILD_DIR/Dockerfile.rudra"

# ---------------------------------------------------------------------------
# Verify the image
# ---------------------------------------------------------------------------
log "Verifying build..."

IMAGE_SIZE=$(docker image inspect "${IMAGE_NAME}:${BUILD_TAG}" --format='{{.Size}}' | awk '{printf "%.0f", $1/1024/1024}')
BUILD_INFO=$(docker run --rm "${IMAGE_NAME}:${BUILD_TAG}" cat /app/build_info.json 2>/dev/null || echo "{}")

log ""
log "============================================="
log "  ${BOLD}BUILD SUCCESSFUL${NC}"
log "============================================="
log "  Image:    ${IMAGE_NAME}:${BUILD_TAG}"
if [ "$VERSION" != "0.0.0" ]; then
    log "  Version:  ${IMAGE_NAME}:${VERSION}"
fi
log "  Size:     ${IMAGE_SIZE} MB"
log "  Git SHA:  ${GIT_SHA} (${GIT_SHORT})"
log "  Metadata: ${BUILD_INFO}"
log ""
log "  Next steps:"
log "    make test-image TAG=${BUILD_TAG}    # Test against image"
log "    make deploy-dev TAG=${BUILD_TAG}    # Deploy to dev"
log "    make deploy-sit TAG=${BUILD_TAG}    # Deploy to SIT"
log "    make deploy-prod TAG=${BUILD_TAG}   # Deploy to prod"
log "============================================="

echo "$BUILD_TAG" > "$INFRA_DIR/.last_build_tag"
