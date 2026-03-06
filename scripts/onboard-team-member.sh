#!/usr/bin/env bash
# =============================================================================
# Add a team member to git-crypt
#
# Usage:
#   ./scripts/onboard-team-member.sh member@email.com
#   ./scripts/onboard-team-member.sh member@email.com /path/to/their-public-key.gpg
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[ONBOARD]${NC} $*"; }
err() { echo -e "${RED}[ONBOARD]${NC} $*"; }

if [ $# -lt 1 ]; then
    echo "Usage: $0 <email> [public-key-file.gpg]"
    exit 1
fi

EMAIL="$1"
KEY_FILE="${2:-}"

# Import their public key if provided
if [ -n "$KEY_FILE" ]; then
    if [ ! -f "$KEY_FILE" ]; then
        err "Key file not found: $KEY_FILE"
        exit 1
    fi
    gpg --import "$KEY_FILE"
    log "Imported GPG public key from $KEY_FILE"
fi

# Verify key exists
if ! gpg --list-keys "$EMAIL" &>/dev/null; then
    err "No GPG public key found for $EMAIL"
    echo "  Ask them to export their key:"
    echo "    gpg --armor --export $EMAIL > their-key.gpg"
    echo "  Then run:"
    echo "    $0 $EMAIL their-key.gpg"
    exit 1
fi

# Add to git-crypt
git-crypt add-gpg-user "$EMAIL"
log "Added $EMAIL to git-crypt"
log "Commit and push to give them access"
echo ""
echo "Tell them to run after pulling:"
echo "  ./scripts/setup-encryption.sh"
