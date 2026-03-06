#!/usr/bin/env bash
# =============================================================================
# One-time setup for git-crypt encryption
#
# Run this after cloning the repo to unlock encrypted .env files.
# Requires: git-crypt, gpg, and your GPG key added to git-crypt.
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[SETUP]${NC} $*"; }
warn() { echo -e "${YELLOW}[SETUP]${NC} $*"; }
err()  { echo -e "${RED}[SETUP]${NC} $*"; }

# Check dependencies
for cmd in git git-crypt gpg; do
    if ! command -v "$cmd" &>/dev/null; then
        err "$cmd is not installed"
        if [ "$cmd" = "git-crypt" ]; then
            echo "  Install: sudo apt-get install git-crypt"
        fi
        exit 1
    fi
done
log "Dependencies OK (git, git-crypt, gpg)"

# Check for GPG key
GPG_EMAIL=$(git config user.email 2>/dev/null || echo "")
if [ -z "$GPG_EMAIL" ]; then
    err "No git user.email configured"
    exit 1
fi

if ! gpg --list-keys "$GPG_EMAIL" &>/dev/null; then
    warn "No GPG key found for $GPG_EMAIL"
    read -p "Generate one now? [Y/n]: " confirm
    if [ "${confirm:-Y}" != "n" ]; then
        gpg --batch --gen-key <<EOF
%no-protection
Key-Type: RSA
Key-Length: 4096
Subkey-Type: RSA
Subkey-Length: 4096
Name-Real: $(git config user.name)
Name-Email: $GPG_EMAIL
Expire-Date: 0
EOF
        log "GPG key generated for $GPG_EMAIL"
    else
        err "GPG key required. Generate one with: gpg --gen-key"
        exit 1
    fi
fi
log "GPG key found for $GPG_EMAIL"

# Unlock the repo
if git-crypt status &>/dev/null; then
    git-crypt unlock 2>/dev/null && log "Repository unlocked — .env files decrypted" || \
        warn "Could not unlock. Your GPG key may not be authorized yet."
else
    warn "git-crypt not initialized in this repo"
fi

# Show status
echo ""
log "Encrypted files:"
git-crypt status -e 2>/dev/null || echo "  (none)"
echo ""
log "Setup complete. Encrypted files are transparent — edit normally, git handles encryption."
