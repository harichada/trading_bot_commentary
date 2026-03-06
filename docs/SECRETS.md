# Secret Management — git-crypt

Encrypted `.env` files live in the repo. git-crypt encrypts them on commit and decrypts on checkout — transparent to your workflow.

## How It Works

```
You (local)          Git repo (GitHub)         Teammate (local)
─────────────        ──────────────────        ─────────────────
.env_dev (plaintext) → git push → .env_dev (AES-256 encrypted) → git pull → .env_dev (plaintext)
                                  ↑ unreadable without GPG key
```

- **Encryption**: AES-256-GCM (via git-crypt 0.7+)
- **Key management**: GPG — each authorized user has their own key
- **Encrypted files**: `.env`, `.env_*`, `*.key`, `secrets/**` (defined in `.gitattributes`)

## Quick Start (New Machine)

```bash
# 1. Install dependencies
sudo apt-get install git-crypt gpg

# 2. Import your GPG private key (from backup)
gpg --import my-private-key.gpg

# 3. Clone and unlock
git clone git@github.com:harichada/trading_bot_commentary.git
cd trading_bot_commentary
git-crypt unlock

# Or use the setup script:
./scripts/setup-encryption.sh
```

After unlocking, `.env_dev` and other encrypted files appear as plaintext. No further steps needed.

## Daily Use

There is nothing special to do. Edit `.env` files normally:

```bash
nano .env_dev                    # Edit in plaintext
git add .env_dev                 # Encrypted automatically on stage
git commit -m "Update API keys"  # Stored encrypted
git push                         # Pushed encrypted to GitHub
```

Verify a file is encrypted in git:

```bash
# Shows GITCRYPT binary header (encrypted in object store)
git show HEAD:.env_dev | head -1

# Shows encryption status of all files
git-crypt status -e
```

## Onboarding a Team Member

### Prerequisites
The new member needs a GPG key. If they don't have one:

```bash
# They run this on their machine:
gpg --gen-key
gpg --armor --export their@email.com > my-public-key.gpg
# They send you my-public-key.gpg (public key only — safe to share)
```

### Add Them

```bash
# Option A: Use the script
./scripts/onboard-team-member.sh their@email.com /path/to/their-public-key.gpg

# Option B: Manual
gpg --import their-public-key.gpg
git-crypt add-gpg-user their@email.com
git push
```

### They Pull and Unlock

```bash
git pull
git-crypt unlock
# Done — .env files are now plaintext for them
```

## GPG Key Backup

**Critical**: If you lose your GPG key, you cannot decrypt the repo. Back it up.

```bash
# Export private key (keep this safe!)
gpg --armor --export-secret-keys harikishorereddy@gmail.com > gpg-private-key.gpg

# Export public key (safe to share)
gpg --armor --export harikishorereddy@gmail.com > gpg-public-key.gpg

# Store gpg-private-key.gpg in:
#   - Password manager (1Password, Bitwarden)
#   - Encrypted USB drive
#   - NOT in the git repo
```

Restore on a new machine:

```bash
gpg --import gpg-private-key.gpg
git-crypt unlock
```

## git-crypt Symmetric Key Backup

Alternative recovery method — export the raw git-crypt key (not tied to any GPG key):

```bash
# Export (store securely alongside GPG backup)
git-crypt export-key /path/to/git-crypt-key.bin

# Unlock with symmetric key (no GPG needed)
git-crypt unlock /path/to/git-crypt-key.bin
```

## CI/CD Integration (GitHub Actions)

Store the symmetric git-crypt key as a GitHub secret for automated deployments:

```bash
# 1. Export and base64-encode the key
git-crypt export-key /tmp/gc-key.bin
base64 < /tmp/gc-key.bin  # Copy this output
rm /tmp/gc-key.bin

# 2. Add as GitHub secret named GIT_CRYPT_KEY
#    Settings → Secrets → Actions → New repository secret
```

In your workflow:

```yaml
- name: Decrypt secrets
  run: |
    echo "${{ secrets.GIT_CRYPT_KEY }}" | base64 -d > /tmp/gc-key.bin
    git-crypt unlock /tmp/gc-key.bin
    rm /tmp/gc-key.bin
```

## Docker Builds

The infra `build.sh` clones the repo to `/tmp/rudra-build`. To decrypt during build:

```bash
# In build pipeline (before docker build):
git-crypt unlock /path/to/key.bin
# .env files are now plaintext in build context
# Dockerfile can COPY them normally
```

The `.dockerignore` should ensure `.env_prod` is NOT baked into the image — inject via `docker-compose --env-file` instead.

## Files Overview

| File | Encrypted | Purpose |
|------|-----------|---------|
| `.env_dev` | Yes | Dev/paper trading secrets |
| `.env_sit` | Yes | SIT environment secrets (when added) |
| `.env_prod` | Yes | Production secrets (when added) |
| `.env.example` | No | Template showing required keys |
| `.gitattributes` | No | Defines which files to encrypt |
| `scripts/setup-encryption.sh` | No | One-time unlock after clone |
| `scripts/onboard-team-member.sh` | No | Add a teammate's GPG key |

## Troubleshooting

### "Error: no GPG secret key available"
Your GPG private key isn't on this machine. Import it:
```bash
gpg --import gpg-private-key.gpg
git-crypt unlock
```

### "Error: this repository has already been locked"
Already encrypted. If you see binary in `.env` files:
```bash
git-crypt unlock
```

### File shows plaintext in `git show` output
The file isn't being encrypted. Check `.gitattributes` matches the filename pattern, then re-add:
```bash
git rm --cached .env_dev
git add .env_dev
git commit -m "Re-encrypt .env_dev"
```

### "gpg: decryption failed: No secret key"
Wrong GPG key or key expired. Check:
```bash
gpg --list-secret-keys
# Verify the key ID matches what was added to git-crypt
```

### New team member can't decrypt
Their GPG public key wasn't added. Run:
```bash
./scripts/onboard-team-member.sh their@email.com their-key.gpg
git push
# They pull again and run: git-crypt unlock
```

## Revoking Access

git-crypt doesn't support removing users directly. To revoke:

1. Rotate the git-crypt key:
   ```bash
   git-crypt lock
   rm -rf .git-crypt
   git-crypt init
   ```
2. Re-add only authorized users:
   ```bash
   git-crypt add-gpg-user authorized@email.com
   ```
3. Rotate all secrets in `.env` files (the revoked user had access to the plaintext).
4. Commit and push.

## Security Notes

- **Never commit** your GPG private key to the repo
- **Never share** GPG passphrases over Slack/email
- **Rotate secrets** if a team member leaves (they had plaintext access)
- **Audit access** periodically: `ls .git-crypt/keys/default/0/` shows authorized key IDs
- The `.env.example` file is **not encrypted** — it contains no secrets, only key names
