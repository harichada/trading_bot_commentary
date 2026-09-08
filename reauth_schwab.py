"""Schwab OAuth re-authentication helper.

Schwab refresh tokens expire 7 days after creation and cannot be
renewed programmatically — a human must log in. Run this whenever
`core.token_health` reports the token expired (the 2026-07-31 outage:
`Refresh token is invalid, expired or revoked` on every stream/quote
call).

Usage (interactive — run yourself, not from an agent):

    conda activate trading-bot
    python reauth_schwab.py

The script prints a Schwab login URL. Open it in a browser, log in,
approve the app, then paste the full https://127.0.0.1/?code=... URL
you are redirected to back into the terminal. The fresh token is
written to the configured token path (token_1.json) after backing up
the old one.

Requires SCHWAB_API_KEY and SCHWAB_SECRET (or SCHWAB_APP_SECRET) in
the environment, same as the bot itself.
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path


def main() -> int:
    from core.config import Config
    from schwab import auth

    cfg = Config()
    api_key = cfg.SCHWAB_API_KEY
    app_secret = cfg.SCHWAB_APP_SECRET
    callback_url = cfg.SCHWAB_CALLBACK_URL
    token_path = Path(cfg.SCHWAB_TOKEN_PATH)

    if not api_key or not app_secret:
        print(
            "ERROR: SCHWAB_API_KEY and SCHWAB_SECRET (or SCHWAB_APP_SECRET) "
            "must be set in the environment.",
            file=sys.stderr,
        )
        return 1

    if token_path.exists():
        backup = token_path.with_name(
            f"{token_path.stem}.expired-{datetime.now():%Y%m%d-%H%M%S}.json"
        )
        shutil.copy2(token_path, backup)
        print(f"Backed up old token to {backup}")

    print(f"Callback URL: {callback_url}")
    print("A Schwab login URL will be printed below. Open it in a browser,")
    print("log in, approve, then paste the full redirect URL back here.\n")

    client = auth.client_from_manual_flow(
        api_key=api_key,
        app_secret=app_secret,
        callback_url=callback_url,
        token_path=str(token_path),
    )

    resp = client.get_account_numbers()
    if resp.status_code == 200:
        print(f"\n✅ Token written to {token_path} and verified "
              f"(get_account_numbers HTTP 200).")
        print("Token is valid for 7 days. Restart the bot to pick it up.")
        return 0

    print(
        f"\n⚠️ Token written to {token_path} but verification call "
        f"returned HTTP {resp.status_code}: {resp.text[:200]}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
