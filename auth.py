"""
OAuth Authentication Module for Gap Fade Trading Bot

Supports Google, GitHub, and Discord OAuth2 login with JWT session cookies.
Graceful degradation: if AUTH_JWT_SECRET is not set, auth is fully disabled.

─── Setup Instructions ───────────────────────────────────────────────────────

1. GOOGLE  (https://console.cloud.google.com/apis/credentials)
   - Create OAuth 2.0 Client ID (Web application)
   - Authorized redirect URI: http://localhost:5173/api/auth/callback/google
     (prod: https://YOUR_DOMAIN/api/auth/callback/google)
   - Set env: GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET

2. GITHUB  (https://github.com/settings/developers → OAuth Apps → New)
   - Authorization callback URL: http://localhost:5173/api/auth/callback/github
     (prod: https://YOUR_DOMAIN/api/auth/callback/github)
   - Set env: GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET

3. DISCORD  (https://discord.com/developers/applications → OAuth2)
   - Redirect URI: http://localhost:5173/api/auth/callback/discord
     (prod: https://YOUR_DOMAIN/api/auth/callback/discord)
   - Scopes: identify, email
   - Set env: DISCORD_CLIENT_ID, DISCORD_CLIENT_SECRET

4. Common env vars:
   - AUTH_JWT_SECRET       — any long random string (required to enable auth)
   - AUTH_FRONTEND_URL     — e.g. http://localhost:5173 (default)
   - AUTH_ALLOWED_EMAILS   — comma-separated whitelist, e.g. "me@gmail.com,you@corp.com"

───────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
import secrets
import time
from typing import Optional

import jwt
from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

logger = logging.getLogger("GapFadeApp.auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ── Configuration ────────────────────────────────────────────────────────────

COOKIE_NAME = "gf_session"
COOKIE_MAX_AGE = 72 * 3600  # 72 hours
JWT_ALGORITHM = "HS256"

# Server-side store for pending mobile OAuth redirects (nonce -> redirect_url)
# Entries expire after 10 minutes. No persistence needed — login retries are cheap.
_pending_mobile_redirects: dict[str, tuple[str, float]] = {}
_MOBILE_NONCE_TTL = 600  # 10 minutes


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def auth_enabled() -> bool:
    """Auth is only active when a JWT secret is configured."""
    return bool(_env("AUTH_JWT_SECRET"))


def _jwt_secret() -> str:
    return _env("AUTH_JWT_SECRET")


def _frontend_url() -> str:
    return _env("AUTH_FRONTEND_URL", "http://localhost:5173")


def _allowed_emails() -> set[str]:
    raw = _env("AUTH_ALLOWED_EMAILS")
    if not raw:
        return set()  # empty = allow all authenticated users
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def _cleanup_expired_nonces():
    """Remove expired mobile redirect nonces."""
    now = time.time()
    expired = [k for k, (_, ts) in _pending_mobile_redirects.items() if now - ts > _MOBILE_NONCE_TTL]
    for k in expired:
        del _pending_mobile_redirects[k]


# ── OAuth provider registration ─────────────────────────────────────────────

oauth = OAuth()

# Google
oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_id=_env("GOOGLE_CLIENT_ID"),
    client_secret=_env("GOOGLE_CLIENT_SECRET"),
    client_kwargs={"scope": "openid email profile"},
)

# GitHub
oauth.register(
    name="github",
    authorize_url="https://github.com/login/oauth/authorize",
    access_token_url="https://github.com/login/oauth/access_token",
    client_id=_env("GITHUB_CLIENT_ID"),
    client_secret=_env("GITHUB_CLIENT_SECRET"),
    client_kwargs={"scope": "read:user user:email"},
    userinfo_endpoint="https://api.github.com/user",
)

# Discord
oauth.register(
    name="discord",
    authorize_url="https://discord.com/api/oauth2/authorize",
    access_token_url="https://discord.com/api/oauth2/token",
    client_id=_env("DISCORD_CLIENT_ID"),
    client_secret=_env("DISCORD_CLIENT_SECRET"),
    client_kwargs={"scope": "identify email"},
    userinfo_endpoint="https://discord.com/api/users/@me",
)

_PROVIDERS = {"google", "github", "discord"}


# ── JWT helpers ──────────────────────────────────────────────────────────────

def _create_token(user: dict) -> str:
    payload = {
        "sub": user["email"],
        "name": user.get("name", ""),
        "picture": user.get("picture", ""),
        "provider": user.get("provider", ""),
        "iat": int(time.time()),
        "exp": int(time.time()) + COOKIE_MAX_AGE,
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, _jwt_secret(), algorithms=[JWT_ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def get_current_user(request: Request) -> Optional[dict]:
    """Extract user from JWT cookie or Authorization Bearer header. Returns None if missing/invalid."""
    if not auth_enabled():
        return None
    # Try Bearer token first (mobile app), then cookie (web)
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        return _decode_token(token)
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    return _decode_token(token)


# ── User-info extractors per provider ────────────────────────────────────────

async def _extract_user_google(token_data: dict, client) -> dict:
    userinfo = token_data.get("userinfo", {})
    return {
        "email": userinfo.get("email", ""),
        "name": userinfo.get("name", ""),
        "picture": userinfo.get("picture", ""),
        "provider": "google",
    }


async def _extract_user_github(token_data: dict, client) -> dict:
    resp = await client.get("https://api.github.com/user", token=token_data)
    profile = resp.json()
    email = profile.get("email", "")
    if not email:
        # GitHub may hide email; fetch from emails endpoint
        emails_resp = await client.get(
            "https://api.github.com/user/emails", token=token_data
        )
        emails = emails_resp.json()
        for e in emails:
            if e.get("primary"):
                email = e["email"]
                break
    return {
        "email": email,
        "name": profile.get("name") or profile.get("login", ""),
        "picture": profile.get("avatar_url", ""),
        "provider": "github",
    }


async def _extract_user_discord(token_data: dict, client) -> dict:
    resp = await client.get(
        "https://discord.com/api/users/@me", token=token_data
    )
    profile = resp.json()
    avatar_hash = profile.get("avatar", "")
    user_id = profile.get("id", "")
    picture = (
        f"https://cdn.discordapp.com/avatars/{user_id}/{avatar_hash}.png"
        if avatar_hash
        else ""
    )
    return {
        "email": profile.get("email", ""),
        "name": profile.get("global_name") or profile.get("username", ""),
        "picture": picture,
        "provider": "discord",
    }


_EXTRACTORS = {
    "google": _extract_user_google,
    "github": _extract_user_github,
    "discord": _extract_user_discord,
}


# ── Routes ───────────────────────────────────────────────────────────────────

@router.get("/login/{provider}")
async def login(provider: str, request: Request):
    if provider not in _PROVIDERS:
        return JSONResponse({"error": f"Unknown provider: {provider}"}, 400)
    if not auth_enabled():
        return JSONResponse({"error": "Auth not configured"}, 501)
    client = oauth.create_client(provider)
    # Derive callback URL from the actual request so it works for both
    # web (via frontend proxy) and mobile (direct server access).
    base = str(request.base_url).rstrip("/")
    redirect_uri = f"{base}/api/auth/callback/{provider}"

    # Mobile flow: store the app's redirect URL server-side keyed by a nonce.
    # The nonce is sent as a plain cookie (not session) so it survives the OAuth redirect chain.
    mobile_redirect = request.query_params.get("mobile_redirect", "")
    resp = await client.authorize_redirect(request, redirect_uri)

    if mobile_redirect:
        _cleanup_expired_nonces()
        nonce = secrets.token_urlsafe(32)
        _pending_mobile_redirects[nonce] = (mobile_redirect, time.time())
        resp.set_cookie(
            "gf_mobile_nonce", nonce,
            max_age=_MOBILE_NONCE_TTL, httponly=True, samesite="lax", path="/",
        )
        logger.info("Mobile OAuth login started: nonce=%s redirect=%s", nonce[:8], mobile_redirect[:50])

    return resp


@router.get("/callback/{provider}")
async def callback(provider: str, request: Request):
    if provider not in _PROVIDERS:
        return JSONResponse({"error": f"Unknown provider: {provider}"}, 400)
    if not auth_enabled():
        return JSONResponse({"error": "Auth not configured"}, 501)

    client = oauth.create_client(provider)
    try:
        token_data = await client.authorize_access_token(request)
    except Exception as exc:
        logger.error("OAuth token exchange failed for %s: %s", provider, exc)
        return RedirectResponse(f"{_frontend_url()}/#/login-error")

    extractor = _EXTRACTORS[provider]
    user = await extractor(token_data, client)
    email = user.get("email", "").lower()

    if not email:
        logger.warning("OAuth %s: no email returned", provider)
        return RedirectResponse(f"{_frontend_url()}/#/access-denied")

    allowed = _allowed_emails()
    if allowed and email not in allowed:
        logger.warning("OAuth %s: email %s not in whitelist", provider, email)
        return RedirectResponse(f"{_frontend_url()}/#/access-denied")

    jwt_token = _create_token(user)

    # Mobile flow: check for pending redirect nonce in cookie (not session)
    mobile_nonce = request.cookies.get("gf_mobile_nonce", "")
    if mobile_nonce and mobile_nonce in _pending_mobile_redirects:
        mobile_redirect, _ = _pending_mobile_redirects.pop(mobile_nonce)
        sep = "&" if "?" in mobile_redirect else "?"
        deep_link = f"{mobile_redirect}{sep}token={jwt_token}"
        logger.info("OAuth mobile login: %s via %s → %s", email, provider, mobile_redirect[:60])
        # Direct 302 redirect to the app's deep link URL.
        # The mobile app uses expo-web-browser's openAuthSessionAsync which
        # opens an in-app browser tab that intercepts custom-scheme redirects.
        response = RedirectResponse(deep_link)
        response.delete_cookie("gf_mobile_nonce", path="/")
        return response

    response = RedirectResponse(f"{_frontend_url()}/")
    response.set_cookie(
        COOKIE_NAME,
        jwt_token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=_frontend_url().startswith("https"),
        path="/",
    )
    logger.info("OAuth login: %s via %s", email, provider)
    return response


@router.get("/mobile-token")
async def mobile_token(nonce: str):
    """Fallback endpoint: mobile app can poll this with its nonce to get the JWT
    token after OAuth completes, in case the deep link redirect didn't work."""
    if not auth_enabled():
        return JSONResponse({"error": "Auth not configured"}, 501)
    entry = _pending_mobile_redirects.get(nonce)
    if not entry:
        return JSONResponse({"error": "Invalid or expired nonce", "ready": False}, 404)
    # The nonce is still pending — OAuth hasn't completed yet, or redirect worked
    return JSONResponse({"ready": False, "message": "Waiting for OAuth to complete"}, 202)


@router.get("/me")
async def me(request: Request):
    if not auth_enabled():
        return JSONResponse({"user": None, "auth_enabled": False})
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Not authenticated", "auth_enabled": True}, 401)
    return {
        "user": {
            "email": user.get("sub", ""),
            "name": user.get("name", ""),
            "picture": user.get("picture", ""),
            "provider": user.get("provider", ""),
        },
        "auth_enabled": True,
    }


@router.get("/logout")
async def logout():
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie(COOKIE_NAME, path="/")
    return response
