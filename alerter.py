"""
Rudra Trading Engine — Trade Alert Notification System.

Supports Telegram and Email channels with throttling, async delivery,
and graceful degradation when credentials are not configured.

Env vars:
    TELEGRAM_BOT_TOKEN      Telegram bot token from @BotFather
    TELEGRAM_CHAT_ID        Target chat/group ID
    ALERT_EMAIL_FROM        Sender email address
    ALERT_EMAIL_PASSWORD    App password (Gmail: Settings > Security > App Passwords)
    ALERT_EMAIL_TO          Recipient email (comma-separated for multiple)
    ALERT_EMAIL_SMTP        SMTP host (default: smtp.gmail.com)
    ALERT_EMAIL_PORT        SMTP port (default: 587)

Usage:
    from alerter import RudraAlerter
    alerter = RudraAlerter()
    await alerter.send('Entry Fill', 'SHORT 2075 BHVN @ $9.09', level='trade')
    await alerter.trade_entry('BHVN', 'short', 2075, 9.09, 9.17, 'exhaustion_gap_fade', gap_pct=7.1)

CLI testing:
    python alerter.py --test-telegram
    python alerter.py --test-email
    python alerter.py --test-all
"""

from __future__ import annotations

import asyncio
import logging
import os
import smtplib
import time
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger('rudra.alerter')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ET = timezone(timedelta(hours=-5))  # US Eastern (approximate; doesn't track DST)


class AlertLevel:
    """Alert severity / category constants."""
    TRADE = 'trade'
    PROFIT = 'profit'
    STOP = 'stop'
    WARNING = 'warning'
    CRITICAL = 'critical'
    DAILY = 'daily'
    SWING = 'swing'
    INFO = 'info'
    ERROR = 'error'
    SUMMARY = 'summary'


_LEVEL_EMOJI = {
    AlertLevel.TRADE: '\U0001f4b0',    # money bag
    AlertLevel.PROFIT: '\U0001f4b0',
    AlertLevel.STOP: '\U0001f534',      # red circle
    AlertLevel.WARNING: '\u26a0\ufe0f',
    AlertLevel.CRITICAL: '\U0001f6a8',  # siren
    AlertLevel.DAILY: '\U0001f4ca',     # chart
    AlertLevel.SWING: '\U0001f30a',     # wave
    AlertLevel.INFO: '\u2139\ufe0f',
    AlertLevel.ERROR: '\U0001f6a8',
    AlertLevel.SUMMARY: '\U0001f4ca',
}

_LEVEL_COLOR = {
    AlertLevel.TRADE: '#2196F3',
    AlertLevel.PROFIT: '#4CAF50',
    AlertLevel.STOP: '#f44336',
    AlertLevel.WARNING: '#FF9800',
    AlertLevel.CRITICAL: '#f44336',
    AlertLevel.DAILY: '#607D8B',
    AlertLevel.SWING: '#9C27B0',
    AlertLevel.INFO: '#607D8B',
    AlertLevel.ERROR: '#f44336',
    AlertLevel.SUMMARY: '#607D8B',
}

# Levels that trigger immediate email (others are batched / telegram-only)
_EMAIL_IMMEDIATE_LEVELS = frozenset({
    AlertLevel.CRITICAL,
    AlertLevel.DAILY,
    AlertLevel.SUMMARY,
})

SEPARATOR = '\u2501' * 20  # ━━━━━━━━━━━━━━━━━━━━


# ---------------------------------------------------------------------------
# Telegram channel
# ---------------------------------------------------------------------------

class _TelegramChannel:
    """Sends messages via Telegram Bot API (aiohttp or urllib fallback)."""

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.enabled = bool(token and chat_id)

    async def send(self, text: str) -> bool:
        """Send HTML-formatted message. Returns True on success."""
        if not self.enabled:
            return False

        url = f'https://api.telegram.org/bot{self.token}/sendMessage'
        payload = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True,
        }

        if _HAS_AIOHTTP:
            return await self._send_aiohttp(url, payload)
        return await self._send_urllib(url, payload)

    async def _send_aiohttp(self, url: str, payload: dict) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                resp = await session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                )
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Telegram API %s: %s", resp.status, body[:200])
                    return False
                return True
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)
            return False

    async def _send_urllib(self, url: str, payload: dict) -> bool:
        """Fallback when aiohttp is not installed — runs in executor."""
        import json
        import urllib.request
        import urllib.error

        def _post():
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={'Content-Type': 'application/json'},
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return resp.status == 200
            except urllib.error.URLError as exc:
                logger.warning("Telegram urllib send failed: %s", exc)
                return False

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _post)


# ---------------------------------------------------------------------------
# Email channel
# ---------------------------------------------------------------------------

class _EmailChannel:
    """Sends HTML emails via SMTP (TLS)."""

    def __init__(
        self,
        from_addr: str,
        password: str,
        to_addrs: str,
        smtp_host: str = 'smtp.gmail.com',
        smtp_port: int = 587,
    ):
        self.from_addr = from_addr
        self.password = password
        self.to_addrs = [a.strip() for a in to_addrs.split(',') if a.strip()]
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.enabled = bool(from_addr and password and self.to_addrs)

    async def send(self, subject: str, html_body: str) -> bool:
        """Send an HTML email. Runs SMTP in executor to avoid blocking."""
        if not self.enabled:
            return False

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._send_sync, subject, html_body)

    def _send_sync(self, subject: str, html_body: str) -> bool:
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_addr
            msg['To'] = ', '.join(self.to_addrs)

            # Plain-text fallback (strip tags crudely)
            import re
            plain = re.sub(r'<[^>]+>', '', html_body)
            msg.attach(MIMEText(plain, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.from_addr, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            return True
        except Exception as exc:
            logger.warning("Email send failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# HTML email builder
# ---------------------------------------------------------------------------

def _build_email_html(
    title: str,
    message: str,
    level: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a styled HTML email body."""
    color = _LEVEL_COLOR.get(level, '#607D8B')
    emoji = _LEVEL_EMOJI.get(level, '')

    rows_html = ''
    if data:
        row_items = []
        for key, val in data.items():
            label = key.replace('_', ' ').title()
            row_items.append(
                f'<tr><td style="padding:4px 12px 4px 0;color:#888;">{label}</td>'
                f'<td style="padding:4px 0;font-weight:600;">{val}</td></tr>'
            )
        rows_html = (
            '<table style="margin-top:12px;border-collapse:collapse;">'
            + ''.join(row_items)
            + '</table>'
        )

    # Escape message newlines for HTML
    message_html = message.replace('\n', '<br>')

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
             background:#f5f5f5;padding:20px;">
  <div style="max-width:480px;margin:0 auto;background:#fff;border-radius:8px;
              overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
    <div style="background:{color};padding:16px 20px;color:#fff;">
      <h2 style="margin:0;font-size:18px;">{emoji} {title}</h2>
    </div>
    <div style="padding:20px;">
      <p style="margin:0 0 8px;color:#333;white-space:pre-line;">{message_html}</p>
      {rows_html}
      <hr style="border:none;border-top:1px solid #eee;margin:16px 0;">
      <p style="margin:0;color:#aaa;font-size:12px;">
        Rudra Trading Engine &mdash; {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}
      </p>
    </div>
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main alerter
# ---------------------------------------------------------------------------

class RudraAlerter:
    """Multi-channel trade alert system for the Rudra Trading Engine.

    Drop-in replacement for the existing AlertNotifier — same `send()` signature.
    Adds higher-level helpers: trade_entry, trade_exit, daily_summary, etc.
    """

    THROTTLE_SECONDS = 60

    def __init__(self, config: Any = None):
        """Initialise from env vars (and optional GapFadeConfig).

        Gracefully degrades: if credentials are missing, that channel
        is silently disabled. The trading bot never crashes due to alerter.
        """
        # Telegram
        tg_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
        tg_chat = os.environ.get('TELEGRAM_CHAT_ID', '')
        if config is not None:
            tg_token = getattr(config, 'alert_telegram_token', '') or tg_token
            tg_chat = getattr(config, 'alert_telegram_chat_id', '') or tg_chat
            self.enabled = getattr(config, 'alert_telegram_enabled', False) or bool(tg_token and tg_chat)
        else:
            self.enabled = bool(tg_token and tg_chat)

        self._telegram = _TelegramChannel(tg_token, tg_chat)

        # Email
        self._email = _EmailChannel(
            from_addr=os.environ.get('ALERT_EMAIL_FROM', ''),
            password=os.environ.get('ALERT_EMAIL_PASSWORD', ''),
            to_addrs=os.environ.get('ALERT_EMAIL_TO', ''),
            smtp_host=os.environ.get('ALERT_EMAIL_SMTP', 'smtp.gmail.com'),
            smtp_port=int(os.environ.get('ALERT_EMAIL_PORT', '587')),
        )

        # Throttle state: key -> last-sent timestamp
        self._last_sent: Dict[str, float] = {}

        channels = []
        if self._telegram.enabled:
            channels.append('Telegram')
        if self._email.enabled:
            channels.append('Email')
        if channels:
            logger.info("RudraAlerter ready: %s", ' + '.join(channels))
        else:
            logger.info("RudraAlerter: no channels configured (alerts disabled)")

    # ------------------------------------------------------------------
    # Backward-compatible send() — matches existing AlertNotifier.send()
    # ------------------------------------------------------------------

    async def send(
        self,
        title: str,
        message: str,
        level: str = 'info',
        throttle_key: str = '',
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send an alert via all configured channels.

        Args:
            title: Short headline (e.g. 'Entry Fill', 'Circuit Breaker').
            message: Body text (newlines OK).
            level: One of AlertLevel constants (or legacy 'info'/'error').
            throttle_key: If set, suppress duplicate sends for THROTTLE_SECONDS.
            data: Optional structured data dict for email table rendering.
        """
        try:
            await self._send_impl(title, message, level, throttle_key, data)
        except Exception as exc:
            # Never let alerter crash the trading bot
            logger.error("RudraAlerter.send() unhandled error: %s", exc, exc_info=True)

    async def _send_impl(
        self,
        title: str,
        message: str,
        level: str,
        throttle_key: str,
        data: Optional[Dict[str, Any]],
    ) -> None:
        # Throttle check
        if throttle_key:
            now = time.monotonic()
            last = self._last_sent.get(throttle_key, 0.0)
            if now - last < self.THROTTLE_SECONDS:
                logger.debug("Alert throttled: %s", throttle_key)
                return
            self._last_sent[throttle_key] = now

        emoji = _LEVEL_EMOJI.get(level, '')

        # -- Telegram --
        tg_text = f"<b>{emoji} {title}</b>\n{SEPARATOR}\n{message}"
        tg_future = self._telegram.send(tg_text)

        # -- Email (immediate for CRITICAL/DAILY, skip for routine trades) --
        email_future: Optional[asyncio.Task] = None
        if level in _EMAIL_IMMEDIATE_LEVELS or level == AlertLevel.ERROR:
            subject = f"Rudra: {title}"
            html = _build_email_html(title, message, level, data)
            email_future = asyncio.ensure_future(self._email.send(subject, html))

        # Await Telegram
        await tg_future

        # Await Email if scheduled
        if email_future is not None:
            await email_future

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    async def trade_entry(
        self,
        symbol: str,
        side: str,
        shares: int,
        price: float,
        stop: float,
        strategy: str,
        gap_pct: float = 0.0,
        rr_ratio: float = 0.0,
    ) -> None:
        """Format and send a trade entry alert."""
        side_upper = side.upper()
        now_et = datetime.now(ET).strftime('%H:%M ET')

        lines = [
            f"Strategy: {strategy}",
            f"Shares: {shares:,} @ ${price:.2f}",
            f"Stop: ${stop:.2f}",
        ]
        if gap_pct:
            lines.append(f"Gap: {gap_pct:+.1f}%")
        if rr_ratio:
            lines.append(f"R/R: {rr_ratio:.1f}:1")
        lines.append(f"Time: {now_et}")

        title = f"ENTRY \u2014 {side_upper} {symbol}"
        message = '\n'.join(lines)

        data = {
            'symbol': symbol,
            'side': side_upper,
            'shares': f"{shares:,}",
            'price': f"${price:.2f}",
            'stop': f"${stop:.2f}",
            'strategy': strategy,
        }
        if gap_pct:
            data['gap'] = f"{gap_pct:+.1f}%"

        await self.send(
            title=f"\U0001f7e2 {title}",
            message=message,
            level=AlertLevel.TRADE,
            throttle_key=f"entry:{symbol}",
            data=data,
        )

    async def trade_exit(
        self,
        symbol: str,
        side: str,
        shares: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        reason: str,
        hold_min: int = 0,
    ) -> None:
        """Format and send a trade exit alert."""
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price else 0.0
        if side.lower() == 'short':
            pnl_pct = -pnl_pct

        is_stop = 'stop' in reason.lower()
        is_profit = pnl > 0

        # Format hold time
        if hold_min >= 60:
            hours = hold_min // 60
            mins = hold_min % 60
            hold_str = f"{hours}h {mins}m"
        else:
            hold_str = f"{hold_min} min"

        pnl_sign = '+' if pnl >= 0 else ''
        lines = [
            f"Exit: ${exit_price:.2f} ({reason})",
            f"P&L: {pnl_sign}${pnl:,.2f} ({pnl_pct:+.1f}%)",
            f"Hold: {hold_str}",
        ]
        message = '\n'.join(lines)

        if is_stop:
            emoji = '\U0001f534'
            title_prefix = "STOP HIT"
            level = AlertLevel.STOP
        elif is_profit:
            emoji = '\U0001f4b0'
            title_prefix = "PROFIT"
            level = AlertLevel.PROFIT
        else:
            emoji = '\U0001f7e1'
            title_prefix = "EXIT"
            level = AlertLevel.TRADE

        data = {
            'symbol': symbol,
            'exit_price': f"${exit_price:.2f}",
            'pnl': f"{pnl_sign}${pnl:,.2f}",
            'pnl_pct': f"{pnl_pct:+.1f}%",
            'reason': reason,
            'hold_time': hold_str,
        }

        await self.send(
            title=f"{emoji} {title_prefix} \u2014 {symbol}",
            message=message,
            level=level,
            throttle_key=f"exit:{symbol}",
            data=data,
        )

    async def daily_summary(
        self,
        trades: int,
        pnl: float,
        equity: float,
        win_rate: float,
        positions: Optional[List[Dict[str, Any]]] = None,
        max_dd_pct: float = 0.0,
    ) -> None:
        """Format and send end-of-day summary."""
        wins = round(trades * win_rate / 100) if trades else 0
        losses = trades - wins
        pnl_sign = '+' if pnl >= 0 else ''

        lines = [
            f"Trades: {trades} ({wins}W / {losses}L)",
            f"P&L: {pnl_sign}${pnl:,.2f}",
            f"Win Rate: {win_rate:.0f}%",
            f"Equity: ${equity:,.2f}",
        ]
        if max_dd_pct:
            lines.append(f"Max DD today: {max_dd_pct:.1f}%")
        if positions:
            pos_strs = [
                f"{p.get('symbol', '?')} ({p.get('side', '?')})"
                for p in positions
            ]
            lines.append(f"Active: {len(positions)} position(s) \u2014 {', '.join(pos_strs)}")

        message = '\n'.join(lines)

        data = {
            'trades': str(trades),
            'win_loss': f"{wins}W / {losses}L",
            'pnl': f"{pnl_sign}${pnl:,.2f}",
            'win_rate': f"{win_rate:.0f}%",
            'equity': f"${equity:,.2f}",
        }

        await self.send(
            title="\U0001f4ca DAILY SUMMARY",
            message=message,
            level=AlertLevel.DAILY,
            data=data,
        )

    async def warning(self, title: str, message: str) -> None:
        """Send a warning alert (e.g. circuit breaker, drawdown)."""
        await self.send(
            title=f"\u26a0\ufe0f WARNING \u2014 {title}",
            message=message,
            level=AlertLevel.WARNING,
            throttle_key=f"warn:{title}",
        )

    async def critical(self, title: str, message: str) -> None:
        """Send a critical alert via ALL channels immediately."""
        # Force email for critical regardless of _EMAIL_IMMEDIATE_LEVELS
        # (already included, but be explicit)
        await self.send(
            title=f"\U0001f6a8 CRITICAL \u2014 {title}",
            message=message,
            level=AlertLevel.CRITICAL,
        )

    async def swing_signal(
        self,
        symbol: str,
        direction: str,
        signal_type: str,
        details: str = '',
    ) -> None:
        """Send a swing trading signal alert."""
        message = f"{direction.upper()} signal: {signal_type}"
        if details:
            message += f"\n{details}"

        await self.send(
            title=f"\U0001f30a SWING \u2014 {symbol}",
            message=message,
            level=AlertLevel.SWING,
            throttle_key=f"swing:{symbol}",
        )


# ---------------------------------------------------------------------------
# CLI test harness
# ---------------------------------------------------------------------------

async def _test_telegram(alerter: RudraAlerter) -> None:
    """Send a test message to Telegram."""
    if not alerter._telegram.enabled:
        print("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return

    print("Sending Telegram test message...")
    now = datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S ET')
    ok = await alerter._telegram.send(
        f"<b>\u2705 Rudra Alert Test</b>\n"
        f"{SEPARATOR}\n"
        f"Telegram alerts are working.\n"
        f"Time: {now}"
    )
    print(f"Telegram: {'OK' if ok else 'FAILED'}")


async def _test_email(alerter: RudraAlerter) -> None:
    """Send a test email."""
    if not alerter._email.enabled:
        print("Email not configured. Set ALERT_EMAIL_FROM, ALERT_EMAIL_PASSWORD, ALERT_EMAIL_TO.")
        return

    print("Sending test email...")
    html = _build_email_html(
        title="Alert Test",
        message="Email alerts are working correctly.\nThis is a test from the Rudra Trading Engine.",
        level=AlertLevel.INFO,
        data={
            'status': 'Connected',
            'channels': 'Email',
            'timestamp': datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S ET'),
        },
    )
    ok = await alerter._email.send("Rudra Alert Test", html)
    print(f"Email: {'OK' if ok else 'FAILED'}")


async def _test_all(alerter: RudraAlerter) -> None:
    """Test all configured channels."""
    await _test_telegram(alerter)
    await _test_email(alerter)

    # Also test a high-level helper
    print("\nSending sample trade entry alert...")
    await alerter.trade_entry(
        symbol='BHVN',
        side='short',
        shares=2075,
        price=9.09,
        stop=9.17,
        strategy='exhaustion_gap_fade',
        gap_pct=7.1,
        rr_ratio=3.2,
    )
    print("Done.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Rudra Trading Engine — Alert System Test')
    parser.add_argument('--test-telegram', action='store_true', help='Send test Telegram message')
    parser.add_argument('--test-email', action='store_true', help='Send test email')
    parser.add_argument('--test-all', action='store_true', help='Test all channels')
    args = parser.parse_args()

    if not any([args.test_telegram, args.test_email, args.test_all]):
        parser.print_help()
        return

    alerter = RudraAlerter()

    if args.test_all:
        asyncio.run(_test_all(alerter))
    elif args.test_telegram:
        asyncio.run(_test_telegram(alerter))
    elif args.test_email:
        asyncio.run(_test_email(alerter))


if __name__ == '__main__':
    main()
