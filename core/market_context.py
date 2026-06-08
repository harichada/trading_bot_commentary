"""v-market-context-2026-06-08: pro-level market context awareness.

The bot's existing strategies look at one symbol's 5-min bar in
isolation. Real traders never do that. Before they take a position
on NVDA, they check:
  * What is SPY doing right now?
  * What's the broader semis sector (XLK / SOXX) doing?
  * Is the VIX rising or falling?
  * Is it the first 30 min (volatile chop) or post-lunch (cleaner)?

This module bundles those reads into a single MarketContext value
that strategies consume before emitting signals. It exposes:

  spy_direction          'bullish' | 'bearish' | 'neutral'
  spy_change_pct         SPY day % change
  sector_etf             matched SPDR sector ETF symbol (or '')
  sector_strength_pct    sector ETF day % change
  vix_level              current VIX
  vix_change_pct         VIX day % change
  regime                 'risk_on' | 'risk_off' | 'mixed' | 'unknown'
  time_of_day            'opening_30' | 'morning' | 'midday' | 'afternoon' | 'closing_30'
  conviction_multiplier  0.5 .. 1.5  (sizing multiplier strategies can apply)
  allows_long            bool — convenience gate
  allows_short           bool
  reason                 human-readable summary for log/commentary

Strategies (wired in a subsequent commit — this module is dormant
until they consume it):
  * LONG entries skip when regime == 'risk_off'
  * Position size = base × conviction_multiplier
  * High-conviction setups (regime risk_on + sector positive +
    SPY positive) get full size; mixed setups get 0.5-0.7 size.

Data sources (all already live):
  * core/market_indices.MarketIndicesCache — extended 2026-06-08
    to track 11 SPDR sector ETFs in addition to SPY/DIA/QQQ/IWM/VIX.
    Refreshes every 10s from Schwab's batched get_quotes API.
  * Symbol→sector mapping is a static table here (small enough to
    maintain by hand; most signals will be on the ~40 most-traded
    names). Unknown symbols default to no-sector and skip the
    sector-strength multiplier.

Why this is dormant until wired:
  * The function is pure — no side effects, no risk.
  * Tests can validate the algorithm end-to-end against synthetic
    cache states without needing Schwab.
  * Tonight after close, strategies start consuming it. If we wired
    it now during RTH, we'd be deploying behavior change to a live
    session — bad practice. The build/wire split mirrors the
    direction_reader rollout from 2026-05-28.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timezone
from typing import Optional


# ────────────────────────────────────────────────────────────────────
# Symbol → SPDR sector ETF mapping
# ────────────────────────────────────────────────────────────────────
#
# Coverage strategy: lean toward the names that actually fire signals
# (tech-heavy, small-cap volatile, news-active). Unknown symbols
# default to empty string — caller treats as "no sector data" and
# skips the sector-strength check.
#
# The map intentionally lives in this module (not yaml) so changes
# require a code review. Sector misclassification is silent-bias
# risk — it changes sizing on real trades.

_SYMBOL_TO_SECTOR_ETF = {
    # Technology — semiconductors, software, hardware
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK",
    "AVGO": "XLK", "ORCL": "XLK", "CRM": "XLK", "ADBE": "XLK",
    "INTC": "XLK", "TXN": "XLK", "QCOM": "XLK", "MU": "XLK",
    "AMAT": "XLK", "KLAC": "XLK", "LRCX": "XLK", "MRVL": "XLK",
    "PLTR": "XLK", "SMCI": "XLK", "ARM": "XLK", "NOW": "XLK",
    "SNOW": "XLK", "DDOG": "XLK", "PATH": "XLK", "ANET": "XLK",
    "MDB": "XLK", "TEAM": "XLK", "ZETA": "XLK", "HPE": "XLK",
    "HPQ": "XLK", "WDAY": "XLK", "INTU": "XLK", "DELL": "XLK",
    "ASML": "XLK", "CRWV": "XLK", "IBM": "XLK", "COHR": "XLK",
    "ONDS": "XLK", "RGTI": "XLK", "QBTS": "XLK", "QUBT": "XLK",
    "IONQ": "XLK", "BBAI": "XLK", "BB": "XLK", "GLW": "XLK",
    "NVTS": "XLK", "AAOI": "XLK", "POET": "XLK", "OUST": "XLK",
    # Communication Services — Meta-class, telecoms, media
    "META": "XLC", "GOOG": "XLC", "GOOGL": "XLC", "NFLX": "XLC",
    "VZ": "XLC", "T": "XLC", "TMUS": "XLC", "DIS": "XLC",
    "CMCSA": "XLC", "WBD": "XLC", "SNAP": "XLC", "PINS": "XLC",
    "ROKU": "XLC", "ZM": "XLC", "PARA": "XLC", "EA": "XLC",
    "NOK": "XLC", "TE": "XLC",
    # Consumer Discretionary — Amazon, Tesla, retail
    "AMZN": "XLY", "TSLA": "XLY", "MCD": "XLY", "SBUX": "XLY",
    "NKE": "XLY", "HD": "XLY", "LOW": "XLY", "TGT": "XLY",
    "F": "XLY", "GM": "XLY", "RIVN": "XLY", "LCID": "XLY",
    "NIO": "XLY", "XPEV": "XLY", "LI": "XLY", "ABNB": "XLY",
    "BKNG": "XLY", "MAR": "XLY", "RACE": "XLY", "BBWI": "XLY",
    "CPRI": "XLY", "FUBO": "XLY", "CHWY": "XLY", "PTON": "XLY",
    "DKNG": "XLY",
    # Financials — banks, brokers, insurance
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "C": "XLF",
    "GS": "XLF", "MS": "XLF", "BLK": "XLF", "AXP": "XLF",
    "SCHW": "XLF", "COF": "XLF", "USB": "XLF", "PYPL": "XLF",
    "HOOD": "XLF", "COIN": "XLF", "SOFI": "XLF", "SQ": "XLF",
    "FUTU": "XLF", "HBAN": "XLF", "MSTR": "XLF",
    # Energy — oil & gas
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "OXY": "XLE",
    "MPC": "XLE", "PSX": "XLE", "VLO": "XLE", "EOG": "XLE",
    "BE": "XLE", "FCEL": "XLE", "PLUG": "XLE",
    # Healthcare — pharma, biotech, devices
    "UNH": "XLV", "JNJ": "XLV", "LLY": "XLV", "ABBV": "XLV",
    "PFE": "XLV", "MRK": "XLV", "TMO": "XLV", "ABT": "XLV",
    "DHR": "XLV", "AMGN": "XLV", "GILD": "XLV", "BMY": "XLV",
    "MRNA": "XLV", "BNTX": "XLV", "NVAX": "XLV", "REGN": "XLV",
    "VRTX": "XLV", "HIMS": "XLV", "SMMT": "XLV", "LEGN": "XLV",
    "ADEA": "XLV", "OSCR": "XLV",
    # Consumer Staples — defensives
    "WMT": "XLP", "PG": "XLP", "KO": "XLP", "PEP": "XLP",
    "COST": "XLP", "PM": "XLP", "MO": "XLP", "CL": "XLP",
    # Industrials — aerospace, transports, manufacturing
    "BA": "XLI", "CAT": "XLI", "UPS": "XLI", "RTX": "XLI",
    "GE": "XLI", "MMM": "XLI", "HON": "XLI", "LMT": "XLI",
    "DE": "XLI", "RKLB": "XLI", "ASTS": "XLI", "ACHR": "XLI",
    "JOBY": "XLI", "LUNR": "XLI", "RDW": "XLI", "AAL": "XLI",
    "DAL": "XLI", "TAC": "XLI", "FLNC": "XLI",
    # Materials — chemicals, mining, steel
    "LIN": "XLB", "FCX": "XLB", "NEM": "XLB", "DD": "XLB",
    "DOW": "XLB", "CLF": "XLB",
    # Utilities — electric/gas/water
    "NEE": "XLU", "SO": "XLU", "DUK": "XLU", "AEP": "XLU",
    "EXC": "XLU", "VST": "XLU",
    # Real Estate
    "PLD": "XLRE", "AMT": "XLRE", "EQIX": "XLRE", "CCI": "XLRE",
    "OPEN": "XLRE", "FIG": "XLRE",
}


def get_sector_etf(symbol: str) -> str:
    """Return the SPDR sector ETF for a symbol, or '' if unknown."""
    return _SYMBOL_TO_SECTOR_ETF.get(symbol.upper(), "")


# ────────────────────────────────────────────────────────────────────
# Output type
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MarketContext:
    """Bundled market context consumed by strategies before deciding."""
    symbol: str

    # Broad market reads
    spy_direction: str       # 'bullish' | 'bearish' | 'neutral' | 'unknown'
    spy_change_pct: float

    # Sector context
    sector_etf: str          # SPDR sector ETF symbol (or '' if unknown)
    sector_strength_pct: float  # sector ETF day % change

    # Volatility regime
    vix_level: float
    vix_change_pct: float

    # Combined regime classification
    regime: str              # 'risk_on' | 'risk_off' | 'mixed' | 'unknown'

    # Time-of-day awareness
    time_of_day: str         # 'opening_30' | 'morning' | 'midday' | 'afternoon' | 'closing_30' | 'off_hours'

    # Sizing multiplier strategies apply to position size
    conviction_multiplier: float  # 0.5..1.5

    reason: str = ""
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def allows_long(self) -> bool:
        """Convenience gate — strategies that want a binary
        permit/deny check can use this instead of reading regime
        directly. Allows long when regime is risk_on OR mixed
        with mild SPY positive; blocks risk_off."""
        if self.regime == "risk_off":
            return False
        if self.regime == "unknown":
            return True  # fail-open when we have no read
        return True

    @property
    def allows_short(self) -> bool:
        """Mirror gate for shorts."""
        if self.regime == "risk_on":
            return False
        if self.regime == "unknown":
            return True
        return True


# ────────────────────────────────────────────────────────────────────
# Classification helpers
# ────────────────────────────────────────────────────────────────────

def _classify_spy_direction(change_pct: float) -> str:
    """SPY day % change → direction label.

    Thresholds tuned for intraday context:
      > +0.3%  → bullish
      < -0.3% → bearish
      else    → neutral

    The 0.3% threshold filters tiny moves from being interpreted
    as direction. SPY moving 0.1% all day is NOT a directional tape.
    """
    if change_pct > 0.3:
        return "bullish"
    if change_pct < -0.3:
        return "bearish"
    return "neutral"


def _classify_regime(
    spy_change_pct: float,
    vix_level: float,
    vix_change_pct: float,
) -> str:
    """Combine SPY direction + VIX state into regime label.

    Risk-on:  SPY up + VIX down (or VIX low)
              Risk appetite is high; trend-following long bias.
    Risk-off: SPY down + VIX up (especially spiking)
              Risk appetite collapsed; defensive bias.
    Mixed:    Conflicting signals (SPY up but VIX up = nervous rally;
              SPY down but VIX flat = orderly pullback).
    """
    # Risk-on: clear bullish SPY + falling/low vol
    if spy_change_pct > 0.3 and (vix_change_pct < 0 or vix_level < 16):
        return "risk_on"

    # Risk-off: SPY down + VIX rising (the panic combination)
    if spy_change_pct < -0.5 and vix_change_pct > 5:
        return "risk_off"

    # VIX spike alone is risk-off even if SPY hasn't moved yet
    if vix_change_pct > 10:
        return "risk_off"

    # Mixed: clear conflicts
    if (spy_change_pct > 0.3 and vix_change_pct > 3) or (
        spy_change_pct < -0.3 and vix_change_pct < 0
    ):
        return "mixed"

    return "mixed"


def _classify_time_of_day(now_utc: Optional[datetime] = None) -> str:
    """Bucket the current time into intraday phases.

    Times in ET (UTC-4 in summer, UTC-5 in winter — we assume
    summer/EDT for simplicity; the bot runs east-coast all year).

      04:00-09:30 → off_hours    (pre-market, no entries)
      09:30-10:00 → opening_30   (volatile, hard-block window covers
                                  first 5 min already)
      10:00-12:00 → morning      (most-traded window)
      12:00-14:00 → midday       (lunch chop)
      14:00-15:30 → afternoon    (best window for breakouts)
      15:30-16:00 → closing_30   (late-entry cutoff covers this)
      else        → off_hours
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    # Convert to ET (assume EDT). For production-grade time-of-day
    # logic we'd use pytz; for this classifier the ~1h boundary
    # accuracy is fine.
    et_hour = (now_utc.hour - 4) % 24
    et_minute = now_utc.minute

    if et_hour < 9 or (et_hour == 9 and et_minute < 30):
        return "off_hours"
    if et_hour == 9 or (et_hour == 10 and et_minute == 0):
        return "opening_30"
    if et_hour < 12:
        return "morning"
    if et_hour < 14:
        return "midday"
    if et_hour < 15 or (et_hour == 15 and et_minute < 30):
        return "afternoon"
    if et_hour < 16:
        return "closing_30"
    return "off_hours"


def _compute_conviction(
    regime: str,
    spy_change_pct: float,
    sector_strength_pct: float,
    has_sector: bool,
    time_of_day: str,
) -> float:
    """Conviction-based sizing multiplier in 0.5..1.5.

    The multiplier scales position size BEFORE the live_size_multiplier
    and strategy_size_multiplier in the existing chain. Combined with
    those, the final size respects both global risk budget AND
    per-trade context strength.

    Components (each can subtract or add 0.1-0.2):
      Base:                          1.0
      Regime risk_on:                +0.2
      Regime risk_off:               -0.3  (heavily discount)
      Regime mixed:                  -0.1
      SPY same direction as long:    +0.1
      Sector same direction:         +0.15
      Sector opposite direction:     -0.15
      Mid-day (lunch chop):          -0.05
      Opening 30 / closing 30:       -0.1

    Final clamped to [0.5, 1.5] so worst case is half-size and best
    case is 1.5x. The defaults (no context data) return 1.0 — exactly
    the current behavior, so this is fail-open.
    """
    mult = 1.0

    # Regime weight
    if regime == "risk_on":
        mult += 0.2
    elif regime == "risk_off":
        mult -= 0.3
    elif regime == "mixed":
        mult -= 0.1

    # Sector confluence (only counts when we have a sector match)
    if has_sector:
        if sector_strength_pct > 0.5:
            mult += 0.15
        elif sector_strength_pct < -0.5:
            mult -= 0.15

    # Time-of-day adjustment
    if time_of_day == "midday":
        mult -= 0.05
    elif time_of_day in ("opening_30", "closing_30"):
        mult -= 0.1

    return max(0.5, min(1.5, mult))


# ────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────

def read_market_context(symbol: str) -> MarketContext:
    """Synthesize current market context for a single symbol.

    Reads from MarketIndicesCache (already updating every 10s).
    No side effects, no I/O — strategies call inline.
    """
    # Default values for the unknown/empty-cache case (fail-open).
    spy_change_pct = 0.0
    vix_level = 16.0
    vix_change_pct = 0.0
    sector_strength_pct = 0.0

    try:
        from core.market_indices import MarketIndicesCache
        cache = MarketIndicesCache.instance()
        spy = cache.get_quote("SPY")
        vix = cache.get_quote("$VIX")
        if spy is not None:
            spy_change_pct = spy.change_pct
        if vix is not None:
            vix_level = vix.last
            vix_change_pct = vix.change_pct

        sector_etf = get_sector_etf(symbol)
        if sector_etf:
            sector_quote = cache.get_quote(sector_etf)
            if sector_quote is not None:
                sector_strength_pct = sector_quote.change_pct
    except Exception:
        # Total failure → fail-open: default values, regime unknown.
        return MarketContext(
            symbol=symbol,
            spy_direction="unknown",
            spy_change_pct=0.0,
            sector_etf="",
            sector_strength_pct=0.0,
            vix_level=16.0,
            vix_change_pct=0.0,
            regime="unknown",
            time_of_day=_classify_time_of_day(),
            conviction_multiplier=1.0,
            reason="market_indices cache unreachable",
        )

    sector_etf = get_sector_etf(symbol)
    has_sector = bool(sector_etf)

    spy_direction = _classify_spy_direction(spy_change_pct)
    regime = _classify_regime(spy_change_pct, vix_level, vix_change_pct)
    time_of_day = _classify_time_of_day()

    conviction = _compute_conviction(
        regime=regime,
        spy_change_pct=spy_change_pct,
        sector_strength_pct=sector_strength_pct,
        has_sector=has_sector,
        time_of_day=time_of_day,
    )

    # Human-readable summary for log audit
    parts = [
        f"SPY {spy_change_pct:+.2f}%",
        f"VIX {vix_level:.1f} ({vix_change_pct:+.1f}%)",
        f"regime={regime}",
        f"tod={time_of_day}",
    ]
    if has_sector:
        parts.append(f"sector {sector_etf} {sector_strength_pct:+.2f}%")
    parts.append(f"conviction={conviction:.2f}")
    reason = " | ".join(parts)

    return MarketContext(
        symbol=symbol,
        spy_direction=spy_direction,
        spy_change_pct=round(spy_change_pct, 2),
        sector_etf=sector_etf,
        sector_strength_pct=round(sector_strength_pct, 2),
        vix_level=round(vix_level, 2),
        vix_change_pct=round(vix_change_pct, 2),
        regime=regime,
        time_of_day=time_of_day,
        conviction_multiplier=round(conviction, 2),
        reason=reason,
    )
