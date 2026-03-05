"""VWAP Gap Fade Strategy — Strategy #2 (PDF Playbook).

Implements the full PDF playbook:
- Gap type classification (breakaway/runaway/exhaustion/common)
- VWAP confirmation (price rejected from below VWAP for shorts)
- Opening Range breakdown (price < OR low for shorts)
- HOD-based stop with buffer
- R:R targets (1R partial, min(2R, prev_close) full)
- 9 EMA trailing stop on 5-min candles
- EMA cross exit
- Delayed entry window (9:45 instead of 9:31)
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal, GapFadeStrategy
from .registry import GapFadeStrategyRegistry

# Default config values
_DEFAULTS = {
    'opening_range_minutes': 5,
    'entry_after_minute': 45,
    'entry_cutoff_hour': 11,
    'entry_cutoff_minute': 30,
    'require_vwap_reject': True,
    'require_or_breakdown': True,
    'hod_stop_buffer_pct': 0.003,
    'max_stop_pct': 0.02,
    'ema_trailing_period': 9,
    'ema_trailing_buffer_pct': 0.001,
    'target_1_r_multiple': 1.0,
    'target_2_r_multiple': 2.0,
    'skip_breakaway_gaps': True,
    'skip_runaway_gaps': True,
    'min_gap_pct_exhaustion': 0.15,  # gaps > 15% more likely exhaustion
    'vol_ratio_exhaustion': 2.0,     # vol_ratio > 2x suggests conviction
}


def _classify_gap(gap_pct: float, vol_ratio: float, catalyst: str) -> str:
    """Classify gap type based on gap size, volume, and catalyst.

    Returns: 'breakaway', 'runaway', 'exhaustion', or 'common'
    """
    gap_abs = abs(gap_pct)

    # Breakaway: first move out of a base, usually with catalyst + high volume
    # Indicators: earnings, FDA, M&A with >2x volume and large gap
    if catalyst in ('earnings', 'fda', 'ma') and vol_ratio > 2.0 and gap_abs > 0.10:
        return 'breakaway'

    # Runaway/continuation: high volume continuation of existing trend
    # Hard to detect without multi-day context, approximate with high vol + moderate gap
    if vol_ratio > 2.5 and gap_abs > 0.05 and catalyst not in ('', 'noise'):
        return 'runaway'

    # Exhaustion: large gap on declining relative volume (blow-off top)
    # The fade opportunity — gap is overextended
    if gap_abs > 0.15 or (gap_abs > 0.08 and vol_ratio < 1.5):
        return 'exhaustion'

    # Common: small gap on normal volume, no significant catalyst
    return 'common'


@GapFadeStrategyRegistry.register('vwap_gap_fade')
class VWAPGapFadeStrategy(GapFadeStrategy):
    """PDF playbook: VWAP confirmation + Opening Range + EMA trailing stop.

    Key differences from Classic:
    - Delayed entry (9:45 vs 9:31)
    - Requires VWAP rejection + OR breakdown for entry
    - HOD-based stop instead of fixed %
    - 9 EMA trailing stop on 5-min candles
    - Gap type classification (skips breakaway/runaway)
    - R:R-based targets instead of midpoint/prev_close
    """

    name = 'VWAP Gap Fade'
    description = (
        'PDF playbook strategy: delayed entry at 9:45 AM with VWAP rejection '
        'and Opening Range breakdown confirmation. HOD-based stop, 9 EMA '
        'trailing on 5-min candles, gap type classification.'
    )
    version = '1.0'

    def get_default_config(self) -> Dict:
        return dict(_DEFAULTS)

    # ── Scanning ──────────────────────────────────────────────────────

    def filter_candidate(self, candidate: dict) -> Tuple[bool, str]:
        """Classify gap type and filter breakaway/runaway gaps."""
        gap_pct = candidate.get('gap_pct', 0)
        vol_ratio = candidate.get('vol_ratio', 1.0)
        catalyst = candidate.get('catalyst', '')

        gap_type = _classify_gap(gap_pct, vol_ratio, catalyst)

        if gap_type == 'breakaway' and self._config.get('skip_breakaway_gaps', True):
            return False, f'breakaway gap (catalyst={catalyst}, vol={vol_ratio:.1f}x)'

        if gap_type == 'runaway' and self._config.get('skip_runaway_gaps', True):
            return False, f'runaway gap (vol={vol_ratio:.1f}x)'

        return True, f'gap_type={gap_type}'

    def score_candidate(self, candidate: dict, regime: Optional[dict] = None) -> Optional[float]:
        """Boost exhaustion gaps, penalize gaps without clear type."""
        gap_pct = candidate.get('gap_pct', 0)
        vol_ratio = candidate.get('vol_ratio', 1.0)
        catalyst = candidate.get('catalyst', '')
        base_score = candidate.get('score', 50.0)

        gap_type = _classify_gap(gap_pct, vol_ratio, catalyst)

        if gap_type == 'exhaustion':
            return base_score + 15.0  # exhaustion gaps are prime fade candidates
        elif gap_type == 'common':
            return base_score + 5.0   # common gaps fade reliably

        return None  # keep engine score

    # ── Entry ─────────────────────────────────────────────────────────

    def get_entry_window(self) -> Optional[Tuple[int, int, int, int]]:
        """Delayed entry: 9:45 to cutoff (default 11:30)."""
        entry_min = self._config.get('entry_after_minute', 45)
        cutoff_h = self._config.get('entry_cutoff_hour', 11)
        cutoff_m = self._config.get('entry_cutoff_minute', 30)
        return (9, entry_min, cutoff_h, cutoff_m)

    def should_enter_now(self, candidate: dict, price: float,
                         tick_data: Optional[dict] = None,
                         now: Optional[datetime] = None) -> Tuple[bool, str]:
        """Require VWAP rejection and OR breakdown for entry."""
        direction = candidate.get('direction', 'short')

        if tick_data is None:
            # No indicator data yet — allow entry (degrade gracefully)
            return True, 'no indicator data available'

        reasons = []

        # VWAP check
        if self._config.get('require_vwap_reject', True):
            vwap = tick_data.get('vwap', 0)
            if vwap > 0:
                if direction == 'short' and price >= vwap:
                    return False, f'price ${price:.2f} >= VWAP ${vwap:.2f} (need rejection from below)'
                elif direction == 'long' and price <= vwap:
                    return False, f'price ${price:.2f} <= VWAP ${vwap:.2f} (need rejection from above)'
                reasons.append(f'VWAP ${vwap:.2f} OK')

        # Opening Range check
        if self._config.get('require_or_breakdown', True):
            or_complete = tick_data.get('or_complete', False)
            if or_complete:
                or_low = tick_data.get('or_low')
                or_high = tick_data.get('or_high')
                if direction == 'short' and or_low and price >= or_low:
                    return False, f'price ${price:.2f} >= OR low ${or_low:.2f} (need breakdown)'
                elif direction == 'long' and or_high and price <= or_high:
                    return False, f'price ${price:.2f} <= OR high ${or_high:.2f} (need breakout)'
                reasons.append('OR confirmed')
            else:
                return False, 'opening range not yet complete'

        return True, '; '.join(reasons) if reasons else 'conditions met'

    # ── Stop & Targets ────────────────────────────────────────────────

    def compute_stop_price(self, entry_price: float, candidate: dict,
                           tick_data: Optional[dict] = None) -> Optional[float]:
        """HOD-based stop: high_of_day * (1 + buffer)."""
        direction = candidate.get('direction', 'short')
        buffer_pct = self._config.get('hod_stop_buffer_pct', 0.003)

        if tick_data and tick_data.get('day_high', 0) > 0:
            hod = tick_data['day_high']
            if direction == 'short':
                hod_stop = round(hod * (1 + buffer_pct), 2)
                max_stop = round(entry_price * (1 + self._config.get('max_stop_pct', 0.02)), 2)
                return min(hod_stop, max_stop)
            else:
                # For longs, day_low would be ideal but we track day_high
                # Fall back to engine default for longs
                return None

        # No tick data — fall back to engine default
        return None

    def compute_targets(self, entry_price: float, candidate: dict,
                        tick_data: Optional[dict] = None) -> Optional[Tuple[float, float]]:
        """R:R based targets: 1R partial, min(2R, prev_close) full."""
        direction = candidate.get('direction', 'short')
        prev_close = candidate.get('prev_close', 0)
        r1_mult = self._config.get('target_1_r_multiple', 1.0)
        r2_mult = self._config.get('target_2_r_multiple', 2.0)

        # Compute risk (R) = distance from entry to stop
        stop = self.compute_stop_price(entry_price, candidate, tick_data)
        if stop is None:
            return None  # no stop = can't compute R-based targets

        risk = abs(entry_price - stop)
        if risk < 0.01:
            return None

        if direction == 'short':
            half_target = entry_price - (risk * r1_mult)
            full_target_rr = entry_price - (risk * r2_mult)
            # Pick the further (lower) target for more profit
            full_target = min(full_target_rr, prev_close) if prev_close > 0 else full_target_rr
            # Midpoint floor: ensure partial target is at least at midpoint
            midpoint = (entry_price + prev_close) / 2 if prev_close > 0 else half_target
            half_target = min(half_target, midpoint)
        else:
            half_target = entry_price + (risk * r1_mult)
            full_target_rr = entry_price + (risk * r2_mult)
            # Pick the further (higher) target for more profit
            full_target = max(full_target_rr, prev_close) if prev_close > 0 else full_target_rr
            # Midpoint floor: ensure partial target is at least at midpoint
            midpoint = (entry_price + prev_close) / 2 if prev_close > 0 else half_target
            half_target = max(half_target, midpoint)

        return round(half_target, 2), round(full_target, 2)

    # ── Exit Management ───────────────────────────────────────────────

    def evaluate_exit(self, position: dict, price: float, high: float,
                      tick_data: Optional[dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """EMA cross exit: close when price crosses above 9 EMA (shorts)."""
        if tick_data is None:
            return None

        direction = position.get('direction', 'short')
        ema_key = f'ema{self._config.get("ema_trailing_period", 9)}'
        ema_val = tick_data.get(ema_key, 0)
        ema_init = tick_data.get('ema_initialized', False)

        if not ema_init or ema_val <= 0:
            return None

        remaining = position.get('remaining_shares', 0)
        if remaining <= 0:
            return None

        # Only trigger EMA cross exit after partial has filled (let the trade develop)
        if not position.get('partial_filled', False):
            return None

        if direction == 'short' and price > ema_val:
            return ExitSignal(
                action='close',
                reason=f'ema_cross (price ${price:.2f} > EMA ${ema_val:.2f})',
                shares=remaining,
            )
        elif direction == 'long' and price < ema_val:
            return ExitSignal(
                action='close',
                reason=f'ema_cross (price ${price:.2f} < EMA ${ema_val:.2f})',
                shares=remaining,
            )

        return None

    def update_trailing_stop(self, position: dict, price: float,
                             tick_data: Optional[dict] = None,
                             now: Optional[datetime] = None) -> Optional[float]:
        """9 EMA trailing stop on 5-min candles + buffer."""
        if tick_data is None:
            return None

        direction = position.get('direction', 'short')
        ema_init = tick_data.get('ema_initialized', False)

        if not ema_init:
            return None

        # Only start trailing after partial fill
        if not position.get('partial_filled', False):
            return None

        if direction == 'short':
            ema_stop = tick_data.get('ema_stop_short')
            if ema_stop and ema_stop > 0:
                current_stop = position.get('stop_price', float('inf'))
                # Only tighten (move stop down for shorts)
                if ema_stop < current_stop:
                    return round(ema_stop, 2)
        else:
            ema_stop = tick_data.get('ema_stop_long')
            if ema_stop and ema_stop > 0:
                current_stop = position.get('stop_price', 0)
                # Only tighten (move stop up for longs)
                if ema_stop > current_stop:
                    return round(ema_stop, 2)

        return None

    # ── Indicators ────────────────────────────────────────────────────

    def get_required_indicators(self) -> List[str]:
        return ['vwap', 'ema', 'opening_range', 'day_high']

    # ── LLM Prompt Overrides ──────────────────────────────────────────

    def get_llm_system_prompt(self) -> Optional[str]:
        return (
            "You are an autonomous trading supervisor for a VWAP-based gap fade strategy.\n"
            "The bot shorts stocks that gap up, using VWAP rejection and Opening Range breakdown "
            "as entry confirmation. For gap-down candidates, it goes long with VWAP/OR confirmation.\n\n"
            "Key strategy facts:\n"
            "- Entry is delayed to 9:45 AM to wait for VWAP/OR to establish.\n"
            "- Gaps are classified: exhaustion and common gaps are faded; breakaway and runaway are skipped.\n"
            "- Stop is set at high-of-day + 0.3% buffer (dynamic, not fixed %).\n"
            "- Targets are R:R based: 1R partial cover, 2R or prev_close full cover.\n"
            "- 9 EMA on 5-min candles acts as trailing stop after partial fill.\n"
            "- VWAP acts as dynamic support/resistance — price below VWAP confirms short thesis.\n\n"
            "Your responsibilities:\n"
            "1. SCAN TIMING: Same as classic — scan pre-market, re-scan at 9:25.\n"
            "2. CANDIDATE SELECTION: Prefer exhaustion gaps. Skip breakaway/runaway.\n"
            "3. ENTRY TIMING: Wait for VWAP rejection + OR breakdown (usually 9:45+).\n"
            "4. RISK MANAGEMENT: HOD stop means risk is defined by volatility, not fixed %.\n"
            "5. After 11:30 AM ET, do NOT enter new positions.\n\n"
            "Respond ONLY with valid JSON. No text outside the JSON object."
        )

    def get_llm_profit_prompt(self) -> Optional[str]:
        return (
            "You are a profit-taking advisor for a VWAP-based gap fade strategy.\n"
            "The bot uses VWAP, Opening Range, and 9 EMA on 5-min candles.\n\n"
            "You evaluate open positions that are IN PROFIT and decide whether to:\n"
            "- CLOSE: Take profit. Only when price crosses back above 9 EMA after partial fill.\n"
            "- HOLD: Let it run. DEFAULT bias — gap fades grind slowly.\n"
            "- TIGHTEN_STOP: Use 9 EMA as trailing stop. Preferred over closing.\n\n"
            "CRITICAL PRINCIPLES:\n"
            "- YOUR DEFAULT SHOULD BE HOLD. The 9 EMA trailing stop handles exits mechanically.\n"
            "- If price is below VWAP, the short thesis is intact — HOLD.\n"
            "- If price is between VWAP and 9 EMA, TIGHTEN to EMA level.\n"
            "- If price crosses above 9 EMA on volume, CLOSE.\n"
            "- Before noon: almost ALWAYS hold unless EMA cross occurs.\n"
            "- Gap fill < 60%: HOLD — trade hasn't reached potential.\n"
            "- Gap fill 60-90%: TIGHTEN_STOP to EMA.\n"
            "- Gap fill > 90%: may CLOSE — easy money is made.\n\n"
            "Respond ONLY with valid JSON. No text outside the JSON object."
        )

    # ── UI / Config ───────────────────────────────────────────────────

    def get_parameter_schema(self) -> Dict:
        return {
            'opening_range_minutes': {
                'type': 'int', 'label': 'Opening Range (min)',
                'default': 5, 'min': 1, 'max': 30,
                'description': 'Minutes after 9:30 to define opening range'
            },
            'entry_after_minute': {
                'type': 'int', 'label': 'Entry After Minute',
                'default': 45, 'min': 31, 'max': 59,
                'description': 'Minute past 9:XX to start entries (e.g. 45 = 9:45 AM)'
            },
            'require_vwap_reject': {
                'type': 'bool', 'label': 'Require VWAP Rejection',
                'default': True,
                'description': 'Require price below VWAP (shorts) for entry'
            },
            'require_or_breakdown': {
                'type': 'bool', 'label': 'Require OR Breakdown',
                'default': True,
                'description': 'Require price below Opening Range low (shorts) for entry'
            },
            'hod_stop_buffer_pct': {
                'type': 'float', 'label': 'HOD Stop Buffer %',
                'default': 0.003, 'min': 0.001, 'max': 0.02,
                'description': 'Buffer above high-of-day for stop placement'
            },
            'max_stop_pct': {
                'type': 'float', 'label': 'Max Stop Distance %',
                'default': 0.02, 'min': 0.005, 'max': 0.05,
                'description': 'Maximum stop distance from entry (caps HOD-based stop)'
            },
            'ema_trailing_period': {
                'type': 'int', 'label': 'EMA Trailing Period',
                'default': 9, 'min': 3, 'max': 50,
                'description': 'EMA period for trailing stop (on 5-min candles)'
            },
            'target_1_r_multiple': {
                'type': 'float', 'label': '1st Target (R multiple)',
                'default': 1.0, 'min': 0.5, 'max': 3.0,
                'description': 'R multiple for partial cover target'
            },
            'target_2_r_multiple': {
                'type': 'float', 'label': '2nd Target (R multiple)',
                'default': 2.0, 'min': 1.0, 'max': 5.0,
                'description': 'R multiple for full cover target'
            },
            'skip_breakaway_gaps': {
                'type': 'bool', 'label': 'Skip Breakaway Gaps',
                'default': True,
                'description': 'Filter out breakaway gaps (earnings/FDA/M&A + high volume)'
            },
            'skip_runaway_gaps': {
                'type': 'bool', 'label': 'Skip Runaway Gaps',
                'default': True,
                'description': 'Filter out runaway/continuation gaps'
            },
        }
