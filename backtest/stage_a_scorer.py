"""
Stage A Scorer - Evaluates strategy performance against locked Stage A floors.

Locked Stage A Floors (DO NOT SOFTEN):
- n >= 150 closed trades
- >= 10 sessions
- PF >= 1.30 (fees + slippage included)
- WR >= 48% (scratches |R| < 0.05 out of rate, in n)
- exp >= +0.05 R
- DD <= 6% allocated
- max_losing_day <= 2.0 R
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd


HANDS_OFF_SYMBOLS = {'MU', 'HQGE', 'SPCX'}

STAGE_A_FLOORS = {
    'min_n': 150,
    'min_sessions': 10,
    'min_pf': 1.30,
    'min_wr': 0.48,
    'min_exp_r': 0.05,
    'max_dd_pct': 0.06,
    'max_losing_day_r': 2.0,
}


@dataclass
class StageAMetrics:
    """Stage A metrics for a single lane."""
    lane: str
    direction: str
    n: int
    sessions: int
    pf: float
    wr: float
    exp_r: float
    max_dd_pct: float
    max_losing_day_r: float
    verdict: str  # PASS / FAIL / INCOMPLETE
    recommendation: str  # LIVE on / LIVE off
    notes: str = ""
    risk_off_excluded: bool = True
    sample_start: Optional[str] = None
    sample_end: Optional[str] = None
    floor_failures: List[str] = None

    def __post_init__(self):
        if self.floor_failures is None:
            self.floor_failures = []

    def to_dict(self) -> Dict:
        return asdict(self)


def check_floors(metrics: StageAMetrics) -> Tuple[str, List[str]]:
    """
    Check metrics against locked Stage A floors.
    Returns (verdict, list of failed floors).
    """
    failures = []
    
    if metrics.n < STAGE_A_FLOORS['min_n']:
        failures.append(f"n={metrics.n} < {STAGE_A_FLOORS['min_n']}")
    
    if metrics.sessions < STAGE_A_FLOORS['min_sessions']:
        failures.append(f"sessions={metrics.sessions} < {STAGE_A_FLOORS['min_sessions']}")
    
    if metrics.pf < STAGE_A_FLOORS['min_pf']:
        failures.append(f"PF={metrics.pf:.3f} < {STAGE_A_FLOORS['min_pf']}")
    
    if metrics.wr < STAGE_A_FLOORS['min_wr']:
        failures.append(f"WR={metrics.wr:.1%} < {STAGE_A_FLOORS['min_wr']:.0%}")
    
    if metrics.exp_r < STAGE_A_FLOORS['min_exp_r']:
        failures.append(f"exp_R={metrics.exp_r:+.3f} < +{STAGE_A_FLOORS['min_exp_r']}")
    
    if metrics.max_dd_pct > STAGE_A_FLOORS['max_dd_pct']:
        failures.append(f"maxDD={metrics.max_dd_pct:.1%} > {STAGE_A_FLOORS['max_dd_pct']:.0%}")
    
    if abs(metrics.max_losing_day_r) > STAGE_A_FLOORS['max_losing_day_r']:
        failures.append(f"max_losing_day={metrics.max_losing_day_r:.2f}R > {STAGE_A_FLOORS['max_losing_day_r']}R")
    
    if not failures:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    
    return verdict, failures


def compute_stage_a_from_trades(
    trades: pd.DataFrame,
    lane: str,
    direction: str,
    exclude_risk_off: bool = True
) -> StageAMetrics:
    """
    Compute Stage A metrics from a DataFrame of resolved trades.
    
    Expected columns:
    - symbol: str
    - session_date: date
    - entry_time: datetime
    - exit_time: datetime  
    - r_multiple: float (realized R)
    - pnl_gross: float
    - fees: float
    - slippage: float
    - risk_off: bool (optional)
    """
    if trades.empty:
        return StageAMetrics(
            lane=lane,
            direction=direction,
            n=0,
            sessions=0,
            pf=0.0,
            wr=0.0,
            exp_r=0.0,
            max_dd_pct=0.0,
            max_losing_day_r=0.0,
            verdict="INCOMPLETE",
            recommendation="LIVE off",
            notes="No trades in sample",
            risk_off_excluded=exclude_risk_off
        )
    
    df = trades.copy()
    df = df[~df['symbol'].isin(HANDS_OFF_SYMBOLS)]
    
    if exclude_risk_off and 'risk_off' in df.columns:
        df = df[~df['risk_off']]
    
    if df.empty:
        return StageAMetrics(
            lane=lane,
            direction=direction,
            n=0,
            sessions=0,
            pf=0.0,
            wr=0.0,
            exp_r=0.0,
            max_dd_pct=0.0,
            max_losing_day_r=0.0,
            verdict="INCOMPLETE",
            recommendation="LIVE off",
            notes="No trades after hands-off/risk_off exclusion",
            risk_off_excluded=exclude_risk_off
        )
    
    n = len(df)
    
    if 'session_date' in df.columns:
        sessions = df['session_date'].nunique()
    else:
        sessions = 0
    
    scratches = df[df['r_multiple'].abs() < 0.05]
    non_scratches = df[df['r_multiple'].abs() >= 0.05]
    
    wins = non_scratches[non_scratches['r_multiple'] > 0]
    losses = non_scratches[non_scratches['r_multiple'] <= 0]
    
    if len(non_scratches) > 0:
        wr = len(wins) / len(non_scratches)
    else:
        wr = 0.0
    
    gross_wins = wins['r_multiple'].sum() if len(wins) > 0 else 0.0
    gross_losses = abs(losses['r_multiple'].sum()) if len(losses) > 0 else 0.0
    
    if gross_losses > 0:
        pf = gross_wins / gross_losses
    elif gross_wins > 0:
        pf = float('inf')
    else:
        pf = 0.0
    
    exp_r = df['r_multiple'].mean()
    
    df_sorted = df.sort_values('exit_time' if 'exit_time' in df.columns else 'entry_time')
    cum_r = df_sorted['r_multiple'].cumsum()
    running_max = cum_r.expanding().max()
    drawdown = cum_r - running_max
    max_dd_r = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
    max_dd_pct = max_dd_r / 100.0 * 6.0 if max_dd_r > 0 else 0.0
    
    if 'session_date' in df.columns:
        daily_r = df.groupby('session_date')['r_multiple'].sum()
        max_losing_day_r = abs(daily_r.min()) if len(daily_r) > 0 and daily_r.min() < 0 else 0.0
    else:
        max_losing_day_r = 0.0
    
    sample_start = str(df['session_date'].min()) if 'session_date' in df.columns else None
    sample_end = str(df['session_date'].max()) if 'session_date' in df.columns else None
    
    metrics = StageAMetrics(
        lane=lane,
        direction=direction,
        n=n,
        sessions=sessions,
        pf=round(pf, 3) if pf != float('inf') else 999.0,
        wr=round(wr, 4),
        exp_r=round(exp_r, 4),
        max_dd_pct=round(max_dd_pct, 4),
        max_losing_day_r=round(max_losing_day_r, 2),
        verdict="INCOMPLETE",
        recommendation="LIVE off",
        risk_off_excluded=exclude_risk_off,
        sample_start=sample_start,
        sample_end=sample_end
    )
    
    verdict, failures = check_floors(metrics)
    metrics.verdict = verdict
    metrics.floor_failures = failures
    
    if verdict == "PASS":
        metrics.recommendation = "LIVE on"
    else:
        metrics.recommendation = "LIVE off"
    
    return metrics


def create_incomplete_metrics(lane: str, direction: str, reason: str) -> StageAMetrics:
    """Create INCOMPLETE metrics placeholder when data is unavailable."""
    return StageAMetrics(
        lane=lane,
        direction=direction,
        n=0,
        sessions=0,
        pf=0.0,
        wr=0.0,
        exp_r=0.0,
        max_dd_pct=0.0,
        max_losing_day_r=0.0,
        verdict="INCOMPLETE",
        recommendation="LIVE off",
        notes=reason
    )


def format_metrics_table(metrics: StageAMetrics) -> str:
    """Format metrics as markdown table row."""
    wr_str = f"{metrics.wr*100:.1f}%" if metrics.wr > 0 else "N/A"
    pf_str = f"{metrics.pf:.3f}" if metrics.pf > 0 else "N/A"
    exp_str = f"{metrics.exp_r:+.3f}R" if metrics.exp_r != 0 else "N/A"
    dd_str = f"{metrics.max_dd_pct*100:.1f}%" if metrics.max_dd_pct > 0 else "N/A"
    mld_str = f"{metrics.max_losing_day_r:.2f}R" if metrics.max_losing_day_r > 0 else "N/A"
    
    return f"| {metrics.n} | {metrics.sessions} | {pf_str} | {wr_str} | {exp_str} | {dd_str} | {mld_str} |"
