"""Side-aware (meta) triple-barrier labeling.

Meta-labeling separates *side decision* from *take-or-pass decision*
(López de Prado, AFML §3.6-3.7). The primary model (rule-based strategy
or a primary classifier) decides whether a bet is long or short. This
module labels each such bet as win (1) or loss (0) based on which
barrier gets touched first, interpreted from the primary's perspective:

  side = +1 (long):  upper barrier → win,  lower → loss,  time → loss
  side = -1 (short): lower barrier → win,  upper → loss,  time → loss

Time-expiry is treated as a loss because a strategy that fails to pay
off within its horizon has not earned its carry cost. This matches the
book's recommendation for meta-labeling (§3.7): the meta-model should
learn to pass on setups that do not decisively resolve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.labels import apply_triple_barrier


def apply_meta_triple_barrier(
    prices: pd.Series,
    events: pd.DataFrame,
    atr: pd.Series,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 60,
) -> pd.DataFrame:
    """Binary side-aware labeling for meta-labeling.

    Parameters
    ----------
    prices:
        Close prices indexed by timestamp.
    events:
        DataFrame indexed by event timestamp with a ``side`` column of
        ``+1`` (long) or ``-1`` (short).
    atr:
        ATR series aligned with ``prices``.
    pt_mult, sl_mult, max_holding:
        Barrier configuration. Barriers are always set symmetrically
        around the entry price; it's the *interpretation* that is
        side-aware (upper wins for longs, lower wins for shorts).

    Returns
    -------
    DataFrame indexed by event timestamp with columns:
        ``bin``        — 1 (win) or 0 (loss)
        ``touch_time`` — first-touch barrier timestamp
        ``ret``        — strategy-PnL return (positive = win, negative = loss)
    """
    if "side" not in events.columns:
        raise KeyError("events must contain a 'side' column")
    sides = events["side"].to_numpy()
    if not np.isin(sides, [1, -1]).all():
        bad = np.setdiff1d(np.unique(sides), [1, -1])
        raise ValueError(f"side must be +1 or -1; got {bad.tolist()}")

    # Delegate first-touch resolution to the generic triple-barrier
    # labeler. Its ``label`` output is +1 / -1 / 0 for upper / lower /
    # time expiry, and ``ret`` is the raw (long-perspective) return.
    base = apply_triple_barrier(
        prices=prices,
        events=events.index,
        atr=atr,
        pt_mult=pt_mult,
        sl_mult=sl_mult,
        max_holding=max_holding,
    )

    base_label = base["label"].to_numpy()
    base_ret = base["ret"].to_numpy()

    # Translate first-touch to strategy-PnL.
    #   Long  + upper = win (+)     Long  + lower = loss (-)
    #   Short + lower = win (+)     Short + upper = loss (-)
    #   Either + time-expiry = loss (flat return → treated as 0 PnL)
    strategy_ret = base_ret * sides
    bin_labels = (base_label == sides).astype(np.int64)

    return pd.DataFrame(
        {
            "bin": bin_labels,
            "touch_time": base["touch_time"].values,
            "ret": strategy_ret,
        },
        index=events.index,
    )
