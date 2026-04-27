"""Triple-barrier labeling and sample uniqueness weights.

Implements López de Prado's labeling methodology from *Advances in
Financial Machine Learning* (2018), chapters 3 and 4.

The naive fixed-horizon label ("was the return after N bars positive?")
ignores path-dependence: it can mark a trade as a winner even when a
stop-loss would have fired earlier. Triple-barrier labeling simulates
the actual execution path (take-profit, stop-loss, or time expiry) so
training labels match what live trading would experience.

Sample uniqueness corrects for the fact that overlapping label windows
share outcomes, which inflates the effective sample count seen by the
model and drives overfitting.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def apply_triple_barrier(
    prices: pd.Series,
    events: Iterable[pd.Timestamp],
    atr: pd.Series,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 60,
) -> pd.DataFrame:
    """Label each event by which of three barriers is touched first.

    Parameters
    ----------
    prices:
        Close prices indexed by timestamp. Must be strictly monotonic.
    events:
        Timestamps at which to start a labeling window. Must be a subset
        of ``prices.index``.
    atr:
        Average True Range at each bar, aligned with ``prices``. Used to
        scale the profit-target and stop-loss barriers per symbol volatility.
    pt_mult:
        Profit-target barrier = entry + pt_mult * ATR. Touched first → +1.
    sl_mult:
        Stop-loss barrier = entry - sl_mult * ATR. Touched first → -1.
    max_holding:
        Time barrier, in number of bars after entry. If neither price
        barrier is hit, label is 0.

    Returns
    -------
    DataFrame indexed by event timestamp with columns:
        - ``label``: +1 (upper), -1 (lower), 0 (time expired)
        - ``touch_time``: when the resolving barrier was hit
        - ``ret``: realized return from entry close to touch close

    Raises
    ------
    ValueError
        If any ATR value at an event is non-positive.
    KeyError
        If an event timestamp is not in ``prices.index``.
    """
    if pt_mult <= 0 or sl_mult <= 0:
        raise ValueError("pt_mult and sl_mult must be positive")
    if max_holding <= 0:
        raise ValueError("max_holding must be positive")

    price_index = prices.index
    records = []

    for event_time in events:
        # KeyError if not present — explicit behavior for callers.
        entry_loc = price_index.get_loc(event_time)
        entry_price = float(prices.iloc[entry_loc])
        atr_at_entry = float(atr.iloc[entry_loc])

        if atr_at_entry <= 0:
            raise ValueError(
                f"ATR at {event_time} is non-positive ({atr_at_entry}); "
                "cannot compute barriers"
            )

        upper = entry_price + pt_mult * atr_at_entry
        lower = entry_price - sl_mult * atr_at_entry

        end_loc = min(entry_loc + max_holding, len(price_index) - 1)
        # Walk the window AFTER entry (barriers cannot resolve at entry bar).
        window = prices.iloc[entry_loc + 1 : end_loc + 1]

        label = 0
        touch_time = price_index[end_loc]
        touch_price = float(prices.iloc[end_loc])

        for ts, px in window.items():
            px = float(px)
            if px >= upper:
                label = 1
                touch_time = ts
                touch_price = px
                break
            if px <= lower:
                label = -1
                touch_time = ts
                touch_price = px
                break

        ret = (touch_price - entry_price) / entry_price
        records.append(
            {"label": label, "touch_time": touch_time, "ret": ret}
        )

    return pd.DataFrame(records, index=pd.Index(list(events), name="event_time"))


def triple_barrier_labels(
    events: pd.DataFrame,
    close: pd.Series,
    atr: pd.Series,
    pt_sl: tuple[float, float] = (2.0, 1.0),
    vertical_barrier: int | pd.Series = 60,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """Triple-barrier labeler with asymmetric barriers and ``min_ret`` filter.

    AFML §3.4. Wraps :func:`apply_triple_barrier` (legacy single-vertical API)
    with the surface required by P4:

    * ``pt_sl`` — tuple ``(pt_mult, sl_mult)``; supports asymmetric barriers.
    * ``vertical_barrier`` — int (uniform bar count) or per-event Series.
    * ``min_ret`` — drop events whose realized ``|ret| < min_ret * atr/entry``
      (AFML p. 47 — discards barely-resolving events that bias the model
      toward the "decision is irrelevant here" regime).
    * ``side`` — optional column on ``events``; +1 long, -1 short. ``bin``
      is reported in the side-adjusted frame, so a short whose price falls
      gets ``bin=+1``.

    Degenerate inputs (event timestamp not in ``close.index``, non-positive
    ATR) are silently dropped — the caller is now feeding many candidate
    events from CUSUM/stride and cannot pre-clean each one. The legacy
    :func:`apply_triple_barrier` still raises on these conditions for
    callers (meta-labeling) that pre-validate.

    Returns
    -------
    DataFrame indexed by event timestamp with columns:
        - ``t1``: barrier-touch timestamp (price barrier, vertical, or last bar).
        - ``ret``: realized decimal return from entry close to touch close
          (in the long frame; sign-flipped for shorts before the bin assignment).
        - ``bin``: +1 if the trade made money in its own direction, -1 if it
          lost, 0 if the vertical fired without barrier hit.
        - ``side``: +1 long, -1 short.
    """
    pt_mult, sl_mult = pt_sl
    if pt_mult <= 0 or sl_mult <= 0:
        raise ValueError("pt_sl multiples must be positive")

    if isinstance(vertical_barrier, pd.Series):
        v_lookup = vertical_barrier.reindex(events.index)
    else:
        v_lookup = pd.Series(int(vertical_barrier), index=events.index, dtype="int64")

    side_series = (
        events["side"].astype(int) if "side" in events.columns
        else pd.Series(1, index=events.index, dtype="int64")
    )

    price_index = close.index
    kept_idx: list[pd.Timestamp] = []
    records: list[dict] = []

    for ts in events.index:
        if ts not in price_index:
            continue
        pos = price_index.get_loc(ts)
        atr_at = float(atr.iloc[pos])
        if not np.isfinite(atr_at) or atr_at <= 0:
            continue
        max_h = int(v_lookup.loc[ts])
        if max_h <= 0:
            continue
        entry = float(close.iloc[pos])

        # Delegate barrier walk to the proven primitive. The labeler walks
        # the long frame; we sign-correct for shorts here.
        leg = apply_triple_barrier(
            prices=close, events=[ts], atr=atr,
            pt_mult=pt_mult, sl_mult=sl_mult, max_holding=max_h,
        ).iloc[0]

        long_ret = float(leg["ret"])

        # AFML p. 47: drop |ret| < min_ret * (atr/entry). Per-event scale
        # avoids over-dropping quiet symbols.
        if min_ret > 0 and abs(long_ret) < min_ret * (atr_at / entry):
            continue

        side = int(side_series.loc[ts])
        kept_idx.append(ts)
        records.append({
            "t1": leg["touch_time"],
            "ret": long_ret * side,
            "bin": int(leg["label"]) * side,
            "side": side,
        })

    if not records:
        return pd.DataFrame(columns=["t1", "ret", "bin", "side"])

    return pd.DataFrame(records, index=pd.Index(kept_idx, name=events.index.name))


def compute_uniqueness(
    labels: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    groups: pd.Series | None = None,
) -> pd.Series:
    """Compute average uniqueness weight per sample (AFML §4.3).

    For each bar in ``price_index``, count how many label windows are
    "alive" (cover that bar). Each sample's uniqueness at a bar is
    ``1 / active_count``. The sample's weight is the mean of its
    uniqueness values across the bars its window spans.

    Parameters
    ----------
    labels:
        DataFrame with ``touch_time`` column, indexed by event start time.
    price_index:
        The full price-bar index (must contain every event and touch bar).
    groups:
        Optional positional group key per sample (same length as
        ``labels``). Samples with different group keys do not contribute
        to each other's concurrency. Required for multi-symbol training
        where a 10:00 AM NVDA event is independent of a 10:00 AM AAPL
        event — without grouping, they would falsely share concurrency
        and collapse both weights toward zero.

    Returns
    -------
    Series of weights in (0, 1], indexed to match ``labels.index``.
    """
    if groups is not None:
        groups_arr = np.asarray(groups)
        if len(groups_arr) != len(labels):
            raise ValueError(
                f"groups length ({len(groups_arr)}) must equal labels length "
                f"({len(labels)})"
            )
        result = np.zeros(len(labels), dtype="float64")
        # Position-based iteration so duplicate-index labels are handled.
        # Use np.where → positional indexing so we're independent of
        # label uniqueness and of pandas 2.x iloc-with-mask behavior.
        for key in pd.unique(groups_arr):
            mask = groups_arr == key
            positions = np.where(mask)[0]
            sub_labels = labels.iloc[positions]
            # Per-group price index is the union of that group's events +
            # touches — no need for a caller-supplied master index.
            sub_index = pd.DatetimeIndex(
                np.unique(
                    np.concatenate([
                        sub_labels.index.to_numpy(),
                        sub_labels["touch_time"].to_numpy(),
                    ])
                )
            ).sort_values()
            result[positions] = compute_uniqueness(sub_labels, sub_index).to_numpy()
        return pd.Series(result, index=labels.index, dtype="float64")

    # Single-timeline path.
    concurrency = pd.Series(0, index=price_index, dtype="int64")
    for start, touch in zip(labels.index, labels["touch_time"]):
        concurrency.loc[start:touch] += 1

    per_bar = 1.0 / concurrency.replace(0, np.nan)

    weights = pd.Series(0.0, index=labels.index, dtype="float64")
    for i, (start, touch) in enumerate(zip(labels.index, labels["touch_time"])):
        weights.iloc[i] = per_bar.loc[start:touch].mean()

    return weights
