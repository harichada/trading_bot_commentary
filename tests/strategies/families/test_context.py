import numpy as np
import pandas as pd


def _ind(**cols):
    n = len(next(iter(cols.values())))
    df = pd.DataFrame(index=pd.RangeIndex(n))
    for k, v in cols.items():
        df[k] = v
    return df


def test_ctx_columns_tuple():
    from strategies.families.context import CTX_COLUMNS
    assert CTX_COLUMNS == (
        "rvol_open5", "rvol_open15", "gap_pct", "rs_20", "rs_60", "rs_120",
        "rs_rank_60", "rvol_rank5", "pct_from_52w_high", "adv_dollars_20",
        "adv_bucket", "atr14_pct", "earnings_today", "earnings_prev",
        "eps_surprise_pct", "news_count_24h", "news_sent_24h",
    )


def test_ctx_returns_float_array_for_numeric_column():
    from strategies.families.context import ctx
    ind = _ind(ctx_gap_pct=[1.0, 2.0, np.nan, 4.0, 5.0])
    out = ctx(ind, "gap_pct")
    assert out.dtype == float
    assert np.allclose(out, [1.0, 2.0, np.nan, 4.0, 5.0], equal_nan=True)


def test_ctx_missing_column_is_all_nan():
    from strategies.families.context import ctx
    ind = _ind(other=[1, 2, 3])
    out = ctx(ind, "rvol_rank5")
    assert len(out) == 3
    assert np.isnan(out).all()


def test_ctx_bool_column_mapped_to_0_1():
    from strategies.families.context import ctx
    ind = _ind(ctx_earnings_today=[True, False, True, False, True])
    out = ctx(ind, "earnings_today")
    assert out.dtype == float
    assert np.array_equal(out, [1.0, 0.0, 1.0, 0.0, 1.0])


def test_ctx_adv_bucket_mapped_and_defaults_to_nan():
    from strategies.families.context import ctx
    ind = _ind(ctx_adv_bucket=["mega", "large", "mid", "small", "unknown"])
    out = ctx(ind, "adv_bucket")
    assert np.array_equal(out[:4], [3.0, 2.0, 1.0, 0.0])
    assert np.isnan(out[4])


def test_in_play_true_when_rank_within_top_n():
    from strategies.families.context import in_play
    ind = _ind(ctx_rvol_rank5=[1, 2, 3, 4, 5])
    out = in_play(ind, 3)
    assert out.dtype == bool
    assert np.array_equal(out, [True, True, True, False, False])


def test_in_play_nan_rank_is_false():
    from strategies.families.context import in_play
    ind = _ind(ctx_rvol_rank5=[1.0, np.nan, 3.0])
    out = in_play(ind, 3)
    assert np.array_equal(out, [True, False, True])
