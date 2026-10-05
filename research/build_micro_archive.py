"""Build tick-rule minute bins for the target symbols from trade_ticks, one
parquet per symbol-day under data/micro_archive/<SYM>/<date>.parquet.
Queries one DAY at a time (BRIN on ts), all symbols at once, then splits."""
import sys, time, os
sys.path.insert(0, '/home/nvidia/claude/grok/trading_bot_commentary')
import pandas as pd
from sqlalchemy import create_engine, text
from backtest.engine import DEFAULT_DSN
from research.strategy_hunt.data.micro import archive_session_window, archive_wall_clock, minute_bins_from_ticks

SYMS = ["MU", "TSLA", "META", "PLTR", "SMCI", "NVDA", "AMD", "AVGO", "AMZN", "MSFT", "GOOGL", "HOOD", "COIN", "INTC",
        "AAPL", "UBER", "SOFI", "SNAP", "ROKU", "DKNG"]
ROOT = "/home/nvidia/claude/grok/trading_bot_commentary/data/micro_archive"
e = create_engine(DEFAULT_DSN)
with e.connect() as c:
    c.execute(text("set statement_timeout = 0"))
    lo, hi = c.execute(text("select min(ts), max(ts) from trade_ticks")).fetchone()
days = pd.bdate_range(pd.Timestamp(lo).tz_convert("America/New_York").date(), pd.Timestamp(hi).tz_convert("America/New_York").date())
print(f"days={len(days)} {days[0].date()}..{days[-1].date()}", flush=True)
q = text("SELECT ts, symbol, price, size FROM trade_ticks WHERE ts >= :d0 AND ts < :d1 AND symbol = ANY(:syms) ORDER BY symbol, ts")
for d in days:
    day = d.strftime("%Y-%m-%d")
    done = all(os.path.exists(f"{ROOT}/{s}/{day}.parquet") for s in SYMS)
    if done:
        continue
    t0 = time.time()
    d0, d1 = archive_session_window(day)
    with e.connect() as c:
        c.execute(text("set statement_timeout = 0"))
        df = pd.read_sql(q, c, params={"d0": d0, "d1": d1, "syms": SYMS})
    n = 0
    if not df.empty:
        df["ts"] = archive_wall_clock(df["ts"])
        df["price"] = df["price"].astype(float); df["size"] = df["size"].astype(float)
        for sym, g in df.groupby("symbol"):
            bins = minute_bins_from_ticks(g[["ts", "price", "size"]])
            if bins.empty:
                continue
            os.makedirs(f"{ROOT}/{sym}", exist_ok=True)
            bins.to_parquet(f"{ROOT}/{sym}/{day}.parquet", index=False)
            n += 1
    for s in SYMS:   # mark symbols absent that day so the day is not re-queried
        p = f"{ROOT}/{s}/{day}.parquet"
        if not os.path.exists(p):
            os.makedirs(f"{ROOT}/{s}", exist_ok=True)
            pd.DataFrame().to_parquet(p, index=False)
    print(f"{day}: rows={len(df)} symbols={n} {time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
