#!/usr/bin/env python
"""v-learning-loop-2026-10-22: add nullable cost columns to bot_trades.

  fees_usd           REAL  total commissions + regulatory fees for the round trip
  entry_signal_price REAL  signal_router accepted entry (for fill-vs-signal slippage)
  exit_signal_price  REAL  price the exit decision was taken at
  slippage_usd       REAL  signed (+ = paid) fill-vs-signal cost

Idempotent (ADD COLUMN IF NOT EXISTS), all NULLable, no defaults, no backfill,
no index -> metadata-only change in Postgres. Dry-run by default; pass --apply.
Run in a quiet window: `python scripts/migrate_fee_columns.py --dsn $POSTGRES_DSN --apply`.
"""
from __future__ import annotations

import argparse
import sys

COLUMNS = (("fees_usd", "REAL"), ("entry_signal_price", "REAL"),
           ("exit_signal_price", "REAL"), ("slippage_usd", "REAL"))


def statements() -> list:
    return [f"ALTER TABLE bot_trades ADD COLUMN IF NOT EXISTS {c} {t}" for c, t in COLUMNS]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn"); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    for s in statements():
        print(s + ";")
    if not a.apply:
        print("-- dry run (pass --apply --dsn ... to execute)")
        return 0
    if not a.dsn:
        print("--dsn required with --apply", file=sys.stderr); return 2
    from sqlalchemy import create_engine, text
    eng = create_engine(a.dsn.replace("+asyncpg", ""))
    with eng.begin() as conn:
        conn.execute(text("SET LOCAL lock_timeout = '3s'"))
        for s in statements():
            conn.execute(text(s))
    print("applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
