"""v-learning-loop-2026-10-22: nightly learning-loop driver.

Run after the strategy-hunt pipeline (scripts/nightly_strategy_hunt.sh), i.e.
AFTER the playbook updater has rewritten playbook.json. Every step is behind
its own flag (all default OFF); with every flag off this exits immediately.

  ENABLE_LEARNING_SCORECARD       data/learning/scorecard-<date>.json
  LEARNING_AUTODEMOTE_SHADOW      data/learning/demotions-<date>.json (would-demote log only)
  LEARNING_AUTODEMOTE_ENFORCE     + apply demotions to <DESK_DATA_DIR>/playbook.json (suspend only)
  LEARNING_EXIT_SHADOW            data/learning/exit_replay-<date>.md/.json
  ENABLE_LOSS_ATTRIBUTION_REPORT  docs/loss_attribution_<date>.md (Fridays, or --attribution-now)

Postgres access is SELECT-only (read-only transactions). Never places orders,
never edits Config, never promotes anything.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from core.learning import demoter
from core.learning.scorecard import rolling_scorecard, write_scorecard
from research.learning.data import DAY_TRADE_STRATEGIES, bars_for_db, load_db, sql_reader, to_learn_trades

logger = logging.getLogger("learning.nightly")
LOOKBACK_DAYS = 120


def flags(cfg: Any) -> Dict[str, bool]:
    names = ("ENABLE_LEARNING_SCORECARD", "LEARNING_AUTODEMOTE_SHADOW", "LEARNING_AUTODEMOTE_ENFORCE",
             "LEARNING_EXIT_SHADOW", "ENABLE_LOSS_ATTRIBUTION_REPORT")
    return {n: getattr(cfg, n, False) is True for n in names}


def run(cfg: Any, reader: Any, now: datetime, out_dir: Path, desk_dir: Path, docs_dir: Path,
        attribution_now: bool = False) -> Dict[str, Any]:
    f = flags(cfg)
    result: Dict[str, Any] = {"flags": f, "wrote": []}
    if not any(f.values()):
        result["skipped"] = "all learning flags off"
        return result
    from research.strategy_hunt.playbook_live import family_for
    deny = list(getattr(cfg, "HANDS_OFF_DENYLIST", ()) or ())
    trades = load_db(reader, now - timedelta(days=LOOKBACK_DAYS), now + timedelta(days=1), deny)
    learn = to_learn_trades(trades, family_for)
    date_str = now.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    if f["ENABLE_LEARNING_SCORECARD"]:
        result["wrote"].append(str(write_scorecard(out_dir, rolling_scorecard(learn), date_str)))
    if f["LEARNING_AUTODEMOTE_SHADOW"] or f["LEARNING_AUTODEMOTE_ENFORCE"]:
        dem = demoter.evaluate(learn, now.date())
        p = out_dir / f"demotions-{date_str}.json"
        p.write_text(json.dumps({"date": date_str, "enforce": f["LEARNING_AUTODEMOTE_ENFORCE"],
                                 "demotions": [d.to_dict() for d in dem]}, indent=1))
        result["wrote"].append(str(p)); result["demotions"] = [d.to_dict() for d in dem]
        for d in dem:
            logger.warning("learning_demote %s %s|%s trigger=%s %s", "APPLY" if f["LEARNING_AUTODEMOTE_ENFORCE"] else "WOULD",
                           d.family, d.side, d.trigger, d.detail)
        if f["LEARNING_AUTODEMOTE_ENFORCE"] and dem:
            from core.desk.playbook import load_playbook, playbook_path, save_playbook
            pp = playbook_path(str(desk_dir))
            book = demoter.apply_demotions(load_playbook(pp), dem, now.isoformat(timespec="seconds"))
            save_playbook(book, pp)
            result["wrote"].append(str(pp))
    need_bars = f["LEARNING_EXIT_SHADOW"] or (f["ENABLE_LOSS_ATTRIBUTION_REPORT"] and (attribution_now or now.weekday() == 4))
    if need_bars:
        from research.learning import exit_replay, loss_attribution
        recent = [t for t in trades if t.strategy in DAY_TRADE_STRATEGIES and t.exit_ts >= pd.Timestamp(now - timedelta(days=30))]
        bars = bars_for_db(reader, recent)
        if f["LEARNING_EXIT_SHADOW"]:
            res = exit_replay.replay(recent, bars); summ = exit_replay.summarize(res)
            p = out_dir / f"exit_replay-{date_str}.md"
            p.write_text(exit_replay.render(res, summ, f"Exit-shadow replay {date_str} (30d)"))
            p.with_suffix(".json").write_text(json.dumps({"summary": summ, "trades": [asdict(r) for r in res]}, indent=1))
            result["wrote"].append(str(p))
        if f["ENABLE_LOSS_ATTRIBUTION_REPORT"] and (attribution_now or now.weekday() == 4):
            wk = [t for t in recent if t.exit_ts >= pd.Timestamp(now - timedelta(days=7))]
            res = loss_attribution.attribute(wk, bars)
            docs_dir.mkdir(parents=True, exist_ok=True)
            p = docs_dir / f"loss_attribution_{date_str}.md"
            p.write_text(loss_attribution.render(res, str((now - timedelta(days=7)).date()), date_str))
            result["wrote"].append(str(p))
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Nightly learning loop (flag-gated, report-first).")
    ap.add_argument("--attribution-now", action="store_true")
    a = ap.parse_args(argv)
    from core.config import Config
    cfg = Config()
    if not any(flags(cfg).values()):
        print("learning: all flags off; nothing to do"); return 0
    dsn = os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL")
    if not dsn:
        print("learning: no POSTGRES_DSN; skipping"); return 0
    try:
        res = run(cfg, sql_reader(dsn), datetime.now(), Path("data/learning"),
                  Path(getattr(cfg, "DESK_DATA_DIR", "data/desk")), Path("docs"), a.attribution_now)
    except Exception as exc:
        logger.error("learning nightly failed: %s", exc)
        return 0  # never fail the nightly pipeline
    print(json.dumps({k: v for k, v in res.items() if k != "demotions"}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
