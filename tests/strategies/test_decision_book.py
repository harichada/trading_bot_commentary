from datetime import datetime, timezone, timedelta

from strategies.decision_book import DecisionBook


def test_ring_counts_and_day_rollover():
    b = DecisionBook(ring=10)
    t0 = datetime(2026, 10, 6, 14, 0, tzinfo=timezone.utc)
    for i in range(25):
        b.record("s", f"SYM{i % 3}", "skip" if i % 5 else "signal_buy", "r", close=i, ts=t0 + timedelta(seconds=i))
    assert len(b.recent("s", 100)) == 10 and b.recent("s", 1)[0]["symbol"] == "SYM0"
    c = b.counts("s")
    assert c["today_total"] == 25 and c["by_action"] == {"skip": 20, "signal_buy": 5}
    b.record("s", "X", "skip", "r", ts=t0 + timedelta(days=1))
    assert b.counts("s")["today_total"] == 1          # counters reset with the day; the ring does not
    assert b.recent("s", 100)[0]["symbol"] == "X" and len(b.recent("s", 100)) == 10


def test_base_strategy_records_into_the_book(monkeypatch):
    from types import SimpleNamespace
    from strategies.base import TradingStrategyWithCommentary
    DecisionBook.instance().clear()
    class Toy(TradingStrategyWithCommentary):
        name = "toy_book"
        async def generate_signal_with_commentary(self, md):
            return None
    s = Toy(SimpleNamespace())
    s._log_decision(SimpleNamespace(symbol="MU", close=1.5), "skip", "why", skip_snapshot=True, gate="g")
    rec = DecisionBook.instance().recent("toy_book")
    assert rec and rec[0]["reason"] == "why" and rec[0]["details"] == {"gate": "g"}
