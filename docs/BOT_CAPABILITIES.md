# What This Bot Is and What It Can Do

*Plain-English capabilities overview. Written 2026-06-09. For the
technical architecture, see `COMPOSITE_VIEW_ARCHITECTURE.md`; for
project history, see `../CHANGELOG.md`.*

---

## What it is

A self-directed stock trading system connected to a Charles Schwab
account. It watches the market in real time, decides what to buy or
sell, places the orders itself, manages the exits, and — its signature
feature — **explains every decision it makes in plain language as it
makes it**, streamed live to a web dashboard.

## How it finds trades

**1. It builds its own watchlist.** A screener continuously pulls the
day's most active stocks, top gainers, and top losers from Yahoo, then
runs them through quality filters (liquidity, price sanity, data
quality). The survivors — a couple hundred candidates, ~37 actively
streamed — get watched tick-by-tick through Schwab's real-time data
stream.

**2. Three strategies look at every candidate:**

- **Breakout** — buys when a stock pushes through its 20-day high on
  strong volume
- **Mean reversion** — buys when a stock is oversold (stretched below
  its Bollinger band with low RSI), betting on a snap back
- **News sentiment** — reads headlines from Yahoo, Google News, and
  MarketWatch, scores the sentiment, and trades when strong positive
  news lines up with the price actually moving that way

**3. An ML layer votes on top.** An ensemble of three models
(RandomForest, XGBoost, LightGBM) trained on 2.7 million samples with
200+ technical indicators scores each signal. It keeps learning online
from outcomes as it trades.

## What stops it from doing something dumb

Most of the engineering went into layers of "no":

- **Market context gate** — it reads the overall regime (SPY trend,
  VIX, sector ETF strength, time of day) and blocks longs when the
  market is risk-off. On the 2026-06-09 mid-day crash it sat out
  entirely — correct behavior.
- **Falling-knife filter** — won't buy a dip when the stock is in a
  real downtrend (below its 50-day average with negative momentum).
  The mirror-image **rising-peak filter** exists for shorts.
- **Price-direction confirmation** — good news alone isn't enough; the
  price has to actually be moving up on real volume before a news
  trade fires.
- **Correlation guard** — won't stack positions in the same cluster
  (e.g., it blocked an ARM entry because SMCI was already held in the
  semiconductors cluster).
- **Risk manager** — position sizing, max positions, max risk per
  trade, and a **daily-loss circuit breaker** that halts all trading
  past a loss threshold. The circuit counts *the bot's own* P&L only,
  so external holdings gapping down can't freeze it.
- **Trading-hours and health gates** — won't enter trades if the data
  stream is dead or it's outside regular hours.

## How it manages trades

Every entry goes in with a **bracket**: a stop-loss and a profit
target placed as an OCO pair (one fills, the other cancels). Targets
are chart-based ("where's the next real resistance") rather than a
blind fixed percentage. It also supports trailing stops, scaling out,
TWAP, and DCA-style entries. It cancels the bracket before closing a
position and verifies fills with Schwab before updating its books.

## How it learns

A "brain" stores every trade as a memory — what the setup was, what
happened, what the lesson is (493 memories and counting). It tracks
its own fear/greed levels, recognizes patterns similar to past trades,
and adapts. It learns from the operator's manual closes too, not just
its own exits.

## How it talks to you

The dashboard (port 9000) shows live positions with Schwab's
authoritative P&L, every decision with its reasoning ("skipped NVDA:
no breakout, volume too low"), market context, sentiment, and charts.
Every skip, entry, and exit is written to a structured audit log, so
"why didn't it trade X?" is always answerable.

## How it practices safely

- **Simulation mode** — full pipeline, fake money
- **Shadow modes** — new features run invisibly and log what they
  *would* have done, gathering live evidence before being trusted with
  real money. Currently in shadow: the SHORT branch, the
  side-classifier, and the news-veto tracker.
- **Backtesting engine** — replays strategies over historical data
  with walk-forward analysis before anything ships

## Where it's headed

The Composite View architecture (see
`COMPOSITE_VIEW_ARCHITECTURE.md`) is the next leap: instead of
strategies independently checking conditions, the bot will build one
unified, real-time assessment per symbol — trend, momentum, volatility
quality, sector, news, fundamentals, and liquidity — the full picture
a human would want but can't compute fast enough across 200 stocks at
once.

---

**In one sentence:** a self-explaining trading machine that hunts
setups across the whole market in real time, refuses trades more often
than it takes them, brackets every position it does take, learns from
every outcome, and proves new ideas in shadow mode before risking a
dollar on them.
