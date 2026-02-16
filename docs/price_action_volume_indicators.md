# Price Action, Volume Flow & Chart Pattern Indicators — A Complete Guide

> This document explains all indicator groups added to the trading bot's rules engine.
> Written for someone with zero trading background.

---

## Table of Contents

1. [The Problem We Solved](#the-problem-we-solved)
2. [How the Rules Engine Works (Big Picture)](#how-the-rules-engine-works)
3. [Indicator Group 1: Candlestick Patterns](#1-candlestick-patterns)
4. [Indicator Group 2: Momentum Divergence](#2-momentum-divergence)
5. [Indicator Group 3: Bollinger Band Squeeze](#3-bollinger-band-squeeze)
6. [Indicator Group 4: OBV (On-Balance Volume)](#4-obv-on-balance-volume)
7. [Indicator Group 5: Classic Chart Patterns](#5-classic-chart-patterns)
8. [How They Work Together](#how-they-work-together)
9. [Safety & Graceful Degradation](#safety--graceful-degradation)
10. [Walk-Forward Validation Engine](#walk-forward-validation-engine)
11. [Glossary](#glossary)

---

## The Problem We Solved

The bot had 5 entry setups (Pullback, MACD Cross, Momentum, Breakout, Bounce), but they were all **trend-following** — they only worked when a stock was already moving in a clear direction. When we tested 3 popular stocks (PLTR, NVDA, TSLA) over 300+ price bars, the bot produced **zero trades**. It was too picky and too blind.

Why? Because it couldn't see:

| What was missing | Why it matters |
|---|---|
| Candlestick patterns | Can't spot reversal candles like hammers or engulfing bars |
| Momentum divergence | Can't detect when a trend is running out of steam |
| Volatility compression | Can't predict breakouts from tight consolidation |
| Volume flow | Can't tell if big money is accumulating or distributing |

After adding these 4 groups, the same bot on SPY produced **47 trades** in one month of 5-minute data. It now has more entry paths and stronger confirmation for existing setups.

---

## How the Rules Engine Works

Think of the engine like a scoring system at each price bar (a "bar" is one candle on a chart — could be 1 minute, 5 minutes, 1 day, etc.):

```
For each bar:
  1. Look at dozens of indicators
  2. Check if any "setup" conditions are met
  3. Each setup adds points to a long_score or short_score
  4. If the score reaches the minimum threshold (default: 2), generate a trade signal
```

- **long_score >= 2** → BUY signal (we think the price will go up)
- **short_score >= 2** → SELL signal (we think the price will go down)
- **Both below 2** → HOLD (do nothing)

The minimum threshold of 2 means the engine needs **at least 2 independent reasons** to enter a trade. This prevents impulsive entries based on a single indicator.

---

## 1. Candlestick Patterns

### What are candlesticks?

Every price bar has 4 values: **Open** (where it started), **High** (highest point), **Low** (lowest point), and **Close** (where it ended). When drawn on a chart, these form shapes that look like candles with wicks.

```
    |         ← upper wick (high above the body)
   ███        ← body (difference between open and close)
   ███
    |         ← lower wick (low below the body)
```

- **Green/bullish candle**: Close > Open (price went up)
- **Red/bearish candle**: Close < Open (price went down)
- **Body**: The thick part (distance between open and close)
- **Wicks**: The thin lines above and below (the extremes the price touched)

### The 18 patterns we detect

> The original 4 (Engulfing, Hammer, Shooting Star) plus 14 new patterns added for complete chart sheet coverage.

#### Bullish Engulfing

```
  Bar 1 (red)    Bar 2 (green)
     ███            |
     ███           ███
      |            ███
                   ███
                    |
```

A small red candle followed by a **bigger** green candle that completely covers ("engulfs") the previous candle's body. This means buyers overwhelmed sellers. It often marks the bottom of a dip.

**Detection rule**: Previous bar was bearish. Current bar is bullish. Current body's high > previous body's high AND current body's low < previous body's low.

#### Bearish Engulfing

The mirror image — a small green candle engulfed by a larger red candle. Sellers overwhelmed buyers. Often marks a top.

#### Hammer

```
    |     ← tiny or no upper wick
   ███    ← small body at the TOP of the bar
    |
    |     ← very long lower wick (2x+ the body)
    |
```

Price dropped hard during the bar but buyers pushed it back up to close near the top. The long lower wick shows **rejection of lower prices**. This is a bullish reversal signal, especially at support levels.

**Detection rule**: Body is less than 40% of the total bar range. Lower wick is at least 2x the body. Upper wick is less than 30% of the range.

#### Shooting Star

```
    |
    |     ← very long upper wick (2x+ the body)
    |
   ███    ← small body at the BOTTOM of the bar
    |     ← tiny or no lower wick
```

The opposite of a hammer. Price surged up but sellers pushed it back down. The long upper wick shows **rejection of higher prices**. This is a bearish reversal signal, especially at resistance levels.

#### Dragonfly Doji (bullish)

```
   ███    ← body at the very top (open ≈ close ≈ high)
    |
    |     ← very long lower wick (60%+ of range)
    |
```

A doji where the entire range was below the open/close. Like a hammer but with virtually no body — extreme indecision that resolved bullish.

#### Gravestone Doji (bearish)

```
    |
    |     ← very long upper wick (60%+ of range)
    |
   ███    ← body at the very bottom (open ≈ close ≈ low)
```

Mirror of dragonfly. Buyers pushed up hard but got completely rejected. Strong bearish signal at resistance.

#### Morning Star (bullish, 3-bar)

```
  Bar 1 (red)   Bar 2 (small)   Bar 3 (green)
     |              ███              |
    ███              |             ███
    ███                            ███
    ███                            ███
     |                              |
```

A large bearish candle, followed by a small indecision candle, followed by a large bullish candle closing above bar 1's midpoint. Classic bottom reversal.

#### Evening Star (bearish, 3-bar)

Mirror of morning star. Large bullish → small → large bearish closing below bar 1's midpoint. Classic top reversal.

#### Three White Soldiers (bullish, 3-bar)

Three consecutive bullish candles, each closing higher, each opening within the previous candle's body. Shows sustained, aggressive buying. Especially powerful after a downtrend.

#### Three Black Crows (bearish, 3-bar)

Three consecutive bearish candles, each closing lower. Mirror of Three White Soldiers. Shows relentless selling pressure.

#### Piercing Line (bullish, 2-bar)

Bearish candle, then a bullish candle that **opens below the previous low** (gap down) but **closes above the previous midpoint**. Buyers aggressively reversed the gap — bullish.

#### Dark Cloud Cover (bearish, 2-bar)

Bullish candle, then a bearish candle that **opens above the previous high** (gap up) but **closes below the previous midpoint**. Rare on intraday charts (requires gaps).

#### Bullish Harami (2-bar)

A large bearish candle followed by a small bullish candle whose body fits entirely inside the previous body. "Harami" means "pregnant" in Japanese. The small candle is contained within the large one, showing selling pressure exhausted.

#### Bearish Harami (2-bar)

Mirror: large bullish candle → small bearish candle contained within.

#### Bullish Marubozu (single bar)

A full-body bullish candle with virtually no wicks (<5% of range on each side). Body is 90%+ of the bar's range. This represents maximum buyer conviction — no hesitation.

#### Bearish Marubozu (single bar)

Full-body bearish candle, no wicks. Maximum seller conviction.

#### Hanging Man (bearish)

Identical shape to a hammer (small body, long lower wick), but appears after an uptrend. Context changes the meaning: at a bottom it means rejection of lows (bullish), but at a top it means support is weakening (bearish).

#### Spinning Top (indecision)

Small body with long wicks on both sides. Neither buyers nor sellers won. Signals potential trend change when seen after a strong move.

### How they're used in the engine

Candlestick patterns don't generate trades by themselves. They act as **confirmation** — they boost the score of existing setups:

- **Reversal candles** (engulfing, hammer, doji, morning/evening star, piercing, harami) boost BOUNCE and DIVERGENCE setups by +1
- **Conviction candles** (marubozu, three soldiers/crows) add +1 when aligned with trend and OBV direction
- **Hanging man** acts as a bearish reversal confirmation
- **Spinning top** provides context (indecision) but doesn't directly boost scores

---

## 2. Momentum Divergence

### The concept

Divergence is when **price says one thing but momentum says another**. It's the #1 reversal signal used by professional traders.

Imagine a runner getting slower with each lap but still ahead in the race. The lap times (momentum) are getting worse even though the position (price) looks fine. Eventually, the runner will fall behind. That's divergence.

### Bullish Divergence

```
Price:    Lower Low  ← "things are getting worse"
RSI:      Higher Low ← "but momentum is actually improving"
                     → likely reversal upward
```

The price makes a new low, but RSI (a momentum oscillator, 0-100) makes a *higher* low. This means selling pressure is weakening even though price hasn't turned yet. Buyers are quietly stepping in.

### Bearish Divergence

```
Price:    Higher High ← "things look great"
RSI:      Lower High  ← "but momentum is fading"
                      → likely reversal downward
```

Price makes a new high, but RSI makes a *lower* high. The rally is losing steam even though it looks strong on the surface.

### How we detect it

1. **Find swing points** — A swing low is a bar where the low is the lowest in a 5-bar window (the bar itself + 2 bars on each side). A swing high is the opposite. These are the natural peaks and valleys of price movement.

2. **Compare consecutive swings** — For bullish divergence, we compare the two most recent swing lows. If price's swing low is lower but RSI's value at that swing is higher, that's divergence.

3. **Freshness filter** — The divergence signal only fires if the most recent swing happened within 8 bars of the current bar. Stale divergences (from 50 bars ago) are meaningless.

4. **Noise buffer** — RSI must differ by at least 2 points (or MACD histogram by 0.001) to count. Tiny differences are noise, not real divergence.

### How it's used in the engine

Divergence creates its own entry setup:

```
DIVERGENCE long:
  - Bullish divergence detected + bullish candle → score = 1
  - If at support level → score = 2 (might be enough to trigger!)
  - If also a reversal candle (engulfing/hammer) → score + 1
  - If OBV confirms accumulation → score + 1
```

This is powerful because divergence catches **reversals** — something the old trend-following setups completely missed.

---

## 3. Bollinger Band Squeeze

### What are Bollinger Bands?

Bollinger Bands are an envelope around price based on volatility:

```
━━━━━━━━━ Upper Band (moving average + 2 standard deviations)
                          ← wide = volatile
   ~~~~~ Price ~~~~       ← price bounces between the bands
                          ← narrow = compressed
━━━━━━━━━ Lower Band (moving average - 2 standard deviations)
```

When volatility is **high**, the bands are wide apart. When volatility is **low**, they squeeze together. The key insight: **low volatility leads to high volatility**. A squeeze is like a coiled spring — it predicts a big move is coming, though not which direction.

### How we detect it

1. Measure the **bandwidth**: `(upper band - lower band) / midpoint`
2. Calculate the **percentile rank** of the current bandwidth over the last 120 bars
3. If the bandwidth is in the **bottom 25%** (tighter than 75% of recent bars), it's a squeeze

### The "squeeze fire" event

The squeeze itself just means "something's coming." The **important moment** is when the squeeze **releases** — when the bands start expanding after being compressed. We detect this as:

```
bb_squeeze_fire = yesterday was in squeeze AND today is NOT in squeeze
```

This is the moment the spring uncoils.

### How it's used in the engine

Squeeze fire enhances two types of entries:

1. **Existing BREAKOUT setups** — If price breaks above resistance AND it's a squeeze release, the breakout score gets boosted to 3. A breakout from a squeeze is much more significant than a regular breakout.

2. **New SQUEEZE BREAKOUT setup** — Squeeze fire + bullish candle + price above EMA9:
   - Base score = 1
   - Volume surge → score = 2
   - ADX rising → score + 1

---

## 4. OBV (On-Balance Volume)

### The concept

Price tells you **what** happened. Volume tells you **how convincingly** it happened. OBV (On-Balance Volume) tracks the cumulative flow of volume to reveal whether big players are accumulating (buying) or distributing (selling).

Think of it like vote counting. If price goes up on a bar, ALL of that bar's volume is counted as "buy votes." If price goes down, all volume counts as "sell votes." The running total reveals the crowd's conviction.

### How we compute it

```
For each bar:
  If price went UP   → add volume to running OBV total
  If price went DOWN  → subtract volume from running OBV total
  If price unchanged  → add nothing

Then:
  1. Smooth OBV with a 10-bar exponential moving average (removes noise)
  2. Take the slope (rate of change) of the smoothed OBV
  3. Normalize by dividing by the 20-bar average volume (so the number is comparable across stocks)
```

The result is a number that's:
- **Positive** → accumulation (smart money buying) — "the crowd is bullish"
- **Negative** → distribution (smart money selling) — "the crowd is bearish"
- **Near zero** → no clear conviction

We use thresholds of **+0.1** (bullish) and **-0.1** (bearish) to filter out noise.

### How it's used in the engine

OBV acts as a **confirmation filter** on ALL setups, not just its own:

```
If we already have a long signal (long_score > 0):
  - OBV bullish (+0.1) → long_score + 1    (volume confirms the buy)
  - OBV bearish (-0.1) → long_score - 1    (volume disagrees — weaken the signal)

If we already have a short signal (short_score > 0):
  - OBV bearish (-0.1) → short_score + 1   (volume confirms the sell)
  - OBV bullish (+0.1) → short_score - 1   (volume disagrees — weaken the signal)
```

Exception: If price just broke through a support/resistance level (a breakout), we don't penalize even if OBV disagrees. Breakouts often precede the volume confirmation by a bar or two.

This means OBV can:
- **Upgrade** a marginal setup (score 1 → 2) into a trade
- **Downgrade** a weak setup (score 1 → 0) to prevent a bad trade

---

## 5. Classic Chart Patterns

### What are chart patterns?

The indicators above look at 1-2 bars at a time. **Chart patterns** are bigger — they form over **15 to 120 bars** and involve the *shape* of the price chart itself. These are the patterns that traders draw trendlines for, and they've been used since the early 1900s.

The core idea: **price geometry repeats**. When a stock traces out a specific shape, it often resolves in a predictable direction. Not always, but often enough that traders watch for them.

### How we detect them

All chart patterns are built from **swing points** — the natural peaks and valleys of price movement:

```
Price
  │    *           *       ← swing highs (peaks)
  │   / \   *     / \
  │  /   \ / \   /   \
  │ /     *   \ /     \   ← swing lows (valleys)
  │/           *
  └─────────────────────── Time
```

The bot finds these swing points using a **5-bar window**: if a bar's high is the highest of the 5 bars around it (2 before, itself, 2 after), it's a swing high. Same logic in reverse for swing lows. The bot keeps track of the most recent 10 swing highs and 10 swing lows.

Then, at each bar, it checks whether the recent swing points match the geometry of any known pattern.

### Tolerance: what "same level" means

Two swing points are considered "at the same level" if their prices are within **0.5x ATR** of each other. ATR (Average True Range) measures how much a stock typically moves per bar. This means:

- For a stock that moves $2/bar (ATR=2), swings within $1 of each other are "same level"
- For a stock that moves $0.10/bar (ATR=0.10), swings within $0.05 are "same level"

This automatically adapts to different price ranges and volatility levels.

---

### Double Top (bearish reversal)

```
  Price
    │    ┌──*──┐    ┌──*──┐       ← two peaks at ~same level
    │   /       \  /       \
    │  /         \/         \     ← trough between them (neckline)
    │ /           *          \
    │/                        \   ← price breaks below neckline = CONFIRMED
    └──────────────────────────
```

**What it means**: The stock tried to go higher twice but failed at the same level both times. Buyers couldn't push past that ceiling. When price drops below the trough between the two peaks (the "neckline"), sellers take control.

**Real-world analogy**: Imagine trying to jump over a wall twice. You reach the same height both times but can't get over. After the second failed attempt, you give up and walk away.

**Detection rules**:
- 2 swing highs at approximately the same price (within 0.5 ATR)
- At least 1 swing low (trough) between them
- 15-80 bars apart
- **Confirmation**: Current price drops below the trough = pattern is active

### Double Bottom (bullish reversal)

```
  Price
    │\                        /
    │ \           *          /
    │  \         /\         /     ← peak between them (neckline)
    │   \       /  \       /
    │    └──*──┘    └──*──┘       ← two valleys at ~same level
    └──────────────────────────
```

The mirror of Double Top. The stock tried to go lower twice but found buyers at the same level both times. When price breaks above the peak between the two valleys, buyers are in charge.

**Real-world analogy**: A ball bouncing off a floor twice at the same spot. The floor is solid — the ball bounces higher.

### Head & Shoulders (bearish reversal)

```
  Price
    │              *              ← HEAD (highest peak)
    │             / \
    │     *      /   \      *     ← LEFT SHOULDER    RIGHT SHOULDER
    │    / \    /     \    / \       (both at ~same level)
    │   /   \  /       \  /   \
    │──/─────*──────────*──────\── ← NECKLINE (connects the two troughs)
    │ /                         \  ← price breaks below neckline = CONFIRMED
    └──────────────────────────────
```

**What it means**: This is the most famous reversal pattern. The stock makes three peaks — the middle one (the "head") is the highest, and the two outer peaks (the "shoulders") are at roughly the same, lower level. The line connecting the two valleys between the peaks is the "neckline."

When price drops below the neckline after the right shoulder, it signals the uptrend is over. The logic: the stock tried to make a higher high (the head succeeded), but then couldn't even reach the head's level again (right shoulder). Momentum has clearly shifted.

**Detection rules**:
- 3 swing highs: left shoulder, head (highest), right shoulder
- Head must be above both shoulders by at least half the tolerance
- Shoulders at approximately the same level
- At least 1 trough between each pair of peaks
- 25-100 bars total span
- **Confirmation**: Price below the neckline (higher of the two troughs)

### Inverse Head & Shoulders (bullish reversal)

The upside-down version — three valleys with the middle one being the deepest. When price breaks above the neckline (connecting the two peaks between the valleys), it signals a bullish reversal. Same logic in reverse: the stock failed to make a lower low on the right shoulder, meaning sellers are losing power.

### Ascending Triangle (bullish breakout)

```
  Price
    │─────────*─────────*─────── ← flat resistance (ceiling)
    │        / \       / \
    │       /   \     /   \
    │      /     \   /     \
    │     /       \ /       \
    │    *         *         *   ← rising support (higher lows)
    └──────────────────────────
```

**What it means**: Price keeps hitting the same ceiling (flat resistance) but the floor keeps rising (higher lows). Buyers are getting more aggressive — they're willing to buy at higher and higher prices. This compresses price into a tighter and tighter range until the buyers finally push through the ceiling.

**Real-world analogy**: A crowd pushing against a locked door. Each push is stronger than the last. Eventually, the door breaks open.

**Detection rules**:
- 2-3 swing highs at approximately the same level (flat top)
- 2+ swing lows that are progressively higher
- 20-80 bars span

### Descending Triangle (bearish breakdown)

The mirror: flat support (floor) with lower swing highs (ceiling descending). Sellers are getting more aggressive, pressing price down until the floor breaks.

### Bull Flag (bullish continuation)

```
  Price
    │         │   ╲                ← FLAG (tight, gentle downward channel)
    │         │    ╲╱╲
    │         │       ╲╱╲
    │        /│           ╲       ← flag stays above pole midpoint
    │       / │
    │      /  │  ← FLAGPOLE (strong upward move, 3+ ATR)
    │     /   │
    │    /    │
    │   /     │
    └──/──────────────────────
```

**What it means**: After a strong upward move (the "flagpole"), price consolidates in a tight, slightly downward-sloping channel (the "flag"). This is a pause, not a reversal — the strong move attracted sellers, but they can't push price down much. When the flag resolves, price typically continues in the original direction.

**Real-world analogy**: A sprinter pausing to catch their breath before continuing the race. The pause doesn't mean they've stopped — they're gathering energy for the next burst.

**Detection rules**:
- Flagpole: a rise of at least 3x ATR in the 40-60 bars before the flag
- Flag: last 10-20 bars have a tight range (< 2.5 ATR) and retrace < 50% of the pole
- Flag slopes flat or slightly downward

### Cup & Handle (bullish continuation)

```
  Price
    │ *                     * ╲   ← LEFT LIP   RIGHT LIP   HANDLE
    │  \                   /   ╲╱╲
    │   \                 /        ← handle is a small pullback
    │    \               /
    │     \             /
    │      \           /          ← CUP (U-shaped bottom)
    │       \         /
    │        \       /
    │         ╲─────╱             ← deepest point near the middle
    └──────────────────────────
```

**What it means**: Price forms a rounded bottom (the "cup") that looks like the letter U, then forms a small pullback near the right lip (the "handle"). The rounded bottom shows a gradual shift from selling to buying. The handle is a final shakeout of weak hands before the breakout.

This pattern was popularized by William O'Neil in his book "How to Make Money in Stocks" and is one of the most reliable continuation patterns.

**Detection rules**:
- 5+ swing lows spanning 30-120 bars
- The lowest swing point is in the middle third (forming the U-shape)
- Left and right lips at approximately the same level (within 1.5x tolerance)
- Cup depth is at least 1 ATR (not too shallow)
- Handle: a small recent pullback that's less than 50% of the cup depth

### Falling Wedge (bullish reversal)

```
  Price
    │ \
    │  \ *                        ← swing highs declining
    │   \  \
    │    \   * \
    │     \    \  \
    │      \    *   \             ← swing lows declining FASTER
    │       \      * \               (lines converging downward)
    │        \        *
    │         ── ── ── ──→        ← breakout upward expected
    └──────────────────────────
```

**What it means**: Both the highs and lows are falling, but the lows are falling faster. This means the range is narrowing — price is being squeezed into a tighter and tighter downward channel. Despite the downward direction, this is actually bullish because the selling pressure is weakening (the range is compressing). When the wedge breaks, it typically breaks upward.

**Real-world analogy**: A river narrowing as it approaches rapids. The energy is building up even though the channel is getting smaller.

**Detection rules**:
- 3+ swing highs and 3+ swing lows
- Both series are declining
- Swing lows decline faster than swing highs (slopes converge)
- Current range is less than 80% of the initial range (proving convergence)
- 20-80 bars span

### Rising Wedge (bearish reversal)

The mirror: both highs and lows are rising, but highs rise slower. The range narrows upward. Despite the upward direction, this is bearish — buying pressure is weakening. Breaks downward.

---

### How chart patterns are scored

Chart patterns never trigger trades by themselves (score = 1). They need **confirmation**:

```
CHART PATTERN long (any bullish pattern + bullish candle):
  Base score = 1
  If volume surge → score = 2   (enough to trigger!)
  If in uptrend → score + 1
  If reversal candle (engulfing/hammer) → score + 1
```

This means a Double Bottom alone won't make the bot buy. But a Double Bottom + volume surge = BUY. Or a Double Bottom + bullish trend + engulfing candle = strong BUY.

### Extended chart patterns (14 new)

In addition to the 10 original patterns above, we now detect 14 more:

#### Triple Top / Triple Bottom

Like Double Top/Bottom but with **3 swing points** at the same level instead of 2. Even stronger reversal signal because the level was tested 3 times.

#### Inverse Cup & Handle (bearish)

Mirror of Cup & Handle — an inverted U-shape (dome) in swing highs with a small handle rally. Bearish continuation pattern.

#### Bear Flag (bearish continuation)

Mirror of Bull Flag — strong downward pole (3+ ATR drop) followed by tight, slightly-upward consolidation.

#### Bullish / Bearish Pennant

Like a flag but the consolidation forms a **small symmetrical triangle** (converging highs and lows) rather than a rectangle. Requires a strong pole (3+ ATR) and the pennant range must shrink to < 60% of its initial width.

#### Rounding Bottom / Rounding Top

Gradual U-shaped (or inverted U-shaped) curve over 30-120 bars. Unlike Cup & Handle, no handle or lip-matching is required. Detected by checking swing lows/highs form a proper left-decline, right-recovery sequence with minimum 1.5 ATR depth.

#### Rectangle Bullish / Bearish

A horizontal channel — both swing highs (resistance) and swing lows (support) are at flat levels. When price breaks above resistance = bullish. When price breaks below support = bearish.

#### Broadening Bottom / Top (Megaphone)

The **opposite** of a triangle — range is **expanding** (higher highs AND lower lows). When price breaks above the expanding top = bullish reversal. Below the expanding bottom = bearish.

#### Symmetrical Triangle Bullish / Bearish

Both swing highs declining AND swing lows rising (converging from both directions). Unlike ascending triangle (flat top) or descending (flat bottom), both sides move. Breakout direction determines the signal.

### Pattern detection frequency

These patterns are **rare by design**. On SPY 5-minute data (1,500 bars = about 1 month):

| Pattern | Typical detections | Category |
|---|---|---|
| Double Top / Double Bottom | 0-5 | Reversal |
| Head & Shoulders / Inv H&S | 0-3 | Reversal |
| Triple Top / Triple Bottom | 0-10 | Reversal |
| Ascending / Descending Triangle | 0-10 | Breakout |
| Symmetrical Triangle | 0-3 | Breakout |
| Bull Flag / Bear Flag | 5-20 | Continuation |
| Bull / Bear Pennant | 30-60 | Continuation |
| Cup & Handle / Inv Cup & Handle | 5-20 | Continuation |
| Rounding Bottom / Top | 50-70 | Reversal |
| Rectangle Bull / Bear | 2-8 | Breakout |
| Broadening Bottom / Top | 1-5 | Reversal |
| Falling / Rising Wedge | 5-50 | Reversal |

Rarer patterns carry more weight precisely because they're harder to form. Patterns with higher frequencies (Rounding, Pennant) provide context but still need confluence triggers to generate trades.

---

## How They Work Together

Here's a realistic example of how these indicators combine:

### Scenario: Catching a bottom reversal

```
Bar 247:
  - Price makes a new low (lower than the previous swing low)
  - RSI at this swing low is 34 — higher than the previous swing's RSI of 28
  → Bullish divergence detected!

  - The candle is a hammer (long lower wick, body at top)
  → Reversal candle detected!

  - Price is near the nearest support level (within 0.3 ATR)
  → At support!

  - OBV slope is +0.25 (accumulation happening despite the low price)
  → Volume confirms buyers stepping in!

Scoring:
  DIVERGENCE long: div_bull + candle_bull → 1
  At support → 2
  Reversal candle (hammer) → +1 = 3
  OBV bullish → +1 = 4

  long_score = 4, min_triggers = 2 → BUY SIGNAL
```

### Scenario: Catching a squeeze breakout

```
Bar 312:
  - BB bands have been tight for 8 bars (squeeze)
  - Today the bands expand (squeeze fire!)
  - Candle is bullish, price is above EMA9
  - Volume is 1.8x average (surge!)
  - ADX is rising (trend strengthening)

Scoring:
  SQUEEZE BREAKOUT: squeeze_fire + bull candle + above EMA9 → 1
  Volume surge → 2
  ADX rising → +1 = 3
  OBV bullish → +1 = 4

  long_score = 4, min_triggers = 2 → BUY SIGNAL
```

### The complete list of setups (old + new)

| # | Setup | Type | New? |
|---|-------|------|------|
| 1 | Pullback to EMA21 in trend | Trend-following | No |
| 2 | MACD cross in trend | Trend-following | No |
| 3 | Momentum acceleration | Trend-following | No |
| 4 | Breakout above R / below S | Breakout | Enhanced with squeeze |
| 5 | Bounce at S/R | Mean-reversion | Enhanced with candles + RSI relaxed |
| 6 | Divergence long/short | Reversal | **New** |
| 7 | Squeeze breakout long/short | Volatility | **New** |
| 8 | OBV confirmation | Filter | **New** (modifies all scores) |
| 9 | Chart pattern long/short | Multi-bar structural | **New** (10 patterns) |

---

## Safety & Graceful Degradation

### What happens if an indicator fails?

Every indicator computation is wrapped in a `_safe()` function. If anything goes wrong (bad data, missing columns, math errors), the indicator silently returns zeros:

```
_safe('candlestick_patterns', _candlestick_patterns)
     ↓
If _candlestick_patterns() throws ANY error:
  → Log a warning
  → The columns _engulf_bull, _engulf_bear, etc. stay missing
  → _col('_engulf_bull', 0.0) returns an array of zeros
  → All checks like "engulf_bull > 0.5" return False
  → The engine behaves exactly as it did before these indicators existed
```

This means:
- **No crashes** — a broken indicator can't take down the bot
- **No phantom signals** — failed indicators default to "not detected" (0.0)
- **No data dependency** — volume-based indicators (OBV) return 0.0 for assets without volume data (like some forex feeds)

### No look-ahead bias

Every indicator only uses **past and current** data, never future data:

| Indicator | What data it uses | Why it's safe |
|---|---|---|
| Engulfing | Current bar + previous bar | `np.roll(x, 1)` shifts data backward |
| Hammer / Shooting Star | Current bar only | Single-bar pattern |
| Divergence swings | 5-bar window centered 2 bars back | Center is `i-2`, all bars are in the past |
| Divergence freshness | 8-bar lookback from current | Counts backward only |
| BB Squeeze | Rolling 120-bar percentile | `rolling()` is backward-looking by definition |
| OBV | Cumulative sum + EMA of past volume | Both are backward-looking |
| Double Top/Bottom | Past swing points + current price vs neckline | Swings confirmed 2 bars late; neckline check is current price |
| Head & Shoulders | Past 3 swing highs/lows + neckline break | Same 2-bar confirmation delay |
| Triangles | Past 2-3 swing points, monotonicity check | Only past swing geometry |
| Bull Flag | Past 60 bars for pole, past 20 for flag | All lookback windows |
| Cup & Handle | Past 5+ swing lows + current price | Only completed swings + current close |
| Wedges | Past 3+ swing highs/lows, slope comparison | Slope computed from past points only |

This is critical for backtesting. If indicators "peeked" at future data, backtest results would be unrealistically good and wouldn't translate to live trading.

---

## Walk-Forward Validation Engine

### What Problem Does This Solve?

You've built a trading strategy with 25+ patterns and indicators. You run the optimizer and it says: "Return: +15%, Win Rate: 65%!" Sounds amazing, right?

**Not necessarily.** The optimizer found parameters that work perfectly on *past* data — but that doesn't mean they'll work on *future* data. This is called **overfitting**: the strategy memorized the noise in historical data instead of learning real patterns.

Think of it like a student who memorizes all the answers to last year's exam. They score 100% on the practice test, but when the real exam has different questions, they fail.

### How Walk-Forward Validation Works

Instead of one big backtest, walk-forward breaks the data into rolling windows:

```
Data: |===================================================================|

Fold 1: |---TRAIN (400 bars)---|---TEST (150 bars)---|
Fold 2:      |---TRAIN (400 bars)---|---TEST (150 bars)---|
Fold 3:           |---TRAIN (400 bars)---|---TEST (150 bars)---|
...and so on, sliding forward by 100 bars each time
```

For each fold:
1. **TRAIN**: The optimizer finds the best parameters on this window (in-sample)
2. **TEST**: Those parameters are tested on the *next* window that the optimizer never saw (out-of-sample)
3. **Compare**: If in-sample return is +10% but OOS is -5%, we know the optimizer overfit

The key insight: **only the TEST results matter**. If a strategy can't make money on data it wasn't optimized for, it won't make money in live trading.

### Pattern Attribution — Which Patterns Actually Help?

Every trade in the OOS test window is tagged with which patterns/indicators were active when the trade was entered. For example:

```
Trade #7: BUY at $595.30
  Active patterns: ENGULF_BULL, OBV_BULL, BULL_TREND, RISING_WEDGE
  Exit: $601.15 (target hit)
  PnL: +$292.50
```

After all folds, we build a **Pattern Scorecard**:

| Pattern | Trades | Win Rate | Avg Win | Avg Loss | Expectancy |
|---------|--------|----------|---------|----------|------------|
| ENGULF_BULL | 18 | 55.6% | $245 | -$180 | +$56.40 |
| DIV_BULL | 12 | 41.7% | $310 | -$195 | +$47.90 |
| BULL_FLAG | 5 | 60.0% | $280 | -$150 | +$108.00 |

**Expectancy** = the average dollar amount you win or lose per trade involving that pattern. Positive = the pattern helps. Negative = the pattern hurts.

Patterns with fewer than 10 trades are marked "not significant" — too small a sample to draw conclusions.

### What the Report Tells You

The walk-forward report includes:

1. **Fold-by-fold results**: Each fold's in-sample vs. out-of-sample return, plus the overfit ratio
2. **Aggregate OOS metrics**: Total return, win rate, profit factor, max drawdown, Sharpe ratio — all from out-of-sample data only
3. **Pattern scorecard**: Which patterns contribute to winning vs. losing trades
4. **Recommendations**: Automated analysis including:
   - **Overfit warnings** if OOS/IS ratio < 0.3 (strategy loses its edge out-of-sample)
   - **Top patterns** with positive expectancy (keep these)
   - **Underperforming patterns** with negative expectancy (consider disabling)
   - **Insufficient data** warnings for patterns with too few trades
   - **Overall viability** assessment based on profit factor and win rate

### The Overfit Ratio

```
Overfit Ratio = OOS Return / IS Return
```

| Ratio | Meaning |
|-------|---------|
| > 0.5 | **Robust** — strategy keeps most of its edge |
| 0.3 - 0.5 | **Acceptable** — some degradation but still viable |
| 0 - 0.3 | **Likely overfit** — most of the edge disappears |
| < 0 | **Severely overfit** — strategy does the *opposite* of what was expected |

### How to Use It

**Via the API endpoint** (`POST /api/walkforward`):

```json
{
  "symbol": "SPY",
  "interval": "5m",
  "adaptive_risk": true,
  "risk_per_trade": 1.0,
  "max_stop_pct": 4.0,
  "train_bars": 400,
  "test_bars": 150,
  "step_bars": 100
}
```

**Window sizes are auto-calculated** based on interval if not specified:

| Interval | Train | Test | Step |
|----------|-------|------|------|
| 1m | 1500 | 500 | 250 |
| 5m | 900 | 300 | 150 |
| 15m | 600 | 200 | 100 |
| 1h | 300 | 100 | 50 |
| 1d | 150 | 50 | 25 |

**Tip**: Use longer date ranges for walk-forward (at least 1 month for 5m, 6 months for 1d). More folds = more reliable statistics.

---

## Glossary

| Term | Meaning |
|---|---|
| **ATR** | Average True Range — measures how much a stock typically moves per bar. Used for stop-loss and target distances. |
| **ADX** | Average Directional Index — measures trend strength (0-100). Above 25 = trending, below 20 = choppy. |
| **Bollinger Bands** | Upper and lower bands around a moving average, based on standard deviation. |
| **EMA** | Exponential Moving Average — a smoothed average that weights recent prices more heavily. EMA9 is faster (reacts quicker) than EMA50. |
| **Long** | A bet that the price will go UP. Buy first, sell later at a higher price. |
| **MACD** | Moving Average Convergence Divergence — measures the relationship between two EMAs. The histogram shows acceleration/deceleration. |
| **OBV** | On-Balance Volume — cumulative volume indicator showing accumulation or distribution. |
| **RSI** | Relative Strength Index — momentum oscillator (0-100). Below 30 = oversold, above 70 = overbought. |
| **Short** | A bet that the price will go DOWN. Sell borrowed shares first, buy back later at a lower price. |
| **Stochastic (K/D)** | Another momentum oscillator (0-100). K crossing above D = bullish, below = bearish. |
| **Support** | A price level where buyers tend to step in (floor). |
| **Resistance** | A price level where sellers tend to step in (ceiling). |
| **Swing High/Low** | A local peak or valley in price — the highest high or lowest low in a small window. |
| **Volume Surge** | Volume more than 1.5x the 20-bar average — indicates strong conviction. |
| **VWAP** | Volume-Weighted Average Price — the average price weighted by volume, used as a fair-value benchmark. |
| **Neckline** | In Double Top/Bottom and H&S patterns, the line connecting the troughs (or peaks) between the main swing points. A break of the neckline confirms the pattern. |
| **Flagpole** | The strong, fast price move that precedes a flag pattern. Must be at least 3x ATR to qualify. |
| **Convergence** | When two trendlines (drawn through swing highs and swing lows) move closer together over time. Seen in wedges and triangles. |
| **Continuation pattern** | A chart pattern that predicts price will keep moving in its prior direction after a pause (e.g., Bull Flag, Cup & Handle). |
| **Reversal pattern** | A chart pattern that predicts price will change direction (e.g., Double Top, Head & Shoulders, Falling Wedge). |
| **Walk-Forward** | A validation method that tests a strategy on data it was never optimized for, using rolling train/test windows. |
| **In-Sample (IS)** | The training data window where the optimizer searches for the best parameters. |
| **Out-of-Sample (OOS)** | The test data window that the optimizer never sees — used to measure real performance. |
| **Overfit Ratio** | OOS return / IS return. Values > 0.5 are robust; values < 0 mean the strategy does worse than random OOS. |
| **Expectancy** | Average profit/loss per trade. Positive expectancy = the strategy makes money over many trades. |
| **Profit Factor** | Gross profits / gross losses. Above 1.5 = good. Below 1.0 = losing money. |
| **Sharpe Ratio** | Risk-adjusted return — how much return per unit of volatility. Above 1.0 is considered good. |
| **Pattern Attribution** | Tagging each trade with which indicators/patterns were active at entry, so we can measure each pattern's contribution. |
