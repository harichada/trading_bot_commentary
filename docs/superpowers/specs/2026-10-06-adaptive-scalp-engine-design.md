# Adaptive Scalp Engine — design for approval (2026-10-06)

Operator directive: scalping must not be set-and-forget. The hold time is not
fixed; the bot re-decides every bar from the tape, and exits early when the
move is about to turn against it. Use the RTX 3090 for the learning.

Status: design only. Nothing built.

## 1. Framing

A scalp here is a long or short held from 1 to ~30 minutes with a target of
0.2–0.5 % and a hard stop of 0.15–0.3 %, on the ~20 most liquid names the bot
already streams. Two learned decisions:

* **Entry trigger** — is now a good moment to be in, in which direction?
* **Stay-or-exit policy** — every minute while in: given the tape right now
  and the position's state, is the expected remaining move still positive
  after costs? If not, exit now; if strongly positive, keep holding even past
  the nominal horizon; the hard stop and the 15:55 flatten are floors, not
  the plan.

The second decision is the one the directive is about, so it is the one
with the most machinery.

## 2. Data

| source | granularity | window | aggressor | use |
|---|---|---|---|---|
| live micro book (`core/micro_book.py`) | 1-min bins, O(1) rings | from 2026-10-06 | quote rule, conflated | live features, forward shadow |
| tick archive → `data/micro_archive/` | 1-min bins | 2024-10-01 → 2025-03-11, 12–17 names | tick rule | training set v1 |
| 1-min bars (`minute_bars`) | 1-min | 2021 → | — | price features, labels when tape absent |

The archive tape is tick-rule and five months of one regime; the live tape is
quote-rule and conflated. The model learns on the archive, is **re-fitted on
the live book's own history** as it accumulates (the forward set), and the
two are never mixed in one fit without the `aggressor_rule` feature present.

## 3. Features (per symbol, per minute; all O(1) from the rings)

Flow: CVD over 1/3/5/15 min normalised by volume; sell share 5/15; large-print
bias 5/15; print rate vs its 20-session hour-of-day median; **live only** EMA
bid/ask imbalance and spread z-score (NaN in the archive; the model gets an
explicit `has_book` flag).
Price: returns 1/3/5/15 min; realised 5-min volatility; distance to VWAP(15)
and to session VWAP; range position in the last 15 min; time of day; regime
label (risk_on/mixed/risk_off).
Position state (exit model only): unrealised R, minutes held, max favourable
and adverse excursion so far, bars since last new high/low.

## 4. Labels (triple barrier)

For every minute `t` and side `s ∈ {long, short}`: target `+τ`, stop `−σ`,
horizon `H` minutes (grid τ ∈ {0.2, 0.3, 0.5}%, σ = 2/3·τ, H ∈ {10, 20, 30}).
`y_entry = +1` if the target is touched before the stop within H, `−1` if the
stop first, `0` if neither. Costs are charged inside the label: entry at the
ask (bid for shorts) plus 0.03 % slippage, exit at the bid (ask).

For the exit policy, the label at every minute **while a hypothetical
position is open** is the realised R from that minute to the eventual exit
under the fixed rule — i.e. "what is left in this trade". That turns the
stay-or-exit decision into a regression on remaining R, which is what the
policy needs.

## 5. Models (trained on the 3090)

1. **Baseline, tabular:** LightGBM (GPU) classifiers for entry (`P(target
   first)`, `P(stop first)`) and a regressor for remaining R. Fast, honest,
   feature importances readable.
2. **Sequence model:** a small temporal conv net (TCN) over the last 60
   one-minute feature rows per symbol, same heads, PyTorch on CUDA. Kept only
   if it beats the GBM out of sample by a margin larger than its variance
   across seeds.
3. **Calibration:** isotonic on out-of-fold predictions, so a `P = 0.6` means
   60 %.

Validation: walk-forward by month (train on months ≤ m, test on m+1), groups
= symbol × session so no session leaks across folds, and every metric is
reported **after costs** against the "always hold to the fixed rule" baseline.

## 6. The policy

```
entry  iff P_target(s) − P_stop(s) ≥ π_entry   and   E[R | enter] ≥ 2 × cost_R
           and the spread z ≤ 2, print rate ≥ its median (the tape is awake),
           and BLOCK_NEW_ENTRIES is off
stay   iff E[remaining R | state] > 0  and  P_stop_next_5min < π_exit
exit   otherwise ("policy_exit"), at market
floors: hard stop σ, 15:55 flatten, max hold 30 min, daily scalp loss cap
size   = scalp sleeve risk × clip(E[R | enter] / R_ref, 0.5, 1.0)
```

π_entry, π_exit are chosen out of fold to maximise net expected R per trade
subject to ≥ 100 trades per month. The policy is re-fitted monthly from the
shadow ledger and promoted through the console ratchet.

## 7. Runtime

* A **scalp loop** at 1-minute cadence (not the 5-minute fit-lane cadence),
  reading `MicroSnapshot`s and the 1-minute bars the micro book already
  keeps; feature build is vectorised numpy, inference on CPU (GBM) in < 5 ms
  per symbol, the TCN on CPU in < 20 ms if adopted; the 3090 is for training.
* Entries go through the normal router (desk limits, playbook gate,
  allocation cap, block switch). Exits add one rule, `policy_exit`, consulted
  every minute for scalp positions only; the existing `PositionState` FSM
  keeps the floors.
* Every minute of every open or hypothetical scalp writes a ledger row
  (features, predictions, decision, realised outcome) so the forward set is
  the same shape as the training set.

## 8. Risk and rollout

* Separate scalp sleeve inside the allocation (proposal: $5,000 of the
  $25,000, max 2 concurrent scalps, daily scalp loss cap 1 % of the sleeve),
  all editable on the Risk Console with the usual ratchet.
* Phase S1 (two evenings): dataset + labels from the archive, GBM baseline,
  walk-forward report, policy thresholds. Deliverable: a report that says
  whether any cell of the τ/H grid beats costs out of sample.
* Phase S2 (two evenings): live feature parity, scalp loop, shadow lane with
  hypothetical entries and policy exits ledgered every minute.
* Phase S3: TCN if it earns it; live via the admission gate after ≥ 4 weeks
  of shadow with positive net R.
* Kill criteria, stated now: S1 shows no τ/H cell with net OOF expected R
  > 0 → stop; S2 shadow net R ≤ 0 after 4 weeks → stop.

## 9. What is honest to expect

Scalping is where the spread eats the signal. The research so far found no
intraday indicator edge; the only thing not yet tested is order flow, and the
archive version of it is the weaker tick-rule proxy. The model's job is to
be right about the next few minutes often enough to pay a 0.05–0.1 % round
trip; the directive's adaptive exit is the part most likely to help, because
it stops paying for trades that have already failed.
