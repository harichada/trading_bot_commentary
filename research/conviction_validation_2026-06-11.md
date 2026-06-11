# Conviction → outcome validation (2026-06-11)

Join: 89 completed bot trades (Apr 20 – Jun 8, bot_trades) matched to
their nearest meta_shadow decision (bot_decisions, ±3 min window on
symbol + entry time).

| meta_proba bucket | n  | win rate | PF   | avg ret |
|---|---|---|---|---|
| < 0.60            | 58 | 46.6%    | 0.44 | −0.21%  |
| 0.60 – 0.70       | 22 | 54.5%    | 3.01 | +0.65%  |
| ≥ 0.70            | 9  | 66.7%    | 5.02 | +1.47%  |

Monotonic gradient: the meta-model separates the bot's earners from
its bleed. Caveats: top bucket n=9; approximate time-join; thresholds
are the model's own logged cut-points, not optimized post-hoc.

Consequences:
1. Conviction sizer (v-conviction-sizer-shadow-2026-06-11) ships in
   shadow with meta_proba as the dominant (0.5-weight) input.
2. The <0.60 bucket carries most trades and ALL the losses — after
   the live shadow confirms, the meta-model should graduate from
   shadow to a low-conviction VETO (rollout flavor B in the original
   meta-labeling plan), not just a size reducer.
