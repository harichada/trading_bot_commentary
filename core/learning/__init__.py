"""v-learning-loop-2026-10-22: modular, flag-gated learning loop.

Pure modules only (no Config, no I/O except the explicit writers):
  stage_a     shared Stage A promotion floors (n AND sessions both required)
  scorecard   rolling per-(family, side, regime) scorecard
  demoter     auto-demote triggers; can only ever *suspend*, never promote
  risk_caps   per-trade 1R clamp, realized-loss halt, stop/R entry validation
"""
