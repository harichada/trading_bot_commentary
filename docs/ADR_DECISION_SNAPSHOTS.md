# ADR: Decision Snapshots for ML Training Pipeline

**Date:** 2026-09-09  
**Status:** Implemented  
**Tag:** v-feature-snapshot-2026-09-09

## Context

The trading bot currently makes decisions using a combination of:
1. Strategy-level gates (RSI, volume, support levels, etc.)
2. Engine-level vetoes (ML model, regime gate, conviction floor)
3. News sentiment from NewsBus

These decisions are logged for audit but not in a machine-readable format suitable for ML training. To evolve from rule-based to adaptive autonomy, we need structured snapshots of what the bot "saw" at each decision point.

### Goals
- Enable offline training of a GPU-based policy model (RTX 3090 path)
- Capture context when the bot skips/vetoes trades (most candidates die at gates)
- Maintain architecture constraints: NewsBus as soft prior, no LLM-from-tweets as order authority

### Non-Goals (Out of Scope)
- Online/live model inference (gated behind separate flag)
- Autonomous trading decisions (remains human-gated)
- LLM-based order generation

## Decision

Implement **DecisionSnapshot** — a compact, machine-readable record of decision context.

### Schema

```python
@dataclass(frozen=True)
class DecisionSnapshot:
    # Identity
    snapshot_id: str       # Deterministic SHA256 hash
    symbol: str
    ts: datetime
    
    # Context
    mode: str              # live, simulation, paper
    strategy_id: str
    
    # Decision
    action: DecisionAction # signal_buy, signal_sell, skip, veto, error
    reason: str
    gate_name: str | None  # Which gate blocked (if skip/veto)
    confidence: float
    
    # Features (17-dimensional vector)
    price_vol: PriceVolumeFeatures
    
    # News context
    news: NewsAggregate    # article_count, sentiment, corroboration, gate_action
    
    # Market regime
    regime: RegimeContext  # SPY slope, VIX, tape classification
    
    # Would-be sizing (if entry was blocked)
    would_entry_price: float | None
    would_stop_loss: float | None
    would_take_profit: float | None
    would_size_shares: int | None
```

### Persistence

Snapshots are persisted to Postgres (`bot_decision_snapshots` table) via fire-and-forget async writes. The write path is isolated from the trading loop — failures are logged but never block decisions.

### Feature Flags

```python
FEATURE_SNAPSHOT_LOGGING_ENABLED = True   # ON by default
FEATURE_SNAPSHOT_INFERENCE_ENABLED = False # OFF by default
```

Logging and inference are intentionally decoupled:
- **Logging ON, Inference OFF**: Collect training data without affecting live behavior
- **Inference ON**: Future milestone — requires explicit opt-in and validation

## Implementation

### Emit Points

1. **Strategy level** (`strategies/base.py`):
   - Every `_log_decision()` call emits a snapshot
   - Covers all strategy gates (gate_a through gate_f for oversold_v2)

2. **Engine level** (`core/engine.py`):
   - ML veto path
   - Regime gate veto
   - (Extensible to other engine-level vetoes)

### API Endpoints

```
GET /api/decision-snapshots?symbol=TSLA&strategy_id=oversold_v2&action=skip&limit=100
GET /api/features/{symbol}  # Latest snapshot with 17-dim feature vector
```

### Dashboard Integration

`DecisionData` interface extended with:
- `snapshotId`: Reference to snapshot for correlation
- `snapshotTs`: Timestamp
- `snapshotStrategy`: Strategy that produced the snapshot

## GPU Inference Sidecar Integration

The future inference sidecar will consume snapshots via:

### Option A: Polling (MVP)
```python
# Sidecar polls every N seconds
snapshots = await fetch("/api/decision-snapshots?since={last_ts}&action=skip")
for snap in snapshots:
    features = snap["price_vol"]
    vector = [features["returns_1"], features["returns_5"], ...]
    prediction = model.predict(vector)
    # Log or surface prediction for comparison
```

### Option B: WebSocket Stream (Future)
```python
# Subscribe to real-time snapshot stream
async for snapshot in ws.connect("/ws/snapshots"):
    # Inference in near-realtime
```

### Option C: Database Trigger (Future)
```sql
-- Postgres NOTIFY on new snapshot
CREATE TRIGGER snapshot_notify
AFTER INSERT ON bot_decision_snapshots
FOR EACH ROW
EXECUTE FUNCTION pg_notify('snapshots', row_to_json(NEW)::text);
```

### Training Pipeline

```python
# Export snapshots for training
snapshots = await fetch("/api/decision-snapshots?limit=10000")
X = np.array([s["price_vol"]["to_vector()"] for s in snapshots])
y = ... # Labels: outcome (profit/loss) or human decision override

model = train_policy_model(X, y)
model.save("policy_v1.pt")
```

## Consequences

### Positive
- Complete audit trail of decision context
- Training data accumulates passively during normal operation
- Clear separation between logging and inference
- Schema matches existing MLFeatureExtractor for consistency

### Negative
- ~200 bytes/snapshot adds DB storage (mitigated by TTL/retention policy)
- Slight latency on each decision (mitigated by async fire-and-forget)

### Neutral
- Inference remains disabled until explicitly enabled
- No changes to autonomous_live default
- NewsGate decisions captured but not controlling

## Migration Path

1. **Phase 1 (This PR)**: Logging enabled, inference disabled. Collect data.
2. **Phase 2**: Train initial policy model offline using collected snapshots.
3. **Phase 3**: Shadow-mode inference — model makes predictions logged but not acted upon.
4. **Phase 4**: A/B test against rule-based decisions (Research Stage metrics).
5. **Phase 5**: Promote to live inference with explicit Config opt-in.

## References

- `core/decision_snapshot.py` — Schema definition
- `data_providers/db_logger.py` — Persistence layer
- `strategies/base.py` — Strategy-level emission
- `api/routes.py` — Read API endpoints
- `tests/test_decision_snapshot.py` — Test coverage
