# Operations Runbook - Institutional Trading System

## Table of Contents
1. [System Overview](#system-overview)
2. [Deployment Procedures](#deployment-procedures)
3. [Monitoring & Health Checks](#monitoring--health-checks)
4. [Incident Response](#incident-response)
5. [Common Issues & Solutions](#common-issues--solutions)
6. [Emergency Procedures](#emergency-procedures)
7. [Maintenance Procedures](#maintenance-procedures)

---

## System Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INSTITUTIONAL TRADING SYSTEM                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   FastAPI    │  │  WebSocket   │  │   Health     │              │
│  │   Gateway    │  │   Server     │  │   Endpoints  │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                       │
│         └────────────┬────┴─────────────────┘                       │
│                      │                                              │
│  ┌───────────────────┴───────────────────┐                         │
│  │         TRADING COORDINATOR           │                         │
│  │   ┌─────────────────────────────┐    │                         │
│  │   │     State Machine           │    │                         │
│  │   │  INIT → CONNECTED → TRADING │    │                         │
│  │   └─────────────────────────────┘    │                         │
│  └───────────────────┬───────────────────┘                         │
│                      │                                              │
│  ┌──────────┬────────┴────────┬──────────┬──────────┐             │
│  │          │                 │          │          │              │
│  ▼          ▼                 ▼          ▼          ▼              │
│ ┌────┐   ┌────────┐   ┌──────────┐  ┌────────┐  ┌────────┐       │
│ │Risk│   │Strategy│   │Execution │  │Market  │  │ML      │       │
│ │Eng │   │Executor│   │Engine    │  │Data    │  │Pipeline│       │
│ └────┘   └────────┘   └──────────┘  └────────┘  └────────┘       │
│                              │                                     │
│                              ▼                                     │
│                    ┌──────────────────┐                           │
│                    │   Schwab API     │                           │
│                    │   (via schwab-py)│                           │
│                    └──────────────────┘                           │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                    OBSERVABILITY                         │     │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │     │
│  │  │Metrics  │  │Logging  │  │Alerting │  │Audit    │   │     │
│  │  │Collector│  │System   │  │System   │  │Trail    │   │     │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │     │
│  └─────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| State Machine | `institutional_core.py` | System state management |
| Trading Coordinator | `institutional_trading_system.py` | Orchestration |
| Risk Intelligence | `risk_intelligence.py` | VaR, correlation, limits |
| ML Pipeline | `ml_pipeline.py` | Feature store, drift detection |
| Observability | `observability.py` | Logging, metrics, alerts |
| Infrastructure | `infrastructure.py` | Config, deployment |

### System States

| State | Description | Allowed Operations |
|-------|-------------|-------------------|
| `INITIALIZING` | System startup | Health checks only |
| `CONNECTED` | API connected | Position queries |
| `RECONCILING` | Syncing positions | Fetch positions |
| `TRADING` | Normal operation | All trading ops |
| `RISK_LOCKED` | Risk limit breach | Close positions only |
| `EMERGENCY_EXIT` | Critical failure | Market sell all |
| `SHUTDOWN` | Graceful termination | None |

---

## Deployment Procedures

### Pre-Deployment Checklist

- [ ] All tests passing
- [ ] Configuration reviewed for target environment
- [ ] Secrets properly set in environment
- [ ] Database migrations applied (if any)
- [ ] Previous deployment backed up
- [ ] Market hours verified (avoid deploying during open)
- [ ] Team notified of deployment

### Deployment Steps

#### 1. Pre-flight Checks

```bash
# Verify environment
echo $TRADING_ENV  # Should be: production/staging/development

# Check configuration
python -c "from infrastructure import ConfigurationManager; cm = ConfigurationManager(); cm.load_config(); print('Config valid')"

# Run quick health check
python -c "from institutional_core import InstitutionalTradingCore; core = InstitutionalTradingCore(); print('Core initializable')"
```

#### 2. Stop Current Instance

```bash
# Send graceful shutdown signal
kill -SIGTERM $(cat /var/run/trading_bot.pid)

# Wait for graceful shutdown (max 60 seconds)
timeout 60 tail -f /var/log/trading/trading_bot.log | grep -q "Graceful shutdown complete"

# Verify process stopped
ps aux | grep trading_bot
```

#### 3. Deploy New Version

```bash
# Pull latest code
cd /opt/trading_bot
git fetch origin
git checkout $DEPLOY_TAG

# Install/update dependencies
pip install -r requirements.txt

# Run database migrations (if any)
# python migrate.py

# Verify deployment
python -c "from institutional_trading_system import InstitutionalTradingSystem; print('Import successful')"
```

#### 4. Start New Instance

```bash
# Start with production config
export TRADING_ENV=production
python institutional_trading_system.py --config config/config.production.yaml &

# Capture PID
echo $! > /var/run/trading_bot.pid

# Verify startup
sleep 10
curl http://localhost:8000/health
```

#### 5. Post-Deployment Verification

```bash
# Check logs for errors
tail -100 /var/log/trading/trading_bot.log | grep -i error

# Verify API connectivity
curl http://localhost:8000/api/status | jq .

# Check position reconciliation
curl http://localhost:8000/api/positions | jq .

# Monitor for 15 minutes
watch -n 30 'curl -s http://localhost:8000/health | jq .'
```

### Rollback Procedure

```bash
# Stop current instance
kill -SIGTERM $(cat /var/run/trading_bot.pid)
sleep 30

# Revert to previous version
git checkout $PREVIOUS_TAG

# Restart
python institutional_trading_system.py --config config/config.production.yaml &
echo $! > /var/run/trading_bot.pid
```

---

## Monitoring & Health Checks

### Health Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `/health` | Overall system health | `{"status": "healthy"}` |
| `/health/ready` | Readiness probe | HTTP 200 if ready |
| `/health/live` | Liveness probe | HTTP 200 if alive |
| `/api/status` | Detailed status | Full system status |

### Key Metrics to Monitor

#### Trading Metrics

| Metric | Warning Threshold | Critical Threshold |
|--------|-------------------|-------------------|
| `daily_pnl` | < -$2,000 | < -$5,000 |
| `drawdown_percent` | > 5% | > 10% |
| `position_count` | > 8 | > 10 |
| `order_rejection_rate` | > 5% | > 15% |

#### System Metrics

| Metric | Warning Threshold | Critical Threshold |
|--------|-------------------|-------------------|
| `api_latency_p99` | > 500ms | > 2000ms |
| `memory_usage_mb` | > 1500 | > 2000 |
| `cpu_usage_percent` | > 70% | > 90% |
| `disk_usage_percent` | > 80% | > 90% |

#### Circuit Breaker Metrics

| Metric | Action Required |
|--------|-----------------|
| `circuit_breaker_api.state = OPEN` | API connectivity issue |
| `circuit_breaker_order.state = OPEN` | Order execution issue |
| `circuit_breaker_data.state = OPEN` | Market data issue |

### Log Locations

```
/var/log/trading/trading_bot.log     # Main application log
/var/log/trading/audit.log           # Audit trail
/var/log/trading/trades.log          # Trade execution log
/var/log/trading/alerts.log          # Alert history
```

### Log Search Commands

```bash
# Find errors in last hour
grep "ERROR" /var/log/trading/trading_bot.log | tail -100

# Find specific order
grep "order_id.*ORD123" /var/log/trading/trading_bot.log

# Find position reconciliation issues
grep "reconciliation" /var/log/trading/trading_bot.log | grep -i "discrepancy\|mismatch"

# Monitor in real-time
tail -f /var/log/trading/trading_bot.log | jq -r 'select(.level == "ERROR")'
```

---

## Incident Response

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| **SEV-1** | System down, trading halted | Immediate | API authentication failure |
| **SEV-2** | Degraded, partial trading | 15 minutes | High latency, partial fills |
| **SEV-3** | Minor issue, full trading | 1 hour | Non-critical alerts |
| **SEV-4** | Informational | Next business day | Log warnings |

### SEV-1: System Down

**Symptoms:**
- No trading activity
- Health endpoint returning unhealthy
- Circuit breakers all open
- Multiple critical alerts

**Immediate Actions:**

1. **Assess the situation** (2 minutes)
   ```bash
   # Check system status
   curl http://localhost:8000/health
   curl http://localhost:8000/api/status
   tail -50 /var/log/trading/trading_bot.log
   ```

2. **Check positions** (2 minutes)
   ```bash
   # Verify current positions with broker directly
   python -c "from schwab import auth; c = auth.easy_client(...); print(c.get_positions().json())"
   ```

3. **Determine cause** (5 minutes)
   - API authentication expired?
   - Network connectivity?
   - Database down?
   - Memory/disk exhaustion?

4. **Take corrective action**
   - Refresh API token if expired
   - Restart if hung
   - Scale resources if exhausted
   - Failover if infrastructure issue

5. **Verify recovery**
   ```bash
   # Watch health for 5 minutes
   watch -n 10 'curl -s http://localhost:8000/health'
   ```

### SEV-2: Degraded Performance

**Symptoms:**
- Slow order execution
- Partial fills not handled
- High API latency
- Some circuit breakers open

**Actions:**

1. Identify degraded component
2. Check rate limits
3. Review recent changes
4. Consider reducing trading activity
5. Monitor closely

---

## Common Issues & Solutions

### Issue: API Authentication Expired

**Symptoms:**
- 401 errors in logs
- `AuthenticationException` in logs
- State machine in `AUTH_REQUIRED`

**Solution:**
```bash
# Re-authenticate with Schwab
python -c "
from schwab import auth
auth.easy_client(
    api_key='YOUR_KEY',
    app_secret='YOUR_SECRET',
    callback_url='https://127.0.0.1',
    token_path='token_1.json',
    asyncio=True
)
print('Token refreshed')
"

# Restart the trading system
kill -SIGTERM $(cat /var/run/trading_bot.pid)
sleep 10
python institutional_trading_system.py &
```

### Issue: Position Reconciliation Mismatch

**Symptoms:**
- `PHANTOM` or `MISSING` positions in logs
- P&L calculation incorrect
- State machine stuck in `RECONCILING`

**Solution:**
```bash
# Force reconciliation
curl -X POST http://localhost:8000/api/reconcile

# If still mismatched, manual review:
# 1. Get broker positions
python -c "..."
# 2. Get local positions
cat trading_state_institutional.json | jq '.positions'
# 3. Manually update local state if needed
```

### Issue: Circuit Breaker Open

**Symptoms:**
- Orders not executing
- "circuit_breaker_open" errors
- Specific component degraded

**Solution:**
```bash
# Check circuit breaker status
curl http://localhost:8000/api/status | jq '.circuit_breakers'

# Wait for automatic recovery (recovery_timeout)
# Or manually reset if issue is resolved:
curl -X POST http://localhost:8000/api/circuit-breaker/reset/order
```

### Issue: High Memory Usage

**Symptoms:**
- OOM warnings in logs
- Slow performance
- Memory metric > threshold

**Solution:**
```bash
# Check memory usage
curl http://localhost:8000/api/status | jq '.health.memory'

# Force garbage collection
curl -X POST http://localhost:8000/api/system/gc

# If persistent, restart during low-activity period
```

### Issue: ML Model Drift Detected

**Symptoms:**
- "drift_detected" alerts
- Prediction accuracy declining
- PSI score > threshold

**Solution:**
```bash
# Check drift status
curl http://localhost:8000/api/ml/drift-status | jq .

# Trigger model retraining
curl -X POST http://localhost:8000/api/ml/retrain

# Monitor retraining progress
watch curl -s http://localhost:8000/api/ml/training-status
```

---

## Emergency Procedures

### Emergency Close All Positions

**When to use:** Critical system failure, market crash, or security breach.

```bash
# Via API (preferred)
curl -X POST http://localhost:8000/api/emergency/close-all \
  -H "Authorization: Bearer $EMERGENCY_TOKEN"

# Via direct script (if API unavailable)
python -c "
from schwab import auth
c = auth.easy_client(...)
positions = c.get_positions().json()
for pos in positions:
    symbol = pos['symbol']
    qty = pos['quantity']
    c.place_order(
        account_hash='...',
        order_spec={'symbol': symbol, 'type': 'MARKET', 'side': 'SELL', 'quantity': qty}
    )
print('All positions closed')
"
```

### Emergency Shutdown

```bash
# Graceful shutdown with position close
curl -X POST http://localhost:8000/api/emergency/shutdown?close_positions=true

# Force kill if unresponsive
kill -9 $(cat /var/run/trading_bot.pid)

# Verify all processes stopped
ps aux | grep -i trading

# Check for orphan orders
# (Manual review in Schwab account required)
```

### Manual Override

**To bypass circuit breakers (use with extreme caution):**

```bash
# Temporarily disable specific circuit breaker
export BYPASS_CIRCUIT_BREAKER=order

# Execute critical operation
curl -X POST http://localhost:8000/api/positions/close/SYMBOL

# Re-enable
unset BYPASS_CIRCUIT_BREAKER
```

---

## Maintenance Procedures

### Daily Checks

| Time (ET) | Task | Command |
|-----------|------|---------|
| 06:00 | Review overnight logs | `grep ERROR /var/log/trading/*.log` |
| 09:25 | Verify market ready | `curl localhost:8000/health` |
| 16:05 | Review daily P&L | `curl localhost:8000/api/pnl/daily` |
| 16:30 | Verify positions | `curl localhost:8000/api/reconcile` |

### Weekly Tasks

- Review and rotate logs
- Check disk space
- Review alert thresholds
- Update feature flags if needed
- Backup state files

### Monthly Tasks

- Review and update risk limits
- Analyze strategy performance
- ML model performance review
- Security audit review
- Dependency updates

### Log Rotation

```bash
# Rotate logs (usually in cron)
logrotate /etc/logrotate.d/trading_bot

# Manual rotation
cd /var/log/trading
mv trading_bot.log trading_bot.log.$(date +%Y%m%d)
kill -USR1 $(cat /var/run/trading_bot.pid)  # Signal log reopen
gzip trading_bot.log.$(date +%Y%m%d)
```

### Backup Procedures

```bash
# Backup state files
tar -czf /backup/trading_state_$(date +%Y%m%d).tar.gz \
  trading_state*.json \
  trading_brain.json \
  models/

# Backup to remote (S3 example)
aws s3 cp /backup/trading_state_$(date +%Y%m%d).tar.gz \
  s3://trading-backups/daily/

# Verify backup
aws s3 ls s3://trading-backups/daily/ | tail -5
```

---

## Contact & Escalation

### On-Call Rotation

| Role | Primary | Backup |
|------|---------|--------|
| Trading Ops | trading-ops@company.com | +1-XXX-XXX-XXXX |
| Platform Eng | platform@company.com | +1-XXX-XXX-XXXX |
| Management | trading-mgmt@company.com | - |

### Escalation Matrix

| Severity | First Contact | Escalate After | Escalate To |
|----------|---------------|----------------|-------------|
| SEV-1 | Trading Ops | 15 min | Platform Eng + Mgmt |
| SEV-2 | Trading Ops | 30 min | Platform Eng |
| SEV-3 | Trading Ops | 2 hours | - |
| SEV-4 | Ticket | Next day | - |

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-21 | Institutional Transformation | Initial version |

---

*This runbook is a living document. Update it as procedures change.*
