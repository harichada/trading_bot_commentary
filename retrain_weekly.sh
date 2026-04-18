#!/usr/bin/env bash
# Weekly meta-model retrain — runs Sunday 22:00 ET via cron.
# Backs up the current model, retrains on the latest 180 days from
# Postgres, and writes a timestamped report for comparison.
#
# Cron entry:
#   0 2 * * 1 cd /home/nvidia/claude/trading_bot_commentary && ./retrain_weekly.sh >> retrain_weekly.log 2>&1

set -euo pipefail

PYTHON="/home/nvidia/anaconda3/envs/trading-bot/bin/python"
PROJECT_DIR="/home/nvidia/claude/trading_bot_commentary"
cd "$PROJECT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "[$TIMESTAMP] === Weekly retrain starting ==="

# 1. Backup current model
if [ -f ml_meta_model.pkl ]; then
    cp ml_meta_model.pkl "model_versions/ml_meta_model_${TIMESTAMP}.pkl"
    echo "[$TIMESTAMP] Backed up ml_meta_model.pkl"
fi
mkdir -p model_versions

# 2. Retrain meta-model (uses latest Postgres data via nightly ingest)
echo "[$TIMESTAMP] Training meta-model..."
$PYTHON train_meta_model.py \
    --report-path "model_versions/ml_meta_training_report_${TIMESTAMP}.json" \
    --model-path ml_meta_model.pkl

# 3. Compare new metrics to baseline
echo "[$TIMESTAMP] Training complete. Report at model_versions/ml_meta_training_report_${TIMESTAMP}.json"

# 4. Also retrain primary v2 model
echo "[$TIMESTAMP] Training v2 primary model..."
$PYTHON train_ml_model_v2.py \
    --pt-mult 1.5 --sl-mult 1.5 \
    --report-path "model_versions/ml_training_report_v2_${TIMESTAMP}.json" \
    --model-path ml_model_v2.pkl

echo "[$TIMESTAMP] === Weekly retrain complete ==="
