#!/usr/bin/env bash
# v-rollback-2026-05-23: one-command revert of the quarter-size launch
# (commit tagged launch-2026-05-26, base tagged pre-launch-2026-05-26).
#
# Use this if Tuesday 2026-05-26 launch behaves badly and you want to
# back out cleanly. It restores Config().yaml from the tracked baseline
# snapshot first, then runs git revert.
#
# Safe to run before or after Tuesday's session. Aborts loudly if the
# expected tags aren't present (e.g. you're on the wrong branch).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO}"

TAG="launch-2026-05-26"
BASE_TAG="pre-launch-2026-05-26"
SNAPSHOT="Config_launch_baseline_2026-05-26.yaml"

# Sanity checks.
if ! git rev-parse "${TAG}" >/dev/null 2>&1; then
  echo "ABORT: tag '${TAG}' not found. Are you on the right branch?" >&2
  exit 1
fi
if ! git rev-parse "${BASE_TAG}" >/dev/null 2>&1; then
  echo "ABORT: tag '${BASE_TAG}' not found." >&2
  exit 1
fi
if [ ! -f "${SNAPSHOT}" ]; then
  echo "ABORT: snapshot file '${SNAPSHOT}' missing. Cannot restore yaml." >&2
  exit 1
fi

cat <<EOF
================================================================
ROLLBACK PLAN — quarter-size launch (${TAG})
================================================================
This will:
  1. Restore Config().yaml from ${SNAPSHOT}
     (current Config().yaml will be moved aside as Config().yaml.rollback-bak)
  2. git revert ${TAG} (creates a new revert commit)

After this:
  * live_size_multiplier reverts to absent (defaults to 1.0)
  * max_daily_loss reverts to 0.05 (pre-launch)
  * enable_breakout_long reverts to true (pre-launch)
  * Late-entry cutoff REMOVED from engine.py
  * 9:35 hard block REMOVED (back to soft veto with meta_proba bypass)
  * token_health check REMOVED from engine startup
  * Strategy multipliers REMOVED from risk_manager

If you also want to roll back the bundled pre-existing dev work
(2026-05-13 through 2026-05-21 side-classifier / schwab fixes),
use 'git reset --hard ${BASE_TAG}' INSTEAD of this script.

================================================================
Press ENTER to proceed, Ctrl-C to abort.
================================================================
EOF
read -r _

# 1. Snapshot the CURRENT Config().yaml before overwriting (in case the
# operator made manual edits since launch that they want to preserve).
if [ -f "Config().yaml" ]; then
  cp "Config().yaml" "Config().yaml.rollback-bak"
  echo "Current Config().yaml backed up to Config().yaml.rollback-bak"
fi

# 2. Restore baseline.
cp "${SNAPSHOT}" "Config().yaml"
echo "Config().yaml restored from ${SNAPSHOT}"

# 3. Revert the launch commit.
git revert --no-edit "${TAG}"
echo
echo "Rollback complete."
echo "  - Snapshot of just-overwritten yaml saved as Config().yaml.rollback-bak"
echo "  - To verify safety dials reverted: ./deploy.sh (it will refuse to start"
echo "    because the expected launch-config no longer matches; that is correct)."
echo
echo "If you also want to drop the rollback revert-commit itself, use:"
echo "  git reset --hard HEAD^"
