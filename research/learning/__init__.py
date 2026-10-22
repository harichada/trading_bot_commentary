"""v-learning-loop-2026-10-22: research side of the learning loop (nightly /
weekly jobs, exit replay, loss attribution). Read-only against Postgres; the
only writes are report files and, when LEARNING_AUTODEMOTE_ENFORCE is on, the
playbook.json demotions."""
