"""v-playbook-desk-b2-2026-09-25: the desk — a human trader's daily routine.

Pure decision modules (daily plan, in-play ranking, adaptive playbook,
journal) plus thin file-backed stores. Nothing here imports `research`; the
JSON/Parquet trees the research pipeline writes are the only channel between
the two sides.
"""
