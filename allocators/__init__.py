"""Allocation layer — decides WHICH strategy deserves the current tape.

Evidence basis and design law: docs/PROFITABILITY_ROADMAP.md §3-4.
Every allocator ships shadow-first behind a config flag with an
append-only NDJSON ledger.
"""
