"""Thin research-side re-export of the shared sentiment backend.

v-playbook-desk-b1-2026-09-25 (Task 7): research code should import
`score`/`available_backends` from here rather than reaching into
`core.sentiment_backend` directly, but both names are the exact same
functions — this module adds no behavior of its own so research and live
always agree on scoring for a given `SENTIMENT_BACKEND`.
"""
from __future__ import annotations

from core.sentiment_backend import available_backends, score

__all__ = ["score", "available_backends"]
