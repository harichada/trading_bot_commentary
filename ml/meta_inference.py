"""Meta-model inference adapter.

Loads the joblib bundle produced by ``train_meta_model.py`` and exposes
a clean inference API: hand it a feature dict and the primary's side,
get back a ``MetaSignal`` with the win probability plus pre-computed
threshold gates for the trade/no-trade decision.

This adapter is deliberately read-only: it never modifies the model,
never writes state, and has no side effects beyond logging. Safe to
import from the live trading loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


_REQUIRED_KEYS = ("classifier", "scaler", "feature_names")


@dataclass(frozen=True)
class MetaSignal:
    """Result of a single meta-model inference call."""

    win_probability: float
    side: int
    would_trade_at_065: bool
    would_trade_at_070: bool
    would_trade_at_075: bool


class MetaInference:
    """Inference wrapper for the meta-model bundle."""

    def __init__(
        self,
        classifier,
        scaler,
        feature_names: list[str],
        primary_strategy: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._classifier = classifier
        self._scaler = scaler
        self.feature_names = list(feature_names)
        self.primary_strategy = primary_strategy
        self.config = dict(config or {})

    @classmethod
    def load(cls, path: str | Path) -> "MetaInference":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Meta-model bundle not found: {path}")

        bundle = joblib.load(path)
        missing = [k for k in _REQUIRED_KEYS if k not in bundle]
        if missing:
            raise KeyError(
                f"Meta-model bundle at {path} is missing keys: {missing}"
            )
        return cls(
            classifier=bundle["classifier"],
            scaler=bundle["scaler"],
            feature_names=bundle["feature_names"],
            primary_strategy=bundle.get("primary_strategy", "unknown"),
            config=bundle.get("config"),
        )

    def predict(self, features: dict[str, float], side: int) -> MetaSignal:
        """Return the meta-model's win probability for this setup."""
        if side not in (1, -1):
            raise ValueError(f"side must be +1 or -1; got {side!r}")

        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise KeyError(
                f"Missing required features: {missing[:5]}"
                + (" (and more)" if len(missing) > 5 else "")
            )

        row = np.array(
            [[float(features[name]) for name in self.feature_names]],
            dtype="float64",
        )
        scaled = self._scaler.transform(row)
        proba = float(self._classifier.predict_proba(scaled)[0, 1])

        return MetaSignal(
            win_probability=proba,
            side=int(side),
            would_trade_at_065=proba >= 0.65,
            would_trade_at_070=proba >= 0.70,
            would_trade_at_075=proba >= 0.75,
        )
