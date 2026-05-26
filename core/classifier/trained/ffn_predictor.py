"""FFN-backed trained predictor.

v-ffn-predictor-2026-05-13. Wraps ``SideClassifierFFN``, loads a
checkpoint from disk, runs inference on a single
``SymbolFeatures``, and returns a ``SymbolSideDecision``.

Lifecycle expectations:
  - Construct once at engine start; reuse across signal evals.
  - The model stays loaded on the configured device (GPU if
    available, else CPU). Inference is ~ms per symbol.
  - On model-load failure, ``FFNPredictor.from_checkpoint`` raises;
    the composer's fail-open path then keeps the bot rule-only.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from core.classifier.trained.feature_encoder import FEATURE_DIM, encode
from core.classifier.trained.model import (
    CLASS_LONG_WINS,
    CLASS_NEITHER,
    CLASS_SHORT_WINS,
    SideClassifierFFN,
)
from core.classifier.types import Side, SymbolFeatures, SymbolSideDecision

logger = logging.getLogger("TradingBot")


class FFNPredictor:
    """Trained predictor backed by ``SideClassifierFFN``.

    Mapping from softmax probabilities → SymbolSideDecision:
      - ``long_score``  = P(LONG_WINS)
      - ``short_score`` = P(SHORT_WINS)
      - Allow LONG if long_score >= long_threshold AND
        long_score > short_score.
      - Mirror for SHORT.
      - If neither side passes, ``allowed_sides`` is empty (= NEITHER).
    """

    def __init__(
        self,
        model: SideClassifierFFN,
        device: str = "cpu",
        long_threshold: float = 0.4,
        short_threshold: float = 0.4,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

        self.model.to(self.device)
        self.model.train(False)   # inference mode (dropout off)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Optional[str] = None,
        long_threshold: float = 0.4,
        short_threshold: float = 0.4,
    ) -> "FFNPredictor":
        """Load a state-dict checkpoint and return a ready predictor.

        Raises
        ------
        FileNotFoundError
            If the checkpoint doesn't exist.
        RuntimeError
            If the state-dict's shape doesn't match the current model
            (almost always means feature schema drift — retrain).
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = SideClassifierFFN()
        state = torch.load(path, map_location=device, weights_only=True)
        # Accept both raw state_dict and {"state_dict": ...} bundles.
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        logger.info(
            "FFNPredictor: loaded checkpoint from %s onto %s (feature_dim=%d)",
            path, device, model.feature_dim,
        )
        return cls(model, device=device,
                   long_threshold=long_threshold,
                   short_threshold=short_threshold)

    def predict(self, features: SymbolFeatures) -> SymbolSideDecision:
        vec = encode(features)
        x = torch.from_numpy(vec).to(self.device)

        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        # probs is shape (3,)
        probs_np = probs.detach().cpu().numpy()
        long_score = float(probs_np[CLASS_LONG_WINS])
        short_score = float(probs_np[CLASS_SHORT_WINS])
        neither_score = float(probs_np[CLASS_NEITHER])

        allowed = set()
        if long_score >= self.long_threshold and long_score > short_score:
            allowed.add(Side.LONG)
        if short_score >= self.short_threshold and short_score > long_score:
            allowed.add(Side.SHORT)

        if allowed == {Side.LONG}:
            primary = f"trained:long_wins_p={long_score:.2f}"
        elif allowed == {Side.SHORT}:
            primary = f"trained:short_wins_p={short_score:.2f}"
        else:
            primary = (
                f"trained:no_edge_long={long_score:.2f}_"
                f"short={short_score:.2f}_neither={neither_score:.2f}"
            )

        return SymbolSideDecision(
            symbol=features.symbol,
            allowed_sides=frozenset(allowed),
            long_score=long_score,
            short_score=short_score,
            primary_reason=primary,
            feature_dump={
                "trained_long_wins_p": long_score,
                "trained_short_wins_p": short_score,
                "trained_neither_p": neither_score,
                "device": str(self.device),
            },
        )
