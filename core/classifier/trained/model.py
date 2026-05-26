"""PyTorch baseline model for the trained side classifier.

v-trained-baseline-model-2026-05-13. Simple feed-forward classifier
that consumes the encoded SymbolFeatures vector (length
``FEATURE_DIM``) and outputs class logits over
``{LONG_WINS, SHORT_WINS, NEITHER}`` — matching the triple-barrier
label encoding in ``research/labels.py``.

This is the **baseline**. It deliberately uses a tiny FFN rather
than TFT for v1 because:
  - Establishing the data pipeline + checkpoint format + composer
    integration matters more than architecture choice in the first
    iteration.
  - The interface (``predict(features) -> SymbolSideDecision``) is
    architecture-agnostic; swapping FFN → TFT later is one file.
  - Faster to train; faster to iterate; easier to debug.

Once this is shadow-validated, the operator can decide whether the
FFN baseline carries enough edge or whether the upgrade to TFT
(see ``.claude/PLAN_trained_side_classifier.md``) is warranted.
"""
from __future__ import annotations

import torch
from torch import nn

from core.classifier.trained.feature_encoder import FEATURE_DIM

# Class indices — must match the integer encoding in
# ``research/labels.py::BarrierLabel`` (alphabetical / enum order).
CLASS_LONG_WINS: int = 0
CLASS_SHORT_WINS: int = 1
CLASS_NEITHER: int = 2
N_CLASSES: int = 3

# Hidden sizes chosen to give roughly 2-3 K parameters total —
# big enough to learn signal, small enough to train in seconds
# on the 3090 (which is hilariously over-spec for this) and small
# enough to never overfit on the available data.
HIDDEN_1: int = 64
HIDDEN_2: int = 32


class SideClassifierFFN(nn.Module):
    """3-layer feedforward over the encoded feature vector.

    Architecture::

        Linear(FEATURE_DIM, 64) -> ReLU -> Dropout(0.2)
        -> Linear(64, 32) -> ReLU -> Dropout(0.2)
        -> Linear(32, 3)   # logits over {LONG, SHORT, NEITHER}
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden1: int = HIDDEN_1,
        hidden2: int = HIDDEN_2,
        n_classes: int = N_CLASSES,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, feature_dim)`` or ``(feature_dim,)`` for
            single-symbol inference. We accept both and reshape
            internally.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)   # (1, feature_dim)
        return self.net(x)


def predict_proba(model: SideClassifierFFN, x: torch.Tensor) -> torch.Tensor:
    """Softmax over the class logits. Returns shape ``(batch, n_classes)``.

    Caller is responsible for putting the model in inference mode
    (``model.train(False)``) and disabling autograd (e.g.
    ``with torch.inference_mode():``).
    """
    logits = model(x)
    return torch.softmax(logits, dim=-1)
