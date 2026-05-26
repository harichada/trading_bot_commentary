"""Trained side-classifier subsystem.

v-trained-classifier-package-2026-05-13. PyTorch baseline + training
infrastructure. Public surface:

  - ``TrainedPredictor`` (Protocol)  — duck-typed contract
  - ``StubPredictor``                — fixed-decision, no GPU
  - ``FFNPredictor``                 — loads checkpoint, runs inference
  - ``SideClassifierFFN``            — the model class
  - ``encode``, ``FEATURE_DIM``       — feature encoding

Training: see ``research/train_classifier.py``.

This subpackage imports torch. The parent ``core.classifier`` package
does NOT import this one — so a torch-less environment can still use
the rule-based classifier without dragging in CUDA / pytorch-lightning.
"""
from core.classifier.trained.feature_encoder import FEATURE_DIM, encode
from core.classifier.trained.ffn_predictor import FFNPredictor
from core.classifier.trained.model import SideClassifierFFN
from core.classifier.trained.predictor_protocol import TrainedPredictor
from core.classifier.trained.stub_predictor import StubPredictor

__all__ = [
    "FEATURE_DIM",
    "encode",
    "FFNPredictor",
    "SideClassifierFFN",
    "StubPredictor",
    "TrainedPredictor",
]
