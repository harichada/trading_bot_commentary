"""v-ffn-predictor-2026-05-13: tests for the FFN predictor wrapper.

We test the wrapper logic (encoding → forward pass → decision mapping
→ checkpoint round-trip) without depending on a trained model. The
training is exercised separately by the smoke test in research/.

These tests need ``torch`` to be installed. They are skipped (not
xfailed) when torch is unavailable, so a torch-less CI environment
still runs the rest of the suite cleanly.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")


def _features(**overrides):
    from core.classifier.types import SymbolFeatures

    base = dict(symbol="TEST", close=100.0)
    base.update(overrides)
    return SymbolFeatures(**base)


class TestModelArchitecture:
    def test_model_runs_on_random_input(self):
        from core.classifier.trained.model import (
            N_CLASSES, SideClassifierFFN,
        )
        from core.classifier.trained.feature_encoder import FEATURE_DIM

        model = SideClassifierFFN()
        x = torch.randn(4, FEATURE_DIM)
        out = model(x)
        assert out.shape == (4, N_CLASSES)

    def test_single_sample_unsqueeze(self):
        """1D input gets reshaped to (1, D) inside forward."""
        from core.classifier.trained.model import (
            N_CLASSES, SideClassifierFFN,
        )
        from core.classifier.trained.feature_encoder import FEATURE_DIM

        model = SideClassifierFFN()
        x = torch.randn(FEATURE_DIM)   # 1D
        out = model(x)
        assert out.shape == (1, N_CLASSES)

    def test_predict_proba_sums_to_one(self):
        from core.classifier.trained.model import (
            SideClassifierFFN, predict_proba,
        )
        from core.classifier.trained.feature_encoder import FEATURE_DIM

        model = SideClassifierFFN()
        model.train(False)
        x = torch.randn(3, FEATURE_DIM)
        with torch.inference_mode():
            probs = predict_proba(model, x)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-5)


class TestFFNPredictor:
    def test_predict_returns_symbol_side_decision(self):
        from core.classifier.trained.ffn_predictor import FFNPredictor
        from core.classifier.trained.model import SideClassifierFFN
        from core.classifier.types import SymbolSideDecision

        model = SideClassifierFFN()
        pred = FFNPredictor(model, device="cpu")
        d = pred.predict(_features())
        assert isinstance(d, SymbolSideDecision)
        assert d.symbol == "TEST"
        # probabilities in [0,1]
        assert 0.0 <= d.long_score <= 1.0
        assert 0.0 <= d.short_score <= 1.0

    def test_satisfies_predictor_protocol(self):
        from core.classifier.trained.ffn_predictor import FFNPredictor
        from core.classifier.trained.model import SideClassifierFFN
        from core.classifier.trained.predictor_protocol import TrainedPredictor

        pred = FFNPredictor(SideClassifierFFN(), device="cpu")
        assert isinstance(pred, TrainedPredictor)

    def test_long_class_wins_yields_long_decision(self):
        """Hand-craft a model whose last layer biases toward LONG_WINS
        so the predictor must surface LONG."""
        from core.classifier.trained.ffn_predictor import FFNPredictor
        from core.classifier.trained.model import (
            CLASS_LONG_WINS, SideClassifierFFN,
        )
        from core.classifier.types import Side

        model = SideClassifierFFN()
        # Zero out the final layer, add a huge bias on the LONG class.
        with torch.no_grad():
            final = model.net[-1]   # last Linear
            final.weight.zero_()
            final.bias.zero_()
            final.bias[CLASS_LONG_WINS] = 10.0
        pred = FFNPredictor(model, device="cpu",
                            long_threshold=0.4, short_threshold=0.4)
        d = pred.predict(_features())
        assert Side.LONG in d.allowed_sides
        assert Side.SHORT not in d.allowed_sides
        assert d.long_score > 0.9

    def test_short_class_wins_yields_short_decision(self):
        from core.classifier.trained.ffn_predictor import FFNPredictor
        from core.classifier.trained.model import (
            CLASS_SHORT_WINS, SideClassifierFFN,
        )
        from core.classifier.types import Side

        model = SideClassifierFFN()
        with torch.no_grad():
            final = model.net[-1]
            final.weight.zero_()
            final.bias.zero_()
            final.bias[CLASS_SHORT_WINS] = 10.0
        pred = FFNPredictor(model, device="cpu",
                            long_threshold=0.4, short_threshold=0.4)
        d = pred.predict(_features())
        assert Side.SHORT in d.allowed_sides
        assert Side.LONG not in d.allowed_sides

    def test_neither_class_wins_yields_empty_allowed(self):
        from core.classifier.trained.ffn_predictor import FFNPredictor
        from core.classifier.trained.model import (
            CLASS_NEITHER, SideClassifierFFN,
        )

        model = SideClassifierFFN()
        with torch.no_grad():
            final = model.net[-1]
            final.weight.zero_()
            final.bias.zero_()
            final.bias[CLASS_NEITHER] = 10.0
        pred = FFNPredictor(model, device="cpu",
                            long_threshold=0.4, short_threshold=0.4)
        d = pred.predict(_features())
        assert d.is_forbidden()


class TestCheckpointRoundTrip:
    def test_save_load_preserves_predictions(self):
        from core.classifier.trained.ffn_predictor import FFNPredictor
        from core.classifier.trained.model import SideClassifierFFN

        model = SideClassifierFFN()
        # Randomize a bit so the test isn't entirely on default init.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Reference prediction
        pred1 = FFNPredictor(model, device="cpu")
        f = _features(sma_20_daily=110.0, sma_50_daily=105.0,
                      rs_vs_spy_5d=0.02)
        d1 = pred1.predict(f)

        # Round-trip through disk
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "test_model.pt"
            torch.save(model.state_dict(), ckpt)

            pred2 = FFNPredictor.from_checkpoint(ckpt, device="cpu")
            d2 = pred2.predict(f)

        assert d1.long_score == pytest.approx(d2.long_score)
        assert d1.short_score == pytest.approx(d2.short_score)

    def test_missing_checkpoint_raises_file_not_found(self):
        from core.classifier.trained.ffn_predictor import FFNPredictor

        with pytest.raises(FileNotFoundError):
            FFNPredictor.from_checkpoint("/nonexistent/checkpoint.pt")
