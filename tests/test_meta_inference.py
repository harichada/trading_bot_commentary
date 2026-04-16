"""Tests for meta-model inference adapter."""
from __future__ import annotations

import joblib
import numpy as np
import pytest

from ml.meta_inference import MetaInference, MetaSignal


class _FakeClassifier:
    """Stand-in for XGBClassifier — records calls, returns scripted probabilities."""

    def __init__(self, return_probas: list[list[float]] | None = None) -> None:
        self.return_probas = return_probas or [[0.3, 0.7]]
        self.calls: list[np.ndarray] = []

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.calls.append(X.copy())
        n = len(X)
        rows = [
            self.return_probas[i % len(self.return_probas)]
            for i in range(n)
        ]
        return np.array(rows, dtype="float64")


class _FakeScaler:
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X * 2.0


class _OrderRecordingClassifier:
    """Classifier that writes its input to a class-level slot.

    Using a class attribute survives joblib's pickle round-trip because
    the list is a module-level singleton, not an instance attribute.
    """
    last_seen: list[float] = []

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _OrderRecordingClassifier.last_seen = X[0].tolist()
        return np.array([[0.5, 0.5]])


def _write_bundle(tmp_path, clf, scaler, feature_names) -> str:
    bundle = {
        "version": "meta-v1",
        "primary_strategy": "mean_reversion",
        "classifier": clf,
        "scaler": scaler,
        "feature_names": feature_names,
        "config": {"rsi_threshold": 30.0},
    }
    path = tmp_path / "meta.pkl"
    joblib.dump(bundle, path)
    return str(path)


class TestMetaInferenceLoader:
    def test_load_from_file(self, tmp_path):
        path = _write_bundle(
            tmp_path, _FakeClassifier(), _FakeScaler(),
            ["a", "b", "c"],
        )

        inf = MetaInference.load(path)

        assert inf.feature_names == ["a", "b", "c"]
        assert inf.primary_strategy == "mean_reversion"

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MetaInference.load(str(tmp_path / "does_not_exist.pkl"))

    def test_load_rejects_bad_bundle(self, tmp_path):
        path = tmp_path / "bad.pkl"
        joblib.dump({"oops": True}, path)

        with pytest.raises(KeyError):
            MetaInference.load(str(path))


class TestMetaInferencePredict:
    def test_predict_returns_scripted_win_probability(self, tmp_path):
        """The loaded pipeline must pass features through scaler then classifier."""
        clf = _FakeClassifier(return_probas=[[0.4, 0.6]])
        path = _write_bundle(tmp_path, clf, _FakeScaler(), ["x", "y"])
        inf = MetaInference.load(path)

        result = inf.predict({"x": 1.0, "y": 2.0}, side=1)

        assert isinstance(result, MetaSignal)
        assert result.win_probability == pytest.approx(0.6)

    def test_predict_feature_order_follows_bundle(self, tmp_path):
        """Use a classifier whose output depends on input ordering.

        This _AssertingClassifier returns a fixed value but records the
        first (post-scaler) row on a module-level list, so we can read
        it back even after joblib's pickle round-trip.
        """
        clf = _OrderRecordingClassifier()
        path = _write_bundle(
            tmp_path, clf, _FakeScaler(), ["first", "second", "third"],
        )
        inf = MetaInference.load(path)

        inf.predict({"second": 2.0, "third": 3.0, "first": 1.0}, side=1)

        # After _FakeScaler's *2: first=2, second=4, third=6.
        np.testing.assert_array_almost_equal(
            np.array(_OrderRecordingClassifier.last_seen),
            np.array([2.0, 4.0, 6.0]),
        )

    def test_predict_missing_feature_raises(self, tmp_path):
        path = _write_bundle(
            tmp_path, _FakeClassifier(), _FakeScaler(), ["a", "b"],
        )
        inf = MetaInference.load(path)

        with pytest.raises(KeyError, match="a"):
            inf.predict({"b": 1.0}, side=1)

    def test_predict_rejects_invalid_side(self, tmp_path):
        path = _write_bundle(
            tmp_path, _FakeClassifier(), _FakeScaler(), ["a"],
        )
        inf = MetaInference.load(path)

        with pytest.raises(ValueError):
            inf.predict({"a": 1.0}, side=0)

    def test_signal_includes_threshold_gates(self, tmp_path):
        clf = _FakeClassifier(return_probas=[[0.3, 0.72]])
        path = _write_bundle(tmp_path, clf, _FakeScaler(), ["a"])
        inf = MetaInference.load(path)

        result = inf.predict({"a": 1.0}, side=-1)

        assert result.would_trade_at_065 is True
        assert result.would_trade_at_070 is True
        assert result.would_trade_at_075 is False
        assert result.side == -1
