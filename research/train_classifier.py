"""Train the side-classifier FFN baseline.

v-train-classifier-2026-05-13. Builds (features, label) examples
from yfinance historical bars, trains ``SideClassifierFFN`` with
cross-entropy loss, saves the best checkpoint by validation loss.

Process-level isolation: imports ``core.classifier`` and ``research.*``
but NEVER ``core.engine``. Safe to run on the same machine as the
live bot; uses the 3090 by default but falls back to CPU.

Usage:
    python research/train_classifier.py \\
        --symbols NVDA,AAPL,MSFT,TSLA,AMD,NVDA,GOOG,META,AMZN,NFLX \\
        --start 2020-01-01 --end 2024-12-31 \\
        --out models/classifier/ffn_v1.pt \\
        --epochs 30 --batch-size 256
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Bootstrap project root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class TrainConfig:
    symbols: List[str]
    start_date: str
    end_date: str
    benchmark: str
    output_path: Path
    epochs: int
    batch_size: int
    lr: float
    val_fraction: float
    seed: int
    device: str


def parse_args(argv: Optional[List[str]] = None) -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train the side-classifier FFN baseline."
    )
    parser.add_argument("--symbols", required=True,
                        help="Comma-separated tickers")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None,
                        help="Default: today")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--out", required=True,
                        help="Output checkpoint path (.pt)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Trailing fraction held out for validation "
                        "(time-based split, not random)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (default: auto)")
    args = parser.parse_args(argv)

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    end_date = args.end or datetime.utcnow().strftime("%Y-%m-%d")

    return TrainConfig(
        symbols=syms,
        start_date=args.start,
        end_date=end_date,
        benchmark=args.benchmark.upper(),
        output_path=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device or ("cuda" if _cuda_available() else "cpu"),
    )


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _load_bars(symbols: List[str], start: str, end: str) -> dict:
    """Pull daily bars from yfinance. Returns {symbol: DataFrame}."""
    import yfinance as yf

    out = {}
    print(f"[train] loading {len(symbols)} symbols", flush=True)
    for sym in symbols:
        df = yf.download(sym, start=start, end=end,
                         progress=False, auto_adjust=True)
        if df.empty:
            print(f"[train] WARN: {sym} returned no bars", flush=True)
            continue
        # Flatten potential MultiIndex columns from yfinance.
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower()
                      for c in df.columns]
        out[sym] = df
    return out


def _build_examples(
    bars_by_symbol: dict,
    benchmark_bars,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """For every symbol-day with sufficient history (≥50 prior bars)
    and a valid forward triple-barrier label, build (X, y) arrays."""
    import numpy as np
    import pandas as pd

    from core.classifier.trained.feature_encoder import encode
    from research.features import build_features
    from research.labels import BarrierLabel, label_dataset

    label_to_class = {
        BarrierLabel.LONG_WINS: 0,
        BarrierLabel.SHORT_WINS: 1,
        BarrierLabel.NEITHER: 2,
    }

    Xs, ys = [], []
    for sym, df in bars_by_symbol.items():
        if len(df) < 70:
            continue

        # ATR-14 for triple-barrier
        high = df["high"]; low = df["low"]; close = df["close"]
        prev = close.shift(1)
        tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                       axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        labels = label_dataset(df, atr, k=1.5, horizon_bars=5)

        for i in range(50, len(df) - 5):   # need 5 forward bars for label
            window = df.iloc[: i + 1]
            bench_window = (benchmark_bars.iloc[: i + 1]
                            if benchmark_bars is not None else None)
            try:
                feats = build_features(
                    symbol=sym, daily_bars=window,
                    benchmark_daily=bench_window,
                )
                vec = encode(feats)
                cls = label_to_class[labels.iloc[i]]
            except Exception as exc:
                print(f"[train] skip {sym}@{i}: {exc}", flush=True)
                continue
            Xs.append(vec)
            ys.append(cls)

    if not Xs:
        raise RuntimeError("no training examples — check symbol availability")

    return np.stack(Xs), np.array(ys, dtype=np.int64)


def _time_split(X, y, val_fraction: float):
    """Held-out validation = trailing tail of the example sequence.
    For symbol-major iteration this isn't strictly time-ordered, but
    it's a fair "later-built examples are held out" baseline. Proper
    purged-CV is in a follow-up iteration."""
    n = len(X)
    n_val = max(1, int(n * val_fraction))
    return X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]


def _train(cfg: TrainConfig, X_train, y_train, X_val, y_val) -> "torch.nn.Module":
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from core.classifier.trained.model import SideClassifierFFN

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    print(f"[train] device={device}", flush=True)

    model = SideClassifierFFN().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    # Class-balanced weights — NEITHER usually dominates.
    counts = torch.tensor(
        [(y_train == c).sum() for c in (0, 1, 2)],
        dtype=torch.float32, device=device,
    )
    weights = counts.sum() / (counts.clamp(min=1) * len(counts))
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    best_val_loss = float("inf")
    best_state = None
    for ep in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.train(False)
        val_loss = 0.0
        val_correct = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                val_loss += loss_fn(logits, yb).item() * len(xb)
                val_correct += (logits.argmax(-1) == yb).sum().item()
        val_loss /= len(val_ds)
        val_acc = val_correct / len(val_ds)

        better = " *" if val_loss < best_val_loss else ""
        print(f"[train] epoch={ep + 1:02d} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}{better}",
              flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main(argv: Optional[List[str]] = None) -> int:
    import numpy as np
    import torch

    cfg = parse_args(argv)
    np.random.seed(cfg.seed)

    bars = _load_bars(cfg.symbols + [cfg.benchmark], cfg.start_date, cfg.end_date)
    bench = bars.pop(cfg.benchmark, None)
    if not bars:
        print("[train] no bars loaded — exiting", flush=True)
        return 1

    print("[train] building examples", flush=True)
    X, y = _build_examples(bars, bench)
    print(f"[train] examples={len(X)} class_counts="
          f"L={int((y == 0).sum())} S={int((y == 1).sum())} N={int((y == 2).sum())}",
          flush=True)

    X_tr, y_tr, X_v, y_v = _time_split(X, y, cfg.val_fraction)
    print(f"[train] train={len(X_tr)} val={len(X_v)}", flush=True)

    model = _train(cfg, X_tr, y_tr, X_v, y_v)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.output_path)
    print(f"[train] saved checkpoint -> {cfg.output_path}", flush=True)
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
