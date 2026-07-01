"""
Evaluation utilities — matches the testing plan in the interim report.

    E1  cross-validation benchmark on the hybrid model
    E2  feature-fusion ablation: opcode-only vs entropy-only vs hybrid
    E3  held-out generalisation test
    E4  per-file inference latency

Usage:
    python -m src.evaluate --csv combined_features.csv --model models/hybrid.joblib \
                           --out reports/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .feature_extractor import (
    BIGRAMS_TRACKED, OPCODES_TRACKED, get_feature_order,
)
from .model import ModelConfig, build_model


ENTROPY_FEATURES = [
    "entropy_mean", "entropy_max", "entropy_min", "entropy_std",
    "entropy_text", "entropy_data", "entropy_rdata", "entropy_rsrc",
    "sliding_window_max", "sliding_window_mean", "sliding_window_std",
    "entry_point_entropy",
]
OPCODE_FEATURES = (
    [f"op_{o}" for o in OPCODES_TRACKED]
    + ["mov_ratio", "call_ratio", "jmp_ratio", "xor_ratio", "push_ratio",
       "pop_ratio", "ret_ratio", "cmp_ratio", "arithmetic_ratio", "control_flow_ratio",
       "opcode_diversity", "total_opcodes", "unique_opcodes"]
    + [f"bg_{a}_{b}" for (a, b) in BIGRAMS_TRACKED]
)


def _metrics(y_true, y_pred, y_proba) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_proba)),
        "fpr":       float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def ablation(csv_path: Path, out_dir: Path, cfg: ModelConfig) -> None:
    """E2 — compare opcode-only, entropy-only, and hybrid feature sets."""
    df = pd.read_csv(csv_path)
    y = df["label"].astype(int).to_numpy()

    variants = {
        "entropy_only": ENTROPY_FEATURES,
        "opcode_only":  OPCODE_FEATURES,
        "hybrid":       get_feature_order(),
    }

    results: List[dict] = []
    for name, cols in variants.items():
        X = df[cols].to_numpy(dtype=np.float64)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=cfg.random_state
        )
        model = build_model(cfg)
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        m = _metrics(yte, pred, proba)
        m["variant"]      = name
        m["n_features"]   = len(cols)
        results.append(m)
        print(f"[ablation] {name:14s} "
              f"f1={m['f1']:.4f}  auc={m['roc_auc']:.4f}  fpr={m['fpr']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "ablation.csv", index=False)
    print(f"[ablation] saved → {out_dir / 'ablation.csv'}")


def plot_diagnostics(bundle_path: Path, csv_path: Path, out_dir: Path,
                     cfg: ModelConfig) -> None:
    bundle = joblib.load(bundle_path)
    model, feature_order = bundle["model"], bundle["feature_order"]

    df = pd.read_csv(csv_path)
    X = df[feature_order].to_numpy(dtype=np.float64)
    y = df["label"].astype(int).to_numpy()

    _, Xte, _, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=cfg.random_state
    )
    proba = model.predict_proba(Xte)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(yte, pred),
                           display_labels=["benign", "malware"]).plot(ax=ax, cmap="Blues")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(yte, proba, ax=ax, name="hybrid stacked")
    fig.tight_layout(); fig.savefig(out_dir / "roc_curve.png", dpi=150); plt.close(fig)

    print(f"[plots] saved to {out_dir}")


def latency(bundle_path: Path, csv_path: Path, n: int = 500) -> None:
    """E4 — per-sample inference latency."""
    bundle = joblib.load(bundle_path)
    model, feature_order = bundle["model"], bundle["feature_order"]

    df = pd.read_csv(csv_path).sample(n=min(n, len(pd.read_csv(csv_path))),
                                      random_state=0)
    X = df[feature_order].to_numpy(dtype=np.float64)

    t0 = time.perf_counter()
    _ = model.predict_proba(X)
    dt = time.perf_counter() - t0
    print(f"[latency] {n} samples: total {dt*1000:.1f} ms  "
          f"per-sample {dt/n*1000:.3f} ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",   required=True, type=Path)
    ap.add_argument("--model", type=Path, help="trained bundle (for diagnostics/latency)")
    ap.add_argument("--out",   type=Path, default=Path("reports"))
    ap.add_argument("--do", choices=["ablation", "plots", "latency", "all"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = ModelConfig(random_state=args.seed)

    if args.do in ("ablation", "all"):
        ablation(args.csv, args.out, cfg)
    if args.do in ("plots", "all"):
        if not args.model:
            raise SystemExit("--model is required for plots")
        plot_diagnostics(args.model, args.csv, args.out, cfg)
    if args.do in ("latency", "all"):
        if not args.model:
            raise SystemExit("--model is required for latency")
        latency(args.model, args.csv)


if __name__ == "__main__":
    main()
