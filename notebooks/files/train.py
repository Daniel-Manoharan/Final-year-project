"""
Train the hybrid stacked model on combined_features.csv.

Workflow:
    1. Load features + label.
    2. Stratified 80/20 split into train / held-out test.
    3. StratifiedKFold cross-validation on the training half — reports
       accuracy, precision, recall, F1, ROC-AUC per fold.
    4. Refit on the full training half, evaluate on the held-out test.
    5. Persist the fitted Pipeline plus the feature-order manifest so
       `predict.py` can reload without re-declaring anything.

CLI:
    python -m src.train --csv combined_features.csv --out models/hybrid.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .feature_extractor import get_feature_order
from .model import ModelConfig, build_model


def load_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)
    feature_order = get_feature_order()

    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing {len(missing)} expected feature columns: {missing[:5]}...")

    X = df[feature_order].to_numpy(dtype=np.float64)
    y = df["label"].astype(int).to_numpy()
    return X, y, feature_order


def cross_validate(X, y, cfg: ModelConfig, n_splits: int = 5) -> None:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        model = build_model(cfg)
        model.fit(Xtr, ytr)

        proba = model.predict_proba(Xva)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        m = {
            "fold":      fold,
            "accuracy":  accuracy_score(yva, pred),
            "precision": precision_score(yva, pred, zero_division=0),
            "recall":    recall_score(yva, pred, zero_division=0),
            "f1":        f1_score(yva, pred, zero_division=0),
            "roc_auc":   roc_auc_score(yva, proba),
        }
        fold_metrics.append(m)
        print(f"[cv] fold {fold}: "
              f"acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
              f"rec={m['recall']:.4f}  f1={m['f1']:.4f}  auc={m['roc_auc']:.4f}")

    means = {k: float(np.mean([m[k] for m in fold_metrics]))
             for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    stds  = {k: float(np.std ([m[k] for m in fold_metrics]))
             for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    print("\n[cv] cross-validation summary (mean ± std)")
    for k in means:
        print(f"       {k:10s}: {means[k]:.4f} ± {stds[k]:.4f}")


def train_and_save(
    csv_path: Path,
    out_path: Path,
    cfg: ModelConfig,
    test_size: float = 0.2,
    do_cv: bool = True,
) -> None:
    X, y, feature_order = load_dataset(csv_path)
    print(f"[data] samples={len(y)}  malware={int(y.sum())}  benign={int((y == 0).sum())}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=cfg.random_state
    )

    if do_cv:
        print("\n[cv] running stratified k-fold on the training half...")
        cross_validate(Xtr, ytr, cfg)

    print("\n[fit] refitting on the full training half...")
    model = build_model(cfg)
    model.fit(Xtr, ytr)

    print("\n[eval] held-out test set")
    proba = model.predict_proba(Xte)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    cm = confusion_matrix(yte, pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print(classification_report(yte, pred, target_names=["benign", "malware"], digits=4))
    print(f"confusion matrix:\n{cm}")
    print(f"false-positive rate: {fpr:.4f}")
    print(f"roc-auc:             {roc_auc_score(yte, proba):.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_order": feature_order}, out_path)
    manifest = {
        "feature_order": feature_order,
        "held_out_test": {
            "n": int(len(yte)),
            "accuracy":  float(accuracy_score(yte, pred)),
            "precision": float(precision_score(yte, pred)),
            "recall":    float(recall_score(yte, pred)),
            "f1":        float(f1_score(yte, pred)),
            "roc_auc":   float(roc_auc_score(yte, proba)),
            "fpr":       float(fpr),
            "confusion_matrix": cm.tolist(),
        },
    }
    with open(out_path.with_suffix(".json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n[save] model  → {out_path}")
    print(f"[save] manifest → {out_path.with_suffix('.json')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the hybrid metamorphic malware detector")
    ap.add_argument("--csv", required=True, type=Path, help="combined_features.csv")
    ap.add_argument("--out", required=True, type=Path, help="output .joblib path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--no-cv", action="store_true", help="skip k-fold CV")
    args = ap.parse_args()

    train_and_save(
        csv_path=args.csv,
        out_path=args.out,
        cfg=ModelConfig(random_state=args.seed),
        test_size=args.test_size,
        do_cv=not args.no_cv,
    )


if __name__ == "__main__":
    main()
