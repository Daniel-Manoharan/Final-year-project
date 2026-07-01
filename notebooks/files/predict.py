"""
End-to-end prediction pipeline for unseen PE files.

Takes one or more executable paths, extracts the 114-feature vector, runs the
trained hybrid model, and emits a structured analysis report.

CLI:
    python -m src.predict --model models/hybrid.joblib file1.exe file2.dll
    python -m src.predict --model models/hybrid.joblib --dir samples/ --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import joblib
import numpy as np

from .feature_extractor import FeatureVector, extract_features


def _iter_paths(paths: List[Path], directory: Optional[Path]) -> Iterable[Path]:
    for p in paths:
        yield p
    if directory:
        for dirpath, _, filenames in os.walk(directory):
            for fname in filenames:
                yield Path(dirpath) / fname


def _classify_confidence(p: float) -> str:
    # Bucket for the report — mirrors typical enterprise triage tiers.
    if p >= 0.90 or p <= 0.10:
        return "high"
    if p >= 0.75 or p <= 0.25:
        return "medium"
    return "low"


def _top_signals(fv: FeatureVector) -> List[str]:
    """Human-readable indicators derived from the feature vector."""
    signals: List[str] = []
    f = fv.features
    if f.get("packer_signature", 0) >= 1:
        signals.append("packer section signature detected")
    if f.get("entropy_max", 0) >= 7.2:
        signals.append(f"very high section entropy ({f['entropy_max']:.2f})")
    if f.get("sliding_window_max", 0) >= 7.5:
        signals.append(f"high localised randomness ({f['sliding_window_max']:.2f})")
    if f.get("suspicious_import_count", 0) >= 3:
        signals.append(f"{int(f['suspicious_import_count'])} suspicious imports")
    if f.get("entry_point_in_text", 1) == 0:
        signals.append("entry point outside .text")
    if f.get("timestamp_valid", 1) == 0:
        signals.append("implausible PE timestamp")
    if f.get("text_vsize_ratio", 1) > 3.0:
        signals.append(f".text virtual/raw ratio {f['text_vsize_ratio']:.1f}")
    if not signals:
        signals.append("no strong static indicators")
    return signals


def predict_file(path: Path, model, feature_order: List[str]) -> dict:
    fv = extract_features(str(path))
    if fv is None:
        return {
            "path":   str(path),
            "error":  "unparseable or non-PE file",
            "sha256": None,
        }
    x = np.array([[fv.features.get(k, 0.0) for k in feature_order]], dtype=np.float64)
    proba = float(model.predict_proba(x)[0, 1])
    verdict = "malware" if proba >= 0.5 else "benign"
    return {
        "path":        str(path),
        "sha256":      fv.sha256,
        "verdict":     verdict,
        "probability": proba,
        "confidence":  _classify_confidence(proba),
        "signals":     _top_signals(fv),
    }


def predict_many(model_path: Path, files: List[Path], out_json: Optional[Path]) -> None:
    bundle = joblib.load(model_path)
    model, feature_order = bundle["model"], bundle["feature_order"]

    results = [predict_file(p, model, feature_order) for p in files]

    report = {
        "generated_at":  datetime.utcnow().isoformat() + "Z",
        "model":         str(model_path),
        "n_files":       len(results),
        "n_malware":     sum(r.get("verdict") == "malware" for r in results),
        "n_benign":      sum(r.get("verdict") == "benign"  for r in results),
        "n_errors":      sum("error" in r for r in results),
        "results":       results,
    }

    # Console summary — the pop-up-style analysis report from the interim plan.
    print(f"\nAnalysis report — {report['n_files']} file(s)")
    print(f"  malware: {report['n_malware']}   benign: {report['n_benign']}   "
          f"errors: {report['n_errors']}\n")
    for r in results:
        if "error" in r:
            print(f"  [!] {r['path']}: {r['error']}")
            continue
        marker = "MAL" if r["verdict"] == "malware" else "BEN"
        print(f"  [{marker}] {Path(r['path']).name}  "
              f"p={r['probability']:.3f}  ({r['confidence']} confidence)")
        for s in r["signals"]:
            print(f"        · {s}")

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\n[report] written to {out_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path, help="individual PE files")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--dir", type=Path, help="scan a directory recursively")
    ap.add_argument("--json", type=Path, help="also write the report as JSON")
    args = ap.parse_args()

    all_files = list(_iter_paths(args.files, args.dir))
    if not all_files:
        raise SystemExit("no files provided (pass paths and/or --dir)")

    predict_many(args.model, all_files, args.json)


if __name__ == "__main__":
    main()
