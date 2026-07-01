"""
Dataset builder — walks benign/ and malware/ directories, extracts the
114-feature vector from every parseable PE, and writes combined_features.csv.

Run inside the isolated analysis VM. Never execute samples.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, List, Optional

from tqdm import tqdm

from .feature_extractor import FeatureVector, extract_features, get_feature_order


def _iter_pe_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            p = Path(dirpath) / fname
            # Cheap filter — the extractor still validates the PE header.
            if p.suffix.lower() in {".exe", ".dll", ".sys", ".ocx", ".scr", ""}:
                yield p


def build_dataset(
    benign_dir: Path,
    malware_dir: Path,
    out_csv: Path,
    limit_per_class: Optional[int] = None,
) -> None:
    feature_order = get_feature_order()
    header = ["sha256", *feature_order, "label"]

    seen_hashes: set[str] = set()
    rows_written = 0

    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for label, folder in ((0, benign_dir), (1, malware_dir)):
            files = list(_iter_pe_files(folder))
            if limit_per_class:
                files = files[:limit_per_class]
            desc = f"{'benign' if label == 0 else 'malware'} ({len(files)} files)"
            for path in tqdm(files, desc=desc):
                fv: Optional[FeatureVector] = extract_features(str(path), label=label)
                if fv is None:
                    continue
                if fv.sha256 in seen_hashes:        # dedupe by content hash
                    continue
                seen_hashes.add(fv.sha256)
                writer.writerow(fv.to_row(feature_order))
                rows_written += 1

    print(f"[dataset] wrote {rows_written} rows to {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build combined_features.csv")
    ap.add_argument("--benign",   required=True, type=Path)
    ap.add_argument("--malware",  required=True, type=Path)
    ap.add_argument("--out",      required=True, type=Path)
    ap.add_argument("--limit-per-class", type=int, default=None)
    args = ap.parse_args()

    build_dataset(args.benign, args.malware, args.out, args.limit_per_class)


if __name__ == "__main__":
    main()
