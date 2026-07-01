# Machine Learning-Based Metamorphic Malware Detection

Static, dual-channel detector combining **opcode analysis** and **Shannon entropy** into a
114-feature vector, classified by a **stacked ensemble** (Random Forest + SVM +
Logistic Regression) with an **XGBoost meta-learner**.

Built to the specification in `Interim_Report.docx` — same feature families,
same feature count (114), same stacked architecture.

## Project layout

```
metamorphic_detector/
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── feature_extractor.py   # PE → 114-feature vector (static only)
    ├── dataset.py             # walk benign/malware trees → combined_features.csv
    ├── model.py               # stacked pipeline (RF+SVM+LR → XGBoost)
    ├── train.py               # k-fold CV + held-out eval + save
    ├── evaluate.py            # ablation, ROC/CM plots, latency
    └── predict.py             # end-to-end inference + analysis report
```

## The 114 features (frozen order)

| Family                | Count | Examples                                                            |
|-----------------------|-------|---------------------------------------------------------------------|
| Entropy               | 12    | `entropy_mean/max/min/std`, `entropy_text`, `sliding_window_max`, `entry_point_entropy` |
| Opcode counts         | 50    | `op_mov`, `op_call`, `op_jmp`, `op_xor`, …                          |
| Opcode ratios & aggs  | 13    | `mov_ratio`, `control_flow_ratio`, `opcode_diversity`, …            |
| Opcode bigrams        | 20    | `bg_push_call`, `bg_mov_ret`, `bg_cmp_jne`, …                       |
| PE structural         | 19    | `section_count`, `imports_count`, `packer_signature`, `has_tls`, …  |

`get_feature_order()` in `feature_extractor.py` is the single source of truth.

## Install

```bash
python -m venv .venv
source .venv/bin/activate           # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

> **Run everything inside an isolated analysis VM** (VirtualBox / VMware,
> host-only network). The extractor never executes samples, but a malformed
> download or sample handling mistake shouldn't be able to reach your host.

## 1 — Build the dataset (Objectives 1, 2, 3)

Point the builder at two folders — benign PEs and malware PEs:

```bash
python -m src.dataset \
    --benign  /path/to/benign_pe/ \
    --malware /path/to/malware_pe/ \
    --out     combined_features.csv \
    --limit-per-class 4000
```

Output is exactly the schema you documented in the interim: `sha256`, then the
114 feature columns in `get_feature_order()`, then `label` (1 = malware,
0 = benign). Duplicates by SHA-256 are dropped automatically.

## 2 — Train the hybrid stacked model (Objective 5)

```bash
python -m src.train \
    --csv combined_features.csv \
    --out models/hybrid.joblib \
    --seed 42
```

This runs stratified 5-fold CV on the training half, refits on all training
data, and evaluates on a stratified 20% held-out test set. It writes:

- `models/hybrid.joblib` — the fitted `Pipeline` + frozen feature order.
- `models/hybrid.json`   — held-out metrics (accuracy, precision, recall, F1,
  ROC-AUC, FPR, confusion matrix).

## 3 — Full evaluation (matches the interim's E1–E4 plan)

```bash
python -m src.evaluate \
    --csv   combined_features.csv \
    --model models/hybrid.joblib \
    --out   reports/ \
    --do    all
```

Produces:

- `reports/ablation.csv`         — E2 ablation (opcode-only vs entropy-only vs hybrid)
- `reports/confusion_matrix.png` — held-out confusion matrix
- `reports/roc_curve.png`        — ROC curve
- console line for E4 per-sample latency

## 4 — Predict on unseen executables

```bash
# Single file
python -m src.predict --model models/hybrid.joblib suspect.exe

# Recursive scan + JSON report
python -m src.predict --model models/hybrid.joblib --dir /samples/ --json out/report.json
```

Console output is the pop-up-style analysis report described in the interim:
verdict, probability, confidence bucket, and the top static indicators
(packer sections, entropy spikes, suspicious imports, TLS/entry-point
anomalies, timestamp sanity).

## Mapping to proposal objectives

| Proposal / interim item                | Where it lives                                      |
|----------------------------------------|-----------------------------------------------------|
| Objective 1 — labelled dataset         | `dataset.py` (walks benign/ and malware/ trees)     |
| Objective 2 — accurate labelling       | `dataset.py` (folder-derived label + SHA-256 dedup) |
| Objective 3 — static disassembly       | `feature_extractor._iter_opcodes` (Capstone)        |
| Objective 3 — Shannon entropy          | `feature_extractor._entropy_features`               |
| Objective 4 — opcode extraction        | `feature_extractor._opcode_features`                |
| Objective 5 — hybrid ML model          | `model.build_model` (stacked → XGBoost)             |
| Novelty — dual-channel static fusion   | `model.py` `passthrough=True`                       |
| Novelty — stacked XGBoost meta-learner | `model._meta_learner`                               |
| Testing plan E1                        | `train.py` cross-validation                         |
| Testing plan E2                        | `evaluate.ablation`                                 |
| Testing plan E3                        | held-out test in `train.train_and_save`             |
| Testing plan E4                        | `evaluate.latency`                                  |
| Analysis report (pop-up)               | `predict.predict_many` console + `--json`           |

## Notes

- All extraction is **static** — no sample is ever executed.
- Extraction is bounded (`max_bytes=2_000_000` of disassembly per file) so
  pathological packers can't stall a run.
- Every per-file failure (bad PE header, malformed section table, unusual
  architecture) returns `None` rather than raising, so a batch run over 8,000
  files won't abort on one bad sample.
- The `feature_order` returned by `get_feature_order()` is **frozen** once you
  train — do not reorder or you will invalidate any saved model.
