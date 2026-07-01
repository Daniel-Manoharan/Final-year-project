"""
Hybrid stacked model — the architecture specified in the proposal.

    Base learners  : Random Forest, SVM (RBF), Logistic Regression
    Meta-learner   : XGBoost
    Wrapper        : median imputation + robust scaling (opcode counts are
                     heavy-tailed) applied via a Pipeline so serialisation
                     round-trips cleanly.

The meta-learner receives *out-of-fold* base-learner probabilities
(StackingClassifier does this automatically), which is what makes stacked
generalisation reduce variance rather than merely averaging.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


@dataclass
class ModelConfig:
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 5           # inner CV used by StackingClassifier


def _base_learners(cfg: ModelConfig):
    """The three base learners named in the proposal + LR for calibration."""
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )
    svm = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        probability=True,          # required to feed probabilities to the meta-learner
        class_weight="balanced",
        random_state=cfg.random_state,
    )
    lr = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=cfg.random_state,
    )
    return [("rf", rf), ("svm", svm), ("lr", lr)]


def _meta_learner(cfg: ModelConfig) -> XGBClassifier:
    """XGBoost meta-learner — correlates the base probabilities with raw features."""
    return XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )


def build_model(cfg: ModelConfig = ModelConfig()) -> Pipeline:
    """Return the full sklearn Pipeline ready for `fit` / `predict_proba`."""
    stack = StackingClassifier(
        estimators=_base_learners(cfg),
        final_estimator=_meta_learner(cfg),
        cv=cfg.cv_folds,
        stack_method="predict_proba",
        passthrough=True,             # meta-learner also sees the raw 114-D vector
        n_jobs=cfg.n_jobs,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  RobustScaler()),
            ("stack",   stack),
        ]
    )
