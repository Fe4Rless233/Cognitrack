"""ML models and hyperparameter tuning for CogniTrack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from .constants import CRITICAL_FATIGUE


@dataclass
class TrainedModels:
    svm_search: GridSearchCV
    logistic_model: LogisticRegression


class CogniTrackTrainer:
    """Train CogniTrack SVM and Logistic Regression classifiers."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    @staticmethod
    def _binary_fatigue_labels(labels: pd.Series | np.ndarray) -> np.ndarray:
        labels_arr = np.asarray(labels)
        return (labels_arr == CRITICAL_FATIGUE).astype(int)

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        cv_splits: int = 5,
    ) -> TrainedModels:
        y_binary = self._binary_fatigue_labels(labels)

        param_grid = {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "kernel": ["rbf", "linear"],
            "class_weight": ["balanced"],
        }

        folds = min(cv_splits, int(np.bincount(y_binary).min()))
        if folds < 2:
            raise ValueError("Need at least 2 samples in each class for cross-validation.")

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)

        svm = SVC(random_state=self.random_state)
        svm_search = GridSearchCV(
            estimator=svm,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
        )
        svm_search.fit(features, y_binary)

        logistic_model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
        )
        logistic_model.fit(features, y_binary)

        return TrainedModels(svm_search=svm_search, logistic_model=logistic_model)


class BurnoutRiskGauge:
    """Risk gauge converting model probabilities to 0-100 percentage score."""

    @staticmethod
    def risk_percent(model: LogisticRegression, features: pd.DataFrame) -> np.ndarray:
        probs = model.predict_proba(features)
        positive_class_index = int(np.where(model.classes_ == 1)[0][0])
        return probs[:, positive_class_index] * 100.0

    @staticmethod
    def is_critical_alert(risk_percent: float, threshold: float = 81.0) -> bool:
        return risk_percent >= threshold



def evaluate_f1(model: LogisticRegression, features: pd.DataFrame, labels: pd.Series) -> float:
    """Evaluate binary fatigue F1 score for convenience in experiments."""
    y_true = (labels == CRITICAL_FATIGUE).astype(int)
    y_pred = model.predict(features)
    return float(f1_score(y_true, y_pred))
