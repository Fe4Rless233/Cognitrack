"""ML models, patient-grouped nested cross-validation, and Burnout Risk Gauge for CogniTrack."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.svm import SVC

from .constants import (
    ALERT_THRESHOLD_PERCENT,
    CRITICAL_FATIGUE,
    INTERVENTION_CRITICAL,
    INTERVENTION_MODERATE,
    INTERVENTION_NONE,
    LOW_LOAD_REST,
    MODERATE_WORKLOAD,
)
from .stats import compute_roc_youden_threshold


@dataclass
class NestedCVFoldResult:
    """Out-of-subject evaluation metrics for a single outer fold in nested CV."""
    fold_index: int
    test_patients: list[str]
    train_patients: list[str]
    best_params: dict
    accuracy: float
    balanced_accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float
    y_true: list[int]
    y_pred: list[int]
    y_prob: list[float]


@dataclass
class NestedCVReport:
    """Comprehensive summary of patient-grouped nested cross-validation."""
    n_outer_folds: int
    n_patients: int
    mean_accuracy: float
    std_accuracy: float
    mean_balanced_accuracy: float
    std_balanced_accuracy: float
    mean_f1: float
    std_f1: float
    mean_roc_auc: float
    std_roc_auc: float
    fold_results: list[NestedCVFoldResult]
    confusion_matrix: np.ndarray
    roc_analysis: dict = field(default_factory=dict)


@dataclass
class TrainedModels:
    """Holds fitted models, hyperparameter searches, and nested CV report."""
    svm_search: GridSearchCV
    logistic_model: LogisticRegression
    rf_search: GridSearchCV | None = None
    nested_cv_report: NestedCVReport | None = None


class CogniTrackTrainer:
    """Train CogniTrack classifiers using patient-grouped nested cross-validation.

    Guarantees:
    - Zero data leakage between participants (all data from one patient stays together in outer test fold).
    - Inner hyperparameter tuning folds also preserve patient grouping.
    - Full backward compatibility for single-subject pilot benchmarks.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    @staticmethod
    def _binary_fatigue_labels(labels: pd.Series | np.ndarray) -> np.ndarray:
        labels_arr = np.asarray(labels)
        return (labels_arr == CRITICAL_FATIGUE).astype(int)

    @staticmethod
    def _extract_model_features(features: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if isinstance(features, pd.DataFrame):
            canonical = ["z_theta_alpha", "z_theta_beta", "z_focus"]
            if all(c in features.columns for c in canonical):
                return features[canonical]
        return features

    def run_nested_group_cv(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        patients: pd.Series | np.ndarray,
        outer_splits: int = 5,
        inner_splits: int = 3,
        model_type: str = "svm",
    ) -> NestedCVReport:
        """Execute strict patient-grouped nested cross-validation.

        Outer Loop: GroupKFold over patients (all epochs from test patients isolated).
        Inner Loop: GroupKFold over training patients for hyperparameter grid search.
        """
        X_df = self._extract_model_features(features)
        X = X_df.values if isinstance(X_df, pd.DataFrame) else np.asarray(X_df)
        y = self._binary_fatigue_labels(labels)
        groups = np.asarray(patients)

        unique_patients = np.unique(groups)
        n_unique_patients = len(unique_patients)

        if n_unique_patients < 2:
            raise ValueError("Patient-grouped nested CV requires at least 2 distinct patients.")

        n_outer = min(outer_splits, n_unique_patients)
        outer_gkf = GroupKFold(n_splits=n_outer)

        if model_type == "rf":
            param_grid = {
                "n_estimators": [50, 100],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
                "class_weight": ["balanced"],
            }
            base_estimator = RandomForestClassifier(random_state=self.random_state)
        else:
            param_grid = {
                "C": [0.1, 1.0, 10.0, 100.0],
                "gamma": ["scale", "auto", 0.01, 0.1],
                "kernel": ["rbf", "linear"],
                "class_weight": ["balanced"],
            }
            base_estimator = SVC(probability=False, random_state=self.random_state)

        fold_results: list[NestedCVFoldResult] = []
        all_y_true: list[int] = []
        all_y_pred: list[int] = []
        all_y_prob: list[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_gkf.split(X, y, groups=groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            groups_train = groups[train_idx]
            test_patient_names = sorted(list(np.unique(groups[test_idx])))
            train_patient_names = sorted(list(np.unique(groups_train)))

            # Inner CV: patient-grouped hyperparameter search
            n_train_pts = len(train_patient_names)
            n_inner = min(inner_splits, n_train_pts)
            inner_cv = GroupKFold(n_splits=n_inner)

            search = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                scoring="f1",
                cv=inner_cv,
                n_jobs=-1,
            )
            search.fit(X_train, y_train, groups=groups_train)

            # Evaluate best estimator on completely unseen outer test patient(s)
            best_model = search.best_estimator_
            y_pred = best_model.predict(X_test)
            if hasattr(best_model, "decision_function"):
                y_prob = best_model.decision_function(X_test)
            elif hasattr(best_model, "predict_proba"):
                y_prob = best_model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred.astype(float)

            acc = float(accuracy_score(y_test, y_pred))
            bal_acc = float(balanced_accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))

            try:
                auc_score = float(roc_auc_score(y_test, y_prob))
            except Exception:
                auc_score = 0.5

            fold_results.append(
                NestedCVFoldResult(
                    fold_index=fold_idx + 1,
                    test_patients=test_patient_names,
                    train_patients=train_patient_names,
                    best_params=search.best_params_,
                    accuracy=acc,
                    balanced_accuracy=bal_acc,
                    f1=f1,
                    precision=prec,
                    recall=rec,
                    roc_auc=auc_score,
                    y_true=y_test.tolist(),
                    y_pred=y_pred.tolist(),
                    y_prob=y_prob.tolist(),
                )
            )

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
            all_y_prob.extend(y_prob.tolist())

        # Aggregate metrics across all outer folds
        accs = [f.accuracy for f in fold_results]
        bal_accs = [f.balanced_accuracy for f in fold_results]
        f1s = [f.f1 for f in fold_results]
        aucs = [f.roc_auc for f in fold_results]

        cm = confusion_matrix(all_y_true, all_y_pred)
        roc_info = compute_roc_youden_threshold(np.array(all_y_true), np.array(all_y_prob))

        return NestedCVReport(
            n_outer_folds=len(fold_results),
            n_patients=n_unique_patients,
            mean_accuracy=float(np.mean(accs)),
            std_accuracy=float(np.std(accs)),
            mean_balanced_accuracy=float(np.mean(bal_accs)),
            std_balanced_accuracy=float(np.std(bal_accs)),
            mean_f1=float(np.mean(f1s)),
            std_f1=float(np.std(f1s)),
            mean_roc_auc=float(np.mean(aucs)),
            std_roc_auc=float(np.std(aucs)),
            fold_results=fold_results,
            confusion_matrix=cm,
            roc_analysis=roc_info,
        )

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        patients: pd.Series | np.ndarray | None = None,
        cv_splits: int = 5,
        enable_nested_cv: bool = True,
    ) -> TrainedModels:
        """Train models with patient grouping when available."""
        y_binary = self._binary_fatigue_labels(labels)
        X_df = self._extract_model_features(features)
        nested_report: NestedCVReport | None = None

        has_groups = patients is not None and len(np.unique(patients)) >= 2

        if has_groups and enable_nested_cv:
            nested_report = self.run_nested_group_cv(
                features=X_df,
                labels=labels,
                patients=patients,
                outer_splits=min(cv_splits, len(np.unique(patients))),
            )

        # Fit final hyperparameter-tuned SVM
        svm_param_grid = {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "kernel": ["rbf", "linear"],
            "class_weight": ["balanced"],
        }

        if has_groups:
            n_splits = min(cv_splits, len(np.unique(patients)))
            cv_strat = GroupKFold(n_splits=n_splits)
            svm_search = GridSearchCV(
                estimator=SVC(probability=True, random_state=self.random_state),
                param_grid=svm_param_grid,
                scoring="f1",
                cv=cv_strat,
                n_jobs=-1,
            )
            if len(X_df) > 2500:
                sample_idx = np.random.default_rng(self.random_state).choice(len(X_df), size=2000, replace=False)
                sub_X = X_df.iloc[sample_idx] if isinstance(X_df, pd.DataFrame) else X_df[sample_idx]
                svm_search.fit(sub_X, y_binary[sample_idx], groups=np.asarray(patients)[sample_idx])
            else:
                svm_search.fit(X_df, y_binary, groups=np.asarray(patients))
        else:
            folds = min(cv_splits, int(np.bincount(y_binary).min()))
            if folds < 2:
                folds = 2
            cv_strat = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
            svm_search = GridSearchCV(
                estimator=SVC(random_state=self.random_state),
                param_grid=svm_param_grid,
                scoring="f1",
                cv=cv_strat,
                n_jobs=-1,
            )
            svm_search.fit(X_df, y_binary)

        # Fit final Logistic Regression for probabilistic Risk Gauge
        logistic_model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
        )
        logistic_model.fit(X_df, y_binary)

        # Fit Random Forest Classifier
        rf_param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10],
            "class_weight": ["balanced"],
        }
        if has_groups:
            rf_search = GridSearchCV(
                estimator=RandomForestClassifier(random_state=self.random_state),
                param_grid=rf_param_grid,
                scoring="f1",
                cv=GroupKFold(n_splits=min(cv_splits, len(np.unique(patients)))),
                n_jobs=-1,
            )
            rf_search.fit(X_df, y_binary, groups=np.asarray(patients))
        else:
            rf_search = None

        return TrainedModels(
            svm_search=svm_search,
            logistic_model=logistic_model,
            rf_search=rf_search,
            nested_cv_report=nested_report,
        )


class BurnoutRiskGauge:
    """Burnout Risk Gauge converting model outputs to 0-100% and prescribing closed-loop actions."""

    @staticmethod
    def risk_percent(model: LogisticRegression, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute 0-100% cognitive fatigue risk probabilities."""
        feat_matrix = features
        if hasattr(model, "feature_names_in_") and isinstance(features, pd.DataFrame):
            missing = [c for c in model.feature_names_in_ if c not in features.columns]
            if not missing:
                feat_matrix = features[model.feature_names_in_]
        elif isinstance(features, pd.DataFrame):
            canonical = ["z_theta_alpha", "z_theta_beta", "z_focus"]
            if all(c in features.columns for c in canonical):
                feat_matrix = features[canonical]

        probs = model.predict_proba(feat_matrix)
        positive_class_index = int(np.where(model.classes_ == 1)[0][0])
        return probs[:, positive_class_index] * 100.0

    @staticmethod
    def is_critical_alert(risk_percent: float, threshold: float = ALERT_THRESHOLD_PERCENT) -> bool:
        """Determine if fatigue risk has breached the validated 81% critical threshold."""
        return risk_percent >= threshold

    @staticmethod
    def classify_operational_tier(risk_percent: float) -> int:
        """Classify into three operational tiers: 0=Low Load, 1=Moderate, 2=Critical."""
        if risk_percent >= ALERT_THRESHOLD_PERCENT:
            return CRITICAL_FATIGUE
        elif risk_percent >= 50.0:
            return MODERATE_WORKLOAD
        else:
            return LOW_LOAD_REST

    @staticmethod
    def prescribe_intervention(risk_percent: float) -> str:
        """Prescribe immediate, data-driven closed-loop recovery interventions."""
        if risk_percent >= ALERT_THRESHOLD_PERCENT:
            return INTERVENTION_CRITICAL
        elif risk_percent >= 50.0:
            return INTERVENTION_MODERATE
        else:
            return INTERVENTION_NONE


def evaluate_f1(model: LogisticRegression, features: pd.DataFrame | np.ndarray, labels: pd.Series) -> float:
    """Evaluate binary fatigue F1 score."""
    y_true = (labels == CRITICAL_FATIGUE).astype(int)
    feat_matrix = features
    if hasattr(model, "feature_names_in_") and isinstance(features, pd.DataFrame):
        missing = [c for c in model.feature_names_in_ if c not in features.columns]
        if not missing:
            feat_matrix = features[model.feature_names_in_]
    elif isinstance(features, pd.DataFrame):
        canonical = ["z_theta_alpha", "z_theta_beta", "z_focus"]
        if all(c in features.columns for c in canonical):
            feat_matrix = features[canonical]
    y_pred = model.predict(feat_matrix)
    return float(f1_score(y_true, y_pred, zero_division=0))

