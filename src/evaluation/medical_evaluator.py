"""Medical evaluation for binary NORMAL vs PNEUMONIA classifiers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThresholdResult:
    """Sensitivity/specificity at one pneumonia decision threshold."""

    threshold: float
    sensitivity: float
    specificity: float
    tp: int
    tn: int
    fp: int
    fn: int


class MedicalEvaluator:
    """Generate comprehensive, clinically-oriented classification metrics."""

    def __init__(
        self,
        model: Any,
        test_images: Any,
        test_labels: Iterable[int],
        class_names: tuple[str, str] = ("NORMAL", "PNEUMONIA"),
        prediction_batch_size: int | None = None,
    ) -> None:
        if len(class_names) != 2:
            raise ValueError("Day 6 evaluator expects exactly two classes.")

        self.model = model
        self.test_images = test_images
        self.test_labels = np.asarray(list(test_labels), dtype=int)
        self.class_names = tuple(class_names)
        self.prediction_batch_size = prediction_batch_size

        if self.test_labels.ndim != 1:
            raise ValueError("test_labels must be a one-dimensional array.")
        if np.any(~np.isin(self.test_labels, [0, 1])):
            raise ValueError("test_labels must contain only 0 and 1.")

    def predict_probabilities(self) -> np.ndarray:
        """Return model probabilities with shape (n_samples, 2)."""
        kwargs = {}
        if self.prediction_batch_size is not None:
            kwargs["batch_size"] = self.prediction_batch_size

        try:
            predictions = self.model.predict(self.test_images, verbose=0, **kwargs)
        except TypeError:
            # Lightweight test doubles may not accept verbose/batch_size.
            predictions = self.model.predict(self.test_images)

        probabilities = np.asarray(predictions, dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError(
                "Model predictions must have shape (n_samples, 2)."
            )
        if probabilities.shape[0] != len(self.test_labels):
            raise ValueError("Prediction count does not match test_labels.")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Model predictions contain non-finite values.")
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("Model probabilities must be in [0, 1].")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Each prediction row must sum to 1.0.")

        return probabilities

    def evaluate_comprehensive(self) -> dict[str, Any]:
        """Generate per-class, aggregate, discrimination, and calibration metrics."""
        probabilities = self.predict_probabilities()
        pred_labels = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)
        pneumonia_probability = probabilities[:, 1]

        report: dict[str, Any] = {}
        for class_idx, class_name in enumerate(self.class_names):
            report[class_name] = self._evaluate_class(
                class_idx, pred_labels, self.test_labels
            )

        report["aggregate"] = self._aggregate_metrics(pred_labels)
        report["confusion_matrix"] = self._confusion_matrix(pred_labels).tolist()
        report["roc_auc"] = self._roc_auc(pneumonia_probability, self.test_labels)
        report["pr_auc"] = self._pr_auc(pneumonia_probability, self.test_labels)
        report["calibration"] = self._check_calibration(
            confidence=confidence,
            correct=(pred_labels == self.test_labels),
        )
        report["pneumonia_calibration"] = self._check_binary_calibration(
            pneumonia_probability,
            self.test_labels,
        )

        return report

    @staticmethod
    def _evaluate_class(
        class_idx: int,
        pred_labels: np.ndarray,
        true_labels: np.ndarray,
    ) -> dict[str, float | int]:
        """Calculate sensitivity, specificity, precision, and F1 for one class."""
        tp = int(np.sum((pred_labels == class_idx) & (true_labels == class_idx)))
        tn = int(np.sum((pred_labels != class_idx) & (true_labels != class_idx)))
        fp = int(np.sum((pred_labels == class_idx) & (true_labels != class_idx)))
        fn = int(np.sum((pred_labels != class_idx) & (true_labels == class_idx)))

        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity)
            else 0.0
        )

        return {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "f1": float(f1),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    def _aggregate_metrics(self, pred_labels: np.ndarray) -> dict[str, float]:
        """Return accuracy plus macro and weighted F1 context."""
        accuracy = float(np.mean(pred_labels == self.test_labels))
        per_class = [
            self._evaluate_class(index, pred_labels, self.test_labels)
            for index in range(2)
        ]
        supports = [
            int(np.sum(self.test_labels == index))
            for index in range(2)
        ]
        macro_f1 = float(np.mean([metric["f1"] for metric in per_class]))
        total = sum(supports)
        weighted_f1 = (
            float(sum(metric["f1"] * support for metric, support in zip(per_class, supports)) / total)
            if total
            else 0.0
        )
        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

    @staticmethod
    def _confusion_matrix(pred_labels: np.ndarray) -> np.ndarray:
        matrix = np.zeros((2, 2), dtype=int)
        for true_label, pred_label in zip(pred_labels * 0 + 0, pred_labels):
            del true_label, pred_label
        return matrix

    def _confusion_matrix(self, pred_labels: np.ndarray) -> np.ndarray:
        matrix = np.zeros((2, 2), dtype=int)
        for true_label, pred_label in zip(self.test_labels, pred_labels):
            matrix[true_label, pred_label] += 1
        return matrix

    @staticmethod
    def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute binary ROC-AUC using the rank-sum formulation."""
        positives = scores[labels == 1]
        negatives = scores[labels == 0]
        if len(positives) == 0 or len(negatives) == 0:
            return 0.0

        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        ranks = np.empty(len(scores), dtype=float)

        start = 0
        while start < len(sorted_scores):
            end = start + 1
            while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            average_rank = (start + 1 + end) / 2.0
            ranks[order[start:end]] = average_rank
            start = end

        positive_rank_sum = ranks[labels == 1].sum()
        n_pos = len(positives)
        n_neg = len(negatives)
        auc = (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return float(auc)

    @staticmethod
    def _pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute average precision using precision-at-rank weighting."""
        positives = int(np.sum(labels == 1))
        if positives == 0:
            return 0.0

        order = np.argsort(-scores, kind="mergesort")
        sorted_labels = labels[order]
        cumulative_tp = np.cumsum(sorted_labels == 1)
        ranks = np.arange(1, len(labels) + 1)
        precision_at_rank = cumulative_tp / ranks
        return float(np.sum(precision_at_rank * (sorted_labels == 1)) / positives)

    @staticmethod
    def _check_calibration(
        confidence: np.ndarray,
        correct: np.ndarray,
        bins: int = 10,
    ) -> dict[str, dict[str, float | int]]:
        """Check whether maximum predicted confidence matches empirical accuracy."""
        edges = np.linspace(0.0, 1.0, bins + 1)
        calibration: dict[str, dict[str, float | int]] = {}

        for index in range(bins):
            lower, upper = edges[index], edges[index + 1]
            if index == bins - 1:
                mask = (confidence >= lower) & (confidence <= upper)
            else:
                mask = (confidence >= lower) & (confidence < upper)

            if np.any(mask):
                interval = f"{lower:.1f}-{upper:.1f}"
                calibration[interval] = {
                    "predicted_confidence": float(np.mean(confidence[mask])),
                    "actual_accuracy": float(np.mean(correct[mask])),
                    "sample_count": int(np.sum(mask)),
                }

        return calibration

    @staticmethod
    def _check_binary_calibration(
        probabilities: np.ndarray,
        labels: np.ndarray,
        bins: int = 10,
    ) -> dict[str, dict[str, float | int]]:
        """Check calibration of the PNEUMONIA probability specifically."""
        edges = np.linspace(0.0, 1.0, bins + 1)
        calibration: dict[str, dict[str, float | int]] = {}

        for index in range(bins):
            lower, upper = edges[index], edges[index + 1]
            if index == bins - 1:
                mask = (probabilities >= lower) & (probabilities <= upper)
            else:
                mask = (probabilities >= lower) & (probabilities < upper)

            if np.any(mask):
                interval = f"{lower:.1f}-{upper:.1f}"
                calibration[interval] = {
                    "predicted_probability": float(np.mean(probabilities[mask])),
                    "observed_pneumonia_rate": float(np.mean(labels[mask] == 1)),
                    "sample_count": int(np.sum(mask)),
                }

        return calibration

    def analyze_thresholds(
        self,
        thresholds: Iterable[float] | None = None,
        min_specificity: float = 0.80,
    ) -> dict[str, Any]:
        """Evaluate sensitivity/specificity across pneumonia thresholds."""
        if not 0.0 <= min_specificity <= 1.0:
            raise ValueError("min_specificity must be in [0, 1].")

        probabilities = self.predict_probabilities()[:, 1]
        threshold_values = (
            np.linspace(0.0, 1.0, 101)
            if thresholds is None
            else np.asarray(list(thresholds), dtype=float)
        )
        if threshold_values.ndim != 1 or np.any((threshold_values < 0) | (threshold_values > 1)):
            raise ValueError("thresholds must contain values in [0, 1].")

        results: list[ThresholdResult] = []
        for threshold in threshold_values:
            pred_binary = (probabilities >= threshold).astype(int)
            result = self._binary_counts(pred_binary, self.test_labels, float(threshold))
            results.append(result)

        valid = [result for result in results if result.specificity >= min_specificity]
        selected = max(valid, key=lambda result: (result.sensitivity, -result.threshold)) if valid else None

        return {
            "min_specificity": float(min_specificity),
            "selected_threshold": selected.threshold if selected else None,
            "selected_sensitivity": selected.sensitivity if selected else None,
            "selected_specificity": selected.specificity if selected else None,
            "points": [result.__dict__ for result in results],
        }

    @staticmethod
    def _binary_counts(
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float,
    ) -> ThresholdResult:
        tp = int(np.sum((predictions == 1) & (labels == 1)))
        tn = int(np.sum((predictions == 0) & (labels == 0)))
        fp = int(np.sum((predictions == 1) & (labels == 0)))
        fn = int(np.sum((predictions == 0) & (labels == 1)))
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        return ThresholdResult(
            threshold=threshold,
            sensitivity=float(sensitivity),
            specificity=float(specificity),
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
        )

    def generate_report(self, filepath: str | Path) -> Path:
        """Write a human-readable medical evaluation report and JSON companion."""
        report_data = self.evaluate_comprehensive()
        threshold_data = self.analyze_thresholds()
        report_data["threshold_analysis"] = threshold_data

        pneumonia = report_data["PNEUMONIA"]
        normal = report_data["NORMAL"]
        matrix = np.asarray(report_data["confusion_matrix"])

        lines = [
            "MEDICAL AI EVALUATION REPORT",
            "=" * 34,
            "",
            "1. PNEUMONIA DETECTION",
            "-" * 24,
            f"Sensitivity (Recall): {pneumonia['sensitivity']:.4f}",
            f"Specificity: {pneumonia['specificity']:.4f}",
            f"Precision: {pneumonia['precision']:.4f}",
            f"F1-Score: {pneumonia['f1']:.4f}",
            f"TP: {pneumonia['tp']}",
            f"FN: {pneumonia['fn']}",
            f"FP: {pneumonia['fp']}",
            f"TN: {pneumonia['tn']}",
            "",
            "2. NORMAL DETECTION",
            "-" * 19,
            f"Sensitivity: {normal['sensitivity']:.4f}",
            f"Specificity: {normal['specificity']:.4f}",
            f"Precision: {normal['precision']:.4f}",
            f"F1-Score: {normal['f1']:.4f}",
            "",
            "3. OVERALL METRICS",
            "-" * 19,
            f"Accuracy: {report_data['aggregate']['accuracy']:.4f}",
            f"Macro F1: {report_data['aggregate']['macro_f1']:.4f}",
            f"Weighted F1: {report_data['aggregate']['weighted_f1']:.4f}",
            f"ROC-AUC: {report_data['roc_auc']:.4f}",
            f"PR-AUC: {report_data['pr_auc']:.4f}",
            "",
            "4. CONFUSION MATRIX",
            "-" * 20,
            f"                Pred NORMAL  Pred PNEUMONIA",
            f"Actual NORMAL       {matrix[0,0]:4d}           {matrix[0,1]:4d}",
            f"Actual PNEUMONIA    {matrix[1,0]:4d}           {matrix[1,1]:4d}",
            "",
            "5. PROBABILITY CALIBRATION",
            "-" * 28,
            "Maximum-confidence calibration:",
        ]

        for interval, data in report_data["calibration"].items():
            lines.append(
                f"  {interval}: confidence={data['predicted_confidence']:.3f}, "
                f"accuracy={data['actual_accuracy']:.3f}, n={data['sample_count']}"
            )

        lines.extend(["", "PNEUMONIA-probability calibration:"])
        for interval, data in report_data["pneumonia_calibration"].items():
            lines.append(
                f"  {interval}: probability={data['predicted_probability']:.3f}, "
                f"observed_rate={data['observed_pneumonia_rate']:.3f}, n={data['sample_count']}"
            )

        lines.extend(
            [
                "",
                "6. THRESHOLD ANALYSIS",
                "-" * 22,
                f"Minimum specificity constraint: {threshold_data['min_specificity']:.2f}",
                f"Selected threshold: {threshold_data['selected_threshold']}",
                f"Selected sensitivity: {threshold_data['selected_sensitivity']}",
                f"Selected specificity: {threshold_data['selected_specificity']}",
                "",
                "Interpretation note: these metrics describe this evaluated test set "
                "and should not be treated as evidence of clinical safety or deployment readiness.",
            ]
        )

        output = Path(filepath)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")

        json_path = output.with_suffix(".json")
        json_path.write_text(
            json.dumps(report_data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        LOGGER.info("Medical evaluation report saved to %s", output)
        return output
