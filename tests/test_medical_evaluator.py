import numpy as np
import pytest

from src.evaluation.medical_evaluator import MedicalEvaluator


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict(self, images, verbose=0, **kwargs):
        return self.probabilities


def make_evaluator():
    probabilities = np.array(
        [
            [0.90, 0.10],
            [0.80, 0.20],
            [0.20, 0.80],
            [0.30, 0.70],
            [0.40, 0.60],
            [0.70, 0.30],
        ]
    )
    labels = np.array([0, 0, 1, 1, 0, 1])
    return MedicalEvaluator(FakeModel(probabilities), np.zeros((6, 1)), labels)


def test_comprehensive_report_contains_required_medical_metrics():
    report = make_evaluator().evaluate_comprehensive()

    assert set(["sensitivity", "specificity", "precision", "f1"]).issubset(
        report["PNEUMONIA"]
    )
    assert set(["sensitivity", "specificity", "precision", "f1"]).issubset(
        report["NORMAL"]
    )
    assert 0.0 <= report["roc_auc"] <= 1.0
    assert 0.0 <= report["pr_auc"] <= 1.0
    assert len(report["confusion_matrix"]) == 2
    assert report["aggregate"]["accuracy"] == pytest.approx(4 / 6)


def test_pneumonia_metrics_are_computed_from_pneumonia_as_positive_class():
    report = make_evaluator().evaluate_comprehensive()
    pneumonia = report["PNEUMONIA"]

    assert pneumonia["tp"] == 2
    assert pneumonia["fn"] == 1
    assert pneumonia["fp"] == 1
    assert pneumonia["tn"] == 2
    assert pneumonia["sensitivity"] == pytest.approx(2 / 3)
    assert pneumonia["specificity"] == pytest.approx(2 / 3)


def test_confidence_calibration_uses_prediction_confidence_not_pneumonia_probability():
    report = make_evaluator().evaluate_comprehensive()
    calibration = report["calibration"]

    populated = list(calibration.values())
    assert populated
    assert all("predicted_confidence" in item for item in populated)
    assert all("actual_accuracy" in item for item in populated)


def test_pneumonia_calibration_uses_class_one_probability():
    report = make_evaluator().evaluate_comprehensive()
    calibration = report["pneumonia_calibration"]

    assert calibration
    assert all("predicted_probability" in item for item in calibration.values())
    assert all("observed_pneumonia_rate" in item for item in calibration.values())


def test_threshold_analysis_enforces_specificity_constraint():
    result = make_evaluator().analyze_thresholds(
        thresholds=[0.2, 0.5, 0.8],
        min_specificity=0.80,
    )

    assert result["selected_threshold"] in {0.5, 0.8}
    assert result["selected_specificity"] >= 0.80
    assert len(result["points"]) == 3


def test_threshold_validation_rejects_invalid_values():
    with pytest.raises(ValueError, match="min_specificity"):
        make_evaluator().analyze_thresholds(min_specificity=1.2)

    with pytest.raises(ValueError, match="thresholds"):
        make_evaluator().analyze_thresholds(thresholds=[-0.1, 0.5])


def test_invalid_predictions_are_rejected():
    bad_model = FakeModel([[0.8, 0.3]])
    evaluator = MedicalEvaluator(bad_model, np.zeros((1, 1)), [0])

    with pytest.raises(ValueError, match="sum to 1.0"):
        evaluator.predict_probabilities()


def test_generate_report_writes_text_and_json(tmp_path):
    output = make_evaluator().generate_report(tmp_path / "medical_report.txt")

    assert output.exists()
    assert output.with_suffix(".json").exists()
    text = output.read_text(encoding="utf-8")
    assert "PNEUMONIA DETECTION" in text
    assert "ROC-AUC" in text
    assert "THRESHOLD ANALYSIS" in text
