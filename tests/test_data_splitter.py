from src.data_pipeline.data_splitter import PatientLevelSplitter
from src.split_strategy.patient_level_split import split_patient_level


def make_rows(patient_count=20):
    rows = []

    for number in range(1, patient_count + 1):
        patient_id = f"p{number:03d}"
        label = "PNEUMONIA" if number % 2 == 0 else "NORMAL"

        rows.append(
            {
                "image_path": f"images/{patient_id}_1.jpg",
                "patient_id": patient_id,
                "label": label,
            }
        )
        rows.append(
            {
                "image_path": f"images/{patient_id}_2.jpg",
                "patient_id": patient_id,
                "label": label,
            }
        )

    return rows


def test_patient_level_splitter_has_zero_patient_overlap():
    rows = make_rows()
    splitter = PatientLevelSplitter(seed=42)

    splits = split_patient_level(rows, seed=42)
    splitter.validate_split(splits, expected_rows=rows)

    train_patients = {row["patient_id"] for row in splits["train"]}
    val_patients = {row["patient_id"] for row in splits["validation"]}
    test_patients = {row["patient_id"] for row in splits["test"]}

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)


def test_summarize_counts():
    rows = make_rows()
    splitter = PatientLevelSplitter(seed=42)

    splits = split_patient_level(rows, seed=42)
    summaries = splitter.summarize(splits)

    assert {summary.split for summary in summaries} == {
        "train",
        "validation",
        "test",
    }

    assert sum(summary.patient_count for summary in summaries) == 20
    assert sum(summary.image_count for summary in summaries) == 40


def test_class_balance_warnings_are_empty_for_balanced_data():
    rows = make_rows()
    splitter = PatientLevelSplitter(seed=42)

    splits = split_patient_level(rows, seed=42)

    assert splitter._class_balance_warnings(splits) == []


def test_write_outputs_creates_manifests_and_report(tmp_path):
    rows = make_rows()
    splitter = PatientLevelSplitter(seed=42)

    splits = split_patient_level(rows, seed=42)

    output_dir = tmp_path / "splits"
    report = tmp_path / "integrity.txt"

    report_path = splitter.write_outputs(
        splits=splits,
        output_dir=output_dir,
        report_path=report,
    )

    assert report_path.exists()
    assert (output_dir / "train.csv").exists()
    assert (output_dir / "validation.csv").exists()
    assert (output_dir / "test.csv").exists()
    assert "Leakage check: PASSED" in report_path.read_text(
        encoding="utf-8"
    )
