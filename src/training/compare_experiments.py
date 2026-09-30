"""CLI for comparing MLflow runs."""

from __future__ import annotations

import argparse

from .experiment_tracker import compare_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chest-disease MLflow runs")
    parser.add_argument(
        "--experiment",
        default="chest_disease_clf",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    runs = compare_experiments(args.experiment)
    if runs.empty:
        print(f"No runs found in experiment: {args.experiment}")
        return

    print(runs.to_string(index=False))


if __name__ == "__main__":
    main()
