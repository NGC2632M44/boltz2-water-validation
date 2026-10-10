from __future__ import annotations

import argparse
import csv
from pathlib import Path

from summarize import svg_scatter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", default="production")
    arguments = parser.parse_args()
    root = arguments.root
    with (root / "metrics" / arguments.run_name / "sample_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    deliverables = root / "deliverables" / arguments.run_name
    svg_scatter(
        rows,
        "ligand_rmsd_A",
        "Boltz-2 ligand pose RMSD vs input water count",
        "Ligand heavy-atom RMSD after protein fit (Å)",
        deliverables / "ligand_rmsd_scatter.svg",
        "#2474a6",
        2.0,
        "Reference: qualified pose RMSD = 2.0 Å",
    )
    svg_scatter(
        rows,
        "mean_one_to_one_distance_A",
        "Predicted water oxygen distance to matched crystal water",
        "Mean one-to-one crystal-water O distance (Å)",
        deliverables / "water_position_scatter.svg",
        "#2c8a69",
        1.0,
        "Reference: qualified water-O accuracy ~ 1.0 Å",
    )
    print(f"figures written to {deliverables}")


if __name__ == "__main__":
    main()
