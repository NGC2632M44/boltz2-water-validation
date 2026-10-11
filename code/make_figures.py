from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from summarize import scatter_coordinate_rows, summarize_by_target, summarize_by_water_count, svg_scatter, write_csv

LIGAND_TITLE = "Boltz-2 ligand pose RMSD vs requested input water count"
LIGAND_YLABEL = "Ligand heavy-atom RMSD after protein fit (Å)"
WATER_TITLE = "Predicted water oxygen distance to matched crystal water"
WATER_YLABEL = "Mean one-to-one crystal-water O distance (Å)"


def read_metric_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate result figures and summary tables (including the 0-water control).")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", default="production")
    parser.add_argument("--control-run", default="baseline0")
    arguments = parser.parse_args()
    root = arguments.root

    rows = read_metric_rows(root / "metrics" / arguments.run_name / "sample_metrics.csv")
    control_path = root / "metrics" / arguments.control_run / "sample_metrics.csv"
    control_rows = read_metric_rows(control_path) if control_path.is_file() else []
    all_rows = rows + control_rows

    deliverables = root / "deliverables" / arguments.run_name
    water_summary = summarize_by_water_count(all_rows)
    target_summary = summarize_by_target(all_rows)
    coordinates = scatter_coordinate_rows(all_rows)
    write_csv(deliverables / "summary_by_water_count.csv", water_summary, list(water_summary[0]) if water_summary else ["water_count"])
    write_csv(deliverables / "summary_by_target_water.csv", target_summary, list(target_summary[0]) if target_summary else ["pdb_id", "water_count"])
    write_csv(
        deliverables / "scatter_coordinates.csv",
        coordinates,
        ["pdb_id", "water_count", "replicate", "status", "ligand_rmsd_x", "ligand_rmsd_y", "water_position_x", "water_position_y"],
    )

    svg_scatter(all_rows, "ligand_rmsd_A", LIGAND_TITLE, LIGAND_YLABEL, deliverables / "ligand_rmsd_scatter.svg", "#2474a6", 2.0, "Reference: qualified pose RMSD = 2.0 Å", 0, 10)
    svg_scatter(all_rows, "mean_one_to_one_distance_A", WATER_TITLE, WATER_YLABEL, deliverables / "water_position_scatter.svg", "#2c8a69", 1.0, "Reference: qualified water-O accuracy ~ 1.0 Å", 0, 10)

    print(json.dumps({"control_rows": len(control_rows), "total_rows": len(all_rows)}, indent=2))
    for entry in water_summary:
        print(json.dumps(entry, ensure_ascii=False))


if __name__ == "__main__":
    main()
