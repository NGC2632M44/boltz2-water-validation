from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import yaml

from common import write_json


def read_metrics(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def numeric(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_water_count(rows: list[dict]) -> list[dict]:
    summary = []
    counts = sorted({int(row["water_count"]) for row in rows})
    for water_count in counts:
        group = [row for row in rows if int(row["water_count"]) == water_count]
        valid = [row for row in group if row["status"] == "ok"]
        rmsd = numeric(valid, "ligand_rmsd_A")
        summary.append(
            {
                "water_count": water_count,
                "requested": len(group),
                "valid": len(valid),
                "failed_or_missing": len(group) - len(valid),
                "ligand_rmsd_mean_A": statistics.mean(rmsd) if rmsd else None,
                "ligand_rmsd_median_A": statistics.median(rmsd) if rmsd else None,
                "ligand_rmsd_sd_A": statistics.stdev(rmsd) if len(rmsd) > 1 else None,
                "water_f1_2p0A_mean": statistics.mean(numeric(valid, "f1_2p0A")) if valid else None,
                "water_precision_2p0A_mean": statistics.mean(numeric(valid, "precision_2p0A")) if valid else None,
                "water_recall_2p0A_mean": statistics.mean(numeric(valid, "recall_2p0A")) if valid else None,
                "mean_one_to_one_water_distance_A": statistics.mean(numeric(valid, "mean_one_to_one_distance_A")) if valid else None,
            }
        )
    return summary


def summarize_by_target(rows: list[dict]) -> list[dict]:
    output = []
    keys = sorted({(row["pdb_id"], int(row["water_count"])) for row in rows})
    for pdb_id, water_count in keys:
        group = [row for row in rows if row["pdb_id"] == pdb_id and int(row["water_count"]) == water_count]
        valid = [row for row in group if row["status"] == "ok"]
        rmsd = numeric(valid, "ligand_rmsd_A")
        output.append(
            {
                "pdb_id": pdb_id,
                "water_count": water_count,
                "requested": len(group),
                "valid": len(valid),
                "ligand_rmsd_mean_A": statistics.mean(rmsd) if rmsd else None,
                "ligand_rmsd_median_A": statistics.median(rmsd) if rmsd else None,
                "water_f1_2p0A_mean": statistics.mean(numeric(valid, "f1_2p0A")) if valid else None,
            }
        )
    return output


def scatter_coordinate_rows(rows: list[dict]) -> list[dict]:
    coordinates = []
    for row in rows:
        coordinates.append(
            {
                "pdb_id": row["pdb_id"],
                "water_count": row["water_count"],
                "replicate": row["replicate"],
                "status": row["status"],
                "ligand_rmsd_x": row["water_count"],
                "ligand_rmsd_y": row["ligand_rmsd_A"],
                "water_position_x": row["water_count"],
                "water_position_y": row["mean_one_to_one_distance_A"],
            }
        )
    return coordinates


def svg_scatter(rows: list[dict], metric_key: str, title: str, y_label: str, output_path: Path, color: str, baseline_value: float | None = None, baseline_label: str = "") -> None:
    points = []
    targets = sorted({row["pdb_id"] for row in rows})
    target_index = {pdb_id: index for index, pdb_id in enumerate(targets)}
    for row in rows:
        if row["status"] != "ok":
            continue
        try:
            value = float(row[metric_key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        jitter_code = (target_index[row["pdb_id"]] * 37 + int(row["replicate"]) * 19) % 101
        jitter = (jitter_code / 100.0 - 0.5) * 0.58
        points.append((int(row["water_count"]) + jitter, value, row["pdb_id"], int(row["water_count"])))
    width, height = 1360, 820
    left, right, top, bottom = 100, 40, 85, 95
    plot_width, plot_height = width - left - right, height - top - bottom
    max_value = max((point[1] for point in points), default=1.0)
    y_max = max(1.0, math.ceil(max_value * 1.05 * 2) / 2)
    to_x = lambda value: left + (value - 0.5) / 10.0 * plot_width
    to_y = lambda value: top + plot_height - value / y_max * plot_height
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="40" text-anchor="middle" font-family="Arial" font-size="24" font-weight="bold">{escape(title)}</text>',
    ]
    y_tick_count = 8
    for tick_index in range(y_tick_count + 1):
        value = y_max * tick_index / y_tick_count
        y_position = to_y(value)
        lines.append(f'<line x1="{left}" y1="{y_position:.1f}" x2="{width-right}" y2="{y_position:.1f}" stroke="#d9dee7"/>')
        lines.append(f'<text x="{left-14}" y="{y_position+5:.1f}" text-anchor="end" font-family="Arial" font-size="14" fill="#394150">{value:.1f}</text>')
    for water_count in range(1, 11):
        x_position = to_x(water_count)
        lines.append(f'<line x1="{x_position:.1f}" y1="{top}" x2="{x_position:.1f}" y2="{top+plot_height}" stroke="#eef1f5"/>')
        lines.append(f'<text x="{x_position:.1f}" y="{top+plot_height+28}" text-anchor="middle" font-family="Arial" font-size="14" fill="#394150">{water_count}</text>')
    lines.append(f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#273142" stroke-width="2"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#273142" stroke-width="2"/>')
    for x_position, value, pdb_id, water_count in points:
        lines.append(f'<circle cx="{to_x(x_position):.2f}" cy="{to_y(value):.2f}" r="2.5" fill="{color}" fill-opacity="0.36"><title>{escape(pdb_id)}; waters={water_count}; {y_label}={value:.3f}</title></circle>')
    for water_count in range(1, 11):
        group = [point[1] for point in points if point[3] == water_count]
        if group:
            median = statistics.median(group)
            median_x = to_x(water_count)
            median_y = to_y(median)
            lines.append(f'<path d="M {median_x-10:.1f} {median_y:.1f} L {median_x+10:.1f} {median_y:.1f}" stroke="#c23b22" stroke-width="3"/>')
    if baseline_value is not None:
        baseline_y = to_y(baseline_value)
        if top <= baseline_y <= top + plot_height:
            lines.append(f'<line x1="{left}" y1="{baseline_y:.1f}" x2="{width-right}" y2="{baseline_y:.1f}" stroke="#7d1d9c" stroke-width="2.5" stroke-dasharray="10 5"/>')
            lines.append(f'<text x="{left+8}" y="{baseline_y-8:.1f}" text-anchor="start" font-family="Arial" font-size="14" font-weight="bold" fill="#7d1d9c">{escape(baseline_label)}</text>')
    lines.append(f'<text x="{left+plot_width/2}" y="{height-26}" text-anchor="middle" font-family="Arial" font-size="17">Requested input waters (number of separate HOH entities)</text>')
    lines.append(f'<text transform="translate(28 {top+plot_height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="17">{escape(y_label)}</text>')
    lines.append(f'<text x="{width-right}" y="{height-26}" text-anchor="end" font-family="Arial" font-size="13" fill="#5d6572">n={len(points)} valid samples; red bars = per-count median</text>')
    lines.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete_metric_rows(rows: list[dict], target_ids: list[str], water_counts: list[int], job_sample_counts: dict[tuple[str, int], int]) -> dict:
    expected_samples = [
        (pdb_id, water_count, replicate)
        for pdb_id in target_ids
        for water_count in water_counts
        for replicate in range(1, job_sample_counts[(pdb_id, water_count)] + 1)
    ]
    expected_set = set(expected_samples)
    rows_by_sample = {}
    unexpected_metric_samples = []
    for row in rows:
        sample = (str(row["pdb_id"]), int(row["water_count"]), int(row["replicate"]))
        if sample not in expected_set:
            unexpected_metric_samples.append(sample)
            continue
        if sample in rows_by_sample:
            raise RuntimeError(f"Duplicate sample metrics row: {sample}")
        rows_by_sample[sample] = row

    missing_metric_samples = []
    complete_rows = []
    for pdb_id, water_count, replicate in expected_samples:
        sample = (pdb_id, water_count, replicate)
        row = rows_by_sample.get(sample)
        sample_record = {"pdb_id": pdb_id, "water_count": water_count, "replicate": replicate}
        if row is None:
            missing_metric_samples.append(sample_record)
            row = {
                **sample_record,
                "water_count": str(water_count),
                "replicate": str(replicate),
                "status": "missing",
                "error": "No metrics row was recorded",
                "prediction_cif": "",
                "paired_pdb": "",
                "ligand_rmsd_A": "",
                "mean_one_to_one_distance_A": "",
            }
        complete_rows.append(row)

    return {
        "expected_samples": expected_samples,
        "complete_rows": complete_rows,
        "missing_metric_samples": missing_metric_samples,
        "unexpected_metric_samples": [
            {"pdb_id": pdb_id, "water_count": water_count, "replicate": replicate}
            for pdb_id, water_count, replicate in sorted(unexpected_metric_samples)
        ],
    }


def resolved_source_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def validate_job_predictions(status_path: Path, output_dir: Path, input_path: Path, sample_count: int, expected_seed: int, root: Path) -> tuple[dict[int, Path], str | None]:
    if not status_path.is_file():
        return {}, "missing_status"
    if not input_path.is_file():
        return {}, "missing_input_yaml"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, "invalid_status_json"
    if not isinstance(status, dict):
        return {}, "invalid_status_document"

    expected_output = output_dir.resolve()
    if resolved_source_path(status.get("output_dir", ""), root) != expected_output:
        return {}, "output_directory_mismatch"
    if not output_dir.is_dir():
        return {}, "missing_output_directory"
    if status.get("requested_samples") != sample_count:
        return {}, "requested_sample_count_mismatch"
    if status.get("seed") != expected_seed:
        return {}, "seed_mismatch"
    if status.get("input_sha256") != sha256(input_path):
        return {}, "input_hash_mismatch"
    if status.get("model") != "boltz2":
        return {}, "model_mismatch"

    recorded_values = status.get("prediction_files")
    if not isinstance(recorded_values, list) or any(not isinstance(value, str) for value in recorded_values):
        return {}, "invalid_prediction_file_list"
    try:
        recorded_paths = [resolved_source_path(value, root) for value in recorded_values]
        actual_paths = sorted(path.resolve(strict=True) for path in output_dir.rglob("*_model_*.cif") if path.is_file())
    except OSError:
        return {}, "prediction_file_unavailable"
    if len(set(recorded_paths)) != len(recorded_paths) or len(set(actual_paths)) != len(actual_paths):
        return {}, "duplicate_prediction_file_path"
    if any(not path.is_relative_to(expected_output) for path in recorded_paths + actual_paths):
        return {}, "prediction_path_outside_current_output"
    if sorted(recorded_paths) != actual_paths:
        return {}, "recorded_prediction_set_mismatch"

    model_paths = {}
    for path in actual_paths:
        match = re.search(r"_model_(\d+)\.cif$", path.name)
        if not match:
            return {}, "invalid_model_filename"
        model_index = int(match.group(1))
        if model_index >= sample_count or model_index in model_paths:
            return {}, "duplicate_or_out_of_range_model_index"
        model_paths[model_index] = path

    actual_indices = sorted(model_paths)
    if status.get("observed_samples") != len(actual_paths) or status.get("prediction_indices") != actual_indices:
        return {}, "prediction_index_set_mismatch"
    hashes = status.get("prediction_sha256")
    if not isinstance(hashes, dict):
        return {}, "missing_prediction_hashes"
    resolved_hashes = {str(resolved_source_path(path, root)): value for path, value in hashes.items()}
    if set(resolved_hashes) != {str(path) for path in actual_paths}:
        return {}, "prediction_hash_path_set_mismatch"
    if any(not isinstance(resolved_hashes[str(path)], str) or resolved_hashes[str(path)] != sha256(path) for path in actual_paths):
        return {}, "prediction_hash_mismatch"
    return model_paths, None


def prediction_completeness(rows: list[dict], root: Path, run_config: dict, target_indices: dict[str, int]) -> dict:
    target_ids = run_config["targets"]
    water_counts = run_config["water_counts"]
    default_sample_count = run_config["samples_per_input"]
    run_name = run_config["run_name"]
    run_root = root / "runs" / run_name
    job_sample_counts = {}
    for pdb_id in target_ids:
        for water_count in water_counts:
            count = default_sample_count
            status_path = run_root / pdb_id / "status" / f"water_{water_count:02d}.json"
            if status_path.is_file():
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    status = None
                if isinstance(status, dict) and isinstance(status.get("requested_samples"), int):
                    count = status["requested_samples"]
            job_sample_counts[(pdb_id, water_count)] = count
    metric_inventory = complete_metric_rows(rows, target_ids, water_counts, job_sample_counts)
    rows_by_sample = {
        (str(row["pdb_id"]), int(row["water_count"]), int(row["replicate"])): row
        for row in rows
    }
    status_errors = []
    job_predictions = {}
    for pdb_id in target_ids:
        for water_count in water_counts:
            status_path = run_root / pdb_id / "status" / f"water_{water_count:02d}.json"
            output_dir = run_root / pdb_id / f"water_{water_count:02d}"
            input_path = root / "inputs" / pdb_id / f"water_{water_count:02d}.yaml"
            expected_seed = run_config["seed_base"] + target_indices[pdb_id] * 1000 + water_count
            files_by_index, error = validate_job_predictions(
                status_path, output_dir, input_path, job_sample_counts[(pdb_id, water_count)], expected_seed, root
            )
            job_predictions[(pdb_id, water_count)] = files_by_index
            if error:
                status_errors.append(
                    {
                        "pdb_id": pdb_id,
                        "water_count": water_count,
                        "status_file": status_path.relative_to(root).as_posix(),
                        "reason": error,
                    }
                )

    missing_prediction_samples = []
    failed_samples = []
    prediction_path_mismatches = []
    available_prediction_paths = []
    for pdb_id, water_count, replicate in metric_inventory["expected_samples"]:
        sample = (pdb_id, water_count, replicate)
        sample_record = {"pdb_id": pdb_id, "water_count": water_count, "replicate": replicate}
        row = rows_by_sample.get(sample)
        current_path = job_predictions[(pdb_id, water_count)].get(replicate - 1)
        if current_path is None:
            job_error = next(
                (item["reason"] for item in status_errors if item["pdb_id"] == pdb_id and item["water_count"] == water_count),
                "requested_model_file_missing",
            )
            missing_prediction_samples.append({**sample_record, "reason": job_error})
        else:
            available_prediction_paths.append(current_path)

        if row is None:
            failed_samples.append({**sample_record, "status": "missing_metric_row"})
            continue
        if row.get("status") != "ok":
            failed_samples.append({**sample_record, "status": row.get("status", "missing_status"), "error": row.get("error", "")})
        metric_value = row.get("prediction_cif")
        metric_path = resolved_source_path(metric_value, root) if metric_value else None
        if metric_path != current_path and (metric_path is not None or current_path is not None):
            mismatch = {**sample_record, "metrics_prediction_cif": str(metric_path) if metric_path else None, "current_prediction_cif": str(current_path) if current_path else None}
            prediction_path_mismatches.append(mismatch)
        if row.get("status") == "ok" and metric_path != current_path:
            raise RuntimeError(f"Successful metric row points to a non-current prediction file: {sample}")
        if row.get("status") == "ok" and current_path is None:
            raise RuntimeError(f"Successful metric row has no verified current prediction file: {sample}")

    unique_paths = sorted({str(path) for path in available_prediction_paths})
    if len(unique_paths) != len(available_prediction_paths):
        raise RuntimeError("A current prediction file is assigned to more than one sample")
    metric_inventory.update(
        {
            "sample_count": default_sample_count,
            "job_sample_counts": {f"{pdb_id}:{water_count}": value for (pdb_id, water_count), value in job_sample_counts.items()},
            "expected_sample_count": len(metric_inventory["expected_samples"]),
            "metric_row_count": len(rows),
            "failed_samples": failed_samples,
            "missing_prediction_samples": missing_prediction_samples,
            "prediction_path_mismatches": prediction_path_mismatches,
            "invalid_or_missing_job_statuses": status_errors,
            "available_prediction_count": len(unique_paths),
            "available_prediction_paths": unique_paths,
        }
    )
    return metric_inventory


def referenced_msa_paths(root: Path, input_paths: list[Path]) -> dict:
    paths = set()
    missing_references = []
    external_references = []
    for input_path in input_paths:
        payload = yaml.safe_load(input_path.read_text(encoding="utf-8")) or {}
        for sequence in payload.get("sequences", []):
            protein = sequence.get("protein")
            if not protein or not protein.get("msa"):
                continue
            msa_path = resolved_source_path(protein["msa"], root)
            reference = {
                "input_yaml": input_path.relative_to(root).as_posix(),
                "msa_path": str(msa_path),
            }
            if not msa_path.is_relative_to(root.resolve()):
                external_references.append(reference)
            elif msa_path.is_file():
                paths.add(msa_path)
            else:
                missing_references.append(reference)
    return {
        "paths": sorted(paths),
        "missing_references": missing_references,
        "external_references": external_references,
    }


def verified_status_msa_paths(root: Path, run_root: Path, status_paths: list[Path]) -> dict:
    paths = set()
    invalid_references = []
    for status_path in status_paths:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(status, dict):
            continue
        pdb_id = status.get("pdb_id")
        water_count = status.get("water_count")
        if not pdb_id or not isinstance(water_count, int):
            continue
        output_dir = run_root / pdb_id / f"water_{water_count:02d}"
        if resolved_source_path(status.get("output_dir", ""), root) != output_dir.resolve():
            continue
        msa_values = status.get("msa_files", [])
        msa_hashes = status.get("msa_sha256", {})
        if not isinstance(msa_values, list) or not isinstance(msa_hashes, dict):
            continue
        for value in msa_values:
            if not isinstance(value, str):
                continue
            msa_path = resolved_source_path(value, root)
            record = {"status_file": status_path.relative_to(root).as_posix(), "msa_path": str(msa_path)}
            if (
                not msa_path.is_relative_to(output_dir.resolve())
                or not msa_path.is_file()
                or msa_hashes.get(str(msa_path)) != sha256(msa_path)
            ):
                invalid_references.append(record)
                continue
            paths.add(msa_path)
    return {"paths": sorted(paths), "invalid_references": invalid_references}


def build_release(root: Path, run_name: str) -> dict:
    root = root.resolve()
    run_metrics = root / "metrics" / run_name
    metric_rows = read_metrics(run_metrics / "sample_metrics.csv")
    deliverables = root / "deliverables" / run_name
    deliverables.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    prepared = json.loads((root / "prepared_manifest.json").read_text(encoding="utf-8"))
    run_root = root / "runs" / run_name
    run_config_path = run_root / "run_config.json"
    if not run_config_path.is_file():
        raise RuntimeError(f"Run configuration is missing: {run_config_path}")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    prepared_target_ids = [item["pdb_id"] for item in prepared["targets"]]
    if any(target not in prepared_target_ids for target in run_config["targets"]):
        raise RuntimeError("Run configuration contains an unprepared target")
    if any(count not in manifest["design"]["input_water_counts"] for count in run_config["water_counts"]):
        raise RuntimeError("Run configuration contains an unsupported water count")
    if run_config["samples_per_input"] < 1:
        raise RuntimeError("Run configuration has an invalid sample count")

    target_indices = {item["pdb_id"]: index for index, item in enumerate(prepared["targets"])}
    completeness = prediction_completeness(metric_rows, root, run_config, target_indices)
    rows = completeness["complete_rows"]
    coordinate_rows = scatter_coordinate_rows(rows)
    water_summary = summarize_by_water_count(rows)
    target_summary = summarize_by_target(rows)
    write_csv(deliverables / "summary_by_water_count.csv", water_summary, list(water_summary[0]) if water_summary else ["water_count"])
    write_csv(deliverables / "summary_by_target_water.csv", target_summary, list(target_summary[0]) if target_summary else ["pdb_id", "water_count"])
    write_csv(
        deliverables / "scatter_coordinates.csv",
        coordinate_rows,
        ["pdb_id", "water_count", "replicate", "status", "ligand_rmsd_x", "ligand_rmsd_y", "water_position_x", "water_position_y"],
    )
    svg_scatter(rows, "ligand_rmsd_A", "Boltz-2 ligand pose RMSD vs input water count", "Ligand heavy-atom RMSD after protein fit (Å)", deliverables / "ligand_rmsd_scatter.svg", "#2474a6", 2.0, "Reference: qualified pose RMSD = 2.0 Å")
    svg_scatter(rows, "mean_one_to_one_distance_A", "Predicted water oxygen distance to matched crystal water", "Mean one-to-one crystal-water O distance (Å)", deliverables / "water_position_scatter.svg", "#2c8a69", 1.0, "Reference: qualified water-O accuracy ~ 1.0 Å")

    pair_root = root / "pairs" / run_name
    successful_rows = [row for row in rows if row["status"] == "ok"]
    pair_files = []
    for row in successful_rows:
        pair_path = resolved_source_path(row.get("paired_pdb", ""), root)
        if not pair_path.is_file() or not pair_path.is_relative_to(pair_root.resolve()):
            raise RuntimeError(f"Successful metric row has no in-run paired PDB: {row['pdb_id']} water={row['water_count']} replicate={row['replicate']}")
        pair_files.append(pair_path)
    if len(set(pair_files)) != len(successful_rows):
        raise RuntimeError("Successful metric rows do not map one-to-one to paired PDB files")
    pair_archive = deliverables / "paired_reference_boltz_pdbs.zip"
    with zipfile.ZipFile(pair_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in pair_files:
            archive.write(path, path.relative_to(pair_root).as_posix())

    target_ids = run_config["targets"]
    water_counts = run_config["water_counts"]
    sample_count = run_config["samples_per_input"]
    expected_status_paths = [
        run_root / pdb_id / "status" / f"water_{count:02d}.json"
        for pdb_id in target_ids
        for count in water_counts
    ]
    status_paths = [path for path in expected_status_paths if path.is_file()]
    all_status_paths = sorted(run_root.glob("*/status/water_*.json"))
    unexpected_status_paths = [path for path in all_status_paths if path not in expected_status_paths]
    missing_status_files = [path.relative_to(root).as_posix() for path in expected_status_paths if not path.is_file()]
    expected_log_paths = []
    for path in status_paths:
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("log"):
            expected_log_paths.append(resolved_source_path(status["log"], root))
    missing_log_files = [str(path) for path in expected_log_paths if not path.is_file()]
    expected_input_paths = [
        root / "inputs" / pdb_id / f"water_{count:02d}.yaml"
        for pdb_id in target_ids
        for count in water_counts
    ]
    msa_references = referenced_msa_paths(root, expected_input_paths)
    status_msa_inventory = verified_status_msa_paths(root, run_root, status_paths)
    msa_paths = sorted(set(msa_references["paths"]) | set(status_msa_inventory["paths"]))
    prediction_paths = [Path(path) for path in completeness["available_prediction_paths"]]
    source_completeness_path = deliverables / "source_completeness.json"
    completeness["available_prediction_files"] = [path.relative_to(root).as_posix() for path in prediction_paths]
    del completeness["available_prediction_paths"]
    del completeness["complete_rows"]
    del completeness["expected_samples"]
    completeness.update(
        {
            "run_configuration": "runs/" + run_name + "/run_config.json",
            "expected_status_count": len(expected_status_paths),
            "available_status_count": len(status_paths),
            "expected_prediction_count": completeness["expected_sample_count"],
            "missing_status_files": missing_status_files,
            "unexpected_status_files": [path.relative_to(root).as_posix() for path in unexpected_status_paths],
            "design_expected_status_count": len(prepared["targets"]) * len(manifest["design"]["input_water_counts"]),
            "design_expected_prediction_count": len(prepared["targets"]) * len(manifest["design"]["input_water_counts"]) * manifest["design"]["diffusion_samples_per_input"],
            "missing_log_files": missing_log_files,
            "msa_files": [path.relative_to(root).as_posix() for path in msa_paths],
            "missing_msa_references": msa_references["missing_references"],
            "external_msa_references": msa_references["external_references"],
            "invalid_status_msa_references": status_msa_inventory["invalid_references"],
        }
    )
    write_json(source_completeness_path, completeness)

    source_archive = deliverables / "source_data_and_metrics.zip"
    run_summary_path = run_root / "run_summary.json"
    included = [
        root / "manifest.json",
        root / "prepared_manifest.json",
        run_config_path,
        run_summary_path,
        run_metrics / "sample_metrics.csv",
        deliverables / "summary_by_water_count.csv",
        deliverables / "summary_by_target_water.csv",
        deliverables / "scatter_coordinates.csv",
        deliverables / "ligand_rmsd_scatter.svg",
        deliverables / "water_position_scatter.svg",
        source_completeness_path,
    ]
    required = list(included)
    required.extend(root / "source-data" / "structures" / f"{item['pdb_id']}.cif" for item in manifest["targets"])
    required.extend(root / "source-data" / "components" / f"{component_id}.cif" for component_id in sorted({item["ligand_ccd"] for item in manifest["targets"]}))
    required.extend(root / "metadata" / f"{item['pdb_id']}.json" for item in prepared["targets"])
    required.extend(
        root / "inputs" / item["pdb_id"] / f"water_{count:02d}.yaml"
        for item in prepared["targets"]
        for count in manifest["design"]["input_water_counts"]
    )
    required.extend(sorted((root / "code").glob("*.py")))
    required.extend([root / "code" / "README.md", root / "code" / "manifest.json"])
    missing_required = sorted({path.relative_to(root).as_posix() for path in required if not path.is_file()})
    if missing_required:
        raise RuntimeError(f"Required release sources are missing: {missing_required[:20]}")

    log_paths = sorted(run_root.glob("**/logs/*.log"))
    included.extend(status_paths)
    included.extend(log_paths)
    included.extend(prediction_paths)
    included.extend(msa_paths)
    included.extend(sorted((root / "source-data").rglob("*.cif")))
    included.extend(sorted((root / "inputs").rglob("*.yaml")))
    included.extend(sorted((root / "metadata").glob("*.json")))
    included.extend(sorted((root / "code").glob("*.py")))
    included.extend([root / "code" / "README.md", root / "code" / "manifest.json"])
    included = list({path.resolve(): path for path in included}.values())
    with zipfile.ZipFile(source_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in included:
            archive.write(path, path.relative_to(root).as_posix())

    expected_prediction_count = completeness["expected_sample_count"]
    design_expected_prediction_count = completeness["design_expected_prediction_count"]
    design_complete = (
        run_config["targets"] == prepared_target_ids
        and run_config["water_counts"] == manifest["design"]["input_water_counts"]
        and sample_count == manifest["design"]["diffusion_samples_per_input"]
    )
    release_complete = (
        design_complete
        and len(metric_rows) == completeness["expected_sample_count"]
        and not completeness["missing_metric_samples"]
        and not completeness["unexpected_metric_samples"]
        and not completeness["failed_samples"]
        and not completeness["missing_prediction_samples"]
        and not completeness["prediction_path_mismatches"]
        and not completeness["invalid_or_missing_job_statuses"]
        and not missing_status_files
        and not unexpected_status_paths
        and not missing_log_files
        and not msa_references["missing_references"]
        and not msa_references["external_references"]
        and not status_msa_inventory["invalid_references"]
        and len(pair_files) == completeness["expected_sample_count"]
    )
    report = {
        "run_name": run_name,
        "design_complete": design_complete,
        "release_complete": release_complete,
        "requested_rows": completeness["expected_sample_count"],
        "metric_row_count": len(metric_rows),
        "successful_rows": sum(row["status"] == "ok" for row in rows),
        "failed_or_missing_rows": sum(row["status"] != "ok" for row in rows),
        "paired_pdb_count": len(pair_files),
        "expected_paired_pdb_count": completeness["expected_sample_count"],
        "scatter_coordinate_rows": len(coordinate_rows),
        "expected_status_count": len(expected_status_paths),
        "available_status_count": len(status_paths),
        "missing_status_files": missing_status_files,
        "unexpected_status_files": [path.relative_to(root).as_posix() for path in unexpected_status_paths],
        "expected_prediction_count": expected_prediction_count,
        "design_expected_prediction_count": design_expected_prediction_count,
        "available_prediction_count": completeness["available_prediction_count"],
        "missing_prediction_samples": completeness["missing_prediction_samples"],
        "missing_metric_samples": completeness["missing_metric_samples"],
        "failed_samples": completeness["failed_samples"],
        "invalid_or_missing_job_statuses": completeness["invalid_or_missing_job_statuses"],
        "prediction_path_mismatches": completeness["prediction_path_mismatches"],
        "missing_log_files": missing_log_files,
        "msa_csv_count": len(msa_paths),
        "missing_msa_references": msa_references["missing_references"],
        "external_msa_references": msa_references["external_references"],
        "invalid_status_msa_references": status_msa_inventory["invalid_references"],
        "source_completeness_manifest": str(source_completeness_path),
        "source_archive_file_count": len(included),
        "paired_pdb_archive": str(pair_archive),
        "paired_pdb_archive_sha256": sha256(pair_archive),
        "source_data_archive": str(source_archive),
        "source_data_archive_sha256": sha256(source_archive),
        "deliverables_directory": str(deliverables),
    }
    write_json(deliverables / "release_manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", default="production")
    arguments = parser.parse_args()
    print(json.dumps(build_release(arguments.root, arguments.run_name), indent=2))


if __name__ == "__main__":
    main()
