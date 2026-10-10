from __future__ import annotations

import argparse
import concurrent.futures
import csv
from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from common import SEED_BASE, write_input_yaml, write_json


BOLTZ = Path(os.environ.get("BOLTZ_BIN", "boltz"))
DEFAULT_CACHE = Path(os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz")))


def prediction_files(output_dir: Path) -> list[Path]:
    return sorted(output_dir.rglob("*_model_*.cif"))


def prediction_indices(output_dir: Path) -> list[int]:
    indices = []
    for path in prediction_files(output_dir):
        match = re.search(r"_model_(\d+)\.cif$", path.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def boltz_version() -> str:
    try:
        return package_version("boltz")
    except PackageNotFoundError:
        return "unknown"


def persist_run_config(root: Path, run_name: str, targets: list[str], counts: list[int], samples: int, gpus: list[int], cache: Path) -> dict:
    config = {
        "run_name": run_name,
        "targets": targets,
        "water_counts": counts,
        "samples_per_input": samples,
        "gpus": gpus,
        "cache": str(cache.resolve()),
        "model": "boltz2",
        "boltz_version": boltz_version(),
        "seed_base": SEED_BASE,
    }
    config_path = root / "runs" / run_name / "run_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in config.items()):
            raise RuntimeError(f"Run configuration differs from the existing immutable config: {config_path}")
        return existing
    write_json(config_path, config)
    return config


def job_status_complete(status: dict, sample_count: int) -> bool:
    expected_indices = list(range(sample_count))
    prediction_files_in_status = status.get("prediction_files")
    prediction_hashes = status.get("prediction_sha256")
    return (
        status.get("exit_code") == 0
        and status.get("requested_samples") == sample_count
        and status.get("observed_samples") == sample_count
        and status.get("prediction_indices") == expected_indices
        and isinstance(prediction_files_in_status, list)
        and len(prediction_files_in_status) == sample_count
        and len(set(prediction_files_in_status)) == sample_count
        and isinstance(prediction_hashes, dict)
        and set(prediction_hashes) == set(prediction_files_in_status)
        and all(isinstance(value, str) and len(value) == 64 for value in prediction_hashes.values())
    )


def output_files_match_status(status: dict, output_dir: Path, sample_count: int) -> bool:
    if not output_dir.is_dir() or not job_status_complete(status, sample_count):
        return False
    files = prediction_files(output_dir)
    actual_paths = [str(path.resolve()) for path in files]
    if (
        len(files) != sample_count
        or actual_paths != status["prediction_files"]
        or str(output_dir.resolve()) != status.get("output_dir")
        or prediction_indices(output_dir) != list(range(sample_count))
    ):
        return False
    try:
        return all(file_sha256(path) == status["prediction_sha256"][str(path.resolve())] for path in files)
    except OSError:
        return False


def archive_existing_output(output_dir: Path, attempt: str) -> Path | None:
    if not output_dir.exists():
        return None
    archive_directory = output_dir.parent / "attempts"
    archive_directory.mkdir(parents=True, exist_ok=True)
    archive_path = archive_directory / f"{output_dir.name}_{attempt}"
    if archive_path.exists():
        raise FileExistsError(f"Output attempt archive already exists: {archive_path}")
    output_dir.rename(archive_path)
    return archive_path


def status_is_complete(status_path: Path, output_dir: Path, input_path: Path, sample_count: int, seed: int) -> bool:
    if not status_path.exists() or not input_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        return (
            output_files_match_status(status, output_dir, sample_count)
            and status.get("seed") == seed
            and status.get("input_sha256") == file_sha256(input_path)
            and prediction_indices(output_dir) == list(range(sample_count))
        )
    except OSError:
        return False


def find_msa_paths(output_dir: Path, protein_chains: list[dict]) -> dict[str, str]:
    matches = [
        path
        for path in output_dir.rglob("*.csv")
        if "msa" in path.parts and path.is_file() and path.stat().st_size > 0
    ]
    by_sequence = {}
    for path in matches:
        try:
            with path.open(newline="", encoding="utf-8") as input_file:
                reader = csv.DictReader(input_file)
                if not reader.fieldnames or "sequence" not in reader.fieldnames:
                    continue
                query_rows = (row for row in reader if str(row.get("key", "")).strip() == "-1")
                first_row = next(query_rows, None)
        except (OSError, UnicodeError, csv.Error):
            continue
        if not first_row or not first_row.get("sequence"):
            continue
        query_sequence = "".join(str(first_row["sequence"]).split()).upper().replace("-", "").replace(".", "")
        if query_sequence:
            by_sequence.setdefault(query_sequence, str(path.resolve()))
    result = {}
    for protein in protein_chains:
        sequence = "".join(str(protein["sequence"]).split()).upper().replace("-", "").replace(".", "")
        if sequence in by_sequence:
            result[protein["input_id"]] = by_sequence[sequence]
    return result if len(result) == len(protein_chains) else {}


def run_job(
    context: dict,
    root: Path,
    run_name: str,
    water_count: int,
    sample_count: int,
    seed: int,
    gpu: int,
    cache: Path,
    use_msa_server: bool,
    max_parallel_samples: int = 1,
) -> dict:
    pdb_id = context["pdb_id"]
    stem = f"{pdb_id}_water_{water_count:02d}"
    input_path = root / "inputs" / pdb_id / f"water_{water_count:02d}.yaml"
    output_dir = root / "runs" / run_name / pdb_id / f"water_{water_count:02d}"
    status_dir = root / "runs" / run_name / pdb_id / "status"
    log_dir = root / "runs" / run_name / pdb_id / "logs"
    status_path = status_dir / f"water_{water_count:02d}.json"
    status_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if status_is_complete(status_path, output_dir, input_path, sample_count, seed):
        return json.loads(status_path.read_text(encoding="utf-8"))

    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_existing_output(output_dir, attempt)
    log_path = log_dir / f"water_{water_count:02d}_{attempt}.log"
    command = [
        str(BOLTZ),
        "predict",
        str(input_path),
        "--out_dir",
        str(output_dir),
        "--cache",
        str(cache),
        "--devices",
        "1",
        "--accelerator",
        "gpu",
        "--diffusion_samples",
        str(sample_count),
        "--max_parallel_samples",
        str(max_parallel_samples),
        "--output_format",
        "mmcif",
        "--num_workers",
        "2",
        "--seed",
        str(seed),
        "--model",
        "boltz2",
        "--override",
    ]
    if use_msa_server:
        command.append("--use_msa_server")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["OMP_NUM_THREADS"] = "4"
    started = time.time()
    exception_text = ""
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            completed = subprocess.run(
                command,
                cwd=str(root),
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
            exit_code = completed.returncode
        except Exception as error:
            exit_code = 127
            exception_text = f"{type(error).__name__}: {error}"
            log_file.write(exception_text + "\n")
    files = prediction_files(output_dir)
    indices = prediction_indices(output_dir)
    resolved_files = [str(path.resolve()) for path in files]
    resolved_msa_files = sorted(set(find_msa_paths(output_dir, context["protein_chains"]).values()))
    status = {
        "pdb_id": pdb_id,
        "water_count": water_count,
        "requested_samples": sample_count,
        "observed_samples": len(files),
        "prediction_indices": indices,
        "prediction_files": resolved_files,
        "prediction_sha256": {str(path.resolve()): file_sha256(path) for path in files},
        "msa_files": resolved_msa_files,
        "msa_sha256": {path: file_sha256(Path(path)) for path in resolved_msa_files},
        "input_sha256": file_sha256(input_path),
        "seed": seed,
        "gpu": gpu,
        "use_msa_server": use_msa_server,
        "max_parallel_samples": max_parallel_samples,
        "exit_code": exit_code,
        "error": exception_text,
        "model": "boltz2",
        "boltz_version": boltz_version(),
        "elapsed_seconds": round(time.time() - started, 2),
        "input_yaml": str(input_path),
        "output_dir": str(output_dir.resolve()),
        "log": str(log_path),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(status_path, status)
    return status


def run_target(
    context: dict,
    root: Path,
    run_name: str,
    water_counts: list[int],
    sample_count: int,
    target_index: int,
    gpu: int,
    cache: Path,
    max_parallel_samples: int = 1,
) -> list[dict]:
    outcomes = []
    msa_paths = {}
    sorted_counts = sorted(set(water_counts))
    first_count = sorted_counts[0]
    for water_count in sorted_counts:
        if water_count != first_count and msa_paths:
            write_input_yaml(
                context,
                water_count,
                root / "inputs" / context["pdb_id"] / f"water_{water_count:02d}.yaml",
                msa_paths,
            )
        use_msa_server = not bool(msa_paths)
        status = run_job(
            context=context,
            root=root,
            run_name=run_name,
            water_count=water_count,
            sample_count=sample_count,
                seed=SEED_BASE + target_index * 1000 + water_count,
            gpu=gpu,
            cache=cache,
            use_msa_server=use_msa_server,
            max_parallel_samples=max_parallel_samples,
        )
        outcomes.append(status)
        if water_count == first_count:
            expected_indices = list(range(sample_count))
            first_job_complete = (
                output_files_match_status(status, Path(status["output_dir"]), sample_count)
                and status.get("prediction_indices") == expected_indices
            )
            msa_paths = (
                find_msa_paths(Path(status["output_dir"]), context["protein_chains"])
                if first_job_complete
                else {}
            )
    return outcomes


def run_grid(root: Path, run_name: str, targets: list[str], counts: list[int], samples: int, gpus: list[int], cache: Path, max_parallel_samples: int = 1) -> list[dict]:
    counts = sorted(set(counts))
    if not targets or not counts or not gpus or any(count < 1 for count in counts):
        raise ValueError("At least one target, positive water count, and GPU are required")
    if samples < 1 or len(set(targets)) != len(targets) or len(set(gpus)) != len(gpus):
        raise ValueError("Samples, targets, and GPU assignments must be positive and unique")
    prepared_path = root / "prepared_manifest.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared["target_count_prepared"] != prepared["target_count_requested"]:
        raise RuntimeError("Prepared manifest is incomplete; rerun prepare.py before prediction")
    contexts = {
        item["pdb_id"]: json.loads((root / "metadata" / f"{item['pdb_id']}.json").read_text(encoding="utf-8"))
        for item in prepared["targets"]
    }
    unknown = sorted(set(targets) - set(contexts))
    if unknown:
        raise ValueError(f"Unprepared targets: {unknown}")
    persist_run_config(root, run_name, targets, counts, samples, gpus, cache)
    target_indices = {item["pdb_id"]: index for index, item in enumerate(prepared["targets"])}
    target_jobs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        for target_position, pdb_id in enumerate(targets):
            target_index = target_indices[pdb_id]
            gpu = gpus[target_position % len(gpus)]
            future = executor.submit(
                    run_target,
                    contexts[pdb_id],
                    root,
                    run_name,
                    counts,
                    samples,
                    target_index,
                    gpu,
                    cache,
                    max_parallel_samples,
                )
            target_jobs[future] = (target_index, pdb_id, gpu)
        outcomes = []
        for future in concurrent.futures.as_completed(target_jobs):
            target_index, pdb_id, gpu = target_jobs[future]
            try:
                outcomes.extend(future.result())
            except Exception as error:
                error_text = f"{type(error).__name__}: {error}"
                for water_count in counts:
                    status_path = root / "runs" / run_name / pdb_id / "status" / f"water_{water_count:02d}.json"
                    input_path = root / "inputs" / pdb_id / f"water_{water_count:02d}.yaml"
                    output_dir = root / "runs" / run_name / pdb_id / f"water_{water_count:02d}"
                    seed = SEED_BASE + target_index * 1000 + water_count
                    if status_is_complete(status_path, output_dir, input_path, samples, seed):
                        outcomes.append(json.loads(status_path.read_text(encoding="utf-8")))
                        continue
                    files = prediction_files(output_dir)
                    resolved_files = [str(path.resolve()) for path in files]
                    resolved_msa_files = sorted(set(find_msa_paths(output_dir, contexts[pdb_id]["protein_chains"]).values()))
                    status = {
                        "pdb_id": pdb_id,
                        "water_count": water_count,
                        "requested_samples": samples,
                        "observed_samples": len(files),
                        "prediction_indices": prediction_indices(output_dir),
                        "prediction_files": resolved_files,
                        "prediction_sha256": {str(path.resolve()): file_sha256(path) for path in files},
                        "msa_files": resolved_msa_files,
                        "msa_sha256": {path: file_sha256(Path(path)) for path in resolved_msa_files},
                        "input_sha256": file_sha256(input_path) if input_path.is_file() else None,
                        "seed": seed,
                        "gpu": gpu,
                        "exit_code": 1,
                        "model": "boltz2",
                        "boltz_version": boltz_version(),
                        "error": f"Target worker aborted: {error_text}",
                        "output_dir": str(output_dir.resolve()),
                        "finished_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    write_json(status_path, status)
                    outcomes.append(status)
    outcomes.sort(key=lambda item: (item["pdb_id"], item["water_count"]))
    status_directory = root / "runs" / run_name
    cumulative_statuses = {}
    for status_path in status_directory.glob("*/status/water_*.json"):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cumulative_statuses[(status.get("pdb_id"), status.get("water_count"))] = status
    for status in outcomes:
        cumulative_statuses[(status["pdb_id"], status["water_count"])] = status
    run_summary = sorted(cumulative_statuses.values(), key=lambda item: (item["pdb_id"], item["water_count"]))
    write_json(status_directory / "run_summary.json", run_summary)
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", default="production")
    parser.add_argument("--targets", default="all")
    parser.add_argument("--counts", default="1-10")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--max-parallel-samples", type=int, default=1)
    arguments = parser.parse_args()
    prepared = json.loads((arguments.root / "prepared_manifest.json").read_text(encoding="utf-8"))
    targets = [item["pdb_id"] for item in prepared["targets"]] if arguments.targets == "all" else arguments.targets.split(",")
    if "-" in arguments.counts:
        start, end = [int(value) for value in arguments.counts.split("-", 1)]
        counts = list(range(start, end + 1))
    else:
        counts = [int(value) for value in arguments.counts.split(",")]
    outcomes = run_grid(arguments.root, arguments.run_name, targets, counts, arguments.samples, [int(value) for value in arguments.gpus.split(",")], arguments.cache, arguments.max_parallel_samples)
    failed = [item for item in outcomes if not job_status_complete(item, item["requested_samples"])]
    print(json.dumps({"jobs": len(outcomes), "samples_requested": sum(item["requested_samples"] for item in outcomes), "samples_observed": sum(item["observed_samples"] for item in outcomes), "failed_jobs": len(failed)}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
