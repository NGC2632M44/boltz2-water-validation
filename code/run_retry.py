from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from common import SEED_BASE, write_json
from run_grid import DEFAULT_CACHE, status_is_complete, run_target


def incomplete_counts(root: Path, run_name: str, pdb_id: str, target_index: int, counts: list[int], samples: int) -> list[int]:
    missing = []
    for water_count in counts:
        status_path = root / "runs" / run_name / pdb_id / "status" / f"water_{water_count:02d}.json"
        output_dir = root / "runs" / run_name / pdb_id / f"water_{water_count:02d}"
        input_path = root / "inputs" / pdb_id / f"water_{water_count:02d}.yaml"
        seed = SEED_BASE + target_index * 1000 + water_count
        if not status_is_complete(status_path, output_dir, input_path, samples, seed):
            missing.append(water_count)
    return missing


def assign_targets(incomplete: dict[str, list[int]], gpu_ids: list[int]) -> tuple[dict[int, list[str]], dict[int, int]]:
    loads = {gpu: 0 for gpu in gpu_ids}
    assignment = {gpu: [] for gpu in gpu_ids}
    for pdb_id in sorted(incomplete, key=lambda item: (-len(incomplete[item]), item)):
        gpu = min(gpu_ids, key=lambda candidate: (loads[candidate], candidate))
        assignment[gpu].append(pdb_id)
        loads[gpu] += len(incomplete[pdb_id])
    return assignment, loads


def run_gpu_queue(
    gpu: int,
    pdb_ids: list[str],
    contexts: dict[str, dict],
    root: Path,
    run_name: str,
    counts: list[int],
    sample_counts: dict[str, int],
    target_indices: dict[str, int],
    cache: Path,
    max_parallel_samples: int = 1,
) -> list[dict]:
    outcomes = []
    for pdb_id in pdb_ids:
        try:
            outcomes.extend(
                run_target(
                    contexts[pdb_id],
                    root,
                    run_name,
                    counts,
                    sample_counts[pdb_id],
                    target_indices[pdb_id],
                    gpu,
                    cache,
                    max_parallel_samples,
                )
            )
        except Exception as error:  # noqa: BLE001 - surface worker failure into the report
            outcomes.append(
                {
                    "pdb_id": pdb_id,
                    "error": f"Retry worker failure on gpu {gpu}: {type(error).__name__}: {error}",
                    "exit_code": 1,
                }
            )
    return outcomes


def rebuild_run_summary(root: Path, run_name: str) -> list[dict]:
    run_root = root / "runs" / run_name
    cumulative = {}
    for status_path in run_root.glob("*/status/water_*.json"):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cumulative[(status.get("pdb_id"), status.get("water_count"))] = status
    summary = sorted(cumulative.values(), key=lambda item: (item.get("pdb_id", ""), item.get("water_count", 0)))
    write_json(run_root / "run_summary.json", summary)
    return summary


def retry(root: Path, run_name: str, counts: list[int], samples: int, gpu_ids: list[int], cache: Path, dry_run: bool = False, max_parallel_samples: int = 1, sample_count_overrides: dict[str, int] | None = None) -> dict:
    overrides = sample_count_overrides or {}
    prepared = json.loads((root / "prepared_manifest.json").read_text(encoding="utf-8"))
    target_indices = {item["pdb_id"]: index for index, item in enumerate(prepared["targets"])}
    sample_counts = {
        item["pdb_id"]: int(overrides.get(item["pdb_id"], samples)) for item in prepared["targets"]
    }
    contexts = {
        item["pdb_id"]: json.loads((root / "metadata" / f"{item['pdb_id']}.json").read_text(encoding="utf-8"))
        for item in prepared["targets"]
    }
    incomplete = {
        item["pdb_id"]: missing
        for item in prepared["targets"]
        if (missing := incomplete_counts(root, run_name, item["pdb_id"], target_indices[item["pdb_id"]], counts, sample_counts[item["pdb_id"]]))
    }
    assignment, loads = assign_targets(incomplete, gpu_ids)
    plan = {
        "run_name": run_name,
        "incomplete_jobs": sum(len(value) for value in incomplete.values()),
        "incomplete_targets": {key: incomplete[key] for key in sorted(incomplete)},
        "target_sample_counts": sample_counts,
        "gpu_assignment": {str(gpu): assignment[gpu] for gpu in gpu_ids},
        "gpu_loads": {str(gpu): loads[gpu] for gpu in gpu_ids},
    }
    print(json.dumps(plan, indent=2), flush=True)
    if dry_run:
        return {**plan, "outcomes": 0, "dry_run": True}
    if not incomplete:
        rebuild_run_summary(root, run_name)
        return {**plan, "outcomes": 0}

    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {
            executor.submit(
                run_gpu_queue,
                gpu,
                assignment[gpu],
                contexts,
                root,
                run_name,
                counts,
                sample_counts,
                target_indices,
                cache,
                max_parallel_samples,
            ): gpu
            for gpu in gpu_ids
            if assignment[gpu]
        }
        for future in concurrent.futures.as_completed(futures):
            outcomes.extend(future.result())
    rebuild_run_summary(root, run_name)
    remaining = {
        item["pdb_id"]: incomplete_counts(root, run_name, item["pdb_id"], target_indices[item["pdb_id"]], counts, sample_counts[item["pdb_id"]])
        for item in prepared["targets"]
    }
    remaining = {key: value for key, value in remaining.items() if value}
    return {**plan, "outcomes": len(outcomes), "still_incomplete": remaining}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", default="production")
    parser.add_argument("--counts", default="1-10")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-parallel-samples", type=int, default=1)
    parser.add_argument(
        "--sample-count-overrides",
        default="",
        help="Comma-separated PDB:count pairs, e.g. '1FIN:16,2HYY:16'.",
    )
    arguments = parser.parse_args()
    overrides = {}
    for item in arguments.sample_count_overrides.split(","):
        item = item.strip()
        if not item:
            continue
        pdb_id, count = item.split(":", 1)
        overrides[pdb_id.strip().upper()] = int(count)
    if "-" in arguments.counts:
        start, end = [int(value) for value in arguments.counts.split("-", 1)]
        counts = list(range(start, end + 1))
    else:
        counts = [int(value) for value in arguments.counts.split(",")]
    result = retry(
        arguments.root,
        arguments.run_name,
        counts,
        arguments.samples,
        [int(value) for value in arguments.gpus.split(",")],
        arguments.cache,
        arguments.dry_run,
        arguments.max_parallel_samples,
        overrides,
    )
    print(json.dumps({key: result[key] for key in ("incomplete_jobs", "outcomes") if key in result}, indent=2))


if __name__ == "__main__":
    main()
