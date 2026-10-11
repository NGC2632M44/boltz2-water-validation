from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from common import SEED_BASE, write_input_yaml
from run_grid import DEFAULT_CACHE, find_msa_paths, run_job, status_is_complete

HEAVY_TARGETS = {"1FIN", "2HYY", "3CS9"}


def prepare_input(root: Path, context: dict) -> bool:
    """Write the 0-water input YAML, reusing the MSA from the production water_01 run."""
    pdb_id = context["pdb_id"]
    msa_paths = {}
    production_output = root / "runs" / "production" / pdb_id / "water_01"
    if production_output.is_dir():
        msa_paths = find_msa_paths(production_output, context["protein_chains"])
    write_input_yaml(context, 0, root / "inputs" / pdb_id / "water_00.yaml", msa_paths or None)
    return bool(msa_paths)


def process_target(root: Path, context: dict, position: int, run_name: str, default_samples: int, heavy_samples: int, gpu: int, cache: Path) -> dict:
    pdb_id = context["pdb_id"]
    sample_count = heavy_samples if pdb_id in HEAVY_TARGETS else default_samples
    seed = SEED_BASE + position * 1000 + 0
    status_path = root / "runs" / run_name / pdb_id / "status" / "water_00.json"
    output_dir = root / "runs" / run_name / pdb_id / "water_00"
    input_path = root / "inputs" / pdb_id / "water_00.yaml"
    if status_is_complete(status_path, output_dir, input_path, sample_count, seed):
        print(f"{pdb_id} skip (complete)", flush=True)
        return json.loads(status_path.read_text(encoding="utf-8"))
    return run_job(
        context=context,
        root=root,
        run_name=run_name,
        water_count=0,
        sample_count=sample_count,
        seed=seed,
        gpu=gpu,
        cache=cache,
        use_msa_server=False,
        max_parallel_samples=1,
    )


def run_gpu_queue(gpu: int, items: list[tuple[int, str]], contexts: dict, root: Path, run_name: str, default_samples: int, heavy_samples: int, cache: Path) -> list[dict]:
    outcomes = []
    for position, pdb_id in items:
        try:
            status = process_target(root, contexts[pdb_id], position, run_name, default_samples, heavy_samples, gpu, cache)
            print(f"{pdb_id} gpu{gpu} exit={status.get('exit_code')} observed={status.get('observed_samples')}/{status.get('requested_samples')}", flush=True)
            outcomes.append(status)
        except Exception as error:  # noqa: BLE001
            print(f"{pdb_id} gpu{gpu} ERROR {type(error).__name__}: {error}", flush=True)
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 0-water (no explicit water) co-folding control, one job per GPU at a time.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", default="baseline0")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--heavy-samples", type=int, default=16)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    arguments = parser.parse_args()

    root = arguments.root
    gpus = [int(value) for value in arguments.gpus.split(",")]
    prepared = json.loads((root / "prepared_manifest.json").read_text(encoding="utf-8"))
    contexts = {
        item["pdb_id"]: json.loads((root / "metadata" / f"{item['pdb_id']}.json").read_text(encoding="utf-8"))
        for item in prepared["targets"]
    }

    for item in prepared["targets"]:
        prepare_input(root, contexts[item["pdb_id"]])
    print(f"targets={len(contexts)} run={arguments.run_name} gpus={gpus}", flush=True)

    assignment = {gpu: [] for gpu in gpus}
    for position, item in enumerate(prepared["targets"]):
        assignment[gpus[position % len(gpus)]].append((position, item["pdb_id"]))

    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(run_gpu_queue, gpu, assignment[gpu], contexts, root, arguments.run_name, arguments.samples, arguments.heavy_samples, arguments.cache): gpu
            for gpu in gpus
            if assignment[gpu]
        }
        for future in concurrent.futures.as_completed(futures):
            outcomes.extend(future.result())

    run_root = root / "runs" / arguments.run_name
    summary = sorted((json.loads(p.read_text(encoding="utf-8")) for p in run_root.glob("*/status/water_00.json")), key=lambda item: item["pdb_id"])
    (run_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    incomplete = [item["pdb_id"] for item in summary if item.get("observed_samples") != item.get("requested_samples")]
    print(json.dumps({"jobs": len(summary), "samples": sum(item.get("observed_samples", 0) for item in summary), "incomplete": incomplete}, indent=2))


if __name__ == "__main__":
    main()
