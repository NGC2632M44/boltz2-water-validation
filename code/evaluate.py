from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import gemmi
import numpy as np
from rdkit import Chem

from common import SEED_BASE, residue_heavy_atoms, write_json
from metrics import apply_transform, component_atom_mapping, kabsch_transform, predicted_atom_name_indices, symmetry_aware_mapping, water_match_metrics
from pdb_export import export_pair
from run_grid import output_files_match_status


def load_structure(path: Path):
    block = gemmi.cif.read_file(str(path)).sole_block()
    return block, gemmi.make_structure_from_block(block)


def residue_position(residue):
    try:
        position = int(residue.label_seq)
    except (AttributeError, TypeError, ValueError):
        position = 0
    if position > 0:
        return ("label", position)
    return ("auth", int(residue.seqid.num), str(residue.seqid.icode).strip("\x00"))


def ca_coordinates(model, protein_chains: list[dict], reference: bool) -> dict:
    coordinates = {}
    for protein in protein_chains:
        if reference:
            candidate_residues = [
                residue
                for chain in model
                for residue in chain
                if str(residue.subchain) == protein["label_asym_id"]
                and residue_position(residue)[0] == "label"
                and (str(residue.entity_id) in {"", ".", "?"} or str(residue.entity_id) == protein["entity_id"])
            ]
        else:
            chain = next((chain for chain in model if chain.name == protein["input_id"]), None)
            candidate_residues = list(chain) if chain else []
        for residue in candidate_residues:
            if residue_position(residue)[0] != "label":
                continue
            ca_atoms = [
                atom
                for atom in residue
                if atom.name.strip().upper() == "CA" and atom.element.name.upper() == "C"
            ]
            if ca_atoms:
                ca_atom = max(ca_atoms, key=lambda atom: atom.occ)
                coordinates[(protein["input_id"], residue_position(residue))] = [
                    float(ca_atom.pos.x), float(ca_atom.pos.y), float(ca_atom.pos.z)
                ]
    return coordinates


def protein_transform(reference_model, prediction_model, protein_chains: list[dict]):
    native_ca = ca_coordinates(reference_model, protein_chains, True)
    predicted_ca = ca_coordinates(prediction_model, protein_chains, False)
    common_keys = sorted(set(native_ca) & set(predicted_ca))
    if len(common_keys) < 3:
        raise ValueError(f"Only {len(common_keys)} common protein C-alpha atoms for superposition")
    source = [predicted_ca[key] for key in common_keys]
    target = [native_ca[key] for key in common_keys]
    return kabsch_transform(source, target), len(common_keys)


def find_reference_ligand(model, context: dict):
    instance = context["ligand_instance"]
    candidates = [
        residue
        for chain in model
        for residue in chain
        if residue.name.upper() == context["ligand_ccd"]
        and str(residue.subchain) == instance["label_asym_id"]
        and chain.name == instance["auth_asym_id"]
        and (not instance["entity_id"] or str(residue.entity_id) == instance["entity_id"])
        and str(residue.seqid.icode).strip("\x00 ") == instance["insertion_code"]
    ]
    exact = [residue for residue in candidates if int(residue.seqid.num) == instance["auth_seq_id"]]
    if len(exact) != 1:
        raise ValueError(f"Expected one exact reference ligand {context['ligand_ccd']} instance; found {len(exact)}")
    return exact[0]


def find_prediction_ligand(model, context: dict, atom_count: int):
    candidates = []
    for chain in model:
        for residue in chain:
            atoms = residue_heavy_atoms(residue)
            if residue.name.upper() == "LIG" and len(atoms) == atom_count:
                candidates.append((chain, residue, atoms))
    if not candidates:
        for chain in model:
            if chain.name != "LIG":
                continue
            for residue in chain:
                atoms = residue_heavy_atoms(residue)
                if len(atoms) == atom_count:
                    candidates.append((chain, residue, atoms))
    if len(candidates) != 1:
        raise ValueError(f"Expected one predicted ligand of {atom_count} heavy atoms; found {len(candidates)}")
    return candidates[0]


def predicted_water_coordinates(model) -> list[list[float]]:
    coordinates = []
    for chain in model:
        for residue in chain:
            if residue.name.upper() not in {"HOH", "WAT", "DOD"}:
                continue
            for atom in residue:
                if atom.element.name.upper() == "O":
                    coordinates.append([float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)])
    return coordinates


def load_native_ligand(context: dict, reference_model, input_molecule, component_atom_ids: dict):
    residue = find_reference_ligand(reference_model, context)
    coordinates = {}
    for atom in residue_heavy_atoms(residue):
        atom_index = component_atom_ids.get(atom.name)
        if atom_index is not None:
            coordinates[atom_index] = [float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)]
    if len(coordinates) != input_molecule.GetNumAtoms():
        missing = input_molecule.GetNumAtoms() - len(coordinates)
        raise ValueError(f"Reference ligand atom-name mapping incomplete: {missing} CCD atoms missing")
    return coordinates


def evaluate_sample(
    context: dict,
    prediction_path: Path,
    pair_path: Path | None,
    reference_structure,
    input_molecule,
    component_atom_ids: dict,
    name_indices: dict,
) -> dict:
    _, prediction_structure = load_structure(prediction_path)
    reference_model = reference_structure[0]
    prediction_model = prediction_structure[0]
    transform, aligned_ca_count = protein_transform(reference_model, prediction_model, context["protein_chains"])
    native_coordinates = load_native_ligand(context, reference_model, input_molecule, component_atom_ids)
    _, _, predicted_atoms = find_prediction_ligand(prediction_model, context, input_molecule.GetNumAtoms())
    predicted_input_coordinates = {}
    for atom in predicted_atoms:
        atom_index = name_indices.get(atom.name.strip())
        if atom_index is not None:
            predicted_input_coordinates[atom_index] = [float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)]
    if len(predicted_input_coordinates) != input_molecule.GetNumAtoms():
        raise ValueError(
            f"Predicted ligand atom-name mapping incomplete: {len(predicted_input_coordinates)}/"
            f"{input_molecule.GetNumAtoms()} atoms"
        )
    aligned_input_coordinates = {
        atom_index: apply_transform([coordinates], transform)[0].tolist()
        for atom_index, coordinates in predicted_input_coordinates.items()
    }
    ligand_rmsd, input_to_reference = symmetry_aware_mapping(
        input_molecule,
        input_molecule,
        aligned_input_coordinates,
        native_coordinates,
    )
    predicted_component_coordinates = {
        input_to_reference[input_index]: coordinates
        for input_index, coordinates in aligned_input_coordinates.items()
    }
    raw_water_coordinates = predicted_water_coordinates(prediction_model)
    aligned_water_coordinates = apply_transform(raw_water_coordinates, transform).tolist() if raw_water_coordinates else []
    water_metrics = water_match_metrics(aligned_water_coordinates, context["native_waters_6A"])
    if pair_path is not None:
        export_pair(
            context,
            reference_structure,
            prediction_structure,
            transform,
            input_molecule,
            component_atom_ids,
            native_coordinates,
            predicted_component_coordinates,
            aligned_water_coordinates,
            pair_path,
        )
    return {
        "status": "ok",
        "error": "",
        "ligand_rmsd_A": ligand_rmsd,
        "aligned_protein_CA": aligned_ca_count,
        "native_waters_6A": len(context["native_waters_6A"]),
        "paired_pdb": str(pair_path) if pair_path is not None else None,
        **water_metrics,
    }


def component_atom_mapping_from_context(context: dict, input_molecule):
    return component_atom_mapping(Path(context["component_cif"]), input_molecule)


def prediction_index(path: Path):
    match = re.search(r"_model_(\d+)\.cif$", path.name)
    return int(match.group(1)) if match else None


def run_evaluation(root: Path, run_name: str, targets: list[str], counts: list[int], samples: int) -> list[dict]:
    prepared = json.loads((root / "prepared_manifest.json").read_text(encoding="utf-8"))
    target_indices = {item["pdb_id"]: index for index, item in enumerate(prepared["targets"])}
    contexts = {
        item["pdb_id"]: json.loads((root / "metadata" / f"{item['pdb_id']}.json").read_text(encoding="utf-8"))
        for item in prepared["targets"]
    }
    rows = []
    for pdb_id in targets:
        context = contexts[pdb_id]
        _, reference_structure = load_structure(Path(context["structure_cif"]))
        input_molecule = Chem.MolFromSmiles(context["ligand_smiles"])
        component_atom_ids = component_atom_mapping_from_context(context, input_molecule)
        name_indices = predicted_atom_name_indices(context["ligand_smiles"])
        for water_count in counts:
            run_dir = root / "runs" / run_name / pdb_id / f"water_{water_count:02d}"
            paths = {}
            prediction_paths = []
            duplicate_indices = set()
            for path in run_dir.rglob("*_model_*.cif"):
                index = prediction_index(path)
                if index is None:
                    continue
                prediction_paths.append(path)
                if index in paths:
                    duplicate_indices.add(index)
                paths[index] = path
            status_path = root / "runs" / run_name / pdb_id / "status" / f"water_{water_count:02d}.json"
            job_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            seed = job_status.get("seed")
            job_samples = job_status.get("requested_samples") or samples
            expected_indices = list(range(job_samples))
            expected_seed = SEED_BASE + target_indices[pdb_id] * 1000 + water_count
            input_path = root / "inputs" / pdb_id / f"water_{water_count:02d}.yaml"
            expected_input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest() if input_path.is_file() else None
            job_healthy = (
                job_status.get("exit_code") == 0
                and job_status.get("requested_samples") == job_samples
                and job_status.get("observed_samples") == job_samples
                and job_status.get("seed") == expected_seed
                and job_status.get("input_sha256") == expected_input_sha256
                and output_files_match_status(job_status, run_dir, job_samples)
                and job_status.get("prediction_indices") == expected_indices
                and len(prediction_paths) == job_samples
                and sorted(paths) == expected_indices
                and not duplicate_indices
            )
            for sample_index in range(job_samples):
                base = {
                    "pdb_id": pdb_id,
                    "ligand_ccd": context["ligand_ccd"],
                    "water_count": water_count,
                    "replicate": sample_index + 1,
                    "seed": seed,
                    "expected_seed": expected_seed,
                    "job_exit_code": job_status.get("exit_code"),
                    "job_observed_samples": job_status.get("observed_samples"),
                    "job_complete": job_healthy,
                    "status": "missing",
                    "error": (
                        f"No sample file; Boltz exit={job_status.get('exit_code')}"
                        if sample_index not in paths
                        else ""
                    ),
                    "ligand_rmsd_A": None,
                    "aligned_protein_CA": None,
                    "native_waters_6A": len(context["native_waters_6A"]),
                    "predicted_count": None,
                    "native_count": len(context["native_waters_6A"]),
                    "paired_pdb": None,
                    "prediction_cif": None,
                }
                prediction_path = paths.get(sample_index)
                pair_path = root / "pairs" / run_name / pdb_id / f"water_{water_count:02d}" / f"rep_{sample_index + 1:03d}.pdb"
                pair_path.unlink(missing_ok=True)
                if prediction_path is not None:
                    if sample_index in duplicate_indices:
                        base.update({"status": "error", "error": "Duplicate model index in run output", "prediction_cif": str(prediction_path)})
                    else:
                        try:
                            result = evaluate_sample(
                                context,
                                prediction_path,
                                pair_path if job_healthy else None,
                                reference_structure,
                                input_molecule,
                                component_atom_ids,
                                name_indices,
                            )
                            base.update(result)
                            base["prediction_cif"] = str(prediction_path)
                            if not job_healthy:
                                base.update({"status": "partial", "error": "Boltz job did not produce the exact requested model-index set with exit code 0"})
                        except Exception as error:
                            base.update({"status": "error", "error": f"{type(error).__name__}: {error}", "prediction_cif": str(prediction_path)})
                rows.append(base)
    columns = [
        "pdb_id", "ligand_ccd", "water_count", "replicate", "seed", "expected_seed", "job_exit_code", "job_observed_samples", "job_complete", "status", "error",
        "ligand_rmsd_A", "aligned_protein_CA", "native_waters_6A", "predicted_count", "native_count",
        "matched_1p0A", "precision_1p0A", "recall_1p0A", "f1_1p0A", "mean_match_distance_1p0A",
        "matched_1p5A", "precision_1p5A", "recall_1p5A", "f1_1p5A", "mean_match_distance_1p5A",
        "matched_2p0A", "precision_2p0A", "recall_2p0A", "f1_2p0A", "mean_match_distance_2p0A",
        "mean_one_to_one_distance_A", "paired_pdb", "prediction_cif",
    ]
    metrics_path = root / "metrics" / run_name / "sample_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        root / "metrics" / run_name / "evaluation_summary.json",
        {
            "rows": len(rows),
            "ok": sum(row["status"] == "ok" for row in rows),
            "partial": sum(row["status"] == "partial" for row in rows),
            "errors": sum(row["status"] == "error" for row in rows),
            "missing": sum(row["status"] == "missing" for row in rows),
            "failed_total": sum(row["status"] != "ok" for row in rows),
            "metrics_csv": str(metrics_path),
        },
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", default="production")
    parser.add_argument("--targets", default="all")
    parser.add_argument("--counts", default="1-10")
    parser.add_argument("--samples", type=int, default=30)
    arguments = parser.parse_args()
    prepared = json.loads((arguments.root / "prepared_manifest.json").read_text(encoding="utf-8"))
    targets = [item["pdb_id"] for item in prepared["targets"]] if arguments.targets == "all" else arguments.targets.split(",")
    if "-" in arguments.counts:
        start, end = [int(value) for value in arguments.counts.split("-", 1)]
        counts = list(range(start, end + 1))
    else:
        counts = [int(value) for value in arguments.counts.split(",")]
    rows = run_evaluation(arguments.root, arguments.run_name, targets, counts, arguments.samples)
    print(json.dumps({"rows": len(rows), "ok": sum(row["status"] == "ok" for row in rows), "failed": sum(row["status"] != "ok" for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
