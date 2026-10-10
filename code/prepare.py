from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import yaml
from rdkit import Chem

from common import SEED_BASE, build_target_context, canonical_smiles_candidates, download, write_input_yaml, write_json
from metrics import component_atom_mapping


HERE = Path(__file__).resolve().parent


def download_target(entry: dict, manifest: dict, data_dir: Path) -> None:
    pdb_id = entry["pdb_id"].upper()
    ligand_ccd = entry["ligand_ccd"].upper()
    download(
        manifest["source_urls"]["structure_cif_template"].format(pdb_id=pdb_id),
        data_dir / "structures" / f"{pdb_id}.cif",
    )
    download(
        manifest["source_urls"]["component_cif_template"].format(ccd=ligand_ccd),
        data_dir / "components" / f"{ligand_ccd}.cif",
    )


def select_validated_ligand_smiles(context: dict) -> tuple[str, dict]:
    component_path = Path(context["component_cif"])
    errors = []
    for program, smiles in canonical_smiles_candidates(component_path, context["ligand_ccd"]):
        molecule = Chem.MolFromSmiles(smiles)
        try:
            atom_ids = component_atom_mapping(component_path, molecule)
        except ValueError as error:
            errors.append(f"{program}: {error}")
            continue
        context["ligand_smiles"] = smiles
        context["ligand_smiles_program"] = program
        context["ligand_atom_ids"] = atom_ids
        return program, atom_ids
    raise ValueError(f"No canonical CCD SMILES preserves ligand graph and stereochemistry: {'; '.join(errors)}")


def prepare(manifest_path: Path, root: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_dir = root / "source-data"
    input_dir = root / "inputs"
    metadata_dir = root / "metadata"
    contexts = []
    target_errors = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(download_target, entry, manifest, data_dir): entry
            for entry in manifest["targets"]
        }
        for future in concurrent.futures.as_completed(futures):
            entry = futures[future]
            try:
                future.result()
                print(f"downloaded {entry['pdb_id']} {entry['ligand_ccd']}", flush=True)
            except Exception as error:
                target_errors.append({"pdb_id": entry["pdb_id"], "error": f"download {type(error).__name__}: {error}"})

    for entry in manifest["targets"]:
        pdb_id = entry["pdb_id"].upper()
        if any(error["pdb_id"] == pdb_id for error in target_errors):
            continue
        try:
            context = build_target_context(entry, data_dir)
            select_validated_ligand_smiles(context)
            contexts.append(context)
            write_json(metadata_dir / f"{pdb_id}.json", context)
            for water_count in manifest["design"]["input_water_counts"]:
                write_input_yaml(
                    context,
                    water_count,
                    input_dir / pdb_id / f"water_{water_count:02d}.yaml",
                )
        except Exception as error:
            target_errors.append({"pdb_id": pdb_id, "error": f"{type(error).__name__}: {error}"})

    prepared = {
        "manifest": str(manifest_path),
        "root": str(root),
        "target_count_requested": len(manifest["targets"]),
        "target_count_prepared": len(contexts),
        "seed_base": SEED_BASE,
        "input_count": len(list(input_dir.glob("*/*.yaml"))),
        "targets": [
            {
                "pdb_id": context["pdb_id"],
                "ligand_ccd": context["ligand_ccd"],
                "ligand_heavy_atoms": len(context["ligand_instance"]["atoms"]),
                "protein_chains": len(context["protein_chains"]),
                "native_waters_within_6A": len(context["native_waters_6A"]),
                "ligand_smiles": context["ligand_smiles"],
                "ligand_smiles_program": context["ligand_smiles_program"],
                "ligand_instance": context["ligand_instance"],
            }
            for context in contexts
        ],
        "errors": target_errors,
    }
    write_json(root / "prepared_manifest.json", prepared)
    expected_inputs = len(manifest["targets"]) * len(manifest["design"]["input_water_counts"])
    if target_errors or len(contexts) != len(manifest["targets"]) or prepared["input_count"] != expected_inputs:
        raise RuntimeError(
            f"Preparation incomplete: {len(contexts)}/{len(manifest['targets'])} targets, "
            f"{prepared['input_count']}/{expected_inputs} inputs; see prepared_manifest.json"
        )
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    result = prepare(arguments.manifest, arguments.root)
    print(json.dumps({key: result[key] for key in ("target_count_prepared", "input_count", "errors")}, indent=2))


if __name__ == "__main__":
    main()
