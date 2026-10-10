from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import gemmi
import numpy as np
import yaml
from rdkit import Chem


PROTEIN_PDB_CHAINS = [letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if letter not in {"L", "N", "W"}]
SEED_BASE = 20261009


def cif_rows(block, tags):
    def unquote(value):
        text = str(value)
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", "\""}:
            return text[1:-1]
        return text

    return [[unquote(value) for value in row] for row in block.find(tags)]


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "Boltz2-water-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"Empty download: {url}")
    path.write_bytes(payload)


def canonical_smiles_candidates(component_cif: Path, component_id: str) -> list[tuple[str, str]]:
    block = gemmi.cif.read_file(str(component_cif)).sole_block()
    descriptor_tags = [
        "_pdbx_chem_comp_descriptor.comp_id",
        "_pdbx_chem_comp_descriptor.type",
        "_pdbx_chem_comp_descriptor.program",
        "_pdbx_chem_comp_descriptor.descriptor",
    ]
    candidates = []
    for row in cif_rows(block, descriptor_tags):
        comp_id, descriptor_type, program, descriptor = row
        descriptor_type = descriptor_type.upper().replace(" ", "_")
        if comp_id != component_id or descriptor_type != "SMILES_CANONICAL":
            continue
        if descriptor in {"", ".", "?"}:
            continue
        program = str(program)
        priority = "CACTVS" in program.upper()
        candidates.append((priority, program, descriptor))
    if not candidates:
        raise ValueError(f"No canonical RCSB SMILES descriptor for {component_id}")
    parsed = []
    for _, program, smiles in sorted(candidates, key=lambda item: item[0], reverse=True):
        if Chem.MolFromSmiles(smiles) is not None:
            parsed.append((program, smiles))
    if not parsed:
        raise ValueError(f"RDKit cannot parse any canonical RCSB SMILES for {component_id}")
    return parsed


def canonical_smiles(component_cif: Path, component_id: str) -> str:
    return canonical_smiles_candidates(component_cif, component_id)[0][1]


def residue_heavy_atoms(residue):
    best_by_name = {}
    for atom in residue:
        if atom.element.name.upper() in {"H", "D"}:
            continue
        previous = best_by_name.get(atom.name)
        if previous is None or atom.occ > previous.occ:
            best_by_name[atom.name] = atom
    return list(best_by_name.values())


def atom_coordinates(atom) -> list[float]:
    return [float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)]


def clean_sequence(sequence: str) -> str:
    return "".join(sequence.replace(";", "").split()).strip("\"'").upper()


def build_target_context(entry: dict, data_dir: Path) -> dict:
    pdb_id = entry["pdb_id"].upper()
    ligand_ccd = entry["ligand_ccd"].upper()
    structure_path = data_dir / "structures" / f"{pdb_id}.cif"
    component_path = data_dir / "components" / f"{ligand_ccd}.cif"
    structure_block = gemmi.cif.read_file(str(structure_path)).sole_block()
    structure = gemmi.make_structure_from_block(structure_block)
    model = structure[0]

    poly_rows = cif_rows(
        structure_block,
        ["_entity_poly.entity_id", "_entity_poly.type", "_entity_poly.pdbx_seq_one_letter_code_can"],
    )
    sequences = {
        str(entity_id): clean_sequence(sequence)
        for entity_id, poly_type, sequence in poly_rows
        if str(poly_type).strip("\"'").startswith("polypeptide")
    }
    asym_entity = {
        str(asym_id): str(entity_id)
        for asym_id, entity_id in cif_rows(structure_block, ["_struct_asym.id", "_struct_asym.entity_id"])
    }
    if not sequences:
        raise ValueError(f"No protein polymer sequence found in {pdb_id}")

    protein_groups = {}
    for chain in model:
        for residue in chain:
            subchain = str(residue.subchain)
            entity_id = str(residue.entity_id or "").strip()
            if entity_id in {"", ".", "?"}:
                entity_id = asym_entity.get(subchain, "")
            if entity_id not in sequences:
                continue
            key = (subchain, entity_id, chain.name)
            protein_groups.setdefault(key, []).append(residue)

    ligand_candidates = []
    for chain in model:
        for residue in chain:
            if residue.name.upper() == ligand_ccd and residue_heavy_atoms(residue):
                ligand_candidates.append((chain.name, residue))
    if not ligand_candidates:
        raise ValueError(f"Ligand {ligand_ccd} not found in {pdb_id}")

    ligand_chain_name, ligand_residue = min(
        ligand_candidates,
        key=lambda item: (
            item[0],
            str(item[1].subchain),
            str(item[1].entity_id),
            int(item[1].seqid.num),
            str(item[1].seqid.icode).strip("\x00 "),
        ),
    )
    ligand_atoms = residue_heavy_atoms(ligand_residue)
    ligand_xyz = np.asarray([atom_coordinates(atom) for atom in ligand_atoms], dtype=float)

    selected = sorted(protein_groups, key=lambda item: (item[0], item[2], item[1]))
    if len(selected) > len(PROTEIN_PDB_CHAINS):
        raise ValueError(f"Too many protein chains to preserve in paired PDB output for {pdb_id}: {len(selected)}")

    protein_chains = []
    for index, (subchain, entity_id, auth_chain) in enumerate(selected):
        protein_chains.append(
            {
                "input_id": chr(ord("A") + index),
                "pdb_chain": PROTEIN_PDB_CHAINS[index],
                "label_asym_id": subchain,
                "auth_asym_id": auth_chain,
                "entity_id": entity_id,
                "sequence": sequences[entity_id],
            }
        )
    if not protein_chains:
        raise ValueError(f"No protein chains found in {pdb_id}")

    waters_by_identity = {}
    for chain in model:
        for residue in chain:
            if residue.name.upper() not in {"HOH", "WAT", "DOD"}:
                continue
            oxygen_atoms = [atom for atom in residue_heavy_atoms(residue) if atom.element.name.upper() == "O"]
            if oxygen_atoms:
                water_key = (
                    chain.name,
                    str(residue.subchain),
                    int(residue.seqid.num),
                    str(residue.seqid.icode).strip("\x00 "),
                    residue.name.upper(),
                )
                oxygen = max(oxygen_atoms, key=lambda atom: atom.occ)
                previous = waters_by_identity.get(water_key)
                if previous is None or oxygen.occ > previous[0]:
                    waters_by_identity[water_key] = (float(oxygen.occ), atom_coordinates(oxygen))
    all_water_xyz = [value[1] for _, value in sorted(waters_by_identity.items())]
    if all_water_xyz:
        water_distances = np.linalg.norm(np.asarray(all_water_xyz)[:, None, :] - ligand_xyz[None, :, :], axis=2).min(axis=1)
        native_waters_6a = [point for point, distance in zip(all_water_xyz, water_distances) if distance <= 6.0]
        display_waters_10a = [point for point, distance in zip(all_water_xyz, water_distances) if distance <= 10.0]
    else:
        native_waters_6a = []
        display_waters_10a = []
    if not native_waters_6a:
        raise ValueError(f"No crystallographic water within 6 A of {pdb_id}:{ligand_ccd}")

    smiles = canonical_smiles(component_path, ligand_ccd)
    return {
        "pdb_id": pdb_id,
        "ligand_ccd": ligand_ccd,
        "ligand_smiles": smiles,
        "structure_cif": str(structure_path),
        "component_cif": str(component_path),
        "ligand_instance": {
            "auth_asym_id": ligand_chain_name,
            "label_asym_id": str(ligand_residue.subchain),
            "entity_id": str(ligand_residue.entity_id),
            "auth_seq_id": int(ligand_residue.seqid.num),
            "insertion_code": str(ligand_residue.seqid.icode).strip("\x00 "),
            "atoms": [
                {"name": atom.name, "element": atom.element.name.upper(), "xyz": atom_coordinates(atom)}
                for atom in ligand_atoms
            ],
        },
        "protein_chains": protein_chains,
        "native_waters_6A": native_waters_6a,
        "display_waters_10A": display_waters_10a,
    }


def write_input_yaml(context: dict, water_count: int, path: Path, msa_paths: dict | None = None) -> None:
    entities = []
    for protein in context["protein_chains"]:
        payload = {"id": protein["input_id"], "sequence": protein["sequence"]}
        if msa_paths and protein["input_id"] in msa_paths:
            payload["msa"] = str(msa_paths[protein["input_id"]])
        entities.append({"protein": payload})
    entities.append({"ligand": {"id": "LIG", "smiles": context["ligand_smiles"]}})
    for water_index in range(1, water_count + 1):
        entities.append({"ligand": {"id": f"W{water_index:02d}", "ccd": "HOH"}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"version": 1, "sequences": entities}, sort_keys=False), encoding="utf-8")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
