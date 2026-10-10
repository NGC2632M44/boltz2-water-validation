from __future__ import annotations

from pathlib import Path

import gemmi

from common import cif_rows, residue_heavy_atoms


_BOND_CACHE = {}

REFERENCE_PROTEIN_BASE = 1
PREDICTION_PROTEIN_BASE = 20000
REFERENCE_LIGAND_BASE = 90000
PREDICTION_LIGAND_BASE = 91000
REFERENCE_WATER_BASE = 95000
PREDICTION_WATER_BASE = 96000


def atom_line(record, serial, name, residue_name, chain_id, residue_number, xyz, element, occupancy=1.0, bfactor=0.0, insertion_code=""):
    element = element.upper()
    atom_name = name.strip()
    if len(atom_name) < 4 and len(element) == 1 and not atom_name[:1].isdigit():
        atom_field = f" {atom_name:<3}"
    else:
        atom_field = f"{atom_name:>4}"
    x_coord, y_coord, z_coord = xyz
    return (
        f"{record:<6}{serial:5d} {atom_field}{' ':1s}{residue_name[:3]:>3s} {chain_id:1s}"
        f"{residue_number:4d}{(insertion_code or ' '):1s}   "
        f"{x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}{occupancy:6.2f}{bfactor:6.2f}          {element:>2s}\n"
    )


def chain_by_name(model, name):
    return next((chain for chain in model if chain.name == name), None)


def export_pair(
    context: dict,
    reference_structure,
    prediction_structure,
    transform,
    component_molecule,
    component_atom_ids: dict,
    native_ligand_coordinates: dict,
    predicted_ligand_coordinates: dict,
    predicted_water_coordinates: list,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_model = reference_structure[0]
    prediction_model = prediction_structure[0]
    rotation, source_center, target_center = transform
    lines = [
        f"REMARK 900 {context['pdb_id']} {context['ligand_ccd']} Boltz-2 paired prediction (single MODEL)\n",
        "REMARK 900 Reference (crystal) = UPPERCASE chains: protein A,B,.. ; ligand L ; waters N\n",
        "REMARK 900 Boltz-2 prediction = lowercase chains: protein a,b,.. ; ligand l ; waters n\n",
        "REMARK 900 The prediction is rigid-body aligned to the reference by common protein C-alpha atoms\n",
        "REMARK 900 Native waters displayed within 10 A of the native ligand; scoring shell is 6 A\n",
        f"REMARK 900 CCD {context['ligand_ccd']} is represented by PDB residue name LIG\n",
    ]

    serial = REFERENCE_PROTEIN_BASE
    for protein in context["protein_chains"]:
        for chain in reference_model:
            for residue in chain:
                if str(residue.subchain) != protein["label_asym_id"]:
                    continue
                if str(residue.entity_id) not in {"", ".", "?", protein["entity_id"]}:
                    continue
                try:
                    if int(residue.label_seq) <= 0:
                        continue
                except (AttributeError, TypeError, ValueError):
                    continue
                for atom in residue_heavy_atoms(residue):
                    lines.append(
                        atom_line(
                            "ATOM",
                            serial,
                            atom.name,
                            residue.name,
                            protein["pdb_chain"],
                            int(residue.seqid.num),
                            [atom.pos.x, atom.pos.y, atom.pos.z],
                            atom.element.name,
                            float(atom.occ),
                            float(atom.b_iso),
                            str(residue.seqid.icode).strip("\x00 "),
                        )
                    )
                    serial += 1
        lines.append(f"TER   {serial:5d}      {protein['pdb_chain']:1s}\n")
        serial += 1

    component_index_to_name = {index: atom_name for atom_name, index in component_atom_ids.items()}
    for component_index, atom_name in component_index_to_name.items():
        if component_index not in native_ligand_coordinates:
            continue
        atom = component_molecule.GetAtomWithIdx(component_index)
        lines.append(
            atom_line(
                "HETATM",
                REFERENCE_LIGAND_BASE + component_index,
                atom_name,
                "LIG",
                "L",
                1,
                native_ligand_coordinates[component_index],
                atom.GetSymbol(),
            )
        )
    for water_index, coordinates in enumerate(context["display_waters_10A"], start=1):
        lines.append(atom_line("HETATM", REFERENCE_WATER_BASE + water_index, "O", "HOH", "N", water_index, coordinates, "O"))

    serial = PREDICTION_PROTEIN_BASE
    for protein in context["protein_chains"]:
        prediction_chain = chain_by_name(prediction_model, protein["input_id"])
        if prediction_chain is None:
            raise ValueError(f"Predicted protein chain {protein['input_id']} missing in {context['pdb_id']}")
        prediction_chain_id = protein["pdb_chain"].lower()
        for residue in prediction_chain:
            try:
                if int(residue.label_seq) <= 0:
                    continue
            except (AttributeError, TypeError, ValueError):
                continue
            for atom in residue_heavy_atoms(residue):
                source_xyz = [atom.pos.x, atom.pos.y, atom.pos.z]
                aligned_xyz = (source_xyz - source_center) @ rotation + target_center
                lines.append(
                    atom_line(
                        "ATOM",
                        serial,
                        atom.name,
                        residue.name,
                        prediction_chain_id,
                        int(residue.seqid.num),
                        aligned_xyz,
                        atom.element.name,
                        float(atom.occ),
                        float(atom.b_iso),
                        str(residue.seqid.icode).strip("\x00"),
                    )
                )
                serial += 1
        lines.append(f"TER   {serial:5d}      {prediction_chain_id:1s}\n")
        serial += 1

    for component_index, atom_name in component_index_to_name.items():
        if component_index not in predicted_ligand_coordinates:
            continue
        atom = component_molecule.GetAtomWithIdx(component_index)
        lines.append(
            atom_line(
                "HETATM",
                PREDICTION_LIGAND_BASE + component_index,
                atom_name,
                "LIG",
                "l",
                1,
                predicted_ligand_coordinates[component_index],
                atom.GetSymbol(),
            )
        )
    for water_index, coordinates in enumerate(predicted_water_coordinates, start=1):
        lines.append(atom_line("HETATM", PREDICTION_WATER_BASE + water_index, "O", "HOH", "n", water_index, coordinates, "O"))

    component_path = context["component_cif"]
    if component_path not in _BOND_CACHE:
        component_block = gemmi.cif.read_file(component_path).sole_block()
        pairs = []
        for atom_id_1, atom_id_2, _, _ in cif_rows(
            component_block,
            [
                "_chem_comp_bond.atom_id_1",
                "_chem_comp_bond.atom_id_2",
                "_chem_comp_bond.value_order",
                "_chem_comp_bond.pdbx_aromatic_flag",
            ],
        ):
            first_index = component_atom_ids.get(str(atom_id_1))
            second_index = component_atom_ids.get(str(atom_id_2))
            if first_index is not None and second_index is not None:
                pairs.append((first_index, second_index))
        _BOND_CACHE[component_path] = pairs
    for first_index, second_index in _BOND_CACHE[component_path]:
        lines.append(f"CONECT{REFERENCE_LIGAND_BASE + first_index:5d}{REFERENCE_LIGAND_BASE + second_index:5d}\n")
        lines.append(f"CONECT{PREDICTION_LIGAND_BASE + first_index:5d}{PREDICTION_LIGAND_BASE + second_index:5d}\n")
    lines.append("END\n")
    output_path.write_text("".join(lines), encoding="ascii", errors="replace")
