from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import linear_sum_assignment

from common import cif_rows


def kabsch_transform(source, target):
    source_array = np.asarray(source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if source_array.shape != target_array.shape or source_array.ndim != 2 or source_array.shape[1] != 3:
        raise ValueError("Kabsch inputs must be matching N×3 arrays")
    if source_array.shape[0] < 3:
        raise ValueError("At least three common Cα coordinates are required")
    source_center = source_array.mean(axis=0)
    target_center = target_array.mean(axis=0)
    covariance = (source_array - source_center).T @ (target_array - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right_t)
    rotation = left @ correction @ right_t
    return rotation, source_center, target_center


def apply_transform(coordinates, transform):
    rotation, source_center, target_center = transform
    return (np.asarray(coordinates, dtype=float) - source_center) @ rotation + target_center


def water_match_metrics(predicted, native) -> dict:
    predicted_array = np.asarray(predicted, dtype=float).reshape((-1, 3))
    native_array = np.asarray(native, dtype=float).reshape((-1, 3))
    if predicted_array.size and native_array.size:
        distances = np.linalg.norm(predicted_array[:, None, :] - native_array[None, :, :], axis=2)
    else:
        distances = np.empty((len(predicted_array), len(native_array)), dtype=float)
    result = {"predicted_count": len(predicted_array), "native_count": len(native_array)}
    for threshold in (1.0, 1.5, 2.0):
        if distances.size:
            assignment_cost = np.where(distances <= threshold, distances, 1.0e6)
            row_indices, column_indices = linear_sum_assignment(assignment_cost)
            matched_distances = [
                float(distances[row_index, column_index])
                for row_index, column_index in zip(row_indices, column_indices)
                if distances[row_index, column_index] <= threshold
            ]
        else:
            matched_distances = []
        matched = len(matched_distances)
        precision = matched / len(predicted_array) if len(predicted_array) else 0.0
        recall = matched / len(native_array) if len(native_array) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        suffix = str(threshold).replace(".", "p")
        result[f"matched_{suffix}A"] = matched
        result[f"precision_{suffix}A"] = precision
        result[f"recall_{suffix}A"] = recall
        result[f"f1_{suffix}A"] = f1
        result[f"mean_match_distance_{suffix}A"] = float(np.mean(matched_distances)) if matched_distances else None
    if distances.size:
        row_indices, column_indices = linear_sum_assignment(distances)
        result["mean_one_to_one_distance_A"] = float(distances[row_indices, column_indices].mean())
    else:
        result["mean_one_to_one_distance_A"] = None
    return result


def component_atom_mapping(component_cif, input_molecule):
    import gemmi
    import networkx as nx

    if input_molecule is None:
        raise ValueError("Input ligand SMILES did not produce a molecule")
    block = gemmi.cif.read_file(str(component_cif)).sole_block()
    atom_tags = [
        "_chem_comp_atom.atom_id",
        "_chem_comp_atom.type_symbol",
        "_chem_comp_atom.charge",
        "_chem_comp_atom.pdbx_stereo_config",
    ]
    atom_ids = {}
    stereo_by_id = {}
    component_builder = Chem.RWMol()
    for atom_id, element, charge, stereo in cif_rows(block, atom_tags):
        element = str(element).upper()
        if element in {"H", "D"}:
            continue
        atom_id = str(atom_id)
        if atom_id in atom_ids:
            continue
        atom = Chem.Atom(element.capitalize())
        charge = str(charge).strip()
        if charge in {"", ".", "?"}:
            raise ValueError(f"CCD formal charge is unknown for atom {atom_id} in {component_cif}")
        try:
            atom.SetFormalCharge(int(charge))
        except (TypeError, ValueError) as error:
            raise ValueError(f"CCD formal charge is invalid for atom {atom_id} in {component_cif}: {charge}") from error
        atom_ids[atom_id] = component_builder.AddAtom(atom)
        stereo_by_id[atom_id] = str(stereo).upper()
    bond_rows = cif_rows(
        block,
        [
            "_chem_comp_bond.atom_id_1",
            "_chem_comp_bond.atom_id_2",
            "_chem_comp_bond.value_order",
            "_chem_comp_bond.pdbx_aromatic_flag",
            "_chem_comp_bond.pdbx_stereo_config",
        ],
    )
    bond_order_map = {
        "SING": Chem.BondType.SINGLE,
        "DOUB": Chem.BondType.DOUBLE,
        "TRIP": Chem.BondType.TRIPLE,
        "QUAD": Chem.BondType.QUADRUPLE,
        "AROM": Chem.BondType.AROMATIC,
        "DELO": Chem.BondType.AROMATIC,
    }
    aromatic_atom_indices = set()
    ccd_bond_stereo = {}
    for first_id, second_id, order, aromatic, stereo in bond_rows:
        first_id = str(first_id)
        second_id = str(second_id)
        if first_id in atom_ids and second_id in atom_ids:
            order = str(order).upper()
            if order not in bond_order_map:
                raise ValueError(f"Unsupported CCD bond order {order} in {component_cif}")
            component_builder.AddBond(atom_ids[first_id], atom_ids[second_id], bond_order_map[order])
            if order in {"AROM", "DELO"}:
                aromatic_atom_indices.update((atom_ids[first_id], atom_ids[second_id]))
            stereo = str(stereo).upper()
            if stereo in {"E", "Z"}:
                ccd_bond_stereo[frozenset((first_id, second_id))] = stereo
    for atom_index in aromatic_atom_indices:
        component_builder.GetAtomWithIdx(atom_index).SetIsAromatic(True)
    try:
        component_molecule = component_builder.GetMol()
        Chem.SanitizeMol(component_molecule)
    except Exception as error:
        raise ValueError(f"CCD graph cannot be chemically normalized for {component_cif}: {error}") from error
    Chem.AssignStereochemistry(input_molecule, cleanIt=True, force=True)

    def molecule_graph(molecule):
        graph = nx.Graph()
        for atom in molecule.GetAtoms():
            graph.add_node(
                atom.GetIdx(),
                element=atom.GetSymbol().upper(),
                charge=atom.GetFormalCharge(),
            )
        for bond in molecule.GetBonds():
            if bond.GetIsAromatic():
                bond_type = "AROMATIC"
            else:
                bond_type = {
                    1.0: "SINGLE",
                    2.0: "DOUBLE",
                    3.0: "TRIPLE",
                    4.0: "QUADRUPLE",
                }.get(float(bond.GetBondTypeAsDouble()), "OTHER")
            graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond=bond_type)
        return graph

    input_graph = molecule_graph(input_molecule)
    component_graph = molecule_graph(component_molecule)
    if input_graph.number_of_nodes() != component_graph.number_of_nodes():
        raise ValueError(
            f"{str(component_cif)} has {component_graph.number_of_nodes()} unique heavy atoms; "
            f"canonical SMILES has {input_graph.number_of_nodes()}"
        )
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        input_graph,
        component_graph,
        node_match=lambda input_node, ccd_node: input_node == ccd_node,
        edge_match=lambda input_edge, ccd_edge: input_edge["bond"] == ccd_edge["bond"],
    )
    component_index_to_name = {index: atom_id for atom_id, index in atom_ids.items()}

    def input_bond_stereo(bond):
        return {
            Chem.BondStereo.STEREOE: "E",
            Chem.BondStereo.STEREOZ: "Z",
        }.get(bond.GetStereo())

    for mapping in matcher.isomorphisms_iter():
        component_to_input = {component_index: input_index for input_index, component_index in mapping.items()}
        stereo_consistent = True
        for input_index, input_atom in enumerate(input_molecule.GetAtoms()):
            ccd_atom_name = component_index_to_name[mapping[input_index]]
            ccd_stereo = stereo_by_id.get(ccd_atom_name, "N")
            input_stereo = input_atom.GetProp("_CIPCode") if input_atom.HasProp("_CIPCode") else None
            if (ccd_stereo in {"R", "S"}) != (input_stereo in {"R", "S"}) or ccd_stereo in {"R", "S"} and ccd_stereo != input_stereo:
                stereo_consistent = False
                break
        if not stereo_consistent:
            continue
        for first_id, second_id in ccd_bond_stereo:
            first_index = component_to_input[atom_ids[first_id]]
            second_index = component_to_input[atom_ids[second_id]]
            bond = input_molecule.GetBondBetweenAtoms(first_index, second_index)
            if bond is None or input_bond_stereo(bond) != ccd_bond_stereo[frozenset((first_id, second_id))]:
                stereo_consistent = False
                break
        if not stereo_consistent:
            continue
        for bond in input_molecule.GetBonds():
            input_stereo = input_bond_stereo(bond)
            if input_stereo is None:
                continue
            first_id = component_index_to_name[mapping[bond.GetBeginAtomIdx()]]
            second_id = component_index_to_name[mapping[bond.GetEndAtomIdx()]]
            if ccd_bond_stereo.get(frozenset((first_id, second_id))) != input_stereo:
                stereo_consistent = False
                break
        if not stereo_consistent:
            continue
        if stereo_consistent:
            return {
                atom_name: input_index
                for input_index, component_index in mapping.items()
                for atom_name in [component_index_to_name[component_index]]
            }
    raise ValueError("No chemically valid atom mapping preserves CCD connectivity and stereochemistry")


def predicted_atom_name_indices(smiles: str) -> dict[str, int]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit cannot parse input ligand SMILES")
    molecule_with_hydrogens = Chem.AddHs(molecule)
    canonical_ranks = list(AllChem.CanonicalRankAtoms(molecule_with_hydrogens))
    return {
        f"{molecule.GetAtomWithIdx(atom_index).GetSymbol().upper()}{canonical_ranks[atom_index] + 1}": atom_index
        for atom_index in range(molecule.GetNumAtoms())
    }


def symmetry_aware_mapping(reference_molecule, input_molecule, predicted_coordinates, reference_coordinates):
    parameters = Chem.SubstructMatchParameters()
    parameters.uniquify = False
    parameters.useChirality = True
    parameters.maxMatches = 10000
    matches = reference_molecule.GetSubstructMatches(input_molecule, parameters)
    if not matches:
        raise ValueError("Reference CCD graph does not match the input ligand graph")
    candidates = []
    for match in matches:
        predicted = np.asarray([predicted_coordinates[index] for index in range(input_molecule.GetNumAtoms())], dtype=float)
        native = np.asarray([reference_coordinates[match[index]] for index in range(input_molecule.GetNumAtoms())], dtype=float)
        rmsd = float(np.sqrt(np.mean(np.sum((predicted - native) ** 2, axis=1))))
        candidates.append((rmsd, match))
    return min(candidates, key=lambda item: item[0])


def symmetry_aware_rmsd(reference_molecule, input_molecule, predicted_coordinates, reference_coordinates) -> float:
    return symmetry_aware_mapping(reference_molecule, input_molecule, predicted_coordinates, reference_coordinates)[0]
