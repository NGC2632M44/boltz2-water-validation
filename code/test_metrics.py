import unittest
import os
import tempfile
import hashlib
import json
import csv
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rdkit import Chem

from common import canonical_smiles
from evaluate import ca_coordinates
from metrics import apply_transform, component_atom_mapping, kabsch_transform, symmetry_aware_rmsd, water_match_metrics
from prepare import select_validated_ligand_smiles
from run_grid import archive_existing_output, find_msa_paths, job_status_complete, output_files_match_status, persist_run_config, status_is_complete
from summarize import build_release, complete_metric_rows, prediction_completeness, referenced_msa_paths, scatter_coordinate_rows, summarize_by_target, summarize_by_water_count


class GeometryTests(unittest.TestCase):
    def test_kabsch_recovers_rigid_transform(self):
        source = np.asarray([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3], [1, 2, 1]], dtype=float)
        rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        target = source @ rotation + np.asarray([5, -2, 4], dtype=float)
        transform = kabsch_transform(source, target)
        self.assertTrue(np.allclose(apply_transform(source, transform), target, atol=1e-10))

    def test_water_matching_is_one_to_one(self):
        result = water_match_metrics([[0, 0, 0], [0.1, 0, 0]], [[0.05, 0, 0]])
        self.assertEqual(result["matched_1p0A"], 1)
        self.assertAlmostEqual(result["precision_1p0A"], 0.5)
        self.assertAlmostEqual(result["recall_1p0A"], 1.0)
        self.assertAlmostEqual(result["f1_1p0A"], 2.0 / 3.0)

    def test_mean_water_distance_does_not_reuse_native_waters(self):
        result = water_match_metrics([[0, 0, 0], [0.1, 0, 0]], [[0, 0, 0], [10, 0, 0]])
        self.assertAlmostEqual(result["mean_one_to_one_distance_A"], 4.95)

    def test_ligand_rmsd_allows_symmetric_atom_mapping(self):
        molecule = Chem.MolFromSmiles("CC")
        predicted = {0: [1, 0, 0], 1: [-1, 0, 0]}
        reference = {0: [-1, 0, 0], 1: [1, 0, 0]}
        self.assertAlmostEqual(symmetry_aware_rmsd(molecule, molecule, predicted, reference), 0.0)

    def test_ligand_mapping_rejects_opposite_stereochemistry(self):
        reference_molecule = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
        opposite_molecule = Chem.MolFromSmiles("N[C@H](C)C(=O)O")
        coordinates = {index: [float(index), float(index % 2), 0.0] for index in range(reference_molecule.GetNumAtoms())}
        with self.assertRaises(ValueError):
            symmetry_aware_rmsd(reference_molecule, opposite_molecule, coordinates, coordinates)


class CcdComponentTests(unittest.TestCase):
    def write_component_cif(self, path, atoms, bonds):
        rows = [
            "data_TEST",
            "loop_",
            "_chem_comp_atom.atom_id",
            "_chem_comp_atom.type_symbol",
            "_chem_comp_atom.charge",
            "_chem_comp_atom.pdbx_stereo_config",
        ]
        rows.extend(f"{atom_id} {element} 0 {stereo}" for atom_id, element, stereo in atoms)
        rows.extend(
            [
                "loop_",
                "_chem_comp_bond.atom_id_1",
                "_chem_comp_bond.atom_id_2",
                "_chem_comp_bond.value_order",
                "_chem_comp_bond.pdbx_aromatic_flag",
                "_chem_comp_bond.pdbx_stereo_config",
            ]
        )
        rows.extend(f"{first} {second} {order} N {stereo}" for first, second, order, stereo in bonds)
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def test_invalid_all_single_aromatic_ring_is_rejected(self):
        atoms = [(f"C{index}", "C", "N") for index in range(1, 7)]
        bonds = [(f"C{index}", f"C{index % 6 + 1}", "SING", "N") for index in range(1, 7)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            component_path = Path(temporary_directory) / "TEST.cif"
            self.write_component_cif(component_path, atoms, bonds)
            molecule = Chem.MolFromSmiles("c1ccccc1")
            with self.assertRaises(ValueError):
                component_atom_mapping(component_path, molecule)

    def test_unknown_ccd_formal_charge_is_rejected(self):
        atoms = [("C1", "C", "N"), ("C2", "C", "N")]
        bonds = [("C1", "C2", "SING", "N")]
        molecule = Chem.MolFromSmiles("CC")
        for unknown_charge in (".", "?"):
            with self.subTest(charge=unknown_charge), tempfile.TemporaryDirectory() as temporary_directory:
                component_path = Path(temporary_directory) / "TEST.cif"
                self.write_component_cif(component_path, atoms, bonds)
                content = component_path.read_text(encoding="utf-8").replace("C1 C 0 N", f"C1 C {unknown_charge} N")
                component_path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "formal charge is unknown"):
                    component_atom_mapping(component_path, molecule)

    def test_ccd_double_bond_stereo_must_match_smiles(self):
        atoms = [(f"C{index}", "C", "N") for index in range(1, 5)]
        bonds = [("C1", "C2", "SING", "N"), ("C2", "C3", "DOUB", "E"), ("C3", "C4", "SING", "N")]
        with tempfile.TemporaryDirectory() as temporary_directory:
            component_path = Path(temporary_directory) / "TEST.cif"
            self.write_component_cif(component_path, atoms, bonds)
            correct_stereo = Chem.MolFromSmiles(r"C/C=C/C")
            wrong_stereo = Chem.MolFromSmiles(r"C/C=C\C")
            self.assertEqual(len(component_atom_mapping(component_path, correct_stereo)), 4)
            with self.assertRaises(ValueError):
                component_atom_mapping(component_path, wrong_stereo)

    def test_ccd_atom_stereochemistry_requires_smiles_stereo(self):
        component_dir = os.environ.get("WATER_VALIDATION_COMPONENT_DIR")
        if not component_dir:
            self.skipTest("WATER_VALIDATION_COMPONENT_DIR is not set")
        component_path = Path(component_dir) / "A1DI0.cif"
        molecule = Chem.MolFromSmiles(canonical_smiles(component_path, "A1DI0"))
        Chem.RemoveStereochemistry(molecule)
        with self.assertRaises(ValueError):
            component_atom_mapping(component_path, molecule)

    def test_conflicting_canonical_descriptors_choose_ccd_consistent_stereo(self):
        component_dir = os.environ.get("WATER_VALIDATION_COMPONENT_DIR")
        if not component_dir:
            self.skipTest("WATER_VALIDATION_COMPONENT_DIR is not set")
        component_path = Path(component_dir) / "B49.cif"
        context = {"component_cif": str(component_path), "ligand_ccd": "B49"}
        program, _ = select_validated_ligand_smiles(context)
        self.assertIn("OpenEye", program)

    def test_canonical_smiles_does_not_fall_back_to_noncanonical_descriptor(self):
        component = """data_XYZ
_pdbx_chem_comp_descriptor.comp_id XYZ
_pdbx_chem_comp_descriptor.type SMILES
_pdbx_chem_comp_descriptor.program RDKit
_pdbx_chem_comp_descriptor.descriptor CCO
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            component_path = Path(temporary_directory) / "XYZ.cif"
            component_path.write_text(component, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No canonical RCSB SMILES"):
                canonical_smiles(component_path, "XYZ")

    def test_ca_selection_uses_polymer_carbon_alpha_only(self):
        def atom(name, element, x):
            return SimpleNamespace(
                name=name,
                element=SimpleNamespace(name=element),
                occ=1.0,
                pos=SimpleNamespace(x=x, y=0.0, z=0.0),
            )

        class Residue:
            def __init__(self, entity_id, label_seq, atom_value):
                self.subchain = "A"
                self.entity_id = entity_id
                self.label_seq = label_seq
                self.seqid = SimpleNamespace(num=label_seq, icode="")
                self.atoms = [atom_value]

            def __iter__(self):
                return iter(self.atoms)

        class Chain:
            name = "A"

            def __init__(self, residues):
                self.residues = residues

            def __iter__(self):
                return iter(self.residues)

        model = [
            Chain(
                [
                    Residue("1", 1, atom("CA", "C", 1.0)),
                    Residue("1", 2, atom("CA", "Ca", 99.0)),
                    Residue("2", 3, atom("CA", "C", 88.0)),
                ]
            )
        ]
        chains = [{"input_id": "A", "label_asym_id": "A", "entity_id": "1"}]
        coordinates = ca_coordinates(model, chains, reference=True)
        self.assertEqual(coordinates, {("A", ("label", 1)): [1.0, 0.0, 0.0]})
        predicted_model = [Chain([Residue("1", 1, atom("CA", "C", 1.0)), Residue("1", 0, atom("CA", "C", 88.0))])]
        predicted_coordinates = ca_coordinates(predicted_model, chains, reference=False)
        self.assertEqual(predicted_coordinates, {("A", ("label", 1)): [1.0, 0.0, 0.0]})

    def test_prepared_ccd_graphs_match_official_smiles(self):
        component_dir = os.environ.get("WATER_VALIDATION_COMPONENT_DIR")
        if not component_dir:
            self.skipTest("WATER_VALIDATION_COMPONENT_DIR is not set")
        component_paths = sorted(Path(component_dir).glob("*.cif"))
        self.assertGreater(len(component_paths), 0)
        for component_path in component_paths:
            component_id = component_path.stem
            with self.subTest(component=component_id):
                context = {"component_cif": str(component_path), "ligand_ccd": component_id}
                program, atom_ids = select_validated_ligand_smiles(context)
                self.assertTrue(program)
                input_molecule = Chem.MolFromSmiles(context["ligand_smiles"])
                self.assertEqual(len(atom_ids), input_molecule.GetNumAtoms(), component_id)
                self.assertEqual(set(atom_ids.values()), set(range(input_molecule.GetNumAtoms())), component_id)


class MsaReuseTests(unittest.TestCase):
    def test_msa_files_match_all_chains_by_query_sequence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            msa_directory = Path(temporary_directory) / "prediction" / "msa"
            msa_directory.mkdir(parents=True)
            first_path = msa_directory / "input_0.csv"
            second_path = msa_directory / "input_1.csv"
            first_path.write_text("key,sequence\n0,AAAT\n-1,AAAA\n", encoding="utf-8")
            second_path.write_text("key,sequence\n-1,BBBB\n0,BBBA\n", encoding="utf-8")
            proteins = [
                {"input_id": "A", "sequence": "AAAA"},
                {"input_id": "B", "sequence": "AAAA"},
                {"input_id": "C", "sequence": "BBBB"},
            ]
            result = find_msa_paths(Path(temporary_directory), proteins)
            self.assertEqual(result, {"A": str(first_path.resolve()), "B": str(first_path.resolve()), "C": str(second_path.resolve())})

    def test_msa_reuse_is_disabled_when_any_sequence_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            msa_directory = Path(temporary_directory) / "prediction" / "msa"
            msa_directory.mkdir(parents=True)
            (msa_directory / "input_0.csv").write_text("key,sequence\n-1,AAAA\n", encoding="utf-8")
            proteins = [
                {"input_id": "A", "sequence": "AAAA"},
                {"input_id": "B", "sequence": "BBBB"},
            ]
            self.assertEqual(find_msa_paths(Path(temporary_directory), proteins), {})


class ResumeStatusTests(unittest.TestCase):
    def test_resume_requires_exact_unique_model_indices_and_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.yaml"
            input_path.write_text("input", encoding="utf-8")
            output_dir = root / "output"
            output_dir.mkdir()
            (output_dir / "sample_model_0.cif").write_text("0", encoding="utf-8")
            (output_dir / "sample_model_1.cif").write_text("1", encoding="utf-8")
            status_path = root / "status.json"
            import hashlib
            import json

            digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
            prediction_files = [str((output_dir / f"sample_model_{index}.cif").resolve()) for index in range(2)]
            status = {
                "exit_code": 0,
                "requested_samples": 2,
                "observed_samples": 2,
                "seed": 17,
                "input_sha256": digest,
                "prediction_indices": [0, 1],
                "prediction_files": prediction_files,
                "prediction_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in prediction_files},
                "output_dir": str(output_dir.resolve()),
            }
            status_path.write_text(json.dumps(status), encoding="utf-8")
            self.assertTrue(job_status_complete(status, 2))
            self.assertTrue(output_files_match_status(status, output_dir, 2))
            self.assertTrue(status_is_complete(status_path, output_dir, input_path, 2, 17))
            status["prediction_indices"] = [0, 0]
            status_path.write_text(json.dumps(status), encoding="utf-8")
            self.assertFalse(job_status_complete(status, 2))
            self.assertFalse(status_is_complete(status_path, output_dir, input_path, 2, 17))

    def test_resume_rejects_mismatched_recorded_paths_and_modified_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            output_dir.mkdir()
            input_path = root / "input.yaml"
            input_path.write_text("input", encoding="utf-8")
            files = []
            for index in range(2):
                path = output_dir / f"sample_model_{index}.cif"
                path.write_text(str(index), encoding="utf-8")
                files.append(path)
            import hashlib
            import json

            paths = [str(path.resolve()) for path in files]
            status = {
                "exit_code": 0,
                "requested_samples": 2,
                "observed_samples": 2,
                "seed": 21,
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "prediction_indices": [0, 1],
                "prediction_files": paths,
                "prediction_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
                "output_dir": str(output_dir.resolve()),
            }
            status_path = root / "status.json"
            status_path.write_text(json.dumps(status), encoding="utf-8")
            status["prediction_files"] = [paths[0], str((root / "stale_model_1.cif").resolve())]
            status["prediction_sha256"] = {status["prediction_files"][0]: hashlib.sha256(files[0].read_bytes()).hexdigest(), status["prediction_files"][1]: "0" * 64}
            status_path.write_text(json.dumps(status), encoding="utf-8")
            self.assertFalse(status_is_complete(status_path, output_dir, input_path, 2, 21))
            status["prediction_files"] = paths
            status["prediction_sha256"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
            status_path.write_text(json.dumps(status), encoding="utf-8")
            files[0].write_text("changed", encoding="utf-8")
            self.assertFalse(status_is_complete(status_path, output_dir, input_path, 2, 21))

    def test_previous_output_is_archived_before_retry(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "water_01"
            output_dir.mkdir()
            old_prediction = output_dir / "sample_model_0.cif"
            old_prediction.write_text("old output", encoding="utf-8")
            archive_path = archive_existing_output(output_dir, "retry001")
            self.assertFalse(output_dir.exists())
            self.assertEqual((archive_path / old_prediction.name).read_text(encoding="utf-8"), "old output")

    def test_run_configuration_is_persisted_and_immutable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            configuration = persist_run_config(root, "test", ["11ES"], [1, 2], 2, [0], root / "cache")
            self.assertEqual(configuration["targets"], ["11ES"])
            self.assertEqual(configuration["water_counts"], [1, 2])
            self.assertEqual(configuration["samples_per_input"], 2)
            self.assertEqual(json.loads((root / "runs/test/run_config.json").read_text(encoding="utf-8")), configuration)
            with self.assertRaises(RuntimeError):
                persist_run_config(root, "test", ["11ES"], [1, 2], 3, [0], root / "cache")


class ScatterCoordinateTests(unittest.TestCase):
    def test_scatter_coordinates_preserve_exact_requested_gradient(self):
        rows = [
            {
                "pdb_id": "11ES",
                "water_count": "3",
                "replicate": "7",
                "status": "ok",
                "ligand_rmsd_A": "1.25",
                "mean_one_to_one_distance_A": "2.5",
            }
        ]
        coordinates = scatter_coordinate_rows(rows)
        self.assertEqual(len(coordinates), 1)
        self.assertEqual(coordinates[0]["ligand_rmsd_x"], "3")
        self.assertEqual(coordinates[0]["water_position_x"], "3")
        self.assertEqual(coordinates[0]["ligand_rmsd_y"], "1.25")
        self.assertEqual(coordinates[0]["water_position_y"], "2.5")

    def test_expected_grid_keeps_wholly_missing_target_water_groups(self):
        inventory = complete_metric_rows(
            [],
            ["11ES", "12EW"],
            [1, 2],
            {(pdb_id, water_count): 2 for pdb_id in ["11ES", "12EW"] for water_count in [1, 2]},
        )
        self.assertEqual(len(inventory["complete_rows"]), 8)
        self.assertEqual(len(inventory["missing_metric_samples"]), 8)
        self.assertEqual(len(summarize_by_water_count(inventory["complete_rows"])), 2)
        target_summary = summarize_by_target(inventory["complete_rows"])
        self.assertEqual(len(target_summary), 4)
        self.assertTrue(all(row["requested"] == 2 and row["valid"] == 0 for row in target_summary))

    def test_prediction_inventory_anchors_relative_paths_and_rejects_stale_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            run_config = {
                "run_name": "test",
                "targets": ["11ES"],
                "water_counts": [1],
                "samples_per_input": 2,
                "seed_base": 17000,
            }
            input_path = root / "inputs/11ES/water_01.yaml"
            input_path.parent.mkdir(parents=True)
            input_path.write_text("input", encoding="utf-8")
            output_dir = root / "runs/test/11ES/water_01"
            prediction_dir = output_dir / "boltz_results/predictions"
            prediction_dir.mkdir(parents=True)
            current_paths = []
            for model_index in range(2):
                prediction_path = prediction_dir / f"water_01_model_{model_index}.cif"
                prediction_path.write_text(f"model {model_index}", encoding="utf-8")
                current_paths.append(prediction_path.resolve())
            stale_path = root / "runs/test/11ES/attempts/water_01_old/water_01_model_0.cif"
            stale_path.parent.mkdir(parents=True)
            stale_path.write_text("stale model", encoding="utf-8")
            status_path = root / "runs/test/11ES/status/water_01.json"
            status_path.parent.mkdir(parents=True)
            status = {
                "output_dir": str(output_dir),
                "requested_samples": 2,
                "observed_samples": 2,
                "prediction_indices": [0, 1],
                "prediction_files": [str(path) for path in current_paths],
                "prediction_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in current_paths
                },
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "seed": 17001,
                "model": "boltz2",
                "exit_code": 0,
            }
            status_path.write_text(json.dumps(status), encoding="utf-8")
            rows = [
                {
                    "pdb_id": "11ES",
                    "water_count": "1",
                    "replicate": "1",
                    "status": "partial",
                    "prediction_cif": "runs/test/11ES/attempts/water_01_old/water_01_model_0.cif",
                },
                {
                    "pdb_id": "11ES",
                    "water_count": "1",
                    "replicate": "2",
                    "status": "partial",
                    "prediction_cif": "runs/test/11ES/water_01/boltz_results/predictions/water_01_model_0.cif",
                },
            ]

            result = prediction_completeness(rows, root, run_config, {"11ES": 0})

        self.assertEqual(result["expected_sample_count"], 2)
        self.assertEqual(result["metric_row_count"], 2)
        self.assertEqual(result["available_prediction_paths"], [str(path) for path in current_paths])
        self.assertEqual(result["available_prediction_count"], 2)
        self.assertEqual(result["missing_prediction_samples"], [])
        self.assertEqual([(item["replicate"], item["metrics_prediction_cif"]) for item in result["prediction_path_mismatches"]], [(1, str(stale_path.resolve())), (2, str(current_paths[0]))])

    def test_missing_status_never_counts_archived_predictions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            input_path = root / "inputs/11ES/water_01.yaml"
            input_path.parent.mkdir(parents=True)
            input_path.write_text("input", encoding="utf-8")
            stale_path = root / "runs/test/11ES/attempts/water_01_old/water_01_model_0.cif"
            stale_path.parent.mkdir(parents=True)
            stale_path.write_text("stale model", encoding="utf-8")
            rows = [{"pdb_id": "11ES", "water_count": "1", "replicate": "1", "status": "partial", "prediction_cif": str(stale_path)}]
            config = {"run_name": "test", "targets": ["11ES"], "water_counts": [1], "samples_per_input": 1, "seed_base": 17000}

            result = prediction_completeness(rows, root, config, {"11ES": 0})

        self.assertEqual(result["available_prediction_count"], 0)
        self.assertEqual(result["available_prediction_paths"], [])
        self.assertEqual(result["missing_prediction_samples"][0]["reason"], "missing_status")

    def test_msa_archive_selection_follows_yaml_references(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            input_path = root / "inputs/11ES/water_02.yaml"
            input_path.parent.mkdir(parents=True)
            msa_path = root / "runs/test/msas/query.csv"
            msa_path.parent.mkdir(parents=True)
            msa_path.write_text("sequence,key\nAAA,-1\n", encoding="utf-8")
            unrelated_path = root / "runs/test/msa/unrelated.csv"
            unrelated_path.parent.mkdir(parents=True)
            unrelated_path.write_text("unrelated", encoding="utf-8")
            input_path.write_text(
                "version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: AAA\n      msa: runs/test/msas/query.csv\n",
                encoding="utf-8",
            )

            result = referenced_msa_paths(root, [input_path])

        self.assertEqual(result["paths"], [msa_path.resolve()])
        self.assertEqual(result["missing_references"], [])
        self.assertEqual(result["external_references"], [])


class ReleaseArchiveTests(unittest.TestCase):
    def test_release_archive_contains_only_current_predictions_and_referenced_msa(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            for directory in ("source-data/structures", "source-data/components", "metadata", "code", "inputs/11ES", "runs/test/11ES/status", "metrics/test", "pairs/test/11ES"):
                (root / directory).mkdir(parents=True, exist_ok=True)
            manifest = {
                "targets": [{"pdb_id": "11ES", "ligand_ccd": "LIG"}],
                "design": {"input_water_counts": [1], "diffusion_samples_per_input": 1},
            }
            prepared = {"targets": [{"pdb_id": "11ES", "ligand_ccd": "LIG"}]}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "prepared_manifest.json").write_text(json.dumps(prepared), encoding="utf-8")
            (root / "source-data/structures/11ES.cif").write_text("structure", encoding="utf-8")
            (root / "source-data/components/LIG.cif").write_text("component", encoding="utf-8")
            (root / "metadata/11ES.json").write_text("{}", encoding="utf-8")
            (root / "code/README.md").write_text("readme", encoding="utf-8")
            (root / "code/manifest.json").write_text("{}", encoding="utf-8")
            (root / "code/runner.py").write_text("pass", encoding="utf-8")

            input_path = root / "inputs/11ES/water_01.yaml"
            output_dir = root / "runs/test/11ES/water_01"
            msa_path = output_dir / "msa/query.csv"
            unrelated_msa = output_dir / "msa/unrelated.csv"
            prediction_path = output_dir / "sample_model_0.cif"
            pair_path = root / "pairs/test/11ES/water_01/replicate_001.pdb"
            log_path = root / "runs/test/11ES/logs/water_01.log"
            for path in (msa_path, unrelated_msa, prediction_path, pair_path, log_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            msa_path.write_text("sequence,key\nAAA,-1\n", encoding="utf-8")
            unrelated_msa.write_text("unreferenced", encoding="utf-8")
            prediction_path.write_text("model", encoding="utf-8")
            pair_path.write_text("paired pdb", encoding="utf-8")
            log_path.write_text("run log", encoding="utf-8")
            input_path.write_text(
                "version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: AAA\n      msa: runs/test/11ES/water_01/msa/query.csv\n",
                encoding="utf-8",
            )
            run_config = {
                "run_name": "test",
                "targets": ["11ES"],
                "water_counts": [1],
                "samples_per_input": 1,
                "seed_base": 17,
            }
            run_root = root / "runs/test"
            (run_root / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
            (run_root / "run_summary.json").write_text("[]", encoding="utf-8")
            (root / "runs/test/11ES/status/water_01.json").write_text(
                json.dumps(
                    {
                        "pdb_id": "11ES",
                        "water_count": 1,
                        "output_dir": str(output_dir),
                        "requested_samples": 1,
                        "observed_samples": 1,
                        "prediction_indices": [0],
                        "prediction_files": [str(prediction_path)],
                        "prediction_sha256": {str(prediction_path): hashlib.sha256(prediction_path.read_bytes()).hexdigest()},
                        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                        "seed": 18,
                        "model": "boltz2",
                        "msa_files": [str(msa_path)],
                        "msa_sha256": {str(msa_path): hashlib.sha256(msa_path.read_bytes()).hexdigest()},
                        "log": str(log_path),
                    }
                ),
                encoding="utf-8",
            )
            metric_row = {
                "pdb_id": "11ES",
                "ligand_ccd": "LIG",
                "water_count": "1",
                "replicate": "1",
                "status": "ok",
                "error": "",
                "ligand_rmsd_A": "1.0",
                "f1_2p0A": "0.5",
                "precision_2p0A": "0.5",
                "recall_2p0A": "0.5",
                "mean_one_to_one_distance_A": "1.2",
                "paired_pdb": str(pair_path),
                "prediction_cif": str(prediction_path),
            }
            metrics_path = root / "metrics/test/sample_metrics.csv"
            with metrics_path.open("w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=list(metric_row))
                writer.writeheader()
                writer.writerow(metric_row)

            report = build_release(root, "test")

            self.assertTrue(report["release_complete"])
            self.assertEqual(report["requested_rows"], 1)
            self.assertEqual(report["available_prediction_count"], 1)
            with zipfile.ZipFile(report["source_data_archive"]) as archive:
                names = set(archive.namelist())
            self.assertIn("runs/test/11ES/water_01/msa/query.csv", names)
            self.assertIn("runs/test/11ES/water_01/sample_model_0.cif", names)
            self.assertIn("deliverables/test/source_completeness.json", names)
            self.assertNotIn("runs/test/11ES/water_01/msa/unrelated.csv", names)


if __name__ == "__main__":
    unittest.main()
