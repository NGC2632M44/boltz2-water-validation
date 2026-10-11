# Boltz-2 explicit-water capability check

This capability check tests whether explicitly requesting 1–10 water molecules changes Boltz-2 ligand pose prediction, and whether predicted oxygen positions reproduce crystallographic waters.

## Dataset

The original list contained 29 PDB IDs. `2J6V` and `4X2N` lack an organic target ligand suitable for ligand-pose RMSD, so they are excluded. `10OT`, `28JH`, and `30GX` are added to yield 30 evaluable complexes. The exact ligand CCD IDs and official RCSB entry, structure-CIF, and chemical-component-CIF templates are recorded in `manifest.json`.

## Prediction design

- 30 complexes × 10 requested water counts (1–10) × 30 diffusion samples = 9,000 requested predictions.
- Each water is a separate CCD `HOH` entity. Boltz receives no crystallographic water or ligand coordinates, templates, or native-pocket constraints.
- Every observed polypeptide chain in the asymmetric unit is included, selected only from polymer/entity metadata; native ligand coordinates are used only after prediction to define the reference water shell and evaluate poses.
- Ligands use an official RCSB `SMILES_CANONICAL` descriptor that passes exact heavy-atom connectivity, bond-order/aromatic normalization, formal-charge, atom-stereo, and double-bond-stereo checks against the CCD graph; the selected descriptor program is recorded per target.
- The exact reference ligand copy is selected deterministically and recorded by author/label chain, entity, residue number, and insertion code; evaluation requires that exact copy and never silently falls back to another.
- Boltz-2 is invoked from the preinstalled server environment. The run records software version, GPU assignment, seed, input YAML, stdout/stderr, and per-sample outputs.

## Evaluation

- For each sample, fit the predicted complex to the reference with common protein Cα atoms; report symmetry-aware ligand heavy-atom RMSD after that receptor fit.
- Define native pocket waters as one occupancy-selected oxygen per crystallographic water residue within 6 Å of the native ligand. After receptor alignment, report one-to-one predicted/native oxygen matches and precision, recall, and F1 at 1.0, 1.5, and 2.0 Å.
- Each paired PDB is a single `MODEL` holding the reference and the Boltz-2 prediction in the same frame: the reference uses UPPERCASE chain IDs (protein `A,B,..`; ligand `L`; crystal waters `N`) and the prediction uses lowercase IDs (protein `a,b,..`; ligand `l`; input waters `n`), rigid-body aligned by common protein C-alpha atoms. Both structures therefore overlap directly in any viewer; hiding the uppercase chains shows the prediction alone.
- Failed or missing samples remain explicit in the metrics table; no failed run is silently counted as a prediction.

## Outputs

The completed run will include the raw downloaded RCSB mmCIF and ligand-component CIF files, generated YAML inputs, MSA CSVs used by Boltz-2, server run logs and model outputs, a sample-level metrics CSV, summary tables, scatter plots, paired PDB files, a source-completeness manifest, archive checksums, and ZIP archives. Missing metric rows, prediction files, run statuses, and logs are enumerated instead of being silently omitted. Large run artifacts (the paired-PDB and source-data archives) are distributed via the GitHub Releases page rather than committed to Git.

## Result status (run `production`, 2026-10-10)

- 300/300 jobs completed; 8,580 of the designed 9,000 samples were produced and all metric rows are `ok`. `design_complete = true`, `release_complete = true`.
- Ligand pose RMSD is flat across requested water counts (median ~2.0 Å, mean ~4.5–4.8 Å); predicted water oxygens rarely reproduce crystallographic waters (F1 at 2.0 Å rises only from 0.019 to 0.036 as 1–10 waters are requested).
- Three 4-chain complexes (`1FIN`, `2HYY`, `3CS9`, ~1,100 tokens each) exceed the 32 GB device limit at 30 diffusion samples (peak ≈ 33.6 GB, deterministic OOM); they were run at **16 diffusion samples** instead. This is the only deviation from the design and is recorded per job in `status`/`job_sample_counts`.
- A **0-water blank control** (protein–ligand co-folding with no explicit `HOH`) was run with the same seeds and MSA for all 30 targets (`run_baseline0.py`); its ligand-pose median (1.87 Å) matches the 1–10 water columns, and it emits no predicted water oxygens.
- Result tables and manifests are under `data/`; figures and the full write-up are under `docs/`. Large archives are attached to GitHub Releases.
