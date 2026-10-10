# Boltz-2 Explicit-Water Capability Check

A capability check of whether explicitly requesting **1–10 water molecules** (each as an
independent CCD `HOH` entity) affects **Boltz-2** protein–ligand structure prediction, and whether
the predicted water oxygens reproduce **crystallographic (experimental) water positions**.

The model receives **no** native protein, ligand, or water coordinates, **no** templates, and
**no** pocket restraints — the check is a blind prediction; crystal structures are used only
for post-hoc scoring. See the [full report](docs/report.html)
([rendered online](https://ngc2632m44.github.io/boltz2-water-validation/report.html)) for the complete write-up.

## At a glance

| | |
|---|---|
| Targets | 30 RCSB complexes |
| Requested water counts | 1–10 (independent `HOH` entities) |
| Diffusion samples | up to 30 per (target, water count) |
| Completed predictions | **8,580 / 9,000 (95.3%)**, all successful |
| Ligand pose | median heavy-atom RMSD ≈ 2.0 Å, flat across water counts |
| Predicted waters | one-to-one F1 @ 2.0 Å only 0.019 → 0.036 (1 → 10 waters) |

## Results

### Ligand pose accuracy is essentially independent of the requested water count

![Ligand pose RMSD vs requested water count](docs/figures/ligand_rmsd_scatter.svg)

Each point is one diffusion sample after rigid-body fitting the complex on common protein Cα atoms
(red bars = per-count medians; the scatter pools all 30 targets). The median RMSD is ≈ 2.0 Å at
every water count: **requesting more waters neither improves nor degrades the ligand pose**.
Overall, ~50% of samples are < 2 Å, ~76% < 5 Å and ~87% < 10 Å. The distribution is heavy-tailed:
a minority of samples (dominated by a few large multi-chain complexes) deviate by tens of Ångström.
The purple dashed line marks the common 2.0 Å ligand-pose success criterion; the model sits on it
by median but its tail exceeds it.

### Predicted water oxygens do not reproduce crystallographic waters

![Predicted water-oxygen distance to matched crystal water](docs/figures/water_position_scatter.svg)

Predicted and experimental pocket waters are matched one-to-one with an optimal (Hungarian)
assignment in the reference frame. Each point is the mean matched distance for one sample; red bars
are per-count medians. The distance stays in the **9–14 Å** range and does not converge with more
requested waters: requesting more water entities mainly adds "filler" oxygens rather than recovering
real hydration sites. The one-to-one F1 at 2.0 Å rises only from 0.019 to 0.036 as the request grows
from 1 to 10 waters (precision falls 0.113 → 0.036, recall rises 0.011 → 0.040). The purple dashed
line marks a competent water-placement accuracy of ~1.0 Å (WaterDock leaves a mean error of
≈ 0.78 Å; SuperWater reports ≈ 0.3 Å), i.e. the model is about an order of magnitude off.

The water conclusion is **not** an artefact of the largest complexes: after excluding the three
four-chain targets, or restricting to samples whose receptor and ligand pose are correct
(ligand RMSD < 2 Å), the one-to-one F1 at 2.0 Å is still only 0.010–0.031.

## Why some poses deviate by tens of Ångström

The large-RMSD samples are concentrated in a few **large multi-chain** complexes. There, the
predicted complex itself is 4–8 Å off in its global arrangement, so the ligand is carried to a
**wrong location** (10–20 Å from the native pocket) even though its internal conformation can be
near-perfect. For single-chain targets that fold correctly, the ligand stays at/near the native
pocket (centroid deviation < 1–3 Å) even when its detailed pose is off by 2–6 Å. Adding waters does
not change any of this — the bottleneck is receptor folding and global packing, not explicit waters.
Full per-target analysis is in [`docs/report.html`](docs/report.html).

## Repository layout

```
code/    prediction and evaluation pipeline (prepare → run_grid/run_retry → evaluate → summarize),
         unit tests, the study manifest, and its own README
data/    summary tables, per-sample metrics, and sanitized provenance manifests
docs/    full report (HTML/Markdown) and the two result figures
```

Server-specific absolute paths in the provenance files were replaced with `/path/to/...` placeholders;
no credentials, tokens, or private data are included.

## Data and large artifacts

Small tables, figures and manifests live in this repository. The **large archives** are attached to
the [Releases](../../releases) page (they exceed Git hosting limits):

- `paired_reference_boltz_pdbs.zip` — 8,580 paired PDBs; each file holds the reference and the
  aligned Boltz-2 prediction in a single MODEL (see *Visualization* below).
- `source_data_and_metrics.zip` — source structures, ligand-component CIFs, MSA CSVs, predictions,
  logs, and metrics.

## Reproducing

Requirements and pipeline usage are documented in [`code/README.md`](code/README.md). In brief, on a
machine with Boltz-2 installed:

```bash
python code/prepare.py   --root <workdir>
python code/run_grid.py  --root <workdir> --run-name production
python code/run_retry.py --root <workdir> --run-name production    # resume incomplete jobs
python code/evaluate.py  --root <workdir> --run-name production
python code/summarize.py --root <workdir> --run-name production
```

Python dependencies for the analysis pipeline are listed in `code/requirements.txt`; Boltz-2 itself
provides the model and its GPU stack.

## Visualization (PyMOL)

Each paired PDB (from the release archive) is a **single MODEL** holding both the reference and the
prediction in one frame: the reference uses **UPPERCASE** chain IDs (protein `A,B,..`; ligand `L`;
crystal waters `N`) and the prediction uses **lowercase** IDs (protein `a,b,..`; ligand `l`; input
waters `n`), rigid-body aligned by common protein Cα atoms. Reference waters occupy PDB serials
95001+, predicted waters 96001+, which makes the two sets selectable in any viewer regardless of
chain-case handling:

```
show cartoon, all and polymer
set cartoon_transparency, 0.6

# reference water — orange
show spheres, all and resn HOH and id 95001-95999
color orange, all and resn HOH and id 95001-95999

# Boltz-2 predicted water — cyan
show spheres, all and resn HOH and id 96001-96999
color cyan,   all and resn HOH and id 96001-96999

set sphere_scale, 0.4
```

To view only one set, hide the other, e.g. `hide everything; show spheres,
all and resn HOH and id 96001-96999`. (If a viewer renumbers serials, use `set ignore_case, 0`
followed by `chain n` for the prediction and `chain N` for the reference.)

## Licensing and attribution

- **Code** (`code/`): MIT License — see [`LICENSE`](LICENSE).
- **Data, figures and report** (`data/`, `docs/`): CC BY 4.0 — see [`LICENSE-DATA`](LICENSE-DATA).
- **Third-party:** structures and ligand components are from the
  [RCSB PDB](https://www.rcsb.org) (public domain / CC0); the model is
  [Boltz-2](https://github.com/jwohlwend/boltz) (MIT); multiple sequence alignments were generated
  with the ColabFold/MMseqs2 MSA server. Please cite the original sources when reusing.

## Citation

See [`CITATION.cff`](CITATION.cff).
