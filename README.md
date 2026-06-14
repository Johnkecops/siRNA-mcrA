# siRNA Computational Design Pipeline — *mcrA* (Methyl-Coenzyme M Reductase α)
**Target organism:** Methanogenic Archaea  
**Goal:** Methane cessation via RNAi-mediated silencing of the MCR alpha subunit    
**Reference pipeline:** Parikesit, Ansori & Kharisma (2022). *Indones. J. Chem.* 22(5):1163–1176. https://doi.org/10.22146/ijc.68415  
**Research problem:** Parikesit (2025). Octopus.ac. https://doi.org/10.57874/gjc79460

---

## Overview

This repository contains the complete Python implementation of a 14-step computational siRNA design pipeline targeting the universally conserved *mcrA* gene across taxonomically diverse methanogenic archaea. The pipeline proceeds from sequence retrieval and multiple sequence alignment through thermodynamic siRNA scoring, 2D/3D structure prediction, RNA–RNA molecular docking, and molecular dynamics stability validation.

The pipeline produces a fully characterised 19-nucleotide siRNA candidate and generates all figures, PDB coordinates, and JSON result files used in the accompanying manuscript.

---

## Repository Contents

| File | Role | Pipeline Steps |
|---|---|---|
| `sirna_pipeline.py` | **Master script** — runs the complete end-to-end pipeline in one call | Steps 1–9 |
| `step2_msa_sirna.py` | Modular MSA, conservation analysis, siRNA design & 2D structure | Steps 2–6 |
| `step7_duplex.py` | **Recommended** docking script — antiparallel A-form duplex + Brownian MD | Steps 7–9 |
| `step7_docking_md.py` | Full coarse-grained docking with Lennard-Jones/Coulomb scoring | Steps 7–9 |
| `step7_fast.py` | Vectorised NumPy docking (faster; used for large datasets) | Steps 7–9 |
| `step7_minimal.py` | 1-bead-per-nt minimal docking (fastest; for quick screening) | Steps 7–9 |
| `step_structures.py` | 2D circular/arc diagrams, 3D A-form helix, co-fold, PDB export | Visualisation |
| `step2_6_results.json` | Intermediate results: MSA, siRNA candidate, ViennaRNA data | — |
| `step7_9_results.json` | Docking and MD results: H-bonds, stacking, RMSD, energy | — |
| `fig6_2d_combined.png` | Four-panel 2D secondary structure figure | — |
| `fig7_3d_complex.png` | Three-view 3D A-form duplex figure | — |
| `fig8_cofold_structure.png` | Co-fold siRNA:mRNA arc diagram | — |
| `siRNA_mcrA_duplex.pdb` | PDB coordinates (CG model, Chain A=mRNA, Chain B=siRNA) | — |
| `siRNA_mcrA_Methanogen_Manuscript.docx` | Full manuscript (Philippine Journal of Science format) | — |

---

## Software Requirements

### Python version
```
Python 3.10 or later
```

### Required packages
```bash
pip install biopython numpy matplotlib ViennaRNA
```

| Package | Tested version | Purpose |
|---|---|---|
| `biopython` | 1.87 | Sequence I/O, pairwise alignment (`Bio.Align.PairwiseAligner`), NCBI Entrez |
| `ViennaRNA` (Python bindings) | 2.7.2 | MFE folding, partition function, co-fold, suboptimal structures |
| `numpy` | ≥1.24 | Numerical arrays, vectorised docking, MD trajectory |
| `matplotlib` | ≥3.7 | All figure generation (non-interactive; `Agg` backend) |

### Installing ViennaRNA with Python bindings
ViennaRNA must be installed with Python bindings enabled. The recommended approach:

```bash
# Conda (recommended)
conda install -c bioconda viennarna

# Or from source (requires SWIG)
wget https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_7_x/ViennaRNA-2.7.2.tar.gz
tar xzf ViennaRNA-2.7.2.tar.gz && cd ViennaRNA-2.7.2
./configure --with-python3
make && sudo make install
```

---

## Execution Order

### Option A — Single master script (recommended for first run)

```bash
python3 sirna_pipeline.py
```

This runs the complete pipeline and produces **all** outputs in one command:
- Retrieves/uses the 8 hardcoded representative *mcrA* sequences
- Performs progressive MSA (Biopython PairwiseAligner)
- Identifies the maximally conserved 40-nt window
- Scores all 19-nt siRNA candidates
- Computes MFE, co-fold MFE, accessibility, and structural diversity
- Runs coarse-grained molecular docking + Brownian MD
- Saves all figures (fig1–fig5) and `step2_6_results.json` + `step7_9_results.json`

---

### Option B — Modular step-by-step execution

Run in this exact order (each script reads the JSON output of the previous step):

#### Step 1 — Sequence data
The 8 representative *mcrA* sequences are hard-coded in `sirna_pipeline.py`. If you run the modular scripts separately, you must first generate `seq_data.json`:

```bash
python3 -c "
import json
# Sequences are embedded in sirna_pipeline.py — run Step 1 section only:
exec(open('sirna_pipeline.py').read().split('# \u2500\u2500 Step 2')[0])
"
```

Or simply run `sirna_pipeline.py` once to generate all JSON files, then use modular scripts for re-analysis.

#### Steps 2–6 — MSA, conservation, siRNA design, 2D structure

```bash
python3 step2_msa_sirna.py
```

**Reads:** `seq_data.json`  
**Writes:** `step2_6_results.json`, `fig1_conservation_profile.png`, `fig2_accessibility_landscape.png`, `fig3_sirna_candidates.png`

**What it does:**
- Progressive pairwise alignment (global mode, match +2, mismatch −1, gap open −2, extend −0.5)
- Per-position conservation scoring with 40-nt sliding window
- All 19-nt siRNA candidates scored: GC content, 5′ asymmetry, homopolymer check
- ViennaRNA `RNA.fold()` for MFE of target mRNA window and siRNA guide
- ViennaRNA `RNA.cofold()` for siRNA:mRNA complex MFE
- `RNA.subopt()` for structural diversity (suboptimal structures within 5 kcal/mol)
- Best candidate selected by most negative co-fold MFE

#### Steps 7–9 — Molecular docking and MD stability

Choose one of the three docking scripts depending on your use case:

```bash
# Most accurate (antiparallel A-form geometry + Brownian MD) — recommended
python3 step7_duplex.py

# Full coarse-grained Lennard-Jones/Coulomb docking
python3 step7_docking_md.py

# Vectorised NumPy fast docking (large screening)
python3 step7_fast.py

# Minimal 1-bead model (quick prototyping)
python3 step7_minimal.py
```

**Reads:** `step2_6_results.json`  
**Writes:** `step7_9_results.json`, `fig4_docking_md.png`, `fig5_interactions.png`

**What `step7_duplex.py` does (recommended):**
- Places siRNA guide antiparallel to the 19-nt mRNA target in A-form geometry
  - Rise = 2.81 Å/bp, twist = 32.7°/nt, phosphate radius = 9.0 Å
- Scores Watson-Crick H-bonds (AU = 2, GC = 3), pi-stacking (inter-base-pair), electrostatics
- Brownian (Langevin) MD at 300 K: D = 0.008 Å²/ps, harmonic restraint k = 1.2 kcal/mol/Å²
- 2,000 steps × dt = 0.005 ps; RMSD recorded every 100 steps

#### 2D/3D Structure visualisation

```bash
python3 step_structures.py
```

**Reads:** `step2_6_results.json`  
**Writes:** `fig6_2d_combined.png`, `fig7_3d_complex.png`, `fig8_cofold_structure.png`, `siRNA_mcrA_duplex.pdb`

**What it does:**
- Circular chord diagrams (ViennaRNA `RNA.simple_circplot_coordinates()`)
- Linear arc diagrams with siRNA target site highlighted
- Co-fold duplex diagram (strand-specific colour coding: red = siRNA, blue = mRNA)
- 3D A-form duplex: 3 coarse-grained sites per nucleotide (P, S, B), three orthogonal views
- PDB export: Chain A = mRNA target, Chain B = siRNA guide (compatible with PyMOL / UCSF Chimera / VMD)

---

## Key Results (pre-computed)

These values are stored in `step2_6_results.json` and `step7_9_results.json`:

| Parameter | Value |
|---|---|
| Target gene | *mcrA* (MCR alpha subunit) |
| Species panel | 8 diverse methanogenic archaea |
| Conserved window conservation | 96.9% |
| siRNA guide strand (5′→3′) | `UGCCUGCUUUGAUGCCUGC` |
| Target mRNA sequence (5′→3′) | `GCAGGCAUCAAAGCAGGCA` |
| GC content | 57.9% |
| 5′ asymmetry criterion | Met (weak 5′ end) |
| Target mRNA MFE | −8.20 kcal/mol |
| siRNA guide MFE | −2.10 kcal/mol |
| Co-fold complex MFE | −41.20 kcal/mol |
| Target accessibility (P_unpaired) | 0.581 |
| siRNA conformations (5 kcal/mol) | 22 |
| mRNA conformations (5 kcal/mol) | 94 |
| Watson-Crick H-bonds | 49 (7 AU + 12 GC × correct valence) |
| Pi-stacking contacts | 36 |
| Metal-coordinating contacts | 5 |
| MD final RMSD | 0.54 Å (stable) |

---

## Adapting the Pipeline to Other Targets

To apply this pipeline to a different gene or organism, modify the `REPRESENTATIVE_SEQS` dictionary at the top of `sirna_pipeline.py`:

```python
REPRESENTATIVE_SEQS = {
    "Species_name_1": "ATGAGCAGC...",   # DNA or RNA sequence (both accepted)
    "Species_name_2": "ATGCTTGAC...",
    # Add 5–50 sequences for robust conservation analysis
}
```

All downstream parameters (alignment, siRNA scoring, ViennaRNA analysis, docking) will automatically adapt to the new sequences. For targets with very high or very low GC content, consider adjusting the GC filter bounds in `step2_msa_sirna.py`:

```python
GC_MIN, GC_MAX = 30, 62   # adjust as needed
```

---

## Output Files Reference

### JSON intermediates

**`step2_6_results.json`** — fields used by downstream scripts:
```
aln_len          — alignment length (nt)
best_w           — start position of conserved window
best_s           — conservation score of best window (0–1)
conserved_rna    — 40-nt conserved mRNA sequence (RNA alphabet)
selected_sirna:
  guide          — 19-nt siRNA guide strand (5′→3′)
  target         — 19-nt mRNA target sequence
  gc             — GC content (%)
  cmfe           — co-fold MFE (kcal/mol)
  asym           — 5′ asymmetry criterion met (boolean)
sirna_struct     — dot-bracket notation of siRNA (ViennaRNA MFE)
mrna_struct      — dot-bracket notation of mRNA conserved region
sirna_mfe        — siRNA guide MFE (kcal/mol)
mrna_mfe         — mRNA conserved region MFE (kcal/mol)
cofold_mfe       — siRNA:mRNA co-fold MFE (kcal/mol)
target_accessibility — mean unpairing probability at target site
sirna_conformations  — number of siRNA suboptimal structures
mrna_conformations   — number of mRNA suboptimal structures
```

**`step7_9_results.json`** — fields used by manuscript builder:
```
best_docking_energy   — total docking score (kcal/mol)
wc_hbonds_total       — Watson-Crick hydrogen bond count
pi_stacking           — pi-stacking interaction count
metal_contacts        — metal-coordinating contact count
md:
  final_rmsd_A        — final RMSD in Ångströms
  stability           — "stable" / "unstable"
  n_steps             — MD steps completed
```

### PDB file

`siRNA_mcrA_duplex.pdb` encodes a coarse-grained A-form duplex model with three sites per nucleotide:
- `P` — phosphate group (r = 9.0 Å from helix axis)
- `S` — sugar C4′ atom (r = 7.0 Å)
- `B` — base centroid (r = 5.0 Å)

Load into PyMOL with:
```
pymol siRNA_mcrA_duplex.pdb
```

Or into UCSF Chimera:
```
chimera siRNA_mcrA_duplex.pdb
```

---

## The 14-Step Pipeline (Parikesit et al. 2022)

The full pipeline, as documented in the validated reference framework:

```
Step 1   — Target gene selection & sequence retrieval (NCBI / Ensembl)
Step 2   — Multiple sequence alignment (MAFFT / ClustalX)        ← step2_msa_sirna.py
Step 3   — Phylogenetic tree construction (IQ-TREE / ClustalX)   ← sirna_pipeline.py
Step 4   — siRNA design from conserved region (RNAxs criterion)  ← step2_msa_sirna.py
Step 5   — MSA visualisation & target localisation (Jalview)     ← step2_msa_sirna.py
Step 6   — Conserved mRNA 2D structure (RNAalifold / ViennaRNA)  ← step2_msa_sirna.py
Step 7   — Individual 2D structures — siRNA & mRNA (RNAfold)     ← step_structures.py
Step 8   — 2D structural diversity (Barriers / RNA.subopt)       ← step2_msa_sirna.py
Step 9   — 3D structure de novo modelling (A-form geometry)      ← step_structures.py
Step 10  — 3D structure validation (MolProbity — external)
Step 11  — Energy protonation & minimisation (AVOGADRO — external)
Step 12  — RNA-RNA molecular docking (HNADOCK / custom)          ← step7_duplex.py
Step 13  — Chemical interaction prediction (IntaRNA — external)
Step 14  — 3D interaction profiling (PLIP / UCSF Chimera — ext.) ← siRNA_mcrA_duplex.pdb
```

Steps marked **external** require tools not implemented in Python here (MolProbity, Avogadro, IntaRNA, PLIP) and are documented in the manuscript Methods section.

---

## Citation

If you use this pipeline or any scripts in this repository, please cite:

> Parikesit AA, Ansori ANM, Kharisma VD. 2022. A Computational Design of siRNA in SARS-CoV-2 Spike Glycoprotein Gene and Its Binding Capability toward mRNA. *Indones. J. Chem.* 22(5):1163–1176. https://doi.org/10.22146/ijc.68415

> Parikesit AA. 2025. Silencing methyl-coenzyme M reductase (*mcr*) gene complex with siRNA as a mean to cessation of methane production in methanogens. Octopus.ac Research Problem. https://doi.org/10.57874/gjc79460

---

## Corresponding Contact

**Dr.rer.nat. Arli Aditya Parikesit**  
Department of Biotechnology, i3L University
Jakarta Timur, DKI Jakarta, Indonesia  
arli(dot)parikesit(at)i3l(dot)ac(dot)id  
ORCID: https://orcid.org/0000-0001-8716-3926

---
**AI Assistance Disclaimer**: This codebase was developed with the assistance of Claude Code. While the AI provided code generation, debugging, and structural support, the human developer maintains full responsibility for reviewing, testing, and maintaining all content and functionality.

*Last updated: June 2026*
