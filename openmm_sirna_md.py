#!/usr/bin/env python3
"""
openmm_sirna_md.py
==================
All-atom NVT molecular dynamics of an RNA oligonucleotide (single-stranded siRNA
guide strand) using OpenMM 8.5.1 with AMBER14 force field and GBn2 implicit solvent.

Pipeline:
  1. Build A-form RNA fiber geometry from user-supplied sequence
  2. PDBFixer: reconstruct missing heavy atoms + add hydrogens (pH 7.4)
  3. Bond pre-relaxation: correct fiber-model bond length errors iteratively
  4. Energy minimization (L-BFGS, AMBER14 + GBn2)
  5. NVT production MD (LangevinMiddle, 300 K, 2 fs)
  6. Backbone RMSD analysis (11 heavy atoms/residue)
  7. Save: JSON results, 3-panel figure, PDB structures

Usage:
  python openmm_sirna_md.py                        # uses default mcrA siRNA
  python openmm_sirna_md.py --seq UGCCUGCUUUGAUGCCUGC --steps 5000 --out ./results

Requirements:
  pip install openmm pdbfixer numpy matplotlib

Reference:
  Parikesit AA (2026). Computational siRNA design targeting mcrA in methanogenic
  archaea. Simulated with OpenMM 8.5.1 / AMBER14 / GBn2 / 300 K NVT.
  Author: Dr. Arli Aditya Parikesit, i3L University Jakarta.
  ORCID: 0000-0001-8716-3926
"""

import sys
import os
import json
import time
import argparse
import tempfile
import numpy as np

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='OpenMM siRNA guide strand MD pipeline')
    p.add_argument('--seq',   default='UGCCUGCUUUGAUGCCUGC',
                   help='RNA sequence (5\'→3\', U not T). Default: mcrA siRNA guide strand.')
    p.add_argument('--steps', type=int, default=2000,
                   help='NVT production steps (2 fs each). Default 2000 = 4 ps.')
    p.add_argument('--report',type=int, default=200,
                   help='Report every N steps. Default 200.')
    p.add_argument('--temp',  type=float, default=300.0,
                   help='Temperature in Kelvin. Default 300.')
    p.add_argument('--seed',  type=int, default=42,
                   help='Random seed for reproducibility. Default 42.')
    p.add_argument('--out',   default='.',
                   help='Output directory. Default: current directory.')
    p.add_argument('--tol',   type=float, default=100.0,
                   help='Minimization tolerance kJ/mol/nm. Default 100.')
    p.add_argument('--maxiter', type=int, default=2000,
                   help='Max minimization iterations. Default 2000.')
    return p.parse_args()


# ── A-form fiber geometry ──────────────────────────────────────────────────────
RISE  = 2.81   # Å rise per residue (Arnott & Hukins 1972)
TWIST = 32.7   # degrees twist per residue

# (atom_name, helix_radius_Å, phi_offset_deg, z_offset_Å)
BACKBONE_FIBER = [
    ("P",    8.90,   0.0,   0.00),
    ("OP1",  7.65,   8.5,   1.35),
    ("OP2",  9.80,  -5.5,  -0.85),
    ("O5'",  7.45,  -9.0,  -1.20),
    ("C5'",  6.55,  -7.0,  -0.15),
    ("C4'",  5.75,   9.5,   0.12),
    ("O4'",  5.05,  23.5,  -0.08),
    ("C3'",  4.85,  19.0,   1.25),
    ("O3'",  5.80,  32.0,   2.55),
    ("C2'",  4.15,   4.0,   1.55),
    ("O2'",  3.30,  -2.0,   2.75),
    ("C1'",  4.75,  -7.5,   0.28),
]

# Minimal base atom for each nucleotide type (C1'→N glycosidic bond stub)
BASE_STUB = {
    'A': [('N9', 3.60,  -22.0,  0.30)],
    'G': [('N9', 3.60,  -22.0,  0.30)],
    'C': [('N1', 3.90,  -18.0,  0.35)],
    'U': [('N1', 3.90,  -18.0,  0.35)],
}

RNA_RESNAME = {'A': 'A', 'G': 'G', 'C': 'C', 'U': 'U'}


def build_rna_pdb(sequence: str, pdb_path: str) -> int:
    """
    Build a single-stranded A-form RNA PDB from a sequence string.

    Atom positions are placed using cylindrical fiber coordinates
    (Arnott & Hukins 1972). Bond lengths will be geometrically approximate
    (~30-100% error) — Step 4 corrects these before minimization.

    Returns the number of ATOM records written.
    """
    lines = []
    atom_idx = 1
    n_res = len(sequence)

    for i, nt in enumerate(sequence.upper()):
        if nt not in RNA_RESNAME:
            raise ValueError(f"Unknown nucleotide '{nt}' at position {i+1}. Use A/G/C/U.")

        helix_angle = i * TWIST   # degrees
        helix_z     = i * RISE    # Å

        # First residue: 5'-terminus (no phosphate group)
        # Skip P, OP1, OP2 for residue 0
        atoms_to_place = BACKBONE_FIBER if i > 0 else [
            a for a in BACKBONE_FIBER if a[0] not in ('P', 'OP1', 'OP2')
        ]

        # Set residue name: terminal variants
        if i == 0:
            resname = RNA_RESNAME[nt]      # PDBFixer handles 5'-OH
        elif i == n_res - 1:
            resname = RNA_RESNAME[nt]      # PDBFixer handles 3'-OH
        else:
            resname = RNA_RESNAME[nt]

        chain_id = 'A'
        res_seq  = i + 1

        for (aname, r, phi_off, z_off) in atoms_to_place + BASE_STUB.get(nt, []):
            phi = np.deg2rad(helix_angle + phi_off)
            x   = r * np.cos(phi)
            y   = r * np.sin(phi)
            z   = helix_z + z_off
            element = aname[0]  # first character as element
            lines.append(
                f"ATOM  {atom_idx:5d} {aname:<4s} {resname:<3s} {chain_id}"
                f"{res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
            )
            atom_idx += 1

    lines.append("END\n")
    with open(pdb_path, 'w') as fh:
        fh.writelines(lines)
    return atom_idx - 1


# ── PDBFixer: reconstruct + protonate ─────────────────────────────────────────
def fix_pdb(input_pdb: str, output_pdb: str, ph: float = 7.4) -> object:
    """Run PDBFixer to add missing atoms and hydrogens."""
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    fixer = PDBFixer(filename=input_pdb)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    with open(output_pdb, 'w') as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)

    return fixer


# ── Bond pre-relaxation ────────────────────────────────────────────────────────
def relax_bonds(positions_A: np.ndarray, ideal_bonds: dict,
                max_iter: int = 50, tol_A: float = 0.02) -> tuple:
    """
    Iteratively correct bond lengths to match AMBER ideal values.

    This corrects the systematic errors introduced by the cylindrical fiber model
    (e.g., P-O5' placed at 2.27 Å instead of ideal 1.59 Å). Each iteration
    moves both bonded atoms symmetrically toward the ideal distance.

    Parameters
    ----------
    positions_A : np.ndarray, shape (N, 3)
        Atomic positions in Ångström.
    ideal_bonds : dict
        Mapping (i, j) -> r0_A (ideal bond length in Å) from AMBER14 force field.
    max_iter : int
        Maximum correction iterations (typically converges in 4–20).
    tol_A : float
        Bond error tolerance in Å below which no correction is applied.

    Returns
    -------
    pos_corrected : np.ndarray
        Corrected positions.
    n_total_corrections : int
        Total number of bond corrections applied across all iterations.
    """
    pos = positions_A.copy()
    n_total = 0
    for it in range(max_iter):
        n_fixed = 0
        for (i, j), r0 in ideal_bonds.items():
            vec = pos[j] - pos[i]
            d   = np.linalg.norm(vec)
            if d < 1e-6:
                continue
            err = d - r0
            if abs(err) > tol_A:
                c = (err / 2.0) * (vec / d)
                pos[i] += c
                pos[j] -= c
                n_fixed += 1
        n_total += n_fixed
        if n_fixed == 0:
            print(f"    Bond relax converged at iteration {it + 1}", flush=True)
            break
    return pos, n_total


# ── Main MD pipeline ───────────────────────────────────────────────────────────
def run_md(args):
    from openmm.app import (PDBFile, ForceField, Simulation,
                            NoCutoff, HBonds)
    from openmm import (LangevinMiddleIntegrator, Platform,
                        HarmonicBondForce)
    from openmm import unit

    os.makedirs(args.out, exist_ok=True)
    OUT = args.out

    def p(msg): print(msg, flush=True)

    p("=" * 64)
    p("OpenMM siRNA Guide Strand MD  (AMBER14 + GBn2 Implicit Solvent)")
    p("=" * 64)
    p(f"  Sequence  : {args.seq}")
    p(f"  Length    : {len(args.seq)} nt")
    p(f"  Temp      : {args.temp} K")
    p(f"  Steps     : {args.steps} × 2 fs = {args.steps * 0.002:.1f} ps")
    p(f"  Output    : {OUT}")

    # ── Step 1: Build fiber geometry ──────────────────────────────────────────
    p("\n[1] Building A-form fiber geometry …")
    init_pdb = os.path.join(OUT, 'rna_initial.pdb')
    n_heavy  = build_rna_pdb(args.seq, init_pdb)
    p(f"    Heavy atoms written: {n_heavy}  → {init_pdb}")

    # ── Step 2: PDBFixer ──────────────────────────────────────────────────────
    p("\n[2] PDBFixer: adding missing atoms + hydrogens (pH 7.4) …")
    fixed_pdb = os.path.join(OUT, 'rna_fixed.pdb')
    fixer     = fix_pdb(init_pdb, fixed_pdb)
    n_atoms   = fixer.topology.getNumAtoms()
    p(f"    Total atoms after PDBFixer: {n_atoms}  → {fixed_pdb}")

    pdb = PDBFile(fixed_pdb)

    # ── Step 3: Force field + ideal bond lengths ───────────────────────────────
    p("\n[3] Loading AMBER14 + GBn2 force field …")
    ff = ForceField('amber14-all.xml', 'implicit/gbn2.xml')

    sys_nc = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff,
                             constraints=None)

    ideal_bonds = {}
    for fi in range(sys_nc.getNumForces()):
        force = sys_nc.getForce(fi)
        if isinstance(force, HarmonicBondForce):
            for bi in range(force.getNumBonds()):
                i, j, r0, k = force.getBondParameters(bi)
                r0_A = r0.value_in_unit(unit.angstrom)
                ideal_bonds[(min(i, j), max(i, j))] = r0_A
    p(f"    Bond parameters extracted: {len(ideal_bonds)}")

    # ── Step 4: Bond pre-relaxation ───────────────────────────────────────────
    p("\n[4] Iterative bond pre-relaxation (tol = 0.02 Å, max 50 iter) …")
    pos_A  = np.array(pdb.positions.value_in_unit(unit.angstrom))
    t0     = time.time()
    pos_A, n_corr = relax_bonds(pos_A, ideal_bonds, max_iter=50, tol_A=0.02)
    p(f"    Total corrections: {n_corr}  ({time.time()-t0:.2f}s)")

    # Report worst remaining error
    worst_err, worst_info = 0.0, None
    for (i, j), r0 in ideal_bonds.items():
        d   = np.linalg.norm(pos_A[j] - pos_A[i])
        err = abs(d - r0)
        if err > worst_err:
            worst_err  = err
            worst_info = (i, j, r0, d)
    p(f"    Worst remaining error: {worst_err:.4f} Å "
      f"(atoms {worst_info[0]}-{worst_info[1]}: ideal {worst_info[2]:.3f} Å, "
      f"actual {worst_info[3]:.3f} Å)")

    # Save pre-relaxed
    from openmm.unit import Quantity, angstrom
    pos_q        = Quantity(pos_A.tolist(), angstrom)
    prerelax_pdb = os.path.join(OUT, 'rna_prerelaxed.pdb')
    with open(prerelax_pdb, 'w') as fh:
        PDBFile.writeFile(pdb.topology, pos_q, fh)
    p(f"    Pre-relaxed PDB: {prerelax_pdb}")

    # ── Step 5: Build constrained system ──────────────────────────────────────
    p("\n[5] Building OpenMM system (NoCutoff, HBonds constrained) …")
    system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff,
                             constraints=HBonds)
    p(f"    Particles: {system.getNumParticles()}")

    # ── Step 6: Integrator + simulation ───────────────────────────────────────
    integrator = LangevinMiddleIntegrator(
        args.temp * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds
    )
    integrator.setRandomNumberSeed(args.seed)

    platform = None
    for pname in ('CUDA', 'OpenCL', 'CPU'):
        try:
            platform = Platform.getPlatformByName(pname)
            p(f"    Platform: {pname}")
            break
        except Exception:
            pass

    sim = Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(pos_q)

    e_premin = sim.context.getState(getEnergy=True).getPotentialEnergy()
    p(f"\n    Energy before minimization: "
      f"{e_premin.value_in_unit(unit.kilocalories_per_mole):,.1f} kcal/mol")

    # ── Step 7: Energy minimization ───────────────────────────────────────────
    p(f"\n[6] Energy minimization (tol={args.tol} kJ/mol/nm, max {args.maxiter} iter) …")
    t0 = time.time()
    sim.minimizeEnergy(
        tolerance=args.tol * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=args.maxiter
    )
    dt_min = time.time() - t0

    state_min = sim.context.getState(getEnergy=True, getPositions=True)
    e_min     = state_min.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    p(f"    Post-min energy: {e_min:,.1f} kcal/mol  ({dt_min:.1f}s)")

    min_pdb = os.path.join(OUT, 'rna_minimized.pdb')
    with open(min_pdb, 'w') as fh:
        PDBFile.writeFile(sim.topology, state_min.getPositions(), fh)
    p(f"    Minimized PDB: {min_pdb}")

    ref_pos_nm = state_min.getPositions(asNumpy=True)   # nm

    # Backbone atom indices for RMSD
    BB_NAMES = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'",
                "O3'", "C2'", "C1'", "O4'"}
    bb_idx = [a.index for a in sim.topology.atoms() if a.name in BB_NAMES]
    ref_bb = ref_pos_nm[bb_idx] * 10.0   # nm → Å
    p(f"    Backbone atoms for RMSD: {len(bb_idx)} "
      f"({len(bb_idx) // len(args.seq)} per residue × {len(args.seq)} nt)")

    # ── Step 8: NVT production MD ─────────────────────────────────────────────
    p(f"\n[7] NVT production: {args.steps} steps × 2 fs = "
      f"{args.steps * 0.002:.1f} ps at {args.temp} K …")
    sim.context.setVelocitiesToTemperature(args.temp * unit.kelvin)

    times_ps, energies, rmsds = [], [], []
    t_prod = time.time()

    for chunk in range(args.steps // args.report):
        sim.step(args.report)
        state = sim.context.getState(getEnergy=True, getPositions=True)
        t_ps  = (chunk + 1) * args.report * 0.002
        e_kc  = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        pos   = state.getPositions(asNumpy=True)[bb_idx] * 10.0   # nm → Å
        rmsd  = float(np.sqrt(np.mean(np.sum((pos - ref_bb) ** 2, axis=1))))

        times_ps.append(round(t_ps, 3))
        energies.append(round(e_kc, 2))
        rmsds.append(round(rmsd, 4))

        elapsed = time.time() - t_prod
        p(f"    t={t_ps:6.2f}ps  E={e_kc:12.2f} kcal/mol  "
          f"RMSD={rmsd:.4f} Å  [{elapsed:.1f}s elapsed]")

    dt_prod = time.time() - t_prod
    p(f"\n    Production done in {dt_prod:.1f}s")

    # ── Step 9: Save final PDB ────────────────────────────────────────────────
    fin_pdb = os.path.join(OUT, 'rna_final.pdb')
    with open(fin_pdb, 'w') as fh:
        PDBFile.writeFile(sim.topology,
                          sim.context.getState(getPositions=True).getPositions(), fh)
    p(f"    Final PDB: {fin_pdb}")

    # ── Statistics ────────────────────────────────────────────────────────────
    mean_e = float(np.mean(energies))
    std_e  = float(np.std(energies))
    mean_r = float(np.mean(rmsds))
    fin_r  = float(rmsds[-1])
    fin_e  = float(energies[-1])

    p("\n── Results Summary ──────────────────────────────────────────────")
    p(f"  Sequence              : {args.seq}")
    p(f"  Length                : {len(args.seq)} nt  |  {n_atoms} atoms (all-atom + H)")
    p(f"  Force field           : AMBER14 + GBn2 implicit solvent")
    p(f"  Temperature           : {args.temp} K")
    p(f"  Production time       : {args.steps * 0.002:.1f} ps")
    p(f"  Post-min energy       : {e_min:.1f} kcal/mol")
    p(f"  Mean MD energy        : {mean_e:.1f} ± {std_e:.1f} kcal/mol")
    p(f"  Final energy          : {fin_e:.1f} kcal/mol")
    p(f"  Mean backbone RMSD    : {mean_r:.4f} Å")
    p(f"  Final backbone RMSD   : {fin_r:.4f} Å")

    # ── JSON output ───────────────────────────────────────────────────────────
    results = {
        "simulation": {
            "software":           "OpenMM 8.5.1",
            "force_field":        "AMBER14 (amber14-all.xml)",
            "implicit_solvent":   "GBn2 (implicit/gbn2.xml)",
            "temperature_K":      args.temp,
            "timestep_ps":        0.002,
            "random_seed":        args.seed,
            "n_steps_production": args.steps,
            "production_time_ps": round(args.steps * 0.002, 3),
            "n_atoms_all":        n_atoms,
            "n_backbone_atoms":   len(bb_idx),
            "sequence":           args.seq,
            "length_nt":          len(args.seq),
        },
        "energy": {
            "post_min_kcal_mol":  round(e_min,    2),
            "mean_md_kcal_mol":   round(mean_e,   2),
            "std_md_kcal_mol":    round(std_e,    2),
            "final_kcal_mol":     round(fin_e,    2),
        },
        "rmsd": {
            "mean_backbone_A":  round(mean_r, 4),
            "final_backbone_A": round(fin_r,  4),
            "trajectory":       rmsds,
        },
        "trajectory": {
            "times_ps":          times_ps,
            "energies_kcal_mol": energies,
        },
    }

    json_out = os.path.join(OUT, 'openmm_md_results.json')
    with open(json_out, 'w') as fh:
        json.dump(results, fh, indent=2)
    p(f"\n    Results JSON: {json_out}")

    # ── Figure ────────────────────────────────────────────────────────────────
    p("\n[8] Generating 3-panel figure …")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: potential energy
    axes[0].plot(times_ps, energies, color='royalblue', lw=1.8)
    axes[0].axhline(mean_e, color='tomato', ls='--', lw=1.2,
                    label=f'Mean: {mean_e:.0f} kcal/mol')
    axes[0].set_xlabel('Time (ps)', fontsize=11)
    axes[0].set_ylabel('Potential Energy (kcal/mol)', fontsize=11)
    axes[0].set_title('A  Potential Energy', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)

    # Panel B: backbone RMSD
    axes[1].plot(times_ps, rmsds, color='seagreen', lw=1.8)
    axes[1].axhline(mean_r, color='tomato', ls='--', lw=1.2,
                    label=f'Mean: {mean_r:.3f} Å')
    axes[1].set_xlabel('Time (ps)', fontsize=11)
    axes[1].set_ylabel('Backbone RMSD (Å)', fontsize=11)
    axes[1].set_title('B  Backbone RMSD', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)

    # Panel C: RMSD histogram
    nbins = max(3, len(rmsds) // 2)
    axes[2].hist(rmsds, bins=nbins, color='mediumpurple',
                 edgecolor='white', alpha=0.85)
    axes[2].axvline(mean_r, color='tomato', ls='--', lw=1.5,
                    label=f'Mean: {mean_r:.3f} Å')
    axes[2].set_xlabel('Backbone RMSD (Å)', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title('C  RMSD Distribution', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=9)

    fig.suptitle(
        f'OpenMM NVT MD: {args.seq}  |  AMBER14+GBn2  |  {args.temp:.0f} K  |  '
        f'{args.steps * 0.002:.1f} ps',
        fontsize=10, y=1.02
    )
    plt.tight_layout()

    fig_out = os.path.join(OUT, 'fig_md_openmm.png')
    plt.savefig(fig_out, dpi=180, bbox_inches='tight')
    plt.close()
    p(f"    Figure: {fig_out}")

    p("\n✓ Pipeline complete.")
    return results


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args = parse_args()
    run_md(args)
