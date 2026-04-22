#!/usr/bin/env python3
"""
OpenMM MD for siRNA-mcrA duplex — v2
Strategy:
  1. Read rna_pdbfixed.pdb
  2. Pre-relax bad bond geometry using AMBER bond parameters
     (move atoms iteratively to satisfy ideal bond lengths)
  3. Minimise (now converges fast because energy is reasonable)
  4. Run 5000-step NVT (10 ps) at 300 K
  5. Save results JSON + 3-panel figure

Force field : AMBER14 + GBn2 implicit solvent
"""
import sys, os, json, time
import numpy as np

OUT      = '/sessions/wizardly-vibrant-fermat/mnt/outputs'
FIXED_PDB = os.path.join(OUT, 'rna_pdbfixed.pdb')

def p(msg): print(msg, flush=True)

p("=" * 60)
p("OpenMM siRNA:mcrA duplex MD  (AMBER14 + GBn2)")
p("=" * 60)

# ── imports ────────────────────────────────────────────────────────────────────
p("\n[1] Importing OpenMM …")
from openmm.app import (PDBFile, ForceField, Simulation, NoCutoff, HBonds)
from openmm import (LangevinMiddleIntegrator, Platform, HarmonicBondForce)
from openmm import unit as u
import openmm as mm
p("    OK")

# ── load structure ─────────────────────────────────────────────────────────────
p(f"\n[2] Reading {FIXED_PDB} …")
pdb = PDBFile(FIXED_PDB)
n_atoms = pdb.topology.getNumAtoms()
p(f"    Atoms: {n_atoms}")

# Extract positions as numpy array (in Ångström)
positions_A = np.array(pdb.positions.value_in_unit(u.angstrom))

# ── build system to get ideal bond lengths ────────────────────────────────────
p("\n[3] Loading force field & extracting ideal bond lengths …")
ff = ForceField('amber14-all.xml', 'implicit/gbn2.xml')

# Build system without constraints first so we can read all bond params
sys_nc = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff,
                         constraints=None)

# Collect ideal bonds: {(i,j): r0_A}
ideal_bonds = {}
for fi in range(sys_nc.getNumForces()):
    force = sys_nc.getForce(fi)
    if isinstance(force, HarmonicBondForce):
        for bi in range(force.getNumBonds()):
            i, j, r0, k = force.getBondParameters(bi)
            r0_A = r0.value_in_unit(u.angstrom)
            ideal_bonds[(min(i,j), max(i,j))] = r0_A
p(f"    Bond parameters read: {len(ideal_bonds)}")

# ── pre-relax: iterative bond-length correction ───────────────────────────────
p("\n[4] Pre-relaxing bad bond lengths (iterative SHAKE-like pass) …")

pos = positions_A.copy()

def correct_bonds(pos, ideal_bonds, max_iter=30, tol_A=0.05):
    """
    Iteratively move atom pairs toward ideal bond lengths.
    On each iteration, for every bond where |d - r0| > tol_A,
    move both atoms half-way to the ideal distance.
    """
    n_corrected_total = 0
    for it in range(max_iter):
        n_fixed = 0
        for (i, j), r0 in ideal_bonds.items():
            vec = pos[j] - pos[i]
            d   = np.linalg.norm(vec)
            if d < 1e-6:
                continue
            err = d - r0
            if abs(err) > tol_A:
                correction = (err / 2.0) * (vec / d)
                pos[i] += correction
                pos[j] -= correction
                n_fixed += 1
        n_corrected_total += n_fixed
        if n_fixed == 0:
            p(f"    Converged at iteration {it+1}")
            break
    return pos, n_corrected_total

t0 = time.time()
pos, n_corr = correct_bonds(pos, ideal_bonds, max_iter=50, tol_A=0.02)
p(f"    Bond corrections applied: {n_corr}  ({time.time()-t0:.2f}s)")

# Check worst remaining bond error
worst = 0.0
worst_bond = None
for (i, j), r0 in ideal_bonds.items():
    d = np.linalg.norm(pos[j] - pos[i])
    err = abs(d - r0)
    if err > worst:
        worst = err
        worst_bond = (i, j, r0, d)
p(f"    Worst remaining bond error: {worst:.4f} Å  (atoms {worst_bond[0]}-{worst_bond[1]}: "
  f"ideal {worst_bond[2]:.3f} Å, actual {worst_bond[3]:.3f} Å)")

# ── save pre-relaxed PDB ───────────────────────────────────────────────────────
from openmm.unit import Quantity, angstrom
pos_q = Quantity(pos.tolist(), angstrom)

prerelax_pdb = os.path.join(OUT, 'rna_prerelaxed.pdb')
with open(prerelax_pdb, 'w') as fh:
    PDBFile.writeFile(pdb.topology, pos_q, fh)
p(f"    Pre-relaxed PDB: {prerelax_pdb}")

# ── build system WITH constraints (for MD) ────────────────────────────────────
p("\n[5] Building final OpenMM system (HBonds constrained) …")
system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff,
                         constraints=HBonds)
p(f"    Particles: {system.getNumParticles()}")

# ── integrator & simulation ────────────────────────────────────────────────────
integrator = LangevinMiddleIntegrator(
    300 * u.kelvin, 1.0 / u.picosecond, 0.002 * u.picoseconds)
integrator.setRandomNumberSeed(42)

for pname in ('CUDA', 'OpenCL', 'CPU'):
    try:
        platform = Platform.getPlatformByName(pname)
        p(f"    Platform: {pname}")
        break
    except Exception:
        pass

sim = Simulation(pdb.topology, system, integrator, platform)
sim.context.setPositions(pos_q)

# ── check initial energy with pre-relaxed coords ──────────────────────────────
e_pre = sim.context.getState(getEnergy=True).getPotentialEnergy()
e_pre_kc = e_pre.value_in_unit(u.kilocalories_per_mole)
p(f"\n    Energy after bond pre-relax: {e_pre_kc:,.1f} kcal/mol")

# ── energy minimization ────────────────────────────────────────────────────────
p("\n[6] Energy minimization (tol=100 kJ/mol/nm, max=2000 iter) …")
t0 = time.time()
sim.minimizeEnergy(
    tolerance=100 * u.kilojoules_per_mole / u.nanometer,
    maxIterations=2000)
dt_min = time.time() - t0

state_min = sim.context.getState(getEnergy=True, getPositions=True)
e_min = state_min.getPotentialEnergy().value_in_unit(u.kilocalories_per_mole)
p(f"    Post-min energy: {e_min:,.1f} kcal/mol  ({dt_min:.1f}s)")

min_pdb = os.path.join(OUT, 'rna_minimized.pdb')
with open(min_pdb, 'w') as fh:
    PDBFile.writeFile(sim.topology, state_min.getPositions(), fh)
p(f"    Minimized PDB: {min_pdb}")

ref_pos_nm = state_min.getPositions(asNumpy=True)  # nm

# ── backbone atoms for RMSD ────────────────────────────────────────────────────
BB_NAMES = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'",
            "C2'", "C1'", "O4'"}
bb_idx = [a.index for a in sim.topology.atoms() if a.name in BB_NAMES]
ref_bb  = ref_pos_nm[bb_idx] * 10.0   # nm → Å
p(f"\n    Backbone atoms for RMSD: {len(bb_idx)}")

# ── NVT production (5000 steps = 10 ps) ───────────────────────────────────────
N_STEPS  = 5000
REPORT   = 500
DT_PS    = 0.002
total_ps = N_STEPS * DT_PS

p(f"\n[7] NVT production: {N_STEPS} steps × {DT_PS} ps = {total_ps:.0f} ps …")
sim.context.setVelocitiesToTemperature(300 * u.kelvin)

times_ps  = []
energies  = []
rmsds     = []

t_prod = time.time()
for chunk in range(N_STEPS // REPORT):
    sim.step(REPORT)
    state = sim.context.getState(getEnergy=True, getPositions=True)
    t_ps  = (chunk + 1) * REPORT * DT_PS
    e_kc  = state.getPotentialEnergy().value_in_unit(u.kilocalories_per_mole)
    pos   = state.getPositions(asNumpy=True)[bb_idx] * 10.0   # Å
    rmsd  = float(np.sqrt(np.mean(np.sum((pos - ref_bb) ** 2, axis=1))))
    times_ps.append(round(t_ps, 3))
    energies.append(round(e_kc, 2))
    rmsds.append(round(rmsd, 4))
    p(f"    t={t_ps:5.1f} ps  E={e_kc:12.2f} kcal/mol  RMSD={rmsd:.4f} Å")

dt_prod = time.time() - t_prod
p(f"\n    Production done in {dt_prod:.1f}s")

# ── save final structure ───────────────────────────────────────────────────────
state_fin = sim.context.getState(getPositions=True)
fin_pdb = os.path.join(OUT, 'rna_final.pdb')
with open(fin_pdb, 'w') as fh:
    PDBFile.writeFile(sim.topology, state_fin.getPositions(), fh)
p(f"    Final PDB: {fin_pdb}")

# ── statistics ─────────────────────────────────────────────────────────────────
mean_e = float(np.mean(energies))
std_e  = float(np.std(energies))
mean_r = float(np.mean(rmsds))
fin_r  = float(rmsds[-1])
fin_e  = float(energies[-1])

p("\n── Summary ──────────────────────────────────────────────────")
p(f"  Pre-relax energy  : {e_pre_kc:>14,.1f} kcal/mol")
p(f"  Post-min  energy  : {e_min:>14,.1f} kcal/mol")
p(f"  Mean MD energy    : {mean_e:>14,.1f} ± {std_e:.1f} kcal/mol")
p(f"  Final energy      : {fin_e:>14,.1f} kcal/mol")
p(f"  Mean backbone RMSD: {mean_r:.4f} Å")
p(f"  Final RMSD        : {fin_r:.4f} Å")

# ── JSON ───────────────────────────────────────────────────────────────────────
results = {
    "simulation": {
        "software":          "OpenMM 8.5.1",
        "force_field":       "AMBER14 (amber14-all.xml)",
        "implicit_solvent":  "GBn2 (implicit/gbn2.xml)",
        "temperature_K":     300,
        "timestep_ps":       DT_PS,
        "n_steps_production":N_STEPS,
        "production_time_ps":total_ps,
        "n_atoms_total":     int(n_atoms),
        "n_backbone_atoms":  len(bb_idx),
    },
    "energy": {
        "pre_relax_kcal_mol":  round(e_pre_kc, 2),
        "post_min_kcal_mol":   round(e_min,    2),
        "mean_md_kcal_mol":    round(mean_e,   2),
        "std_md_kcal_mol":     round(std_e,    2),
        "final_kcal_mol":      round(fin_e,    2),
    },
    "rmsd": {
        "mean_backbone_A":  round(mean_r, 4),
        "final_backbone_A": round(fin_r,  4),
        "trajectory_A":     rmsds,
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

# ── figure ─────────────────────────────────────────────────────────────────────
p("\n[8] Generating figure …")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

axes[0].plot(times_ps, energies, color='royalblue', lw=1.8)
axes[0].axhline(mean_e, color='tomato', ls='--', lw=1.2,
                label=f'Mean: {mean_e:.0f} kcal/mol')
axes[0].set_xlabel('Time (ps)', fontsize=11)
axes[0].set_ylabel('Potential Energy (kcal/mol)', fontsize=11)
axes[0].set_title('A  Potential Energy', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)

axes[1].plot(times_ps, rmsds, color='seagreen', lw=1.8)
axes[1].axhline(mean_r, color='tomato', ls='--', lw=1.2,
                label=f'Mean: {mean_r:.3f} Å')
axes[1].set_xlabel('Time (ps)', fontsize=11)
axes[1].set_ylabel('Backbone RMSD (Å)', fontsize=11)
axes[1].set_title('B  Backbone RMSD', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)

axes[2].hist(rmsds, bins=max(3, len(rmsds) // 2), color='mediumpurple',
             edgecolor='white', alpha=0.85)
axes[2].axvline(mean_r, color='tomato', ls='--', lw=1.5,
                label=f'Mean: {mean_r:.3f} Å')
axes[2].set_xlabel('Backbone RMSD (Å)', fontsize=11)
axes[2].set_ylabel('Frequency', fontsize=11)
axes[2].set_title('C  RMSD Distribution', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=9)

fig.suptitle(
    f'OpenMM MD: siRNA–mcrA mRNA duplex  |  AMBER14 + GBn2  |  300 K  |  {total_ps:.0f} ps NVT',
    fontsize=11, y=1.01)
plt.tight_layout()

fig_out = os.path.join(OUT, 'fig_md_openmm.png')
plt.savefig(fig_out, dpi=180, bbox_inches='tight')
p(f"    Figure: {fig_out}")
p("\n✓ All done.")
