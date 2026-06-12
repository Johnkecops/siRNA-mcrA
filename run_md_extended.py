#!/usr/bin/env python3
"""
run_md_extended.py
==================
Extended, resumable all-atom NVT molecular dynamics of an RNA oligonucleotide
(siRNA guide strand) using OpenMM with the AMBER14 force field and GBn2 implicit
solvent. This is the optimised successor to openmm_sirna_md.py, addressing the
reviewer request to extend the production run far beyond the original 4 ps.

Optimisations vs. openmm_sirna_md.py
------------------------------------
  * Hydrogen-mass repartitioning (HMR, 1.5 amu) + 4 fs timestep -> ~2x throughput
    on CPU while preserving energy conservation (Hopkins et al., 2015, JCTC).
  * Checkpoint / resume: simulation state is serialised to disk so a long run can
    be accumulated across many short wall-clock windows (needed on a laptop / in a
    time-limited sandbox). Each invocation advances the trajectory and is safe to
    interrupt.
  * Per-frame backbone coordinates are stored, enabling backbone RMSD, per-residue
    RMSF and radius of gyration (Rg) without external trajectory libraries.
  * CUDA / OpenCL / CPU auto-detection; thread count honoured via OPENMM_CPU_THREADS.

For a definitive validation, run on a GPU with --target-ns 100 (or 200). On a
consumer CPU the same target is reached over a longer wall-clock; the resume
mechanism lets the run be split into convenient windows.

Usage
-----
  python run_md_extended.py init  --seq UGCCUGCUUUGAUGCCUGC --workdir ./md_run
  python run_md_extended.py chunk --workdir ./md_run --seconds 38   # repeat as needed
  python run_md_extended.py finalize --workdir ./md_run --out ./results
  # or, on capable hardware, a single shot:
  python run_md_extended.py auto  --seq UGCCUGCUUUGAUGCCUGC --target-ns 100 --workdir ./md_run --out ./results

Author : Dr. Arli Aditya Parikesit, i3L University, Jakarta (ORCID 0000-0001-8716-3926)
"""
import os, sys, json, time, argparse
import numpy as np

os.environ.setdefault("OPENMM_CPU_THREADS", "4")

DEFAULT_SEQ = "UGCCUGCUUUGAUGCCUGC"
TIMESTEP_PS = 0.004          # 4 fs (HMR)
HMR_AMU     = 1.5
TEMP_K      = 300.0
FRAME_STEPS = 250            # record a frame every 250 steps = 1.0 ps
BB_NAMES = {"P","OP1","OP2","O5'","C5'","C4'","C3'","O3'","C2'","C1'","O4'"}

# ---- geometry / fixer helpers reused from the validated upstream script ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openmm_sirna_md import build_rna_pdb, fix_pdb, relax_bonds


def _platform():
    from openmm import Platform
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            return Platform.getPlatformByName(name)
        except Exception:
            continue
    return None


def _build_simulation(workdir):
    """Construct Simulation object (system is rebuilt deterministically each call)."""
    from openmm.app import PDBFile, ForceField, NoCutoff, HBonds, Simulation
    from openmm import LangevinMiddleIntegrator, unit
    pdb = PDBFile(os.path.join(workdir, "topology.pdb"))
    ff = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff,
                             constraints=HBonds, hydrogenMass=HMR_AMU * unit.amu)
    integ = LangevinMiddleIntegrator(TEMP_K * unit.kelvin, 1.0 / unit.picosecond,
                                     TIMESTEP_PS * unit.picoseconds)
    sim = Simulation(pdb.topology, system, integ, _platform())
    return sim, pdb


def cmd_init(args):
    from openmm.app import PDBFile, ForceField, NoCutoff, HBonds, Simulation
    from openmm import LangevinMiddleIntegrator, HarmonicBondForce, unit
    wd = args.workdir
    os.makedirs(wd, exist_ok=True)
    # 1-4: fiber build, fixer, bond pre-relax
    init_pdb = os.path.join(wd, "rna_initial.pdb")
    build_rna_pdb(args.seq, init_pdb)
    fix_pdb(init_pdb, os.path.join(wd, "topology.pdb"))
    pdb = PDBFile(os.path.join(wd, "topology.pdb"))
    ff = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    sys_nc = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=None)
    ideal = {}
    for fi in range(sys_nc.getNumForces()):
        f = sys_nc.getForce(fi)
        if isinstance(f, HarmonicBondForce):
            for bi in range(f.getNumBonds()):
                i, j, r0, k = f.getBondParameters(bi)
                ideal[(min(i, j), max(i, j))] = r0.value_in_unit(unit.angstrom)
    pos = np.array(pdb.positions.value_in_unit(unit.angstrom))
    pos, ncorr = relax_bonds(pos, ideal, 50, 0.02)
    posq = unit.Quantity(pos.tolist(), unit.angstrom)
    # build production system + minimise
    sim, _ = _build_simulation(wd)
    sim.context.setPositions(posq)
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    t0 = time.time()
    sim.minimizeEnergy(tolerance=100 * unit.kilojoule_per_mole / unit.nanometer, maxIterations=2000)
    st = sim.context.getState(getEnergy=True, getPositions=True)
    emin = st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    sim.context.setVelocitiesToTemperature(TEMP_K * unit.kelvin)
    sim.saveState(os.path.join(wd, "state.xml"))
    # reference backbone (minimised) in Angstrom
    bb_idx = [a.index for a in sim.topology.atoms() if a.name in BB_NAMES]
    refpos = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    np.save(os.path.join(wd, "bb_idx.npy"), np.array(bb_idx))
    np.save(os.path.join(wd, "ref_bb.npy"), refpos[bb_idx])
    meta = {"seq": args.seq, "n_atoms": sim.system.getNumParticles(),
            "n_bb": len(bb_idx), "timestep_ps": TIMESTEP_PS, "temp_K": TEMP_K,
            "hmr_amu": HMR_AMU, "e_premin_kcal": round(e0, 1), "e_min_kcal": round(emin, 1),
            "frame_steps": FRAME_STEPS, "force_field": "AMBER14 (amber14-all.xml)",
            "implicit_solvent": "GBn2 (implicit/gbn2.xml)",
            "software": "OpenMM " + __import__("openmm").version.version,
            "steps_done": 0}
    json.dump(meta, open(os.path.join(wd, "meta.json"), "w"), indent=2)
    # empty trajectory arrays
    np.savez(os.path.join(wd, "frames.npz"),
             times_ps=np.zeros(0), energies=np.zeros(0),
             bb=np.zeros((0, len(bb_idx), 3)))
    print(f"INIT done: {meta['n_atoms']} atoms, {len(bb_idx)} bb atoms; "
          f"E_premin={e0:.1f}, E_min={emin:.1f} kcal/mol; bond corrections={ncorr}", flush=True)


def cmd_chunk(args):
    from openmm import unit
    wd = args.workdir
    meta = json.load(open(os.path.join(wd, "meta.json")))
    bb_idx = np.load(os.path.join(wd, "bb_idx.npy"))
    d = np.load(os.path.join(wd, "frames.npz"))
    times = list(d["times_ps"]); energies = list(d["energies"]); bb = list(d["bb"])
    sim, _ = _build_simulation(wd)
    sim.loadState(os.path.join(wd, "state.xml"))
    steps_done = meta["steps_done"]
    t_start = time.time()
    nblocks = 0
    while time.time() - t_start < args.seconds:
        sim.step(FRAME_STEPS)
        steps_done += FRAME_STEPS
        nblocks += 1
        st = sim.context.getState(getEnergy=True, getPositions=True)
        e = st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        times.append(steps_done * TIMESTEP_PS)
        energies.append(e)
        bb.append(pos[bb_idx])
    sim.saveState(os.path.join(wd, "state.xml"))
    meta["steps_done"] = steps_done
    json.dump(meta, open(os.path.join(wd, "meta.json"), "w"), indent=2)
    np.savez(os.path.join(wd, "frames.npz"),
             times_ps=np.array(times), energies=np.array(energies), bb=np.array(bb))
    print(f"CHUNK +{nblocks} frames ({nblocks*FRAME_STEPS*TIMESTEP_PS:.1f} ps) in "
          f"{time.time()-t_start:.1f}s | total {steps_done*TIMESTEP_PS/1000:.4f} ns "
          f"({len(times)} frames) | lastE={energies[-1]:.1f}", flush=True)


def _rmsd_kabsch(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Wt.T @ V.T))
    D = np.diag([1, 1, d])
    R = Wt.T @ D @ V.T
    Pr = Pc @ R.T
    return np.sqrt(np.mean(np.sum((Pr - Qc) ** 2, axis=1))), Pr, Qc


def cmd_finalize(args):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    wd = args.workdir; out = args.out; os.makedirs(out, exist_ok=True)
    meta = json.load(open(os.path.join(wd, "meta.json")))
    d = np.load(os.path.join(wd, "frames.npz"))
    times = d["times_ps"]; energies = d["energies"]; bb = d["bb"]
    ref = np.load(os.path.join(wd, "ref_bb.npy"))
    n = len(times)
    seq = meta["seq"]; nres = len(seq); per = ref.shape[0] // nres
    # superposed RMSD vs minimised reference + aligned frames for RMSF
    rmsd = np.zeros(n); aligned = np.zeros_like(bb); rg = np.zeros(n)
    for i in range(n):
        r, Pr, Qc = _rmsd_kabsch(bb[i], ref)
        rmsd[i] = r
        aligned[i] = Pr + ref.mean(0)
        c = bb[i] - bb[i].mean(0)
        rg[i] = np.sqrt(np.mean(np.sum(c ** 2, axis=1)))
    # RMSF per atom over the second half (equilibrated) then averaged per residue
    half = n // 2 if n > 4 else 0
    traj = aligned[half:]
    mean_struct = traj.mean(0)
    rmsf_atom = np.sqrt(np.mean(np.sum((traj - mean_struct) ** 2, axis=2), axis=0))
    rmsf_res = np.array([rmsf_atom[k*per:(k+1)*per].mean() for k in range(nres)])
    res = {
        "simulation": {**{k: meta[k] for k in ("software","force_field","implicit_solvent",
                          "temp_K","timestep_ps","hmr_amu","n_atoms","n_bb","seq")},
                       "production_ns": round(float(times[-1])/1000, 4),
                       "production_ps": round(float(times[-1]), 1),
                       "n_frames": int(n), "frame_interval_ps": meta["frame_steps"]*meta["timestep_ps"]},
        "energy": {"post_min_kcal_mol": meta["e_min_kcal"],
                   "mean_kcal_mol": round(float(energies.mean()),2),
                   "std_kcal_mol": round(float(energies.std()),2),
                   "final_kcal_mol": round(float(energies[-1]),2)},
        "rmsd": {"mean_A": round(float(rmsd.mean()),3), "final_A": round(float(rmsd[-1]),3),
                 "plateau_mean_secondhalf_A": round(float(rmsd[half:].mean()),3),
                 "plateau_std_secondhalf_A": round(float(rmsd[half:].std()),3)},
        "rmsf_per_nt_A": [round(float(x),3) for x in rmsf_res],
        "rg": {"mean_A": round(float(rg.mean()),3), "std_A": round(float(rg.std()),3)},
    }
    json.dump(res, open(os.path.join(out, "openmm_md_extended_results.json"), "w"), indent=2)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0,0].plot(times, energies, color="royalblue", lw=1.0)
    ax[0,0].axhline(energies.mean(), color="tomato", ls="--", lw=1,
                    label=f"mean {energies.mean():.0f} kcal/mol")
    ax[0,0].set_xlabel("Time (ps)"); ax[0,0].set_ylabel("Potential energy (kcal/mol)")
    ax[0,0].set_title("A  Potential energy", fontweight="bold"); ax[0,0].legend(fontsize=8)
    ax[0,1].plot(times, rmsd, color="seagreen", lw=1.0)
    ax[0,1].axhline(rmsd[half:].mean(), color="tomato", ls="--", lw=1,
                    label=f"plateau {rmsd[half:].mean():.2f} ± {rmsd[half:].std():.2f} Å")
    ax[0,1].set_xlabel("Time (ps)"); ax[0,1].set_ylabel("Backbone RMSD (Å)")
    ax[0,1].set_title("B  Backbone RMSD vs minimised", fontweight="bold"); ax[0,1].legend(fontsize=8)
    ax[1,0].bar(range(1, nres+1), rmsf_res, color="mediumpurple", edgecolor="white")
    ax[1,0].set_xlabel("Guide-strand nucleotide position (5'->3')"); ax[1,0].set_ylabel("RMSF (Å)")
    ax[1,0].set_title("C  Per-nucleotide RMSF (2nd half)", fontweight="bold")
    ax[1,0].set_xticks(range(1, nres+1)); ax[1,0].set_xticklabels(list(seq), fontsize=7)
    ax[1,1].plot(times, rg, color="darkorange", lw=1.0)
    ax[1,1].axhline(rg.mean(), color="tomato", ls="--", lw=1, label=f"mean {rg.mean():.2f} Å")
    ax[1,1].set_xlabel("Time (ps)"); ax[1,1].set_ylabel("Radius of gyration (Å)")
    ax[1,1].set_title("D  Radius of gyration", fontweight="bold"); ax[1,1].legend(fontsize=8)
    fig.suptitle(f"OpenMM all-atom MD (AMBER14 + GBn2, {meta['temp_K']:.0f} K, HMR/4 fs) | "
                 f"{seq} | {times[-1]/1000:.3f} ns", y=1.0, fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(out, "fig_md_extended.png"), dpi=180, bbox_inches="tight")
    print("FINALIZE: wrote openmm_md_extended_results.json and fig_md_extended.png", flush=True)
    print(json.dumps(res["rmsd"]), json.dumps(res["rg"]), flush=True)


def cmd_auto(args):
    cmd_init(args)
    target_steps = int(args.target_ns * 1000 / TIMESTEP_PS)
    while json.load(open(os.path.join(args.workdir, "meta.json")))["steps_done"] < target_steps:
        cmd_chunk(args)
    cmd_finalize(args)


def main():
    ap = argparse.ArgumentParser(description="Extended resumable OpenMM RNA MD")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("init", "chunk", "finalize", "auto"):
        s = sub.add_parser(name)
        s.add_argument("--workdir", default="./md_run")
        s.add_argument("--seq", default=DEFAULT_SEQ)
        s.add_argument("--seconds", type=float, default=38.0)
        s.add_argument("--out", default="./results")
        s.add_argument("--target-ns", type=float, default=100.0, dest="target_ns")
    a = ap.parse_args()
    {"init": cmd_init, "chunk": cmd_chunk, "finalize": cmd_finalize, "auto": cmd_auto}[a.cmd](a)


if __name__ == "__main__":
    main()
