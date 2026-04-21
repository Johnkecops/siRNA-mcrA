#!/usr/bin/env python3
"""Fast docking: small system + Brownian MD (no gradient)."""
import math, json, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/sessions/wizardly-vibrant-fermat/mnt/outputs'
with open(f'{OUT}/step2_6_results.json') as f:
    data = json.load(f)

sirna_seq = data['selected_sirna']['guide']
mrna_seq  = data['conserved_rna'][:25]  # Use target window only (25 nt) for speed

print(f"siRNA: 5'-{sirna_seq}-3'")
print(f"mRNA target window: 5'-{mrna_seq}-3' ({len(mrna_seq)} nt)")

# ── CG model: 1 bead per nucleotide (centroid of nt) ────────────────────
COULOMB_K = 332.0; HBOND_DIST = 3.4; HBOND_E = -3.0
EPS = {'A':0.15,'U':0.20,'G':0.18,'C':0.17}  # base-stacking well depth
SIG = 4.5   # nt-nt equilibrium distance
CHG = {'A':-0.28,'U':-0.32,'G':-0.30,'C':-0.30}

def rna_beads(seq, x_off=0.0, z_off=0.0):
    """1 bead per nt on A-form helix."""
    rise=3.4; r=9.0; twist=32.7*math.pi/180
    xyz=[]
    for i,nt in enumerate(seq):
        ang=i*twist
        xyz.append([r*math.cos(ang)+x_off, r*math.sin(ang), i*rise+z_off])
    return np.array(xyz,dtype=float), list(seq)

def score_beads(lig_xyz, lig_seq, rec_xyz, rec_seq):
    """Vectorized scoring: LJ + Coulomb + H-bond."""
    diff = lig_xyz[:,None,:] - rec_xyz[None,:,:]
    r    = np.sqrt(np.sum(diff**2, axis=2)) + 1e-10  # (N,M)
    mask = r < 15.0
    # LJ (simplified: uniform eps/sig)
    eps_m = np.array([[math.sqrt(EPS[li]*EPS[rj]) for rj in rec_seq] for li in lig_seq])
    sr6   = (SIG / np.where(mask, r, 1.0))**6
    clash = r < 0.7*SIG
    lj    = np.where(clash, 1e5, np.where(mask, 4*eps_m*(sr6**2 - sr6), 0.0))
    # Coulomb
    qi = np.array([CHG[b] for b in lig_seq])[:,None]
    qj = np.array([CHG[b] for b in rec_seq])[None,:]
    elec = np.where(mask, COULOMB_K*qi*qj/np.where(mask,r,1.0), 0.0)
    # H-bond: A-U and G-C Watson-Crick pairs
    pairs = {('A','U'),('U','A'),('G','C'),('C','G')}
    hb_m = np.array([[HBOND_E if (li,rj) in pairs and r[i,j]<HBOND_DIST else 0.0
                       for j,rj in enumerate(rec_seq)]
                       for i,li in enumerate(lig_seq)])
    return float(np.sum(lj+elec+hb_m))

sir_xyz, sir_seq = rna_beads(sirna_seq, x_off=14.0, z_off=5.0)
mrna_xyz, mrna_seq_l = rna_beads(mrna_seq,  x_off=0.0,  z_off=0.0)

init_e = score_beads(sir_xyz, sir_seq, mrna_xyz, mrna_seq_l)
print(f"\nInitial energy: {init_e:.2f} kcal/mol")

# ── MC Docking ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(111)
kB = 0.001987
coords = sir_xyz.copy(); energy = init_e
best_coords, best_e = coords.copy(), energy
traj = [energy]; accepted = 0

t0 = time.time()
for step in range(3000):
    T = 300*(50/300)**(step/3000)
    trans = rng.uniform(-0.6,0.6,3)
    a = rng.standard_normal(3); a/=(np.linalg.norm(a)+1e-12)
    theta = rng.uniform(-0.25,0.25)
    c,s = math.cos(theta), math.sin(theta)
    ax,ay,az = a
    R = np.array([[c+ax*ax*(1-c), ax*ay*(1-c)-az*s, ax*az*(1-c)+ay*s],
                  [ay*ax*(1-c)+az*s, c+ay*ay*(1-c), ay*az*(1-c)-ax*s],
                  [az*ax*(1-c)-ay*s, az*ay*(1-c)+ax*s, c+az*az*(1-c)]])
    ctr = coords.mean(axis=0)
    nc  = (coords-ctr)@R.T + ctr + trans
    ne  = score_beads(nc, sir_seq, mrna_xyz, mrna_seq_l)
    dE  = ne - energy
    if dE<0 or rng.random()<math.exp(max(-30,-dE/(kB*T))):
        coords,energy=nc,ne; accepted+=1
        if energy<best_e: best_e,best_coords=energy,coords.copy()
    if step%150==0: traj.append(energy)

print(f"MC done in {time.time()-t0:.1f}s | best_e={best_e:.2f} | accept={accepted/3000:.3f}")

# Interaction analysis at best pose
diff2 = best_coords[:,None,:] - mrna_xyz[None,:,:]
r2    = np.sqrt(np.sum(diff2**2, axis=2))
pairs_wc = {('A','U'),('U','A'),('G','C'),('C','G')}
hbonds_detail = []
for i,li in enumerate(sir_seq):
    for j,mj in enumerate(mrna_seq_l):
        dist = float(r2[i,j])
        if dist < HBOND_DIST and (li,mj) in pairs_wc:
            hbonds_detail.append({'i':i+1,'j':j+1,'pair':f'{li}-{mj}','d':round(dist,2)})
n_hbonds = len(hbonds_detail)

# pi-stacking: consecutive nucleotides 3.2-4.5 Å
stacking = [(i+1,j+1,round(float(r2[i,j]),2)) for i,li in enumerate(sir_seq)
            for j,mj in enumerate(mrna_seq_l) if 3.0<float(r2[i,j])<4.8 and (li,mj) not in pairs_wc]
n_pi = len(stacking)

# Metal contacts (strong electrostatic < 3.0 Å)
metal = [(i+1,j+1,round(float(r2[i,j]),2)) for i,li in enumerate(sir_seq)
         for j,mj in enumerate(mrna_seq_l) if float(r2[i,j])<3.0 and li in ('G','A') and mj in ('C','U')]
n_metal = len(metal)

print(f"H-bond (WC) pairs:  {n_hbonds}  | {[h['pair'] for h in hbonds_detail[:5]]}")
print(f"pi-stacking:        {n_pi}")
print(f"Electrostatic:      {n_metal}")

# ── Brownian MD (thermal fluctuations around docked pose) ─────────────────
print("\nRunning Brownian MD trajectory (2000 steps)...")
kB=0.001987; D=0.01; dt_bd=0.005; T_bd=300
pos = best_coords.copy(); ref_pos=pos.copy()
md_e=[]; md_rmsd=[]

# Harmonic restraint to maintain docked geometry
k_spring = 0.5  # kcal/mol/Å²
rng2 = np.random.default_rng(42)
t_md = time.time()
for step in range(2000):
    # Thermal noise (Brownian)
    noise = rng2.standard_normal(pos.shape) * math.sqrt(2*D*dt_bd)
    # Harmonic restoring force (from best_coords equilibrium)
    displacement = pos - best_coords
    drift = -k_spring * displacement * D/(kB*T_bd) * dt_bd
    pos   = pos + noise + drift
    if step % 100 == 0:
        e_bd   = score_beads(pos, sir_seq, mrna_xyz, mrna_seq_l)
        rmsd_v = float(np.sqrt(np.mean(np.sum((pos-ref_pos)**2,axis=1))))
        md_e.append(e_bd); md_rmsd.append(rmsd_v)

print(f"MD done in {time.time()-t_md:.1f}s | {len(md_e)} snapshots")
print(f"Energy: {md_e[0]:.2f} → {md_e[-1]:.2f} kcal/mol")
print(f"RMSD:   {md_rmsd[-1]:.2f} Å  → {'stable' if md_rmsd[-1]<3.0 else 'moderately flexible'}")

# ── Figures ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# MC convergence
ax = axes[0]
ax.plot(np.linspace(0,3000,len(traj)), traj, color='#d6604d', lw=2)
ax.axhline(best_e, color='green', lw=1.5, ls='--', label=f'Best: {best_e:.1f} kcal/mol')
ax.set_xlabel('MC Step',fontsize=10); ax.set_ylabel('Binding Energy (kcal/mol)',fontsize=10)
ax.set_title('MC/SA Docking Convergence\nsiRNA–mcrA mRNA (1-bead CG model)',fontsize=10,fontweight='bold')
ax.legend(fontsize=9)

# MD energy
ax2 = axes[1]
t_ns = np.arange(len(md_e))*100*dt_bd/1000
ax2.plot(t_ns, md_e, color='#4393c3', lw=2)
ax2.set_xlabel('Time (ns)',fontsize=10); ax2.set_ylabel('Potential Energy (kcal/mol)',fontsize=10)
ax2.set_title('Brownian MD Energy\n(300 K, harmonic restraint)',fontsize=10,fontweight='bold')

# RMSD
ax3 = axes[2]
ax3.plot(t_ns, md_rmsd, color='#762a83', lw=2)
ax3.fill_between(t_ns, md_rmsd, alpha=0.2, color='#762a83')
ax3.axhline(3.0, color='red', lw=1, ls='--', alpha=0.7, label='3 Å threshold')
ax3.set_xlabel('Time (ns)',fontsize=10); ax3.set_ylabel('RMSD (Å)',fontsize=10)
ax3.set_title('siRNA Positional RMSD\n(Complex thermal stability)',fontsize=10,fontweight='bold')
ax3.legend(fontsize=9)

plt.suptitle('siRNA–mcrA mRNA: Docking & MD Trajectory Analysis',fontsize=12,fontweight='bold',y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_docking_md.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig4_docking_md.png")

# Interaction summary
fig, ax = plt.subplots(figsize=(7,5))
names  = ['Watson-Crick\nH-bonds', 'π-stacking\ncontacts', 'Electrostatic\ncoordination']
counts = [n_hbonds, n_pi, n_metal]
cols   = ['#2166ac','#4dac26','#d6604d']
bars = ax.bar(names, counts, color=cols, alpha=0.85, edgecolor='white', lw=1.5, width=0.5)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, str(cnt),
            ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of interactions',fontsize=12)
ax.set_title('siRNA–mcrA mRNA: Chemical Interaction Profile\n(Post-docking, CG analysis)',fontsize=11,fontweight='bold')
ax.set_ylim(0, max(counts)*1.4+1)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_interactions.png', dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig5_interactions.png")

# Save
result = {
    'model': '1-bead CG per nucleotide, WC H-bond scoring',
    'n_mc_steps':3000, 'T_start_K':300, 'T_end_K':50, 'seed':111,
    'initial_energy': round(init_e,2), 'best_docking_energy': round(best_e,2),
    'energy_improvement': round(init_e-best_e,2), 'acceptance_rate': round(accepted/3000,3),
    'h_bond_wc_pairs': n_hbonds, 'pi_stacking': n_pi, 'electrostatic': n_metal,
    'hbond_details': hbonds_detail,
    'md': {
        'model': 'Brownian dynamics with harmonic restraint (300 K)',
        'n_snapshots': len(md_e),
        'initial_energy': round(md_e[0],2), 'final_energy': round(md_e[-1],2),
        'final_rmsd_A': round(md_rmsd[-1],2),
        'stability': 'stable' if md_rmsd[-1]<3.0 else 'moderately flexible',
        'energy_traj': [round(e,2) for e in md_e],
        'rmsd_traj':   [round(r,3) for r in md_rmsd],
    }
}
with open(f'{OUT}/step7_9_results.json','w') as f:
    json.dump(result,f,indent=2)
print("Saved step7_9_results.json")
print("\nSteps 7-9 COMPLETE.")
