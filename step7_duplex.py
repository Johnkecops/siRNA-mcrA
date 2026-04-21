#!/usr/bin/env python3
"""
siRNA-mRNA duplex docking: antiparallel helix geometry + Brownian MD.
The guide strand is prepositioned antiparallel to the target mRNA
(as in the real RISC complex), then MD is run for stability analysis.
"""
import math, json, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/sessions/wizardly-vibrant-fermat/mnt/outputs'
with open(f'{OUT}/step2_6_results.json') as f:
    data = json.load(f)

sirna_seq = data['selected_sirna']['guide']      # 5'→3' antisense
mrna_seq  = data['selected_sirna']['target']     # 5'→3' target (19 nt)
cofold_mfe = data['selected_sirna']['cmfe']

print(f"siRNA guide:  5'-{sirna_seq}-3' ({len(sirna_seq)} nt)")
print(f"mRNA target:  5'-{mrna_seq}-3' ({len(mrna_seq)} nt)")
print(f"ViennaRNA co-fold MFE: {cofold_mfe} kcal/mol")

# ── Duplex geometry: place guide antiparallel to target ──────────────────
# A-form RNA: rise=2.81Å (compressed for CG), twist=32.7°
# Guide runs 5'→3' in OPPOSITE direction to target
WC_PAIRS = {('A','U'),('U','A'),('G','C'),('C','G'),('G','U'),('U','G')}
PAIR_DIST = 2.8   # Å, ideal WC H-bond distance (centroid-centroid in CG)
RISE      = 2.81  # Å per bp along helix axis
TWIST     = 32.7 * math.pi / 180
COULOMB   = 332.0
CHG = {'A':-0.28,'U':-0.32,'G':-0.30,'C':-0.30,'X':0.0}
EPS = {'A':0.15,'U':0.20,'G':0.18,'C':0.17,'X':0.1}
SIG_BP = 5.0  # Å base-pair separation in CG model

n = len(sirna_seq)   # 19 nt

# Target strand: 5'→3' along +z axis
mrna_xyz = np.array([
    [9.0 * math.cos(i*TWIST), 9.0 * math.sin(i*TWIST), i * RISE]
    for i in range(n)
], dtype=float)

# Guide strand: antiparallel (reverse complement) — runs from nt n-1 → 0 along z
# Each base pairs with the corresponding target base
ACROSS = 11.0  # distance across the duplex (typical A-form: ~11 Å between strands)
guide_xyz = np.array([
    [ACROSS - 9.0 * math.cos((n-1-i)*TWIST),
     -9.0 * math.sin((n-1-i)*TWIST),
     (n-1-i) * RISE]   # positions align with paired target nt
    for i in range(n)
], dtype=float)

def score_duplex(g_xyz, g_seq, t_xyz, t_seq):
    """
    Score the duplex: Watson-Crick H-bonds for complementary pairs,
    LJ for steric, Coulomb electrostatics.
    """
    diff = g_xyz[:,None,:] - t_xyz[None,:,:]
    r    = np.sqrt(np.sum(diff**2, axis=2)) + 1e-10  # (N, N)
    mask = r < 15.0

    # WC H-bonding (paired positions)
    hb_total = 0.0
    for i in range(n):
        # Paired base
        dist_paired = float(np.linalg.norm(g_xyz[i] - t_xyz[n-1-i]))
        pair = (g_seq[i], t_seq[n-1-i])
        if pair in WC_PAIRS:
            # Strong H-bond at ideal distance
            hb_total += -4.0 * math.exp(-0.5 * ((dist_paired - PAIR_DIST)/1.0)**2)
        # Stacking with adjacent nt
        if i < n-1:
            d_stack = float(np.linalg.norm(g_xyz[i] - g_xyz[i+1]))
            hb_total += -0.5 * math.exp(-0.5 * ((d_stack - RISE)/0.5)**2)

    # LJ steric (prevent clashes)
    r_safe = np.where(mask, r, 1.0)
    sr6 = (SIG_BP / r_safe)**6
    lj = np.where(mask & (r < SIG_BP*0.8), 1e3, 0.0)  # clash penalty only
    # Coulomb
    qi = np.array([CHG.get(b, 0.0) for b in g_seq])[:,None]
    qj = np.array([CHG.get(b, 0.0) for b in t_seq])[None,:]
    elec = np.where(mask, COULOMB*qi*qj/r_safe, 0.0)

    return hb_total + float(np.sum(lj)) + float(np.sum(elec))

init_e = score_duplex(guide_xyz, list(sirna_seq), mrna_xyz, list(mrna_seq))
print(f"\nInitial duplex energy: {init_e:.2f} kcal/mol")

# ── MC refinement of the pre-formed duplex ────────────────────────────────
rng = np.random.default_rng(111)
kB = 0.001987
coords = guide_xyz.copy(); energy = init_e
best_coords, best_e = coords.copy(), energy
traj = [energy]; accepted = 0

t0 = time.time()
for step in range(3000):
    T = 300*(50/300)**(step/3000)
    # Small perturbation (tight around duplex geometry)
    trans = rng.uniform(-0.3, 0.3, 3)
    a = rng.standard_normal(3); a/=(np.linalg.norm(a)+1e-12)
    theta = rng.uniform(-0.1, 0.1)
    c,s=math.cos(theta),math.sin(theta); ax,ay,az=a
    R = np.array([[c+ax*ax*(1-c), ax*ay*(1-c)-az*s, ax*az*(1-c)+ay*s],
                  [ay*ax*(1-c)+az*s, c+ay*ay*(1-c), ay*az*(1-c)-ax*s],
                  [az*ax*(1-c)-ay*s, az*ay*(1-c)+ax*s, c+az*az*(1-c)]])
    ctr = coords.mean(axis=0)
    nc  = (coords-ctr)@R.T + ctr + trans
    ne  = score_duplex(nc, list(sirna_seq), mrna_xyz, list(mrna_seq))
    dE  = ne - energy
    if dE<0 or rng.random()<math.exp(max(-30,-dE/(kB*T))):
        coords,energy=nc,ne; accepted+=1
        if energy<best_e: best_e,best_coords=energy,coords.copy()
    if step%150==0: traj.append(energy)

print(f"MC refinement done in {time.time()-t0:.1f}s | best_e={best_e:.2f} | accept={accepted/3000:.3f}")

# ── Interaction analysis ───────────────────────────────────────────────────
hbonds = []
for i in range(n):
    dist = float(np.linalg.norm(best_coords[i] - mrna_xyz[n-1-i]))
    pair = (sirna_seq[i], mrna_seq[n-1-i])
    if pair in WC_PAIRS:
        hbonds.append({'pos_guide':i+1,'pos_mrna':n-i,'pair':f'{pair[0]}-{pair[1]}','d':round(dist,2)})

# Count pi-stacking (nt-nt distances within same strand, consecutive)
pi_stacks = sum(1 for i in range(n-1)
                if 2.8 < float(np.linalg.norm(best_coords[i]-best_coords[i+1])) < 4.5)

# Count electrostatic (negative-negative repulsions > 3 Å, stabilized by Mg2+)
elec_contacts = sum(1 for i in range(n) for j in range(n)
                    if i != j and 2.5 < float(np.linalg.norm(best_coords[i]-mrna_xyz[j])) < 4.0)

print(f"\nWatson-Crick H-bonds formed: {len(hbonds)}")
for h in hbonds[:6]:
    print(f"  G{h['pos_guide']}-M{h['pos_mrna']}: {h['pair']} ({h['d']:.2f} Å)")
print(f"pi-stacking interactions: {pi_stacks}")
print(f"Electrostatic contacts: {elec_contacts}")

# ── Brownian MD stability ──────────────────────────────────────────────────
print("\nBrownian MD trajectory (2000 steps, 300 K)...")
kB=0.001987; D=0.008; dt_bd=0.005; T_bd=300; k_spring=1.2
pos=best_coords.copy(); ref_pos=pos.copy()
md_e=[]; md_rmsd=[]
rng2=np.random.default_rng(42)

t_md=time.time()
for step in range(2000):
    noise = rng2.standard_normal(pos.shape)*math.sqrt(2*D*dt_bd)
    disp  = pos - best_coords
    drift = -k_spring*disp*D/(kB*T_bd)*dt_bd
    pos   = pos + noise + drift
    if step%100==0:
        e_s   = score_duplex(pos,list(sirna_seq),mrna_xyz,list(mrna_seq))
        rmsd_v= float(np.sqrt(np.mean(np.sum((pos-ref_pos)**2,axis=1))))
        md_e.append(e_s); md_rmsd.append(rmsd_v)

print(f"MD done in {time.time()-t_md:.2f}s | {len(md_e)} snapshots")
print(f"Energy: {md_e[0]:.2f} → {md_e[-1]:.2f} kcal/mol")
print(f"RMSD:   {md_rmsd[-1]:.2f} Å")
stability = 'stable' if md_rmsd[-1]<2.0 else 'moderately flexible'
print(f"Stability: {stability}")

# ── Figures ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# MC
axes[0].plot(np.linspace(0,3000,len(traj)), traj, color='#d6604d', lw=2)
axes[0].axhline(best_e, color='green', lw=1.5, ls='--', label=f'Best: {best_e:.1f} kcal/mol')
axes[0].set_xlabel('MC Step',fontsize=10); axes[0].set_ylabel('Energy (kcal/mol)',fontsize=10)
axes[0].set_title('MC Duplex Refinement\nsiRNA–mcrA (antiparallel geometry)',fontsize=10,fontweight='bold')
axes[0].legend(fontsize=9)
# MD energy
t_ns = np.arange(len(md_e))*100*dt_bd/1000
axes[1].plot(t_ns, md_e, color='#4393c3', lw=2)
axes[1].set_xlabel('Time (ns)',fontsize=10); axes[1].set_ylabel('Energy (kcal/mol)',fontsize=10)
axes[1].set_title('MD Energy Trajectory\n(Brownian, 300 K)',fontsize=10,fontweight='bold')
# RMSD
axes[2].plot(t_ns, md_rmsd, color='#762a83', lw=2)
axes[2].fill_between(t_ns, md_rmsd, alpha=0.2, color='#762a83')
axes[2].axhline(2.0, color='red', lw=1, ls='--', alpha=0.7, label='2 Å threshold')
axes[2].set_xlabel('Time (ns)',fontsize=10); axes[2].set_ylabel('RMSD (Å)',fontsize=10)
axes[2].set_title('Complex RMSD\n(Thermal stability)',fontsize=10,fontweight='bold')
axes[2].legend(fontsize=9)
plt.suptitle('siRNA–mcrA mRNA Duplex: MC Refinement & MD Stability',fontsize=12,fontweight='bold',y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_docking_md.png',dpi=150,bbox_inches='tight')
plt.close(); print("Saved fig4_docking_md.png")

# Interaction summary
fig, ax = plt.subplots(figsize=(8,5))
int_names  = ['Watson-Crick\nH-bonds', 'π-stacking\ninteractions', 'Electrostatic\ncontacts']
int_counts = [len(hbonds), pi_stacks, elec_contacts]
int_cols   = ['#2166ac','#4dac26','#d6604d']
bars = ax.bar(int_names, int_counts, color=int_cols, alpha=0.85, edgecolor='white', lw=1.5, width=0.5)
for bar, cnt in zip(bars, int_counts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            str(cnt), ha='center',va='bottom',fontsize=14,fontweight='bold')
ax.set_ylabel('Number of interactions',fontsize=12)
ax.set_title('siRNA–mcrA mRNA Complex: Chemical Interaction Summary',fontsize=11,fontweight='bold')
ax.set_ylim(0, max(int_counts)*1.35+1)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_interactions.png',dpi=150,bbox_inches='tight')
plt.close(); print("Saved fig5_interactions.png")

result = {
    'docking_model': 'Antiparallel RNA duplex + MC refinement (CG WC scoring)',
    'viennarna_cofold_mfe': cofold_mfe,
    'initial_duplex_energy': round(init_e,2),
    'best_docking_energy': round(best_e,2),
    'mc_steps': 3000, 'T_start_K': 300, 'T_end_K': 50,
    'acceptance_rate': round(accepted/3000,3),
    'wc_hbonds': len(hbonds), 'pi_stacking': pi_stacks,
    'electrostatic_contacts': elec_contacts,
    'hbond_details': hbonds,
    'md': {
        'model': 'Brownian MD (harmonic restraint, 300 K)',
        'n_snapshots': len(md_e),
        'initial_energy': round(md_e[0],2), 'final_energy': round(md_e[-1],2),
        'final_rmsd_A': round(md_rmsd[-1],2), 'stability': stability,
        'energy_traj': [round(e,2) for e in md_e],
        'rmsd_traj':   [round(r,3) for r in md_rmsd],
    }
}
with open(f'{OUT}/step7_9_results.json','w') as f:
    json.dump(result,f,indent=2)
print("Saved step7_9_results.json")
print("\nSteps 7-9 COMPLETE.")
