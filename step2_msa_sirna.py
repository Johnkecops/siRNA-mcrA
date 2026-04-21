#!/usr/bin/env python3
"""Step 2-6: MSA, Conservation, siRNA design, 2D structure, suboptimal analysis."""
import RNA, re, math, json, random
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Bio.Align import PairwiseAligner

OUT = '/sessions/wizardly-vibrant-fermat/mnt/outputs'

with open(f'{OUT}/seq_data.json') as f:
    data = json.load(f)
SEQS = data['seqs']

# ── Step 2: Progressive MSA ──────────────────────────────────────────────
seqs_list = list(SEQS.items())
ref_id, ref_seq = seqs_list[0]
aligner = PairwiseAligner()
aligner.mode = 'global'
aligner.match_score = 2
aligner.mismatch_score = -1
aligner.open_gap_score = -2
aligner.extend_gap_score = -0.5

aligned = {ref_id: ref_seq.replace('U', 'T')}
for sid, s in seqs_list[1:]:
    try:
        aln = aligner.align(ref_seq.replace('U', 'T'), s.replace('U', 'T'))[0]
        aligned[sid] = aln[1]
    except Exception:
        aligned[sid] = s.replace('U', 'T')

aln_len = max(len(v) for v in aligned.values())
padded = {k: v.ljust(aln_len, '-') for k, v in aligned.items()}
seqs_arr = list(padded.values())
print(f"MSA: {len(seqs_arr)} sequences, alignment length {aln_len} nt")

# ── Step 3: Conservation profile ────────────────────────────────────────
def cons_score(col):
    bases = [b.upper() for b in col if b != '-']
    if not bases:
        return 0.0
    return Counter(bases).most_common(1)[0][1] / len(bases)

scores = []
for i in range(aln_len):
    col = [s[i] for s in seqs_arr if i < len(s)]
    scores.append(cons_score(col))

WINDOW = 40
best_w, best_s = 0, 0
for i in range(aln_len - WINDOW):
    a = sum(scores[i:i+WINDOW]) / WINDOW
    if a > best_s:
        best_s, best_w = a, i

ref_aligned = padded[ref_id]
cons_dna = ref_aligned[best_w:best_w+WINDOW].replace('-', '')
cons_rna = cons_dna.upper().replace('T', 'U')
print(f"Best conserved window: pos {best_w}–{best_w+WINDOW}, mean conservation = {best_s:.3f}")
print(f"Conserved mRNA region ({len(cons_rna)} nt): {cons_rna}")

# Conservation figure
fig, ax = plt.subplots(figsize=(12, 4))
colors = ['#d73027' if c >= 0.9 else '#fc8d59' if c >= 0.75 else
          '#fee090' if c >= 0.5 else '#91bfdb' for c in scores]
ax.bar(range(len(scores)), scores, color=colors, width=1.0, edgecolor='none')
ax.axhline(0.75, color='#d73027', lw=1.5, ls='--', label='75% conservation threshold')
ax.axvline(best_w, color='green', lw=2, ls='-', alpha=0.8)
ax.axvline(best_w + WINDOW, color='green', lw=2, ls='-', alpha=0.8,
           label=f'siRNA target window (pos {best_w}–{best_w+WINDOW})')
ax.set_xlabel('Alignment position (nt)', fontsize=11)
ax.set_ylabel('Conservation score', fontsize=11)
ax.set_title('Per-Position Conservation Profile — mcrA (Methanogenic Archaea)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_conservation_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_conservation_profile.png")

# ── Step 4: siRNA Design ────────────────────────────────────────────────
comp_map = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}

def guide_from_target(target):
    return ''.join(comp_map.get(b, 'N') for b in reversed(target))

candidates = []
for i in range(len(cons_rna) - 18):
    target = cons_rna[i:i+19]
    guide = guide_from_target(target)
    if 'N' in guide:
        continue
    gc = (guide.count('G') + guide.count('C')) / 19
    if not (0.30 <= gc <= 0.62):
        continue
    if re.search(r'([ACGU])\1{3,}', guide):
        continue
    fiveprime_gc  = (guide[:3].count('G')  + guide[:3].count('C'))  / 3
    threeprime_gc = (guide[-3:].count('G') + guide[-3:].count('C')) / 3
    asymmetry_ok = fiveprime_gc <= threeprime_gc
    gs, gmfe = RNA.fold(guide)
    ts, tmfe = RNA.fold(target)
    _, cmfe  = RNA.cofold(guide + '&' + target)
    candidates.append({
        'pos': i + 1,
        'target': target,
        'guide':  guide,
        'gc':     round(gc * 100, 1),
        'tmfe':   round(tmfe, 2),
        'gmfe':   round(gmfe, 2),
        'cmfe':   round(cmfe, 2),
        'tstruct': ts,
        'gstruct': gs,
        'asym':   asymmetry_ok,
    })

candidates.sort(key=lambda x: x['cmfe'])  # most negative = tightest binding
top5 = candidates[:5]
top  = top5[0]
print(f"\nsiRNA candidates evaluated: {len(candidates)}")
print(f"\nTop 5 candidates:")
print(f"{'Rank':>4}  {'Pos':>4}  {'Target mRNA':>21}  {'Guide strand':>21}  {'GC%':>5}  {'tMFE':>7}  {'gMFE':>7}  {'cofold':>8}")
for rank, c in enumerate(top5, 1):
    marker = '*' if rank == 1 else ' '
    print(f"{marker}{rank:>3}  {c['pos']:>4}  {c['target']:>21}  {c['guide']:>21}  {c['gc']:>5}  {c['tmfe']:>7}  {c['gmfe']:>7}  {c['cmfe']:>8}")

print(f"\n★ SELECTED siRNA:")
print(f"  Position in conserved region: {top['pos']}")
print(f"  Target mRNA:  5'-{top['target']}-3'")
print(f"  Guide strand: 5'-{top['guide']}-3'")
print(f"  GC content: {top['gc']}%")
print(f"  Target MFE: {top['tmfe']} kcal/mol  | struct: {top['tstruct']}")
print(f"  Guide MFE:  {top['gmfe']} kcal/mol  | struct: {top['gstruct']}")
print(f"  Co-fold MFE: {top['cmfe']} kcal/mol")
print(f"  Asymmetry criterion (weak 5'): {'met' if top['asym'] else 'not met'}")

# ── Step 5: Full 2D structure analysis ────────────────────────────────
mrna_struct, mrna_mfe = RNA.fold(cons_rna)
sirna_struct, sirna_mfe = RNA.fold(top['guide'])
cofold_struct, cofold_mfe = RNA.cofold(top['guide'] + '&' + cons_rna)

print(f"\n2D Structure Analysis:")
print(f"  mRNA conserved region MFE: {mrna_mfe:.2f} kcal/mol  struct: {mrna_struct}")
print(f"  siRNA guide MFE: {sirna_mfe:.2f} kcal/mol  struct: {sirna_struct}")
print(f"  Co-fold MFE: {cofold_mfe:.2f} kcal/mol")

# Accessibility (partition function)
n = len(cons_rna)
fc = RNA.fold_compound(cons_rna)
_, mfe2 = fc.mfe()
fc.exp_params_rescale(mfe2)
fc.pf()
bpp = fc.bpp()
p_paired = np.zeros(n)
for i in range(1, n+1):
    for j in range(1, n+1):
        if i != j:
            p = bpp[min(i,j)][max(i,j)]
            p_paired[i-1] += p
p_unpaired = 1.0 - np.clip(p_paired, 0, 1)

tgt_start = top['pos'] - 1
tgt_end   = min(tgt_start + 19, n)
target_access = float(np.mean(p_unpaired[tgt_start:tgt_end]))
print(f"  siRNA target accessibility: {target_access:.3f}")

# ── Step 6: Suboptimal conformations ─────────────────────────────────
sirna_sub = RNA.subopt(top['guide'], int(5.0 * 100))
mrna_sub  = RNA.subopt(cons_rna, int(5.0 * 100))
unique_sirna = set(s.structure for s in sirna_sub)
unique_mrna  = set(s.structure for s in mrna_sub)
print(f"\nConformational diversity:")
print(f"  siRNA: {len(unique_sirna)} unique structures within 5 kcal/mol")
print(f"  mRNA target: {len(unique_mrna)} unique structures within 5 kcal/mol")

# Figure 2: accessibility + energy landscape
fig, axes = plt.subplots(2, 1, figsize=(12, 7))
ax = axes[0]
ax.bar(range(1, n+1), p_unpaired, color='#2166ac', alpha=0.8, width=0.9)
ax.axvspan(tgt_start+1, tgt_end, alpha=0.25, color='red', label='siRNA target region')
ax.axhline(0.5, color='red', lw=1, ls='--', alpha=0.7, label='50% accessibility')
ax.set_xlabel('Position (nt)', fontsize=10)
ax.set_ylabel('P(unpaired)', fontsize=10)
ax.set_title('mRNA Accessibility Profile — mcrA Conserved Region', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)

ax2 = axes[1]
sirna_e = sorted([s.energy/100.0 for s in sirna_sub])[:15]
mrna_e  = sorted([s.energy/100.0 for s in mrna_sub])[:15]
ax2.plot(range(1, len(sirna_e)+1), sirna_e, 'o-', color='#d6604d', lw=2,
         label=f'siRNA ({len(unique_sirna)} conformations)')
ax2.plot(range(1, len(mrna_e)+1), mrna_e, 's-', color='#4393c3', lw=2,
         label=f'mRNA ({len(unique_mrna)} conformations)')
ax2.set_xlabel('Structure rank', fontsize=10)
ax2.set_ylabel('Free energy (kcal/mol)', fontsize=10)
ax2.set_title('Suboptimal Structure Energy Landscape', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_accessibility_landscape.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_accessibility_landscape.png")

# siRNA candidates figure
fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(top5))
labels = [f"Pos {c['pos']}" for c in top5]
cofold_vals = [-c['cmfe'] for c in top5]
tmfe_vals   = [abs(c['tmfe']) for c in top5]
b1 = ax.bar([xi - 0.2 for xi in x], cofold_vals, 0.35,
            label='-Co-fold MFE (kcal/mol)', color='#2166ac', alpha=0.85)
b2 = ax.bar([xi + 0.2 for xi in x], tmfe_vals, 0.35,
            label='|Target MFE| (kcal/mol)', color='#d73027', alpha=0.85)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Energy magnitude (kcal/mol)', fontsize=10)
ax.set_title('siRNA Candidate Comparison — mcrA Conserved Region', fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.bar_label(b1, fmt='%.2f', fontsize=8)
ax.bar_label(b2, fmt='%.2f', fontsize=8)
ax.annotate('★ Selected', xy=(0, cofold_vals[0]+0.1),
            fontsize=10, color='green', fontweight='bold', ha='center')
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_sirna_candidates.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_sirna_candidates.png")

# Save intermediate results
result = {
    'conserved_rna': cons_rna, 'best_w': best_w, 'best_s': round(best_s, 4),
    'aln_len': aln_len, 'n_seqs': len(seqs_arr),
    'top5': top5,
    'selected_sirna': top,
    'mrna_mfe': round(mrna_mfe, 2), 'mrna_struct': mrna_struct,
    'sirna_mfe': round(sirna_mfe, 2), 'sirna_struct': sirna_struct,
    'cofold_mfe': round(cofold_mfe, 2), 'cofold_struct': cofold_struct,
    'target_accessibility': round(target_access, 3),
    'sirna_conformations': len(unique_sirna),
    'mrna_conformations': len(unique_mrna),
}
with open(f'{OUT}/step2_6_results.json', 'w') as f:
    json.dump(result, f, indent=2)
print("Saved step2_6_results.json")
print("\nSteps 2-6 COMPLETE.")
