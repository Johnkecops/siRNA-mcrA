
import json, random, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Bio import Seq, SeqRecord
from Bio.Align import MultipleSeqAlignment, PairwiseAligner
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import Bio.Phylo as Phylo
from io import StringIO

META = {
    "Mbrev_ruminantium":   {"label":"M. ruminantium",         "strain":"M1",       "acc":"CP001719.1","loc":"New Zealand",         "order":"Methanobacteriales"},
    "Msaeta_thermophila":  {"label":"M. thermophila",         "strain":"PT",       "acc":"CP000477.1","loc":"USA (Virginia)",       "order":"Methanosarcinales"},
    "Msarcina_mazei":      {"label":"M. mazei",               "strain":"Go1",      "acc":"AE008384.1","loc":"Germany",             "order":"Methanosarcinales"},
    "Mbacter_formicicum":  {"label":"M. formicicum",          "strain":"DSM 1535", "acc":"AF414152.1","loc":"USA (Indiana)",        "order":"Methanobacteriales"},
    "Mccoccus_jannaschii": {"label":"M. jannaschii",          "strain":"DSM 2661", "acc":"L77117.1",  "loc":"Pacific Ocean",       "order":"Methanococcales"},
    "Mpyrus_kandleri":     {"label":"M. kandleri",            "strain":"AV19",     "acc":"AE009439.1","loc":"Russia (Kamchatka)",  "order":"Methanopyrales"},
    "Mthermo_thermaut":    {"label":"M. thermautotrophicus",  "strain":"dH",       "acc":"AE000666.1","loc":"USA (Wisconsin)",     "order":"Methanobacteriales"},
    "Mculleus_marisnigri": {"label":"M. marisnigri",          "strain":"JR1",      "acc":"CP000562.1","loc":"Black Sea",           "order":"Methanomicrobiales"},
}
ORDER_COL = {
    "Methanobacteriales": "#1565C0",
    "Methanosarcinales":  "#C62828",
    "Methanococcales":    "#E65100",
    "Methanopyrales":     "#AD1457",
    "Methanomicrobiales": "#4E342E",
}
def tip_color(key): return ORDER_COL.get(META[key]["order"], "#555555")
def label2key(lbl):
    for k in META:
        if META[k]["label"] in lbl: return k
    return None

# Load & align
with open("seq_data.json") as f:
    raw = json.load(f)
seqs = raw["seqs"]
records = {}
keys = list(META.keys())
for kid in keys:
    records[kid] = seqs[kid].replace("U","T").replace("-","")

aligner = PairwiseAligner()
aligner.mode = "global"; aligner.match_score = 2; aligner.mismatch_score = -1
aligner.open_gap_score = -2; aligner.extend_gap_score = -0.5

def align_two(s1, s2):
    aln = aligner.align(s1, s2)[0]
    ln = str(aln).split("\n")
    return ln[0], ln[2]

ref_seq = records[keys[0]]
aligned = {keys[0]: ref_seq}
for k in keys[1:]:
    a_ref, a_new = align_two(ref_seq, records[k])
    if len(a_ref) > len(ref_seq):
        for ek in list(aligned.keys()):
            old = aligned[ek]; new_s = []; idx = 0
            for c in a_ref:
                if c == "-": new_s.append("-")
                else: new_s.append(old[idx] if idx < len(old) else "-"); idx += 1
            aligned[ek] = "".join(new_s)
        ref_seq = a_ref
    aligned[k] = a_new
maxlen = max(len(v) for v in aligned.values())
for k in aligned: aligned[k] = aligned[k].ljust(maxlen, "-")

bio_recs = [SeqRecord.SeqRecord(Seq.Seq(aligned[k]), id=k,
             name=META[k]["label"], description="") for k in keys]
msa = MultipleSeqAlignment(bio_recs)

# Build NJ tree + bootstrap
calc = DistanceCalculator("identity")
ctor = DistanceTreeConstructor(calc, "nj")
tree = ctor.build_tree(msa)

def resample(aln):
    nc = aln.get_alignment_length()
    cols = random.choices(range(nc), k=nc)
    return MultipleSeqAlignment([SeqRecord.SeqRecord(Seq.Seq("".join(str(r.seq)[c] for c in cols)),
                  id=r.id, name=r.name, description="") for r in aln])

def clades_of(t):
    return {frozenset(x.name for x in cl.get_terminals())
            for cl in t.find_clades() if len(list(cl.get_terminals())) > 1}

orig_cl = clades_of(tree)
sup = {c: 0 for c in orig_cl}
random.seed(42)
for i in range(1000):
    try:
        bt = ctor.build_tree(resample(msa))
        for c in orig_cl:
            if c in clades_of(bt): sup[c] += 1
    except: pass

for cl in tree.find_clades():
    taxa = frozenset(t.name for t in cl.get_terminals())
    if len(taxa) > 1:
        cl.confidence = round(sup.get(taxa, 0) / 10.0, 1)

# Rename terminals
for cl in tree.get_terminals():
    k = cl.name
    if k in META:
        cl.name = META[k]["label"] + " " + META[k]["strain"]

# ── Cladogram layout ─────────────────────────────────────────────────────────
def cladogram_layout(tree):
    depth = {}
    def set_depth(cl, d):
        depth[id(cl)] = d
        for ch in cl.clades: set_depth(ch, d+1)
    set_depth(tree.root, 0)
    max_depth = max(depth.values())
    ymap = {}
    tip_counter = [0]
    def assign_y(cl):
        if cl.is_terminal():
            tip_counter[0] += 1
            ymap[id(cl)] = tip_counter[0]
        else:
            for ch in cl.clades: assign_y(ch)
            ys = [ymap[id(ch)] for ch in cl.clades]
            ymap[id(cl)] = (min(ys) + max(ys)) / 2.0
    assign_y(tree.root)
    xmap = {}
    def assign_x(cl):
        if cl.is_terminal():
            xmap[id(cl)] = max_depth
        else:
            for ch in cl.clades: assign_x(ch)
            child_xs = [xmap[id(ch)] for ch in cl.clades]
            xmap[id(cl)] = min(child_xs) - 1
    assign_x(tree.root)
    return xmap, ymap, max_depth, tip_counter[0]

# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE A: RECTANGULAR CLADOGRAM (same as before, it looked good)
# ═══════════════════════════════════════════════════════════════════════════
xmap, ymap, max_d, ntax = cladogram_layout(tree)
fig1, ax1 = plt.subplots(figsize=(11, 6))
fig1.patch.set_facecolor("white"); ax1.set_facecolor("white")
for cl in tree.find_clades(order="level"):
    x, y = xmap[id(cl)], ymap[id(cl)]
    if not cl.is_terminal():
        child_ys = [ymap[id(c)] for c in cl.clades]
        ax1.plot([x,x],[min(child_ys),max(child_ys)], color="#333333", lw=1.4, zorder=3)
        for child in cl.clades:
            cx = xmap[id(child)]; cy = ymap[id(child)]
            col = "#888888"
            if child.is_terminal():
                k2 = label2key(child.name)
                if k2: col = tip_color(k2)
            ax1.plot([x,cx],[cy,cy], color=col, lw=2.0, zorder=2)
        if hasattr(cl,"confidence") and cl.confidence is not None and cl.confidence >= 50:
            ax1.text(x-0.08, y, str(int(cl.confidence)), fontsize=7.5,
                     color="#B71C1C", fontweight="bold", ha="right", va="center", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.15",fc="white",ec="#B71C1C",lw=0.6,alpha=0.9))
root_x = xmap[id(tree.root)]
ax1.plot([root_x-0.8, root_x],[ymap[id(tree.root)],ymap[id(tree.root)]], color="#333333", lw=1.4, zorder=3)
label_x = max_d + 0.2
for term in tree.get_terminals():
    tx = xmap[id(term)]; ty = ymap[id(term)]
    k2 = label2key(term.name)
    col = tip_color(k2) if k2 else "#555555"
    ax1.plot(tx, ty, "o", color=col, ms=8, zorder=7, markeredgecolor="white", markeredgewidth=0.7)
    sp_name = META[k2]["label"] if k2 else term.name
    strain  = META[k2]["strain"] if k2 else ""
    acc = META[k2]["acc"] if k2 else ""
    loc = META[k2]["loc"] if k2 else ""
    ax1.text(label_x, ty+0.18, sp_name+" "+strain,
             fontsize=9.5, fontstyle="italic", color="#111111", va="bottom", ha="left", zorder=8)
    ax1.text(label_x, ty-0.18, acc+"  |  "+loc,
             fontsize=7.5, color="#555555", va="top", ha="left", zorder=8, family="monospace")
ax1.text(root_x-0.8, 0.35, "Cladogram (topology-only; branch lengths not proportional)",
         fontsize=7.5, color="#666666", style="italic", va="bottom")
order_labels = sorted(set(META[k]["order"] for k in keys))
handles = [mpatches.Patch(color=ORDER_COL[o], label=o) for o in order_labels]
ax1.legend(handles=handles, loc="lower left", fontsize=8,
           title="Taxonomic Order", title_fontsize=8.5,
           framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.0, 0.05))
ax1.set_xlim(root_x-1.5, max_d+4.2)
ax1.set_ylim(0.2, ntax+0.8)
ax1.set_title("Neighbour-Joining Phylogenetic Tree — Methanogen mcrA Gene\n(1000 bootstrap replicates; red values = % support at internal nodes)",
              fontsize=11.5, fontweight="bold", pad=10)
ax1.axis("off")
plt.tight_layout(pad=1.5)
fig1.savefig("fig10_phylo_rectangular.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print("fig10 saved")

# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE B: RADIAL CLADOGRAM — proper sector-arc algorithm
# ═══════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(13, 13))
fig2.patch.set_facecolor("white"); ax2.set_facecolor("white")

# Step 1: Assign angles to tips — equal sector spacing
tips_order = list(tree.get_terminals())
n_tips = len(tips_order)
tip_sector_angle = {}          # the angle of each tip
for i, t in enumerate(tips_order):
    # Start from top (90 deg), go clockwise
    tip_sector_angle[id(t)] = math.pi/2 - 2*math.pi*i/n_tips

# Step 2: Each internal node gets min and max angle of its subtree terminals
def get_tip_angles(cl):
    if cl.is_terminal():
        return tip_sector_angle[id(cl)], tip_sector_angle[id(cl)]
    ranges = [get_tip_angles(c) for c in cl.clades]
    all_a = [a for mn,mx in ranges for a in (mn,mx)]
    return min(all_a), max(all_a)

node_mid_angle = {}
def assign_mid_angle(cl):
    mn, mx = get_tip_angles(cl)
    mid = (mn + mx) / 2.0
    node_mid_angle[id(cl)] = mid
    for c in cl.clades:
        assign_mid_angle(c)
assign_mid_angle(tree.root)

# Step 3: Radius = depth from root (cladogram style), tips at r=1
depth3 = {}
def set_d(cl, d):
    depth3[id(cl)] = d
    for ch in cl.clades: set_d(ch, d+1)
set_d(tree.root, 0)
max_d3 = max(depth3.values())
def get_r(cl): return depth3[id(cl)] / max_d3

def p2xy(r, a): return r*math.cos(a), r*math.sin(a)

# Draw each clade: arc at parent radius, then radial to children
for cl in tree.find_clades(order="level"):
    r_cl = get_r(cl)
    ang_cl = node_mid_angle[id(cl)]
    if not cl.is_terminal():
        # Arc spans from leftmost child angle to rightmost child angle
        child_angs = [node_mid_angle[id(c)] for c in cl.clades]
        arc_min = min(child_angs)
        arc_max = max(child_angs)
        arc_angles = np.linspace(arc_min, arc_max, 80)
        ax2.plot([r_cl*math.cos(a) for a in arc_angles],
                 [r_cl*math.sin(a) for a in arc_angles],
                 color="#444444", lw=1.5, zorder=3, solid_capstyle="round")
        # Radial lines from arc to each child
        for child in cl.clades:
            r_ch = get_r(child)
            ac = node_mid_angle[id(child)]
            col = "#888888"
            if child.is_terminal():
                k2 = label2key(child.name)
                if k2: col = tip_color(k2)
            else:
                # color by majority tip color in subtree
                ctips = list(child.get_terminals())
                tip_keys = [label2key(t.name) for t in ctips if label2key(t.name)]
                if tip_keys:
                    from collections import Counter
                    most = Counter(tip_color(tk) for tk in tip_keys).most_common(1)[0][0]
                    col = most
            # Draw from arc point to child node
            arc_pt = p2xy(r_cl, ac)
            ch_pt  = p2xy(r_ch, ac)
            ax2.plot([arc_pt[0], ch_pt[0]], [arc_pt[1], ch_pt[1]],
                     color=col, lw=2.3, zorder=4, solid_capstyle="round")
        # Bootstrap label
        if hasattr(cl,"confidence") and cl.confidence is not None and cl.confidence >= 50:
            # Place label just outside this node's arc midpoint
            lbl_r = r_cl - 0.04
            lbl_a = (arc_min + arc_max) / 2.0
            lx, ly = p2xy(lbl_r, lbl_a)
            ax2.text(lx, ly, str(int(cl.confidence)),
                     fontsize=7.5, ha="center", va="center",
                     color="#B71C1C", fontweight="bold", zorder=8,
                     bbox=dict(boxstyle="round,pad=0.12",fc="white",ec="#B71C1C",lw=0.6,alpha=0.92))

# Tip dots and labels
LABEL_R = 1.10
for t in tips_order:
    a = node_mid_angle[id(t)]
    r = get_r(t)
    k2 = label2key(t.name)
    col = tip_color(k2) if k2 else "#555555"
    # Dot at tip
    ax2.plot(r*math.cos(a), r*math.sin(a), "o", color=col, ms=9, zorder=9,
             markeredgecolor="white", markeredgewidth=0.8)
    # Thin line from tip to label
    lx, ly = p2xy(LABEL_R, a)
    ax2.plot([r*math.cos(a), lx*0.995],[r*math.sin(a), ly*0.995],
             color=col, lw=0.7, alpha=0.5, zorder=5)
    # Label: two lines, rotated along branch
    sp_name = META[k2]["label"] if k2 else t.name
    strain  = META[k2]["strain"] if k2 else ""
    acc     = META[k2]["acc"] if k2 else ""
    loc     = META[k2]["loc"] if k2 else ""
    rot_deg = math.degrees(a)
    if math.cos(a) < 0:   # left half: flip text so it reads outward
        rot_deg = rot_deg + 180
        ha = "right"
    else:
        ha = "left"
    ax2.text(lx, ly, sp_name+" "+strain,
             ha=ha, va="center", fontsize=8.5,
             rotation=rot_deg, rotation_mode="anchor",
             fontstyle="italic", color="#111111", zorder=10)
    ax2.text(p2xy(LABEL_R+0.10, a)[0], p2xy(LABEL_R+0.10, a)[1],
             acc+"  "+loc,
             ha=ha, va="center", fontsize=7,
             rotation=rot_deg, rotation_mode="anchor",
             color="#666666", zorder=10, family="monospace")

# Legend
handles2 = [mpatches.Patch(color=ORDER_COL[o], label=o)
            for o in sorted(set(META[k]["order"] for k in keys))]
ax2.legend(handles=handles2, loc="lower right", fontsize=8,
           title="Taxonomic Order", title_fontsize=8.5,
           framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(1.02,-0.02))

ax2.set_xlim(-1.9, 1.9); ax2.set_ylim(-1.9, 1.9)
ax2.set_aspect("equal"); ax2.axis("off")
ax2.set_title("Radial NJ Phylogenetic Tree — Methanogen mcrA Gene\n(1000 bootstrap replicates; red values = % support at internal nodes)",
              fontsize=11.5, fontweight="bold", pad=10)
plt.tight_layout(pad=1.5)
fig2.savefig("fig11_phylo_radial.png", dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print("fig11 saved")
print("ALL DONE")
