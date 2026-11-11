"""Clean tables-only page, styled to match what the group has been posting."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
t2 = pd.read_csv(f"{R}/task2_test_results.csv")
t3 = pd.read_csv(f"{R}/task3_summary.csv")
bk = pd.read_csv(f"{R}/task3_similarity_buckets.csv")
meta = json.load(open(f"{R}/task2_meta.json"))

S2 = {"B0 global mean": "global mean", "B1 ligand mean": "ligand mean",
      "B2 target mean": "target mean", "B3 bias only": "bias only",
      "M1 target kNN": "target-kNN", "M2 biased MF": "biased MF",
      "M3 kNN + MF blend": "kNN + MF blend", "random ranking": "random ranking"}
S3 = {"C0 ligand-mean fallback": "C0 ligand-mean (fallback)",
      "C1 guilt-by-association kNN": "C1 guilt-by-association kNN",
      "C2 sequence -> MF factors": "C2 sequence &rarr; MF factors",
      "C3 content regression": "C3 content regression",
      "C4 similarity-gated blend": "C4 similarity-gated blend",
      "random ranking": "random ranking"}


def tbl(headers, rows, best=None):
    best = best or {}
    th = "".join(f'<th class="{"l" if i == 0 else "n"}">{h}</th>' for i, h in enumerate(headers))
    tr = []
    for r in rows:
        tds = []
        for i, c in enumerate(r):
            cls = "l" if i == 0 else "n"
            if best.get(i) is not None and str(c) == best[i]:
                cls += " b"
            tds.append(f'<td class="{cls}">{c}</td>')
        tr.append("<tr>" + "".join(tds) + "</tr>")
    return f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(tr)}</tbody></table>'


def t2_tbl(ds):
    g = t2[t2.dataset == ds]; hp = meta[ds.lower()]
    spec = [("mse", 3), ("mse_strong", 3), ("ci_global", 3), ("ci_per_ligand", 3),
            ("hit@1", 3), ("hit@5", 3), ("mrr", 3)]
    real = g[g.model != "random ranking"]
    best = {j + 1: f"{(real[c].min() if c.startswith('mse') else real[c].max()):.{n}f}"
            for j, (c, n) in enumerate(spec)}
    rows = []
    for _, r in g.iterrows():
        nm = S2[r["model"]]
        if nm == "target-kNN": nm += f" (k={hp['best_knn']['k']})"
        if nm == "biased MF": nm += f" (k={hp['best_mf']['k']})"
        cells = [nm] + ["&mdash;" if not np.isfinite(r[c]) else f"{r[c]:.{n}f}" for c, n in spec]
        rows.append(cells)
    return tbl(["Model", "MSE", "MSE strong", "CI", "CI per-ligand", "Hit@1", "Hit@5", "MRR"],
               rows, best)


def t3_tbl(ds):
    g = t3[t3.dataset == ds]
    spec = [("mse", 3), ("ci_global", 3), ("ci_per_ligand", 3), ("hit@1", 3), ("mrr", 3)]
    real = g[g.model != "random ranking"]
    best = {}
    for j, (c, n) in enumerate(spec):
        v = real[f"{c}_mean"].min() if c.startswith("mse") else real[f"{c}_mean"].max()
        sd = real[real[f"{c}_mean"] == v][f"{c}_std"].iloc[0]
        best[j + 1] = f'{v:.{n}f}<span class="sd">&plusmn;{sd:.{n}f}</span>'
    rows = []
    for _, r in g.iterrows():
        cells = [S3[r["model"]]]
        for c, n in spec:
            m, s = r[f"{c}_mean"], r[f"{c}_std"]
            cells.append("&mdash;" if not np.isfinite(m)
                         else f'{m:.{n}f}<span class="sd">&plusmn;{s:.{n}f}</span>')
        fb = r["fallback_frac_mean"]
        cells.append("&mdash;" if not np.isfinite(fb) else f"{fb*100:.0f}%")
        rows.append(cells)
    return tbl(["Model", "MSE", "CI", "CI per-ligand", "Hit@1", "MRR", "Fallback"], rows, best)


def bk_tbl(ds):
    rows = []
    for _, r in bk[bk.dataset == ds].iterrows():
        gain = r["CI gain C1 over C0"]
        cls = "up" if gain > 0.02 else "flat"
        rows.append([r["bucket"].split(" (")[0], r["sim_range"], int(r["n_targets"]),
                     f'{r["CI C0"]:.3f}', f'{r["CI C1"]:.3f}', f'{r["CI C4"]:.3f}',
                     f'<span class="{cls}">{gain:+.3f}</span>',
                     f'{r["MSE C0"]:.3f}', f'{r["MSE C1"]:.3f}', f'{r["MSE C4"]:.3f}'])
    return tbl(["Cold target", "Similarity", "N", "CI C0", "CI C1", "CI C4",
                "C1 gain", "MSE C0", "MSE C1", "MSE C4"], rows)


d, k = meta["davis"], meta["kiba"]
HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DAVIS / KIBA - Task 2 and Task 3 results</title>
<style>
:root{{--bg:#15181d;--panel:#1b1f26;--ink:#e8ecf2;--mut:#8b94a3;--line:#2b313a;
--hi:#7aa2ff;--good:#55c08a;--bad:#e2795f;}}
*{{box-sizing:border-box}}
body{{margin:0;padding:36px 22px 64px;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto}}
h1{{font-size:24px;margin:0 0 4px;letter-spacing:-.01em}}
.sub{{color:var(--mut);font-size:14px;margin:0 0 30px}}
h2{{font-size:18px;margin:38px 0 4px;padding-top:18px;border-top:1px solid var(--line)}}
h3{{font-size:13px;margin:22px 0 8px;color:var(--mut);font-weight:600;
text-transform:uppercase;letter-spacing:.08em}}
p{{margin:8px 0;font-size:14px}}
.cap{{color:var(--mut);font-size:13px;margin:4px 0 0}}
table{{width:100%;border-collapse:collapse;margin:8px 0 4px;font-size:13.5px;
background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden}}
th,td{{padding:8px 11px;border-bottom:1px solid var(--line)}}
thead th{{background:#222733;color:var(--ink);font-weight:600;font-size:11.5px;
text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}}
tbody tr:last-child td{{border-bottom:none}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap}}
td.l,th.l{{text-align:left}}
td.b{{color:var(--hi);font-weight:700}}
.sd{{color:var(--mut);font-size:10.5px;margin-left:3px}}
.up{{color:var(--good);font-weight:700}} .flat{{color:var(--bad);font-weight:700}}
.note{{background:var(--panel);border-left:3px solid var(--hi);padding:11px 15px;
margin:14px 0;font-size:13.5px;color:var(--ink);border-radius:0 6px 6px 0}}
.note.w{{border-left-color:var(--bad)}}
.note b{{color:#fff}}
</style></head><body><div class="wrap">

<h1>DAVIS / KIBA &mdash; Task 2 and Task 3</h1>
<p class="sub">Atharva &middot; test folds scored once &middot; hyperparameters selected on validation</p>

<h2>Task 2 &mdash; warm split</h2>
<p class="cap">70/15/15 random split over observed pairs, every evaluation ligand and target present in
training. DAVIS on pKd, KIBA on the supplied scale. MSE strong = MSE on true strong binders only
(DAVIS pKd &ge; 7, KIBA &ge; 12.1), since 69.6% of DAVIS labels sit at the pKd = 5 screening floor.
CI per-ligand ranks each ligand's own held-out targets.</p>

<h3>DAVIS &mdash; {d['n_test']:,} test pairs, 68 drugs</h3>
{t2_tbl("DAVIS")}
<h3>KIBA &mdash; {k['n_test']:,} test pairs, 2,111 drugs</h3>
{t2_tbl("KIBA")}

<div class="note"><b>Read:</b> CF beats every baseline on both datasets. The kNN + MF blend is best on
MSE and CI; on DAVIS plain target-kNN still wins Hit@1 (0.456 vs 0.382) because the blend weight was
tuned on validation MSE &mdash; MSE-optimal is not ranking-optimal. Note <b>ligand mean</b> scores
0.751 global CI but exactly 0.500 per-ligand: global CI rewards getting the drug-level offset right
and hides that the model cannot rank targets at all.</div>

<h2>Task 3 &mdash; cold targets</h2>
<p class="cap">Whole target <b>sequence groups</b> held out, so test targets have no training rows.
DAVIS lists 442 target IDs but only 379 distinct sequences, so an ID-level split would leak identical
proteins across folds. 5 seeds, mean &plusmn; sd. Fallback = share of test pairs with no usable
neighbour, which drops back to the ligand mean.</p>

<h3>DAVIS &mdash; cold targets, 5 seeds</h3>
{t3_tbl("DAVIS")}
<h3>KIBA &mdash; cold targets, 5 seeds</h3>
{t3_tbl("KIBA")}

<div class="note"><b>Read:</b> C4 (blend of C1 and C2, weighted by how similar the cold target is to
its nearest training target) is best on MSE for both datasets. The factor-regression routes C2 and C3
are weak &mdash; C3 barely beats the fallback on KIBA. Neighbours do nearly all the work.</div>

<h2>Where guilt-by-association actually works</h2>
<p class="cap">Each held-out target scored on its own: across all ligands measured against it, how well
does the model order them. Targets grouped into thirds by similarity to the nearest training target.</p>

<h3>DAVIS</h3>
{bk_tbl("DAVIS")}
<h3>KIBA</h3>
{bk_tbl("KIBA")}

<div class="note w"><b>The finding.</b> Guilt-by-association is a homology lookup, not cold-target
generalisation. For near-homologue targets C1 adds
{bk[(bk.dataset=='DAVIS')&(bk.bucket.str.startswith('near'))].iloc[0]['CI gain C1 over C0']:+.3f} CI on DAVIS and
{bk[(bk.dataset=='KIBA')&(bk.bucket.str.startswith('near'))].iloc[0]['CI gain C1 over C0']:+.3f} on KIBA over the fallback. For orphan targets it adds
{bk[(bk.dataset=='DAVIS')&(bk.bucket.str.startswith('orphan'))].iloc[0]['CI gain C1 over C0']:+.3f} and {bk[(bk.dataset=='KIBA')&(bk.bucket.str.startswith('orphan'))].iloc[0]['CI gain C1 over C0']:+.3f} &mdash; nothing. Spearman(similarity, C1
ranking quality) = +0.573 on DAVIS, +0.281 on KIBA. C4 is the only model that improves on the fallback
in the orphan bucket.</div>

<div class="note w"><b>Protocol &mdash; we need to align on this.</b> Our random-ranking baselines
disagree, so we are not ranking the same candidate set (DAVIS random Hit@1: 0.044 / 0.018 / 0.028
across us). Two things that affect everyone's numbers: prediction ties must be broken at random, or a
constant predictor inherits the array order and beats chance (my global mean scored 0.088 before I
fixed this, 0.044 after); and DAVIS ranks ~66 candidates per ligand against KIBA's ~10, with 252 of
1,750 KIBA ligands having all held-out scores tied &mdash; which is why KIBA's random Hit@1 is 0.308,
not near zero. Hit@1 is not comparable across the two datasets.</div>

<p class="cap" style="margin-top:34px">Scale reference (different split protocol, ballpark only):
DeepDTA reports DAVIS MSE 0.261 / CI 0.878 and KIBA MSE 0.194 / CI 0.863. Our warm MF lands at
DAVIS {t2[(t2.dataset=='DAVIS')&(t2.model=='M2 biased MF')].iloc[0]['mse']:.3f} / {t2[(t2.dataset=='DAVIS')&(t2.model=='M2 biased MF')].iloc[0]['ci_global']:.3f} and KIBA {t2[(t2.dataset=='KIBA')&(t2.model=='M2 biased MF')].iloc[0]['mse']:.3f} / {t2[(t2.dataset=='KIBA')&(t2.model=='M2 biased MF')].iloc[0]['ci_global']:.3f}.</p>
</div></body></html>"""
open(f"{R}/results_tables.html", "w").write(HTML)
print(f"wrote results_tables.html ({len(HTML):,} bytes)")
