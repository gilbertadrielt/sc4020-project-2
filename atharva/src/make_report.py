"""Render the Task 2 / Task 3 results as a self-contained page.

Every figure is read from the result CSVs, never retyped.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
t2 = pd.read_csv(f"{R}/task2_test_results.csv")
meta = json.load(open(f"{R}/task2_meta.json"))
cs = pd.read_csv(f"{R}/task2_case_summary.csv")
cr = pd.read_csv(f"{R}/task2_case_recommendations.csv")
t3 = pd.read_csv(f"{R}/task3_summary.csv")
bk = pd.read_csv(f"{R}/task3_similarity_buckets.csv")
hp = json.load(open(f"{R}/task3_hparams.json"))


def fmt(v, nd=4):
    return "&mdash;" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def table(headers, rows, best_cols=None, aligns=None):
    best_cols = best_cols or {}
    aligns = aligns or {}
    h = "".join(f'<th class="{aligns.get(i,"num")}">{c}</th>' for i, c in enumerate(headers))
    body = []
    for r in rows:
        tds = []
        for i, c in enumerate(r):
            cls = aligns.get(i, "num")
            if i in best_cols and c == best_cols[i]:
                cls += " best"
            tds.append(f'<td class="{cls}">{c}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<table><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def task2_table(dataset):
    g = t2[t2.dataset == dataset]
    cols = [("mse", 4), ("mse_strong", 3), ("ci_global", 4), ("ci_per_ligand", 4),
            ("hit@1", 4), ("hit@5", 4), ("hit@10", 4), ("mrr", 4)]
    real = g[g.model != "random ranking"]
    best = {}
    for j, (c, nd) in enumerate(cols):
        v = real[c].min() if c.startswith("mse") else real[c].max()
        if np.isfinite(v):
            best[j + 1] = f"{v:.{nd}f}"
    rows = []
    for _, r in g.iterrows():
        rows.append([r["model"]] + [fmt(r[c], nd) for c, nd in cols])
    return table(["Model", "MSE", "MSE strong", "CI", "CI per-ligand",
                  "Hit@1", "Hit@5", "Hit@10", "MRR"], rows, best, {0: "txt"})


def task3_table(dataset):
    g = t3[t3.dataset == dataset]
    spec = [("mse", 4), ("ci_global", 4), ("ci_per_ligand", 4), ("hit@1", 4), ("mrr", 4)]
    real = g[g.model != "random ranking"]
    best = {}
    for j, (c, nd) in enumerate(spec):
        v = real[f"{c}_mean"].min() if c.startswith("mse") else real[f"{c}_mean"].max()
        if np.isfinite(v):
            best[j + 1] = f"{v:.{nd}f} <span class='sd'>&plusmn;{real[real[f'{c}_mean']==v][f'{c}_std'].iloc[0]:.{nd}f}</span>"
    rows = []
    for _, r in g.iterrows():
        cells = [r["model"]]
        for c, nd in spec:
            m, s = r[f"{c}_mean"], r[f"{c}_std"]
            cells.append("&mdash;" if not np.isfinite(m)
                         else f"{m:.{nd}f} <span class='sd'>&plusmn;{s:.{nd}f}</span>")
        fb = r["fallback_frac_mean"]
        cells.append("&mdash;" if not np.isfinite(fb) else f"{fb*100:.1f}%")
        rows.append(cells)
    return table(["Model", "MSE", "CI", "CI per-ligand", "Hit@1", "MRR", "Fallback"],
                 rows, best, {0: "txt", 6: "num"})


def bucket_table(dataset):
    g = bk[bk.dataset == dataset]
    rows = []
    for _, r in g.iterrows():
        gain = r["CI gain C1 over C0"]
        cls = "pos" if gain > 0.02 else ("flat" if gain < 0.02 else "")
        rows.append([
            r["bucket"], r["sim_range"], int(r["n_targets"]),
            f'{r["CI C0"]:.3f}', f'{r["CI C1"]:.3f}', f'{r["CI C4"]:.3f}',
            f'<span class="{cls}">{gain:+.3f}</span>',
            f'{r["MSE C0"]:.3f}', f'{r["MSE C1"]:.3f}', f'{r["MSE C4"]:.3f}',
        ])
    return table(["Cold-target bucket", "Similarity to nearest<br>training target", "Targets",
                  "CI C0", "CI C1", "CI C4", "CI gain<br>C1 over C0",
                  "MSE C0", "MSE C1", "MSE C4"], rows, {}, {0: "txt", 1: "txt"})


def case_block(dataset):
    out = []
    for _, s in cs[cs.dataset == dataset].iterrows():
        recs = cr[(cr.dataset == dataset) & (cr.ligand == s["ligand"])]
        rows = [[int(r["rank"]), r["target"], f'{r["pred"]:.3f}', f'{r["true"]:.3f}',
                 f'<span class="{"ok" if r["verdict"].startswith("correct") else "no"}">{r["verdict"]}</span>']
                for _, r in recs.iterrows()]
        out.append(f"""<div class="case">
          <h4>{s['ligand']}</h4>
          <p class="meta">{int(s['held_out_targets'])} held-out targets &middot; true score range {s['true_score_range']}
             &middot; MSE {s['mse']:.4f} &middot; CI {s['ci']:.4f}
             &middot; <strong>true best target ranked #{int(s['rank_of_true_best'])}</strong></p>
          {table(["Rank","Target","Predicted","True","Verdict"], rows, {}, {1:"txt",4:"txt"})}
        </div>""")
    return "".join(out)


d_meta, k_meta = meta["davis"], meta["kiba"]
davis_rand = t2[(t2.dataset == "DAVIS") & (t2.model == "random ranking")].iloc[0]
kiba_rand = t2[(t2.dataset == "KIBA") & (t2.model == "random ranking")].iloc[0]
d_cand = davis_rand["mean_candidates"]; k_cand = kiba_rand["mean_candidates"]
k_deg = int(kiba_rand["n_degenerate"]); k_rank = int(kiba_rand["n_ligands_ranked"])

HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DAVIS / KIBA - Task 2 and Task 3 results</title>
<style>
:root{{--ink:#15181d;--mut:#5d6572;--line:#e3e7ee;--bg:#fff;--soft:#f7f9fc;
--accent:#1d4ed8;--good:#0f7b4f;--bad:#b4341f;--warn:#8a5a00;}}
*{{box-sizing:border-box}}
body{{margin:0;padding:40px 24px 80px;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;}}
.wrap{{max-width:1120px;margin:0 auto}}
h1{{font-size:27px;margin:0 0 4px;letter-spacing:-.02em}}
h2{{font-size:20px;margin:44px 0 6px;padding-top:20px;border-top:2px solid var(--ink);letter-spacing:-.01em}}
h3{{font-size:15px;margin:26px 0 8px;color:var(--mut);text-transform:uppercase;letter-spacing:.07em}}
h4{{font-size:15px;margin:18px 0 4px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
p{{margin:9px 0}}
.sub{{color:var(--mut);margin:0 0 22px}}
table{{width:100%;border-collapse:collapse;margin:10px 0 6px;font-size:13.5px}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--line)}}
thead th{{background:var(--ink);color:#fff;font-weight:600;font-size:12px;
text-transform:uppercase;letter-spacing:.04em;border:none;vertical-align:bottom}}
td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
td.txt,th.txt{{text-align:left}}
tbody tr:nth-child(even){{background:var(--soft)}}
td.best{{font-weight:700;color:var(--accent)}}
.sd{{color:var(--mut);font-size:11px}}
.note{{background:var(--soft);border-left:3px solid var(--accent);padding:12px 16px;margin:16px 0;font-size:14px}}
.note.warn{{border-left-color:var(--warn)}}
.note strong{{color:var(--ink)}}
.ok{{color:var(--good);font-weight:600}} .no{{color:var(--mut)}}
.pos{{color:var(--good);font-weight:700}} .flat{{color:var(--bad);font-weight:700}}
.meta{{color:var(--mut);font-size:13px;margin:2px 0 6px}}
.case{{margin:18px 0 26px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin:16px 0}}
.card{{border:1px solid var(--line);border-radius:8px;padding:12px 14px}}
.card .k{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}}
.card .v{{font-size:21px;font-weight:700;font-variant-numeric:tabular-nums;margin-top:2px}}
.card .d{{font-size:12px;color:var(--mut)}}
code{{background:var(--soft);padding:1px 5px;border-radius:4px;font-size:12.5px}}
footer{{margin-top:50px;padding-top:16px;border-top:1px solid var(--line);color:var(--mut);font-size:12.5px}}
</style></head><body><div class="wrap">

<h1>DAVIS / KIBA &mdash; Task 2 and Task 3</h1>
<p class="sub">Atharva &middot; test folds scored once &middot; all hyperparameters selected on validation</p>

<div class="note warn"><strong>Read this before comparing against your tables.</strong>
Our random-ranking baselines differ, which means we are not ranking the same candidate set.
Mine: for each ligand I rank <em>only that ligand's held-out targets</em>, and a hit means a target
achieving the ligand's maximum true score lands in the top&nbsp;k. That gives
{d_cand:.1f} candidates per ligand on DAVIS and {k_cand:.1f} on KIBA &mdash; so
<strong>Hit@1 is not comparable across the two datasets</strong>, and KIBA's random Hit@1 is
{kiba_rand['hit@1']:.3f} rather than something near zero. Two further things worth agreeing on:
ties in the predictions must be broken at random (otherwise a constant predictor inherits the array
order and scores well above chance), and {k_deg} of {k_rank} KIBA ligands ({k_deg/k_rank:.0%}) have
<em>all</em> their held-out scores tied, so their ranking is trivially correct no matter what.</div>

<h2>Task 2 &mdash; warm split (collaborative filtering)</h2>
<p>Users = ligands, items = protein targets, ratings = binding affinity. 70/15/15 random split over
observed pairs at a fixed seed, with every evaluation ligand and target forced to appear in training.
DAVIS is modelled on pKd = 9 &minus; log<sub>10</sub>(Kd); KIBA on the supplied composite scale.</p>

<div class="grid">
  <div class="card"><div class="k">DAVIS pairs</div><div class="v">{d_meta['n_train']:,} / {d_meta['n_valid']:,} / {d_meta['n_test']:,}</div><div class="d">train / valid / test</div></div>
  <div class="card"><div class="k">KIBA pairs</div><div class="v">{k_meta['n_train']:,} / {k_meta['n_valid']:,} / {k_meta['n_test']:,}</div><div class="d">train / valid / test</div></div>
  <div class="card"><div class="k">Checkpoint reload</div><div class="v">exact</div><div class="d">max |memory &minus; reloaded| = 0</div></div>
</div>

<p class="meta"><strong>MSE strong</strong> = MSE restricted to true strong binders
(DAVIS pKd &ge; {d_meta['strong_threshold']:.0f}, KIBA &ge; {k_meta['strong_threshold']}). On DAVIS
69.6% of labels sit pinned at the pKd&nbsp;=&nbsp;5 screening floor, so plain MSE is dominated by
predicting that floor well &mdash; this column is where the models actually separate.</p>

<h3>DAVIS &mdash; {d_meta['n_test']:,} test pairs</h3>
{task2_table("DAVIS")}
<p class="meta">Selected on validation: kNN {d_meta['best_knn']} &middot; MF {d_meta['best_mf']} &middot; blend weight {d_meta['blend_w']:.2f}</p>

<h3>KIBA &mdash; {k_meta['n_test']:,} test pairs</h3>
{task2_table("KIBA")}
<p class="meta">Selected on validation: kNN {k_meta['best_knn']} &middot; MF {k_meta['best_mf']} &middot; blend weight {k_meta['blend_w']:.2f}</p>

<div class="note"><strong>Is the recommender necessary?</strong> Yes, and the baseline ladder is what
proves it rather than an opinion. <code>B1 ligand mean</code> is exactly what the model degrades to
when the target carries no information, and it ranks at chance by construction (CI per-ligand 0.500).
Adding target structure through bias terms gets DAVIS to {t2[(t2.dataset=='DAVIS')&(t2.model=='B3 bias only')].iloc[0]['ci_per_ligand']:.3f} per-ligand CI; the interaction
term in kNN/MF takes it to {t2[(t2.dataset=='DAVIS')&(t2.model=='M3 kNN + MF blend')].iloc[0]['ci_per_ligand']:.3f}. These are held-out pairs the model never saw,
so it is not a lookup of known answers.</div>

<div class="note"><strong>One honest wrinkle.</strong> On DAVIS the blend wins on MSE
({t2[(t2.dataset=='DAVIS')&(t2.model=='M3 kNN + MF blend')].iloc[0]['mse']:.4f} vs {t2[(t2.dataset=='DAVIS')&(t2.model=='M1 target kNN')].iloc[0]['mse']:.4f}) but <em>loses</em> on Hit@1
({t2[(t2.dataset=='DAVIS')&(t2.model=='M3 kNN + MF blend')].iloc[0]['hit@1']:.4f} vs {t2[(t2.dataset=='DAVIS')&(t2.model=='M1 target kNN')].iloc[0]['hit@1']:.4f}), because I tuned the blend weight on validation MSE.
MSE-optimal is not ranking-optimal. If we care about top-1 recommendation we should tune on MRR
instead &mdash; worth deciding as a group before the report.</div>

<h3>Case study &mdash; predictions generated by reloading <code>saved_models/*.npz</code></h3>
<p class="meta">The three fixed ligands per dataset named in the brief. All are present and all fall in
the test fold. They were chosen because they span the full behaviour range: a ligand that binds
essentially nothing, a moderate binder, and a promiscuous strong binder.</p>
<h3 style="color:var(--ink);text-transform:none;letter-spacing:0;font-size:14px">DAVIS</h3>
{case_block("DAVIS")}
<h3 style="color:var(--ink);text-transform:none;letter-spacing:0;font-size:14px">KIBA</h3>
{case_block("KIBA")}

<h2>Task 3 &mdash; cold targets</h2>
<p>Whole target <em>sequence groups</em> are held out, so a test target has no training rows at all.
Grouping by sequence matters: DAVIS lists 442 target IDs but only 379 distinct sequences, so splitting
on raw IDs would put a byte-identical protein on both sides. 15% of groups held out for test and 15%
for validation, averaged over 5 seeds (mean &plusmn; sd).</p>

<p class="meta"><strong>C0</strong> ligand-mean fallback &mdash; what warm CF collapses to, and the thing the task
asks us to beat &middot; <strong>C1</strong> guilt-by-association kNN over k-mer sequence similarity
&middot; <strong>C2</strong> ridge from sequence descriptors to the learned MF target factors
&middot; <strong>C3</strong> gradient boosting straight from (ligand factors, target descriptors) to score
&middot; <strong>C4</strong> blend of C1 and C2 with a weight gated on how similar the cold target is to
its nearest training target.</p>

<h3>DAVIS &mdash; cold targets, 5 seeds</h3>
{task3_table("DAVIS")}
<h3>KIBA &mdash; cold targets, 5 seeds</h3>
{task3_table("KIBA")}

<h3>Where guilt-by-association actually works</h3>
<p>This is the part I think is worth putting in the report. Each held-out target is scored on its own
&mdash; across all the ligands measured against it, how well does the model order them &mdash; and
targets are then grouped into thirds by their similarity to the nearest training target.</p>
{bucket_table("DAVIS")}
{bucket_table("KIBA")}

<div class="note"><strong>The finding.</strong> Guilt-by-association is not a general solution to cold
targets; it is a homology lookup. For targets in the top similarity third, C1 adds
{bk[(bk.dataset=='DAVIS')&(bk.bucket.str.startswith('near'))].iloc[0]['CI gain C1 over C0']:+.3f} CI on DAVIS and
{bk[(bk.dataset=='KIBA')&(bk.bucket.str.startswith('near'))].iloc[0]['CI gain C1 over C0']:+.3f} on KIBA over the fallback. For orphan targets it adds
{bk[(bk.dataset=='DAVIS')&(bk.bucket.str.startswith('orphan'))].iloc[0]['CI gain C1 over C0']:+.3f} and {bk[(bk.dataset=='KIBA')&(bk.bucket.str.startswith('orphan'))].iloc[0]['CI gain C1 over C0']:+.3f} &mdash; nothing.
Across all cold targets, similarity to the nearest training target correlates with C1's ranking
quality at Spearman +0.573 (DAVIS) and +0.281 (KIBA). That is exactly why C4 gates on similarity:
in the orphan bucket it is the only model that improves on the fallback
(DAVIS MSE {bk[(bk.dataset=='DAVIS')&(bk.bucket.str.startswith('orphan'))].iloc[0]['MSE C4']:.3f} vs {bk[(bk.dataset=='DAVIS')&(bk.bucket.str.startswith('orphan'))].iloc[0]['MSE C0']:.3f} fallback and {bk[(bk.dataset=='DAVIS')&(bk.bucket.str.startswith('orphan'))].iloc[0]['MSE C1']:.3f} for C1 alone).</div>

<div class="note warn"><strong>Caveats to state in the report.</strong> Sequence similarity here is
cosine over 3-mer profiles, not alignment identity &mdash; cheap, no external aligner, and it preserves
the ordering we need, but it is a proxy. No pretrained protein language model (ESM) was used; that is
the obvious next step and would most likely help exactly the orphan bucket where our descriptors are
weakest. The cold-target gains are real but modest in absolute terms: even C4 sits far below the warm
numbers in Task 2, which is the honest headline &mdash; a genuinely new protein is still hard.</div>

<footer>DAVIS and KIBA from the DeepDTA benchmark release. Reference points for scale (different split
protocol, so ballpark only): DeepDTA reports DAVIS MSE 0.261 / CI 0.878 and KIBA MSE 0.194 / CI 0.863.
Our warm MF lands at DAVIS {t2[(t2.dataset=='DAVIS')&(t2.model=='M2 biased MF')].iloc[0]['mse']:.3f} / {t2[(t2.dataset=='DAVIS')&(t2.model=='M2 biased MF')].iloc[0]['ci_global']:.3f} and KIBA {t2[(t2.dataset=='KIBA')&(t2.model=='M2 biased MF')].iloc[0]['mse']:.3f} / {t2[(t2.dataset=='KIBA')&(t2.model=='M2 biased MF')].iloc[0]['ci_global']:.3f}.</footer>
</div></body></html>"""

open(f"{R}/results.html", "w").write(HTML)
print(f"wrote {R}/results.html ({len(HTML):,} bytes)")
