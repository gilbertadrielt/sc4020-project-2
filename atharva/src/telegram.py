"""Render the results as monospace tables for pasting into the group chat."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
t2 = pd.read_csv(f"{R}/task2_test_results.csv")
t3 = pd.read_csv(f"{R}/task3_summary.csv")
bk = pd.read_csv(f"{R}/task3_similarity_buckets.csv")
meta = json.load(open(f"{R}/task2_meta.json"))

SHORT2 = {
    "B0 global mean": "global mean", "B1 ligand mean": "ligand mean",
    "B2 target mean": "target mean", "B3 bias only": "bias only",
    "M1 target kNN": "target-kNN", "M2 biased MF": "biased MF",
    "M3 kNN + MF blend": "kNN+MF blend", "random ranking": "random",
}
SHORT3 = {
    "C0 ligand-mean fallback": "C0 ligand-mean (fallback)",
    "C1 guilt-by-association kNN": "C1 guilt-by-assoc kNN",
    "C2 sequence -> MF factors": "C2 seq -> MF factors",
    "C3 content regression": "C3 content regression",
    "C4 similarity-gated blend": "C4 sim-gated blend",
    "random ranking": "random",
}


def box(headers, rows, bold_row=None):
    """Draw a box table, padding each column to its widest cell."""
    cols = list(zip(*([headers] + rows)))
    w = [max(len(str(c)) for c in col) for col in cols]
    def line(l, m, r):
        return l + m.join("─" * (x + 2) for x in w) + r
    def row(cells, left=True):
        out = []
        for i, c in enumerate(cells):
            c = str(c)
            out.append(" " + (c.ljust(w[i]) if i == 0 else c.rjust(w[i])) + " ")
        return "│" + "│".join(out) + "│"
    body = [line("┌", "┬", "┐"), row(headers), line("├", "┼", "┤")]
    for r in rows:
        body.append(row(r))
    body.append(line("└", "┴", "┘"))
    return "\n".join(body)


def task2(ds):
    g = t2[t2.dataset == ds]
    hp = meta[ds.lower()]
    rows = []
    for _, r in g.iterrows():
        nm = SHORT2[r["model"]]
        if nm == "target-kNN":
            nm += f" (k={hp['best_knn']['k']})"
        if nm == "biased MF":
            nm += f" (k={hp['best_mf']['k']})"
        f = lambda v, n=3: "  —  " if not np.isfinite(v) else f"{v:.{n}f}"
        rows.append([nm, f(r["mse"]), f(r["mse_strong"]), f(r["ci_global"]),
                     f(r["ci_per_ligand"]), f(r["hit@1"]), f(r["hit@5"]), f(r["mrr"])])
    return box(["Model", "MSE", "MSE strong", "CI", "CI/lig", "Hit@1", "Hit@5", "MRR"], rows)


def task3(ds):
    g = t3[t3.dataset == ds]
    rows = []
    for _, r in g.iterrows():
        f = lambda c, n=3: ("  —  " if not np.isfinite(r[f"{c}_mean"])
                            else f"{r[f'{c}_mean']:.{n}f}")
        sd = lambda c, n=3: ("  —  " if not np.isfinite(r[f"{c}_std"])
                             else f"±{r[f'{c}_std']:.{n}f}")
        fb = r["fallback_frac_mean"]
        rows.append([SHORT3[r["model"]], f("mse"), sd("mse"), f("ci_global"),
                     sd("ci_global"), f("ci_per_ligand"), f("hit@1"), f("mrr"),
                     "  —  " if not np.isfinite(fb) else f"{fb*100:.0f}%"])
    return box(["Model", "MSE", "±sd", "CI", "±sd", "CI/lig", "Hit@1", "MRR", "Fallbk"], rows)


def buckets(ds):
    g = bk[bk.dataset == ds]
    rows = []
    for _, r in g.iterrows():
        rows.append([r["bucket"].split(" (")[0], r["sim_range"], int(r["n_targets"]),
                     f'{r["CI C0"]:.3f}', f'{r["CI C1"]:.3f}', f'{r["CI C4"]:.3f}',
                     f'{r["CI gain C1 over C0"]:+.3f}'])
    return box(["Cold target", "Sim", "N", "CI C0", "CI C1", "CI C4", "C1 gain"], rows)


if __name__ == "__main__":
    parts = []
    parts.append("### TASK 2 MESSAGE ###\n")
    parts.append(f"DAVIS ({meta['davis']['n_test']:,} test pairs, 68 drugs)\n" + task2("DAVIS"))
    parts.append(f"\nKIBA ({meta['kiba']['n_test']:,} test pairs, 2,111 drugs)\n" + task2("KIBA"))
    parts.append("\n\n### TASK 3 MESSAGE ###\n")
    parts.append("DAVIS (cold targets, 5 seeds)\n" + task3("DAVIS"))
    parts.append("\nKIBA (cold targets, 5 seeds)\n" + task3("KIBA"))
    parts.append("\n\n### BUCKETS ###\n")
    parts.append("DAVIS\n" + buckets("DAVIS"))
    parts.append("\nKIBA\n" + buckets("KIBA"))
    out = "\n".join(parts)
    open(f"{R}/telegram_tables.txt", "w").write(out)
    print(out)
