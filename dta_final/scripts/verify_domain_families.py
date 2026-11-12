"""Check two claims in Section 2.3 that have no stored provenance:

  (a) what share of DAVIS targets the seven domain-pair families actually are;
  (b) whether excluding them lowers the reported variance explained, and by how much.

(b) is the robustness claim: if those families inflate our numbers, removing them
should RAISE variance explained, not lower it.
"""
import numpy as np
import pandas as pd

from dta import data as D
from dta import metrics as MT
from dta import models as M
from dta import splits as S
from dta.config import SPLIT_SEEDS

LOW_CORR = ["JAK1", "TYK2", "RSK1", "RSK3", "RSK4", "RPS6KA4", "RPS6KA5"]

proc = D.load_processed("davis")
pairs, tg = proc["pairs"], proc["targets"]
raw = D.load_raw("davis")

# --- (a) how many source identifiers and merged entries do the 7 families cover?
id_col = [c for c in tg.columns if "ids" in c or "source" in c]
print("targets columns:", tg.columns.tolist())
n_src_total = sum(len(str(r).split("|")) for r in tg[id_col[0]]) if id_col else None
hits, src_hits = [], 0
for i, row in tg.iterrows():
    ids = str(row[id_col[0]]) if id_col else ""
    members = [m.strip() for m in ids.replace("|", ",").split(",") if m.strip()]
    base = {m.split("(")[0].strip() for m in members}
    if base & set(LOW_CORR):
        hits.append(i)
        src_hits += len(members)
print(f"(a) merged entries touched : {len(hits)} of {len(tg)}  = {100*len(hits)/len(tg):.2f}%")
print(f"    source identifiers     : {src_hits} of {n_src_total} = {100*src_hits/n_src_total:.2f}%")

# --- (b) variance explained with and without those targets in the test set
drop = set(hits)
rows = []
for seed in SPLIT_SEEDS:
    assign = S.load_split("davis", "S1_pair", seed)
    tr = pairs[assign == "train"]
    va = pairs[assign == "val"]
    te = pairs[assign == "test"]
    model = M.BiasedMF(k=32, lam=0.05).fit(tr, proc["n_lig"], proc["n_tgt"], val=va)
    y = te["score"].to_numpy(dtype=float)
    p = model.predict(te)
    keep = ~te["t"].isin(drop).to_numpy() if "t" in te.columns else None
    if keep is None:
        col = [c for c in te.columns if c.startswith("t")][0]
        keep = ~te[col].isin(drop).to_numpy()
    def r2(yy, pp):
        return 1.0 - MT.mse(yy, pp) / float(np.var(yy))
    rows.append({"seed": seed, "r2_all": r2(y, p), "r2_excl": r2(y[keep], p[keep]),
                 "n_all": len(y), "n_excl": int(keep.sum())})

d = pd.DataFrame(rows)
print()
print(d.round(4).to_string(index=False))
print()
print("(b) variance explained, all targets      : %.4f" % d.r2_all.mean())
print("    variance explained, 7 families out   : %.4f" % d.r2_excl.mean())
print("    change in percentage points          : %+.2f" % (100 * (d.r2_excl.mean() - d.r2_all.mean())))
