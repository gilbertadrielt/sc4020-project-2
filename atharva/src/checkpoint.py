"""Persist and restore fitted CF state so predictions reproduce exactly."""
import numpy as np
import models


def save_cf(path, ds, mf, knn, blend_w, split_seed, hparams):
    np.savez_compressed(
        path,
        dataset=ds.name, score_scale="pKd" if ds.name == "DAVIS" else "KIBA composite",
        score_lo=mf.lo, score_hi=mf.hi, strong_threshold=ds.strong_threshold,
        ligand_ids=np.asarray(ds.ligand_ids, dtype=object),
        target_ids=np.asarray(ds.target_ids, dtype=object),
        mf_mu=mf.mu, mf_b=mf.b, mf_c=mf.c, mf_U=mf.U, mf_V=mf.V,
        knn_sim=knn.sim, knn_k=knn.k,
        knn_mu=knn.base.mu, knn_b=knn.base.b, knn_c=knn.base.c,
        knn_obs_rows=knn._fit_rows, knn_obs_cols=knn._fit_cols, knn_obs_resid=knn._fit_resid,
        blend_w=blend_w, split_seed=split_seed,
        hparams=np.asarray(str(hparams), dtype=object),
    )


class LoadedCF:
    """Rebuilds MF, kNN and the blend from a checkpoint file."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.z = z
        self.dataset = str(z["dataset"])
        self.ligand_ids = list(z["ligand_ids"])
        self.target_ids = list(z["target_ids"])
        self.lo, self.hi = float(z["score_lo"]), float(z["score_hi"])
        self.blend_w = float(z["blend_w"])

        mf = models.BiasedMF()
        mf.mu = float(z["mf_mu"]); mf.b = z["mf_b"]; mf.c = z["mf_c"]
        mf.U = z["mf_U"]; mf.V = z["mf_V"]; mf.lo, mf.hi = self.lo, self.hi
        self.mf = mf

        base = models.BiasOnly()
        base.mu = float(z["knn_mu"]); base.b = z["knn_b"]; base.c = z["knn_c"]
        base.lo, base.hi = self.lo, self.hi
        knn = models.TargetKNN(k=int(z["knn_k"]))
        knn.base = base; knn.sim = z["knn_sim"]; knn.lo, knn.hi = self.lo, self.hi
        knn._restore_support(z["knn_obs_rows"], z["knn_obs_cols"], z["knn_obs_resid"])
        self.knn = knn

    def predict(self, rows, cols):
        rows = np.asarray(rows); cols = np.asarray(cols)
        p = self.blend_w * self.knn.predict(rows, cols) + (1 - self.blend_w) * self.mf.predict(rows, cols)
        return np.clip(p, self.lo, self.hi)

    def predict_mf(self, rows, cols):
        return self.mf.predict(np.asarray(rows), np.asarray(cols))
