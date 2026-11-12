"""KronRLS is checked against an independent computation, not against itself.

The iterative solver is the one used in production (it is the only one that
handles an incomplete matrix), so its correctness has to be demonstrated.
Three independent references are used:

1. the closed-form eigendecomposition solution, on a complete grid;
2. an explicit dense Kronecker solve on a small problem, which is the
   definition of the model with no tricks at all;
3. the model's own defining equation, checked as a residual.
"""
import numpy as np
import pytest

from dta.kronrls import KronRLS


def _toy(n_d=6, n_t=5, complete=True, seed=0, drop=4):
    """A small random problem with valid PSD kernels."""
    rng = np.random.default_rng(seed)
    import pandas as pd

    Xd = rng.normal(size=(n_d, 3))
    Xt = rng.normal(size=(n_t, 3))
    KD = Xd @ Xd.T + np.eye(n_d) * 0.5
    KT = Xt @ Xt.T + np.eye(n_t) * 0.5

    rows, cols = np.meshgrid(np.arange(n_d), np.arange(n_t), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    y = rng.normal(size=len(rows))
    if not complete:
        keep = rng.permutation(len(rows))[: len(rows) - drop]
        rows, cols, y = rows[keep], cols[keep], y[keep]

    pairs = pd.DataFrame({"lig_idx": rows, "tgt_idx": cols, "score": y})
    return pairs, n_d, n_t, {"drug_structural": KD, "target_sequence": KT}


def _dense_reference(pairs, n_d, n_t, kernels, lam, mu):
    """The definition: build the full pair kernel and solve it directly."""
    KD, KT = kernels["drug_structural"], kernels["target_sequence"]
    li = pairs["lig_idx"].to_numpy()
    ti = pairs["tgt_idx"].to_numpy()
    y = pairs["score"].to_numpy(dtype=float) - mu
    n = len(li)
    K = np.empty((n, n))
    for a in range(n):
        for b in range(n):
            K[a, b] = KD[li[a], li[b]] * KT[ti[a], ti[b]]
    alpha = np.linalg.solve(K + lam * np.eye(n), y)
    return K @ alpha + mu  # fitted values on the training pairs


def test_cg_matches_eigen_on_a_complete_grid():
    """The two solvers are different algorithms; on a complete grid they must agree."""
    pairs, n_d, n_t, ks = _toy(complete=True)
    out = {}
    for solver in ("eigen", "cg"):
        m = KronRLS(lam=0.7, solver=solver, center="global", cg_tol=1e-12, cg_maxiter=5000)
        m.fit(pairs, n_d, n_t, kernels=ks)
        out[solver] = m.predict(pairs)
    assert np.allclose(out["eigen"], out["cg"], atol=1e-7)


def test_cg_matches_a_dense_kronecker_solve_when_incomplete():
    """Against the literal definition, with an incomplete observation pattern."""
    pairs, n_d, n_t, ks = _toy(complete=False, drop=7)
    lam = 0.9
    m = KronRLS(lam=lam, solver="cg", center="global", cg_tol=1e-13, cg_maxiter=10000)
    m.fit(pairs, n_d, n_t, kernels=ks)
    got = m.predict(pairs)
    want = _dense_reference(pairs, n_d, n_t, ks, lam, mu=float(pairs["score"].mean()))
    assert np.allclose(got, want, atol=1e-6)


def test_solution_satisfies_the_normal_equations():
    """Residual check of (P K P + lam I) A = Y on the observed set."""
    pairs, n_d, n_t, ks = _toy(complete=False, drop=5)
    lam = 1.3
    m = KronRLS(lam=lam, solver="cg", center="global", cg_tol=1e-13, cg_maxiter=10000)
    m.fit(pairs, n_d, n_t, kernels=ks)

    li, ti = pairs["lig_idx"].to_numpy(), pairs["tgt_idx"].to_numpy()
    Y = np.zeros((n_d, n_t))
    Y[li, ti] = pairs["score"].to_numpy(dtype=float) - float(pairs["score"].mean())
    obs = np.zeros((n_d, n_t), bool)
    obs[li, ti] = True
    lhs = np.where(obs, m.KD_ @ np.where(obs, m.A_, 0.0) @ m.KT_, 0.0) + lam * m.A_
    assert np.abs(lhs - Y).max() < 1e-6


def test_auto_solver_picks_eigen_only_for_complete_grids():
    pairs, n_d, n_t, ks = _toy(complete=True)
    assert KronRLS(solver="auto", center="global").fit(
        pairs, n_d, n_t, kernels=ks
    ).solver_used_ == "eigen"

    pairs, n_d, n_t, ks = _toy(complete=False)
    assert KronRLS(solver="auto", center="global").fit(
        pairs, n_d, n_t, kernels=ks
    ).solver_used_ == "cg"


def test_eigen_solver_refuses_an_incomplete_grid():
    pairs, n_d, n_t, ks = _toy(complete=False)
    with pytest.raises(ValueError, match="every pair observed"):
        KronRLS(solver="eigen", center="global").fit(pairs, n_d, n_t, kernels=ks)


def test_larger_lambda_shrinks_the_interaction():
    """Basic sanity: more regularisation means a flatter prediction surface."""
    pairs, n_d, n_t, ks = _toy(complete=True)
    norms = []
    for lam in (0.1, 1.0, 100.0):
        m = KronRLS(lam=lam, solver="eigen", center="global").fit(pairs, n_d, n_t, kernels=ks)
        norms.append(np.linalg.norm(m.predict_interaction(pairs)))
    assert norms[0] > norms[1] > norms[2]


def test_predicts_for_a_target_never_seen_in_training():
    """The whole point of the model: a cold column still gets a prediction.

    The held-out target contributes no training pair, so a collaborative
    model could only return a constant for it. Here its kernel column exists,
    so predictions vary across drugs.
    """
    pairs, n_d, n_t, ks = _toy(complete=True, n_t=6)
    cold = 5
    train = pairs[pairs["tgt_idx"] != cold]
    test = pairs[pairs["tgt_idx"] == cold]

    m = KronRLS(lam=0.5, solver="cg", center="bias").fit(train, n_d, n_t, kernels=ks)
    pred = m.predict(test)
    assert len(pred) == n_d
    assert np.ptp(pred) > 1e-8, "a cold target must not collapse to one constant"
