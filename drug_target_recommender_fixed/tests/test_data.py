import numpy as np
from src.data import canonical_smiles, convert_scores


def test_score_scales_and_missingness():
    values = np.array([10000., 10., 0., -1., np.nan, np.inf, -np.inf])
    result, valid = convert_scores(values, 'Kd_nM')
    np.testing.assert_allclose(result[:2], [5., 8.])
    assert valid.sum() == 2 and np.isnan(result[2:]).all()
    result, valid = convert_scores([0., -4., 5., 1e9, np.inf], 'DeepDTA_KIBA')
    np.testing.assert_array_equal(result[:4], [0., -4., 5., 1e9])
    assert valid.sum() == 4
    result, _ = convert_scores([5., 8.], 'pKd')
    np.testing.assert_array_equal(result, [5., 8.])


def test_canonical_identity():
    assert canonical_smiles('OCC') == canonical_smiles('CCO')
    assert canonical_smiles('C[C@H](O)F') != canonical_smiles('C[C@@H](O)F')


def test_sequence_normalization_and_duplicate_provenance():
    import pandas as pd
    from src.data import normalize_sequence, aggregate_observations
    assert normalize_sequence(' acd\n EF ') == 'ACDEF'
    f = pd.DataFrame({'drug_id': [0, 0], 'target_id': [0, 0], 'affinity': [5., 7.],
                      'censored': [True, False], 'suspicious': [False, False],
                      'original_drug_id': ['a', 'b'], 'original_target_id': ['x', 'y'],
                      'source_cell': ['0:0', '1:1']})
    row = aggregate_observations(f).iloc[0]
    assert row.affinity == 6 and row.n_merged == 2 and row.censored_fraction == .5
    assert row.original_drug_ids == 'a|b' and row.source_cells == '0:0|1:1'
