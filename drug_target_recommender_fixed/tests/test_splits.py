import numpy as np
import pandas as pd
import pytest
from src.splits import warm_split, cold_split, validate_split


def fixture():
    return pd.DataFrame([(d, t, f'AAA{chr(65+t)}') for d in range(8) for t in range(10)], columns=['drug_id', 'target_id', 'sequence'])


def test_deterministic_warm_and_cold():
    frame = fixture()
    a, _ = warm_split(frame)
    np.testing.assert_array_equal(a, warm_split(frame)[0])
    validate_split(frame, a, 'task2')
    c = cold_split(frame)
    np.testing.assert_array_equal(c, cold_split(frame))
    validate_split(frame, c, 'task3')


def test_exact_sequence_aliases_stay_together():
    frame = fixture()
    frame.loc[frame.target_id == 1, 'sequence'] = 'AAAA'
    c = cold_split(frame)
    assert len(set(c[frame.target_id.isin([0, 1])])) == 1
    validate_split(frame, c, 'task3')


def test_coverage_repair_and_detect_leakage():
    frame = fixture()
    extra = pd.DataFrame({'drug_id': [20], 'target_id': [30], 'sequence': ['CCCC']})
    frame = pd.concat([frame, extra], ignore_index=True)
    labels, _ = warm_split(frame)
    validate_split(frame, labels, 'task2')
    c = cold_split(frame)
    c[0] = 'test' if c[0] == 'train' else 'train'
    with pytest.raises(AssertionError):
        validate_split(frame, c, 'task3')
