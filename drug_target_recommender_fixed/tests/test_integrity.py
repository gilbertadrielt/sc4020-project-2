import json
import numpy as np
import pandas as pd
import pytest
from src.common import sha256, write_immutable_csv
from src.baselines import TargetMean
from src.cf import Means, BiasOnly
from src.features import protein_features, protein_composition


def test_frozen_data_cannot_be_silently_replaced(tmp_path):
    path = tmp_path / 'frozen.csv'
    frame = pd.DataFrame({'key': [1], 'value': [3.]})
    write_immutable_csv(frame, path)
    digest = sha256(path)
    write_immutable_csv(frame, path)
    assert sha256(path) == digest
    with pytest.raises(RuntimeError, match='frozen'):
        write_immutable_csv(frame.assign(value=4), path)
    assert sha256(path) == digest


def test_raw_checksum_rejection_precedes_pickle_loading(tmp_path, monkeypatch):
    from src import data
    root = tmp_path / 'data/raw/davis'
    root.mkdir(parents=True)
    (root / 'Y').write_bytes(b'not a pickle and must never be opened as one')
    (root / 'source.json').write_text(json.dumps({'files': {'Y': {'sha256': '0' * 64}}}))
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='checksum mismatch'):
        data.load_raw('davis')


def test_frozen_split_hash_tampering_rejected(tmp_path, monkeypatch):
    from src import splits
    directory = tmp_path / 'data/processed/splits/task2/davis'
    directory.mkdir(parents=True)
    source = tmp_path / 'data/processed/pairs_davis.csv'
    source.write_text('drug_id,target_id,affinity\n0,0,3\n')
    path = directory / 'split.csv'
    path.write_text('pair_index,drug_id,target_id,split\n0,0,0,train\n')
    entities = directory / 'target_entities.csv'
    entities.write_text('target_id,sequence\n0,AAAA\n')
    manifest = {'clean_sha256': sha256(source), 'split_sha256': sha256(path), 'entity_sha256': sha256(entities)}
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    path.write_text(path.read_text().replace('train', 'test'))
    monkeypatch.setattr(splits, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Frozen artifact changed'):
        splits.load_splits('davis', 'task2')


def test_mean_baselines_only_training_and_unknown_fallback():
    train = pd.DataFrame({'drug_id': [0, 0, 1], 'target_id': [0, 1, 1], 'affinity': [1., 3., 5.]})
    query = pd.DataFrame({'drug_id': [0, 2], 'target_id': [0, 2], 'affinity': [999., -999.]})
    ligand, target, bias = Means(train, 3), TargetMean(train, 3), BiasOnly(train, 3, 3)
    np.testing.assert_array_equal(ligand.predict(query), [2., 3.])
    np.testing.assert_array_equal(target.predict(query), [1., 3.])
    assert bias.predict(query)[1] == 3.
    original = bias.predict(query)
    query.affinity = 0
    np.testing.assert_array_equal(bias.predict(query), original)


def test_fixed_protein_features_ignore_other_query_sequences():
    alone = protein_features(['ACDEACDE']).toarray()[0]
    together = protein_features(['ACDEACDE', 'ZZZZYYYY']).toarray()[0]
    np.testing.assert_array_equal(alone, together)
    assert protein_features(['ACDE']).shape[1] == 26**3 + 20 + 400
    assert protein_composition(['AAAA']).sum() == 1
