import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data import prepare
from src.config import DATASETS, ROOT
from src.common import setup, write_immutable_csv
if __name__ == '__main__':
    setup()
    summaries = [prepare(dataset) for dataset in DATASETS]
    keys = ['dataset', 'raw_ligands', 'raw_targets', 'raw_cells', 'raw_finite', 'missing_nan', 'invalid_infinite', 'invalid_nonpositive_kd', 'invalid_entities', 'entity_excluded_observations', 'valid_observations', 'unique_ligands', 'unique_targets', 'unique_pairs', 'merged_groups', 'conflicting_groups', 'missing_fraction', 'censored_raw_measurements', 'censored_clean_pairs', 'suspicious_raw_measurements']
    table = pd.DataFrame([{k: s[k] for k in keys} for s in summaries])
    write_immutable_csv(table, ROOT / 'data/processed/cleaning_log.csv')
    print(table.to_string(index=False))
