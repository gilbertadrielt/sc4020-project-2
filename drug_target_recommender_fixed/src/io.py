"""Read prediction score columns with Python's exact round-trip float parser."""
import pandas as pd

SCORES = ('affinity', 'global_mean', 'ligand_mean', 'target_mean', 'bias_only',
          'target_knn', 'MF', 'sequence_neighbors', 'factor_ridge', 'max_train_similarity')


def read_predictions(path):
    # Explicit converters avoid a faster CSV parser rounding adjacent doubles to
    # the same value, which could change exact ranking tie credit after reload.
    return pd.read_csv(path, converters={column: float for column in SCORES})
