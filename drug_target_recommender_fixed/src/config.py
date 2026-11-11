"""Experiment policy; changes require a new output workspace/version."""
from pathlib import Path
import os

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('DTR_ROOT', CODE_ROOT)).resolve()
VERSION = 'v1'
SEED = 2026
COLD_SEEDS = tuple(range(SEED, SEED + 5))
DATASETS = ('davis', 'kiba')
MF_GRID = {'factors': (0, 4, 8, 16, 32), 'regularization': (.01, .1), 'learning_rate': (.02,)}
MF_EPOCHS = 160
MF_PATIENCE = 16
KNN_K = (5, 10, 20, 50)
KNN_BETA = (0., 10., 50., 100.)
COLD_K = (5, 15, 50)
COLD_POWER = (1., 4.)
RIDGE_ALPHA = (.1, 1., 10., 100.)
SIMILARITY_BINS = (0., .3, .5, .7, 1.000001)
CLIP_PREDICTIONS = False
