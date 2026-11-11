import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments import run_cf
from src.config import DATASETS
p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=[*DATASETS, 'all'], default='all')
p.add_argument('--smoke', action='store_true')
a = p.parse_args()
if a.smoke:
    from src.config import ROOT, CODE_ROOT
    if ROOT == CODE_ROOT:
        p.error('Smoke training must use an isolated DTR_ROOT; run scripts/run_pipeline.py --smoke')
for dataset in DATASETS if a.dataset == 'all' else [a.dataset]:
    run_cf(dataset, a.smoke)
