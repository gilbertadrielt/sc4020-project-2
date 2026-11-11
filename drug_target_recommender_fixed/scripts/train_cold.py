import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments import run_cold
from src.config import DATASETS, COLD_SEEDS, ROOT, CODE_ROOT
p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=[*DATASETS, 'all'], default='all')
p.add_argument('--seeds', nargs='+', type=int, default=list(COLD_SEEDS))
p.add_argument('--smoke', action='store_true')
a = p.parse_args()
if a.smoke and ROOT == CODE_ROOT:
    p.error('Smoke training requires an isolated DTR_ROOT; run scripts/run_pipeline.py --smoke')
# Complete seed 2026 on BOTH datasets before extending to further splits.
for seed in a.seeds:
    for dataset in DATASETS if a.dataset == 'all' else [a.dataset]:
        run_cold(dataset, seed, a.smoke)
