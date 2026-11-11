import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATASETS
from src.splits import make_splits
if __name__ == '__main__':
    for dataset in DATASETS:
        make_splits(dataset)
