"""Fresh interpreter prediction check: query file contains NO affinity labels."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from src.cf import MatrixFactorization
from src.cold import ColdTargetNeighbors, ColdFactorRidge
p = argparse.ArgumentParser()
p.add_argument('kind', choices=['mf', 'neighbors', 'ridge'])
p.add_argument('model')
p.add_argument('query')
p.add_argument('output')
a = p.parse_args()
model = {'mf': MatrixFactorization, 'neighbors': ColdTargetNeighbors, 'ridge': ColdFactorRidge}[a.kind].load(a.model)
query = pd.read_csv(a.query)
assert 'affinity' not in query
prediction = model.predict(query)
if isinstance(prediction, tuple):
    prediction = prediction[0]
np.save(a.output, prediction, allow_pickle=False)
