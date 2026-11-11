import hashlib
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from .config import ROOT, SEED, VERSION


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_immutable_csv(frame, path):
    path = Path(path)
    content = frame.to_csv(index=False, float_format='%.12g')
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f'Refusing to overwrite frozen data: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content)


def log(phase, message):
    path = ROOT / 'results/implementation_log.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(f'\n- {timestamp()} | {phase} | {message}\n')


def timestamp():
    return datetime.now(ZoneInfo('Asia/Singapore')).isoformat()


def git_commit():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip() or 'unavailable'


def provenance():
    return {'version': VERSION, 'seed': SEED, 'date_sgt': timestamp(), 'git_commit': git_commit(),
            'python': sys.version, 'test_used_for_selection': False}


def setup():
    for name in ['tables', 'figures', 'predictions', 'logs']:
        (ROOT / 'results' / name).mkdir(parents=True, exist_ok=True)
    (ROOT / 'saved_models').mkdir(parents=True, exist_ok=True)
