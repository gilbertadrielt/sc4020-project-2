"""Download only the pinned, trusted DeepDTA files; never execute remote code."""
import hashlib
import json
import sys
import ssl
import certifi
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, write_json

COMMIT = 'a546a8433a6822e958f36171c4356ad6f414d623'
BASE = f'https://raw.githubusercontent.com/hkmztrk/DeepDTA/{COMMIT}/data'


def main():
    for dataset in ['davis', 'kiba']:
        directory = ROOT / 'data/raw' / dataset
        directory.mkdir(parents=True, exist_ok=True)
        previous = json.loads((directory / 'source.json').read_text()) if (directory / 'source.json').exists() else None
        files = {}
        for name in ['ligands_can.txt', 'proteins.txt', 'Y']:
            path = directory / name
            url = f'{BASE}/{dataset}/{name}'
            content = path.read_bytes() if path.exists() else urlopen(
                url, timeout=60, context=ssl.create_default_context(cafile=certifi.where())).read()
            digest = hashlib.sha256(content).hexdigest()
            if previous and digest != previous['files'][name]['sha256']:
                raise ValueError(f'Raw file changed: {path}')
            if not path.exists():
                path.write_bytes(content)
            files[name] = {'url': url, 'sha256': digest, 'bytes': len(content)}
        manifest = {'repository': 'https://github.com/hkmztrk/DeepDTA', 'commit': COMMIT,
                    'input_scale': 'Kd_nM' if dataset == 'davis' else 'DeepDTA_KIBA', 'files': files}
        write_json(directory / 'source.json', manifest)
        print(dataset, {key: val['bytes'] for key, val in files.items()}, flush=True)


if __name__ == '__main__':
    main()
