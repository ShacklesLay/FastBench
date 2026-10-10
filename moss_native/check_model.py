"""Check the exact public MOSS-VL-Realtime snapshot without copying weights into the fork."""
import argparse
import hashlib
import json
from pathlib import Path


def check(path, weights=False):
    manifest = json.loads(Path(__file__).with_name('model_manifest.json').read_text())
    checked = []
    for item in manifest['files']:
        if not weights and item['file'].endswith('.safetensors'):
            continue
        file = path / item['file']
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if file.stat().st_size != item['size'] or digest.hexdigest() != item['sha256']:
            raise ValueError(f'Public MOSS-VL-Realtime snapshot mismatch: {item["file"]}')
        checked.append(item['file'])
    return {'ok': True, 'model': manifest['id'], 'revision': manifest['revision'],
            'weights_checked': weights, 'checked_files': checked}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--weights', action='store_true')
    args = p.parse_args()
    print(json.dumps(check(args.model_path, args.weights), indent=2))


if __name__ == '__main__':
    main()
