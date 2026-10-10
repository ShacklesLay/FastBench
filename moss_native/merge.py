"""Merge disjoint successful shards of one predeclared scope; never mix split task runs."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .clock import read_jsonl, write_jsonl
from .policy import describe, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    receipts = [json.loads((run / 'receipt.json').read_text()) for run in args.runs]
    assert all(r['ok'] and r['api_failures'] == 0 and r['protocol'] == describe() for r in receipts)
    for key in ('num_shards', 'scope', 'dataset_sha256', 'config_sha256'):
        assert len({r[key] for r in receipts}) == 1, key
    assert all(r['max_samples'] == 0 for r in receipts), 'Do not present a smoke subset as complete'
    assert {r['shard_index'] for r in receipts} == set(range(receipts[0]['num_shards']))
    assert len(receipts) == receipts[0]['num_shards']
    predictions, raw = [], []
    for run, receipt in zip(args.runs, receipts):
        assert sha(run / 'predictions.jsonl') == receipt['predictions_sha256']
        assert sha(run / 'original_HTTP.jsonl') == receipt['raw_sha256']
        predictions.extend(read_jsonl(run / 'predictions.jsonl'))
        raw.extend(read_jsonl(run / 'original_HTTP.jsonl'))
    assert len({str(s['id']) for s in predictions}) == len(predictions), 'Duplicate video writers'
    counts = Counter({})
    for receipt in receipts:
        counts.update(receipt['task_counts'])
    scope = receipts[0]['scope']
    expected = {'all': (300, {'forward': 168, 'instant': 73, 'backward': 65}),
                'forward': (167, {'forward': 168}),
                'instant_backward': (136, {'instant': 73, 'backward': 65})}[scope]
    assert len(predictions) == expected[0] and counts == expected[1], 'Missing source samples/questions'
    args.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output / 'predictions.jsonl', predictions)
    write_jsonl(args.output / 'original_HTTP.jsonl', raw)
    (args.output / 'receipt.json').write_text(json.dumps({
        'ok': True, 'protocol': describe(), 'scope': scope, 'videos': len(predictions),
        'qa': sum(counts.values()), 'task_counts': dict(counts), 'api_failures': 0,
        'predictions_sha256': sha(args.output / 'predictions.jsonl'),
        'raw_sha256': sha(args.output / 'original_HTTP.jsonl'),
        'shards': [str(p) for p in args.runs]}, indent=2) + '\n')


if __name__ == '__main__':
    main()
