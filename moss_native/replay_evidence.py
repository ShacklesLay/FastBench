"""Re-score saved outputs with saved exact-input judgments, offline; no model or API."""
import json
from .clock import read_jsonl, seconds
from .policy import ROOT, sha, verify_upstream
from .score import evaluate


def main():
    verify_upstream()
    import yaml
    import run_stream_eval as up
    up.init_runtime_constants(yaml.safe_load((ROOT / 'config.yaml').read_text()))
    up.time_to_seconds = seconds
    total, scores = 0, 0
    for name in ('forward', 'instant_backward'):
        run = ROOT / 'moss_native/evidence/runs' / name
        receipt = json.loads((run / 'receipt.json').read_text())
        assert receipt['ok'] and receipt['evidence_export_only']
        for file, digest in receipt['files'].items():
            assert sha(run / file) == digest, file
        old = read_jsonl(run / 'result_details.jsonl')
        judgments = {}
        for row in old:
            if row['category'] == 'PartlyCorrect' and not row['is_objective']:
                key = (row['question'], row['response'], row['answer'])
                value = {'score': row['score'] / 20, 'explanation': row['explanation']}
                assert key not in judgments or judgments[key] == value
                judgments[key] = value
        class SavedJudge:
            def judge(self, question, response, answer):
                return judgments[(question, response, answer)]
        rows = evaluate(up, read_jsonl(run / 'predictions.jsonl'), SavedJudge())
        key = lambda r: (r['sample_id'], r['question_time'], r['question'])
        reference = {key(r): r for r in old}
        assert len(rows) == receipt['qa'] == len(reference)
        assert {key(r) for r in rows} == set(reference)
        for row in rows:
            for field in ('response', 'response_time', 'response_anchor_seconds', 'category', 'score', 'explanation'):
                assert row[field] == reference[key(row)][field], (key(row), field)
        mean = sum(r['score'] for r in rows) / len(rows)
        print(name, len(rows), 'QA:', mean, 'PASS')
        total += len(rows); scores += sum(r['score'] for r in rows)
    assert total == 306
    print('Weighted split-run mean:', scores / total, '(not a new joint full run)')


if __name__ == '__main__':
    main()
