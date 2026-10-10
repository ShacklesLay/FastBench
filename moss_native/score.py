"""Original FastBench answer selection and judging, with fractional input-slot times."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import threading

from .clock import read_jsonl, write_jsonl, seconds, exact_selected_responses
from .policy import ROOT, CLOCK_ID, sha, verify_upstream


def evaluate(upstream, predictions, judge):
    rows = []
    for sample in predictions:
        result = upstream.evaluate_sample(sample, judge, 4, native_moss=True)
        choices = exact_selected_responses(upstream, sample)
        assert len(result) == len(choices)
        for row, choice in zip(result, choices):
            row.update(clock_id=CLOCK_ID, metric='upstream_strict_fractional_input_event_clock',
                       response_anchor_seconds=choice['seconds'] if choice else None,
                       response_anchor_time=choice['timestamp'] if choice else None)
        rows.extend(result)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-scorers', type=int, default=16)
    p.add_argument('--require-full', action='store_true', help='Require exactly 300 videos/306 QA')
    args = p.parse_args()
    verify_upstream()
    if args.num_scorers < 1:
        p.error('Need at least one scorer')
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {'ok': False}
    try:
        source = json.loads((args.run / 'receipt.json').read_text())
        assert source['ok'] and source['api_failures'] == 0
        assert sha(args.run / 'predictions.jsonl') == source['predictions_sha256']
        predictions = read_jsonl(args.run / 'predictions.jsonl')
        sys.path.insert(0, str(ROOT))
        import run_stream_eval as upstream
        config, _ = upstream.load_release_config(args.config)
        upstream.init_runtime_constants(config)
        upstream.time_to_seconds = seconds  # ONLY parser changes; selection/windows/GT are unchanged.
        counts = Counter(q['task_type'] for s in predictions for q in upstream.parse_logical_questions(s['sqa']))
        if args.require_full:
            assert len(predictions) == 300 and counts == {'forward': 168, 'instant': 73, 'backward': 65}
        assert config['judger']['temperature'] == 0 and config['judger']['max_new_tokens'] == 256
        assert config['judger']['model'].lower() in {
            'qwen/qwen3-235b-a22b-2507', 'qwen/qwen3-235b-a22b-instruct-2507'
        }, 'Explicitly configure the original Qwen3-235B judge; do not use the YAML fallback model/server'
        original_config, _ = upstream.load_release_config(ROOT / 'config.yaml')
        assert config['judger']['prompt_template'] == original_config['judger']['prompt_template']
        needed = set()
        class SelectOnly:
            def judge(self, question, text, gt):
                needed.add((question, text, gt))
                return {'score': 0, 'explanation': 'Selection pass only, never a reported score'}
        evaluate(upstream, predictions, SelectOnly())
        judge = upstream.load_llm_judger(config)
        assert judge is not None, 'A real original LLMJudger backend is required'
        original = judge.backend.generate
        lock = threading.Lock()
        with (args.output / 'judge_calls.jsonl').open('x', buffering=1) as log:
            def logged(messages, max_new_tokens=256, **kwargs):
                result = original(messages, max_new_tokens=max_new_tokens, **kwargs)
                with lock:
                    log.write(json.dumps({'request': {'messages': messages, 'max_new_tokens': max_new_tokens,
                        'model': config['judger']['model'], 'temperature': 0}, 'result': result},
                        ensure_ascii=False) + '\n')
                return result
            judge.backend.generate = logged
            judgments = {}
            with ThreadPoolExecutor(max_workers=args.num_scorers) as pool:
                pending = {pool.submit(judge.judge, *key): key for key in sorted(needed)}
                for future in as_completed(pending):
                    value = future.result()
                    assert value['explanation'] and 'judger parse error' not in value['explanation'].lower(), value
                    judgments[pending[future]] = value
        write_jsonl(args.output / 'judgments.jsonl', [dict(question=k[0], response=k[1], answer=k[2], **v)
                                                     for k, v in sorted(judgments.items())])
        class Fixed:
            def judge(self, question, text, gt):
                return judgments[(question, text, gt)]
        rows = evaluate(upstream, predictions, Fixed())
        assert len(rows) == sum(counts.values())
        assert len({(r['sample_id'], r['question_time'], r['question']) for r in rows}) == len(rows)
        write_jsonl(args.output / 'result_details.jsonl', rows)
        receipt.update(ok=True, videos=len(predictions), qa=len(rows), task_counts=dict(counts),
                       score_100={task: sum(r['score'] for r in rows if r['task_type'] == task) / n
                                  for task, n in counts.items()},
                       mean=sum(r['score'] for r in rows) / len(rows),
                       mean_label='overall' if args.require_full else 'evaluated_subset_mean',
                       judge_failures=0, unique_judged_inputs=len(judgments),
                       categories=dict(Counter(r['category'] for r in rows)), clock_id=CLOCK_ID,
                       predictions_sha256=source['predictions_sha256'],
                       details_sha256=sha(args.output / 'result_details.jsonl'),
                       judge={'backend': config['judger']['backend'], 'model': config['judger']['model'],
                              'temperature': 0, 'max_new_tokens': 256, 'scorers': args.num_scorers},
                       config_sha256=sha(args.config), upstream_scorer_sha256=sha(ROOT / 'run_stream_eval.py'))
        print(json.dumps(receipt, indent=2))
    except Exception as error:
        import traceback
        traceback.print_exc()
        receipt['error'] = repr(error)
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return 0 if receipt['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
