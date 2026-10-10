"""One independent native model server per shard, using the author's sampler/runner."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen

from .policy import ROOT, describe, install_client, prompt_text, sha, verify_upstream
from .clock import read_jsonl, reconstruct, mapped_predictions, write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT / 'dataset/proactive_perception_annotations.json')
    p.add_argument('--video-root', type=Path, default=ROOT)
    p.add_argument('--model-path', type=Path, required=True, help='Same local MOSS-VL-Realtime snapshot as the server; tokenizer audit only')
    p.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    p.add_argument('--api-base', default='http://127.0.0.1:8000/v1')
    p.add_argument('--trace-dir', type=Path, required=True, help='Same directory as the paired native server')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scope', choices=['all', 'forward', 'instant_backward'], default='all')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--max-samples', type=int, default=0, help='Smoke only; labels result as a subset')
    args = p.parse_args()
    verify_upstream()
    if args.max_samples < 0 or not 0 <= args.shard_index < args.num_shards:
        p.error('Invalid shard/sample bounds')
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    receipt = {'ok': False, 'protocol': describe()}
    try:
        endpoint = args.api_base.removesuffix('/v1').rstrip('/') + '/moss-native/protocol'
        with urlopen(endpoint, timeout=10) as response:
            if json.load(response) != describe():
                raise ValueError('Endpoint is not the matching native framewise server')
        sys.path.insert(0, str(ROOT))
        import run_stream_eval as upstream
        config, _ = upstream.load_release_config(args.config)
        upstream.init_runtime_constants(config)
        config['prompts']['streaming']['system_prompt'] = prompt_text()
        config['prompts']['streaming']['silent_word'] = '<|silence|>'
        cfg = config['models']['moss_vl_realtime']
        cfg.update(api_base=args.api_base, temperature=0, max_new_tokens=1024, max_retries=0)
        # Duration must be measured, never inferred from an annotated answer event.
        allowed = {'forward', 'instant', 'backward'} if args.scope == 'all' else (
            {'forward'} if args.scope == 'forward' else {'instant', 'backward'})
        dataset_rows = read_dataset(args.dataset)
        selected = []
        for index, original_row in enumerate(dataset_rows):
            row = dict(original_row)
            if args.scope != 'all':
                if 'verified_responses' not in row or 'sqa' in row:
                    raise ValueError('Subset selection requires the released verified_responses schema')
                row['verified_responses'] = [q for q in row['verified_responses']
                                             if str(q['time_type']).lower() in allowed]
                if not row['verified_responses']:
                    continue
            sample = upstream.normalize_sample(row, index, 'phostream', args.dataset,
                args.video_root, args.output / 'cache', need_video_info=False)
            if not any(e.get('question') for e in sample['sqa']):
                continue
            # Use supplied real video_info when available; otherwise probe the media once.
            original_info = original_row.get('video_info')
            info = original_info or upstream.probe_video(sample['video'])
            if not info or float(info.get('duration', 0)) <= 0:
                raise ValueError(f'Cannot determine real video duration for {sample["id"]}')
            sample['video_info'] = info
            sample['stream_addr'] = str(args.output / 'cache' / sample['uuid'])
            selected.append(sample)
        selected = selected[args.shard_index::args.num_shards]
        if args.max_samples:
            selected = selected[:args.max_samples]
        if not selected:
            raise ValueError('No samples in selected scope/shard')
        with (args.output / 'client_trace.jsonl').open('x', buffering=1) as stream:
            install_client(upstream, lambda row: stream.write(json.dumps(row, ensure_ascii=False) + '\n'))
            options = upstream.InferenceOptions(
                sparse_mode=False, active_window=2, max_retries=0, chunk_seconds=1.0,
                model_video_fps=2.0, trim_fps=2.0, force_focus=False, focus_window_seconds=0,
                high_fps=None, full_video_high_fps=False, max_context_frames=120,
                dialog_dump_root=args.output / 'dialog', chunk_cache_root=args.output / 'cache',
                native_moss=True)
            runner = upstream.UnifiedInferenceRunner(config, args.config.parent, 'moss_vl_realtime',
                'streaming', '', options, None, args.video_root, args.output / 'cache')
            raw = []
            with (args.output / 'original_HTTP.jsonl').open('x', buffering=1) as rawfile:
                for sample in selected:
                    result = runner.run_sample(sample, 'phostream')
                    rawfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    raw.append(result)
                    if any(r['status_code'] != 200 for r in result['responses']):
                        raise RuntimeError('Native execution failure; preserve traces and do not score as zero')
        questions = {str(s['id']): {upstream.render_user_prompt(e).strip(): upstream.time_to_seconds(e['timestamp'])
                                  for e in s['sqa'] if e.get('question')} for s in raw}
        slots = reconstruct(args.trace_dir / 'handshake.jsonl', raw, questions)
        from transformers import AutoTokenizer
        from .check_model import check
        from .audit import audit
        check(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        actual_audit = audit(raw, args.output / 'client_trace.jsonl', args.trace_dir / 'handshake.jsonl', tokenizer)
        (args.output / 'audit.json').write_text(json.dumps(actual_audit, indent=2) + '\n')
        write_jsonl(args.output / 'slots.jsonl', slots)
        write_jsonl(args.output / 'predictions.jsonl', mapped_predictions(raw, slots))
        tasks = Counter(q['task_type'] for s in raw for q in upstream.parse_logical_questions(s['sqa']))
        receipt.update(ok=True, videos=len(raw), qa=sum(tasks.values()), task_counts=dict(tasks),
                       shard_index=args.shard_index, num_shards=args.num_shards, scope=args.scope,
                       max_samples=args.max_samples, api_failures=0,
                       dataset_sha256=sha(args.dataset), config_sha256=sha(args.config),
                       raw_sha256=sha(args.output / 'original_HTTP.jsonl'),
                       predictions_sha256=sha(args.output / 'predictions.jsonl'),
                       trace_sha256=sha(args.trace_dir / 'handshake.jsonl'),
                       client_sha256=sha(args.output / 'client_trace.jsonl'),
                       slots=len(slots), policy_context='All question types included together by default')
    except Exception as error:
        import traceback
        traceback.print_exc()
        receipt['error'] = repr(error)
    receipt['elapsed_seconds'] = time.time() - began
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return 0 if receipt['ok'] else 1


def read_dataset(path):
    if path.suffix == '.jsonl':
        return read_jsonl(path)
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, list) else raw['samples']


if __name__ == '__main__':
    raise SystemExit(main())
