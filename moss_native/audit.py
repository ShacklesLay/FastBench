"""Actual causal calendars, single-frame inputs, fresh gates and native FIFO audit."""
from collections import defaultdict
import hashlib
import math
from pathlib import Path

from .clock import read_jsonl, seconds
from .gate import OutputDecoder, SPECIAL


def audit(raw, client_trace, server_trace, tokenizer):
    samples = {str(s['id']): s for s in raw}
    client = read_jsonl(client_trace)
    calendars = defaultdict(list)
    states = {}
    for row in client:
        sid = row['sample_id']
        if row['type'] == 'state':
            assert sid not in states
            states[sid] = row
            assert row['input_keys'] == sorted(['id', 'uuid', 'video', 'stream_addr', 'video_info', 'sqa'])
            assert all(set(q) <= {'timestamp', 'question', 'options'} for q in row['questions'])
        else:
            assert row['type'] == 'step'
            calendars[sid].append(row)
    assert set(states) == set(samples) == set(calendars)
    for sid, rows in calendars.items():
        duration = float(samples[sid]['video_info']['duration'])
        assert [r['time'] for r in rows] == list(range(1, math.ceil(duration) + 1))
        assert all(r['has_chunk'] and r['chunk_start'] == r['time'] - 1 for r in rows)
        assert [r['time'] for r in rows if r['finished']] == [math.ceil(duration)]
        actual_questions = [q for r in rows for q in r['questions']]
        assert actual_questions == states[sid]['questions']
        for row in rows:
            assert all(seconds(q['timestamp']) == row['chunk_start'] for q in row['questions'])
        assert len(samples[sid]['responses']) == len(rows)
    events, prefills, consumed, generated, gates = {}, {}, set(), defaultdict(list), {}
    decoder = OutputDecoder(tokenizer)
    trace = read_jsonl(server_trace)
    for row in trace:
        bid = row.get('batch_id')
        if row['type'] == 'startup':
            decoder.buffer.clear()
        elif row['type'] == 'enqueue':
            events[bid] = row
            assert row['frames'] == (0 if row['question_batch'] else 1)
            assert row['prompts'] == (1 if row['question_batch'] else 0)
            for pixel in row['input_pixels']:
                assert hashlib.sha256(Path(pixel['image']).read_bytes()).hexdigest() == pixel['image_sha256']
        elif row['type'] == 'prefill':
            prefills[bid] = row
            assert 151671 not in row['input_ids']
            assert row['input_ids'].count(151655) == row['actual_images'] == events[bid]['frames']
        elif row['type'] == 'consumed':
            assert bid in prefills and bid not in consumed
            consumed.add(bid)
            if row['question_batch']:
                decoder.buffer.clear()
        elif row['type'] == 'generated':
            assert bid in consumed
            assert decoder.feed(row['token_id']) == row['decoded_output_chunk']
            generated[bid].append(row)
        elif row['type'] == 'gate':
            assert bid in consumed and row['source'] in ('answer', 'silence')
            question = events[bid]['question_batch']
            assert row['timeout_seconds'] == (30 if question else 60)
            release = next(t for t in generated[bid] if t['token_id'] == 151671 or
                           (question and t['decoded_output_chunk'] and not SPECIAL.match(t['decoded_output_chunk'].strip())))
            assert release['token_index'] == row['release_token_index']
            assert row['source'] == ('silence' if release['token_id'] == 151671 else 'answer')
            gates[bid] = row
    assert set(events) == set(prefills) == consumed == set(gates)
    # Compare the full frame list, not only the last visible timestamp.
    expected_frames = {}
    for sid, sample in samples.items():
        cache = Path(sample['stream_addr']) / 'native_moss_frames'
        images = [row for file in sorted(cache.glob('*/frames.json')) for row in read_manifest(file)]
        images.sort(key=lambda x: (float(x['timestamp']), x['image']))
        expected_frames[sid] = images
    actual = defaultdict(list)
    for row in trace:
        if row['type'] == 'request':
            actual[str(row['sample_id'])].extend(row['frames'])
    for sid in samples:
        # Paths and decoder PTS are exact native sampler outputs.
        assert actual[sid] == expected_frames[sid], 'Missing, duplicated or reordered native frame'
    return {'ok': True, 'videos': len(raw), 'HTTP': sum(len(v) for v in calendars.values()),
            'visual_slots': sum(not e['question_batch'] for e in events.values()),
            'question_slots': sum(e['question_batch'] for e in events.values()),
            'fresh_silence_gates': sum(g['source'] == 'silence' for g in gates.values()),
            'answer_gates': sum(g['source'] == 'answer' for g in gates.values()),
            'timeouts': 0, 'forced_silence': 0}


def read_manifest(path):
    import json
    return json.loads(path.read_text())
