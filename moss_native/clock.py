"""Lossless native producer/consumer FIFO reconstruction, including late prompt text.

Anchors are model-visible input times, not wall-clock latency. No answer window
is consulted when assigning text to its actual generating input slot.
"""
import ast
from collections import defaultdict, deque
import importlib.util
import json
from pathlib import Path
import re
import sys
from .policy import ROOT


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def original_cleaner():
    path = ROOT / 'hf_openai_server.py'
    tree = ast.parse(path.read_text())
    names = {'NATIVE_MOSS_CONTROL_TOKENS', 'NATIVE_MOSS_SILENCE'}
    nodes = [n for n in tree.body if
             (isinstance(n, ast.FunctionDef) and n.name == 'prepare_native_moss_output') or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['prepare_native_moss_output']


def seconds(value):
    return sum(float(v) * 60 ** i for i, v in enumerate(reversed(value.split(':'))))


def label(value):
    minutes, sec = divmod(round(float(value), 6), 60)
    hours, minutes = divmod(int(minutes), 60)
    tail = f'{sec:04.1f}' if sec % 1 else f'{int(sec):02d}'
    return f'{hours:02d}:{minutes:02d}:{tail}' if hours else f'{minutes:02d}:{tail}'

def load_scorer(path):
    spec = importlib.util.spec_from_file_location('frame_clock_original_scorer', path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module

def exact_selected_responses(up, sample):
    responses = sorted(sample['responses'], key=lambda r: seconds(r['timestamp']))
    used = set(); selected = []
    for q in up.parse_logical_questions(sample['sqa']):
        found = None
        for index, response in enumerate(responses):
            t = seconds(response['timestamp'])
            if index in used or not q['question_time_sec'] <= t <= q['answer_event_time_sec'] + 4: continue
            text = up.prepare_native_moss_scored_text(response['response'])
            if text != q['ground_truth'] and up.is_placeholder(text, up.NATIVE_MOSS_SCORE_PLACEHOLDERS): continue
            if t == q['question_time_sec'] and t < q['answer_event_time_sec'] - 2: continue
            found = {'timestamp': response['timestamp'], 'seconds': t, 'text': text, 'index': index}
            used.add(index); break
        selected.append(found)
    return selected

def reconstruct(trace_file, raw, questions):
    clean = original_cleaner()
    rows = read_jsonl(trace_file)
    events, slots, order, consumed, prefills = {}, {}, [], set(), {}
    producer = deque(); requests = []; collected = set(); last_index = None
    for row in rows:
        kind, bid = row['type'], row.get('batch_id')
        if kind == 'gate':
            assert row['source'] in ('silence', 'answer')
            assert row['timeout_seconds'] == (30 if events[bid]['question_batch'] else 60)
            if not events[bid]['question_batch']: assert row['source'] == 'silence'
        if kind == 'startup':
            assert not producer and row['chunks'] == ['<|silence|>']
            last_index = None
        elif kind == 'enqueue':
            assert bid not in events and (not order or order[-1] in collected)
            events[bid] = row; order.append(bid)
        elif kind == 'prefill':
            assert bid in events and bid not in prefills
            assert 151671 not in row['input_ids']
            prefills[bid] = row
        elif kind == 'consumed':
            assert bid in prefills and bid not in consumed
            if last_index is not None:
                assert row['token_index'] == last_index
            last_index = row['token_index']; consumed.add(bid)
            e, p = events[bid], prefills[bid]
            if e['question_batch']:
                assert e['frames'] == p['actual_images'] == 0 and len(e['prompt_texts']) == 1
                anchor = questions[str(e['sample_id'])][e['prompt_texts'][0]]
                producer.append((bid, '<|round_start|>'))
            else:
                assert e['frames'] == p['actual_images'] == 1
                stamps = re.findall(r'<\|time_start\|>([0-9.]+) seconds<\|time_end\|>', p['actual_text'])
                assert stamps == [f"{e['timestamps'][0]:.1f}"]
                anchor = float(stamps[0])
            slots[bid] = {'sample_id': str(e['sample_id']), 'request_id': e['request_id'],
                          'batch_id': bid, 'role': e['role'], 'anchor_seconds': anchor,
                          'timestamp': label(anchor), 'generated_chunks': []}
        elif kind == 'generated':
            assert bid in consumed and row['token_index'] == last_index + 1
            last_index = row['token_index']
            if row['decoded_output_chunk']:
                producer.append((bid, row['decoded_output_chunk']))
        elif kind == 'collection':
            assert bid in consumed and bid not in collected
            for chunk in row['chunks']:
                assert producer, 'Collected output absent from actual native generated trace'
                origin, expected = producer.popleft()
                assert chunk == expected, 'Missing, reordered or changed FIFO chunk'
                slots[origin]['generated_chunks'].append(chunk)
            collected.add(bid)
        elif kind == 'request':
            requests.append(row)
    assert not producer, 'Uncollected output cannot be silently discarded'
    assert set(events) == set(slots) == set(prefills) == consumed == collected
    for bid in order:
        slots[bid]['response'] = clean(''.join(slots[bid]['generated_chunks']))
    actual = [(str(s['id']), r) for s in raw for r in s['responses']]
    assert len(actual) == len(requests)
    by_request = defaultdict(list)
    for row in rows:
        if row['type'] == 'collection':
            by_request[events[row['batch_id']]['request_id']].extend(row['chunks'])
    for request, (sid, response) in zip(requests, actual):
        assert str(request['sample_id']) == sid and response['status_code'] == 200
        assert clean(''.join(by_request[request['request_id']])) == request['response'] == response['response']
    return [slots[bid] for bid in order]

def mapped_predictions(raw, slots):
    by_sample = defaultdict(list)
    for slot in slots:
        by_sample[slot['sample_id']].append({'timestamp': slot['timestamp'], 'response': slot['response'],
                                           'status_code': 200, 'source_batch_id': slot['batch_id']})
    assert set(by_sample) == {str(s['id']) for s in raw}
    return [dict(sample, responses=by_sample[str(sample['id'])]) for sample in raw]
