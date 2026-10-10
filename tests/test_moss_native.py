"""Causal schedule/gate/FIFO regressions; no model or judge API calls."""
from collections import deque
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest

from moss_native.clock import reconstruct, seconds, load_scorer
from moss_native.gate import QuestionGate
from moss_native.policy import ROOT, clean_sample, install_client, verify_upstream
from moss_native.server import StartupGate, groups
from moss_native.score import evaluate


class Trace:
    def record(self, row):
        pass


class Tokenizer:
    def convert_tokens_to_ids(self, value):
        table = {'<|silence|>': 151671, '<|...|>': 20, '�': 30}
        return [table[v] for v in value] if isinstance(value, list) else table[value]

    def decode(self, ids, **kwargs):
        return ''.join({10: 'A', 11: 'B', 151671: '<|silence|>', 151672: '<|response|>'}[i] for i in ids)


class State:
    def __init__(self, sample, chunk_seconds=1, **kwargs):
        self.events_queue = deque(sample['sqa'])
        self.duration = sample['video_info']['duration']
        self.sample_id = str(sample['id'])
        self.current_time = 0
        self.chunk_seconds = chunk_seconds

    def step(self):
        self.current_time += 1
        return {'chunk_start_seconds': self.current_time - 1, 'stream_chunk': 'native',
                'is_finished': self.current_time >= int(self.duration),
                'new_events': self._collect_current_events()}


class NativeProtocolTests(unittest.TestCase):
    def sample(self, duration=4.2):
        return {'id': 1, 'uuid': 'v', 'video': 'original', 'stream_addr': 'cache',
                'video_info': {'duration': duration}, 'timestamp_focus': 'GT',
                'sqa': [{'timestamp': '00:00', 'type': 'forward', 'question': 'Q0', 'response': 'GT'},
                        {'timestamp': '00:04', 'type': 'instant', 'question': 'Q4', 'response': 'GT'}]}

    def calendar(self, sample):
        module = types.SimpleNamespace(StreamingState=State, UnifiedInferenceRunner=type('Runner', (), {}),
                                       time_to_seconds=seconds)
        rows = []
        install_client(module, rows.append)
        state = module.StreamingState(sample)
        steps = []
        while True:
            result = state.step()
            steps.append(result)
            if result['is_finished']:
                break
        return steps, rows

    def test_checked_upstream_and_exact_realtime_prompt(self):
        self.assertEqual(verify_upstream(), ROOT)

    def test_questions_before_causal_chunks_and_partial_tail(self):
        steps, _ = self.calendar(self.sample())
        self.assertEqual([s['chunk_start_seconds'] for s in steps], [0, 1, 2, 3, 4])
        self.assertEqual([[q['question'] for q in s['new_events']] for s in steps], [['Q0'], [], [], [], ['Q4']])

    def test_integer_end_does_not_add_empty_tick(self):
        sample = self.sample(5)
        self.assertEqual(len(self.calendar(sample)[0]), 5)

    def test_gold_annotations_cannot_change_model_schedule(self):
        sample = self.sample()
        changed = copy.deepcopy(sample)
        changed['timestamp_focus'] = '99:00'
        for e in changed['sqa']:
            e.update(response='different', type='backward', timestamp_focus='99:00')
        changed['sqa'].append({'timestamp': '99:00', 'response': 'extra answer'})
        self.assertEqual(clean_sample(sample), clean_sample(changed))
        self.assertEqual(self.calendar(sample), self.calendar(changed))

    def test_scope_preserves_original_forward_answer_event(self):
        module = load_scorer(ROOT / 'run_stream_eval.py')
        raw = [{'time_type': 'forward', 'timestamp_question': '00:01',
                'timestamp_proactive': '00:04', 'user_query': 'Q', 'response': 'GT'},
               {'time_type': 'instant', 'timestamp_question': '00:02',
                'user_query': 'Q2', 'response': 'GT2'}]
        scoped = [q for q in raw if q['time_type'] == 'forward']
        events = module.convert_verified_responses_to_sqa(scoped)
        question = module.parse_logical_questions(events)[0]
        self.assertEqual(question['ground_truth'], 'GT')
        self.assertEqual(question['answer_event_time_sec'], 4)

    def test_question_has_zero_images_and_all_visual_inputs_one(self):
        frames = [{'timestamp': 11}, {'timestamp': 11.5}]
        self.assertEqual(groups(frames, ['Q']), [([], ['Q'], 'question'),
                          ([frames[0]], [], 'visual'), ([frames[1]], [], 'visual')])

    def gate(self, question=False):
        gate = StartupGate(Trace(), 151671, Tokenizer())
        gate.consumed({'batch_id': 1, 'requires_gate': True, 'question_batch': question})
        return gate

    def test_256_text_tokens_cannot_release_visual(self):
        gate = self.gate()
        for _ in range(256):
            gate.observe(10)
        with self.assertRaises(TimeoutError):
            gate.wait(1, timeout=0)
        gate.observe(151671)
        self.assertEqual(gate.wait(1, timeout=0), 'silence')

    def test_startup_silence_cannot_release_first_visual(self):
        gate = StartupGate(Trace(), 151671, Tokenizer())
        gate.observe(151671)
        self.assertTrue(gate.startup_silence)
        gate.consumed({'batch_id': 1, 'requires_gate': True, 'question_batch': False})
        with self.assertRaises(TimeoutError):
            gate.wait(1, timeout=0)

    def test_emit_start_control_cannot_release_question(self):
        gate = self.gate(True)
        gate.observe(151672)
        self.assertIsNone(gate.states[1]['release'])
        gate.observe(10)
        self.assertEqual(gate.wait(1, timeout=0), 'answer')

    def fifo_fixture(self):
        q = {'batch_id': 1, 'sample_id': '1', 'request_id': 1, 'question_batch': True,
             'frames': 0, 'prompt_texts': ['Q'], 'role': 'question'}
        v = {'batch_id': 2, 'sample_id': '1', 'request_id': 1, 'question_batch': False,
             'frames': 1, 'timestamps': [11.5], 'role': 'visual'}
        rows = [dict(q, type='enqueue'), dict(q, type='prefill', actual_images=0, input_ids=[1]),
                dict(q, type='consumed', token_index=0),
                {'type': 'generated', 'batch_id': 1, 'token_index': 1, 'token_id': 10, 'decoded_output_chunk': 'A'},
                {'type': 'gate', 'batch_id': 1, 'source': 'answer', 'timeout_seconds': 30},
                {'type': 'collection', 'batch_id': 1, 'chunks': ['<|round_start|>', 'A']},
                {'type': 'generated', 'batch_id': 1, 'token_index': 2, 'token_id': 11, 'decoded_output_chunk': 'B'},
                dict(v, type='enqueue'), dict(v, type='prefill', actual_images=1, input_ids=[151655],
                    actual_text='<|time_start|>11.5 seconds<|time_end|>'),
                dict(v, type='consumed', token_index=2),
                {'type': 'generated', 'batch_id': 2, 'token_index': 3, 'token_id': 151671, 'decoded_output_chunk': '<|silence|>'},
                {'type': 'gate', 'batch_id': 2, 'source': 'silence', 'timeout_seconds': 60},
                {'type': 'collection', 'batch_id': 2, 'chunks': ['B', '<|silence|>']},
                {'type': 'request', 'sample_id': '1', 'request_id': 1, 'response': 'AB'}]
        raw = [{'id': 1, 'responses': [{'timestamp': '00:12', 'response': 'AB', 'status_code': 200}]}]
        return rows, raw

    def replay(self, rows, raw):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / 'trace.jsonl'
            trace.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            return reconstruct(trace, raw, {'1': {'Q': 11}})

    def test_late_question_chunks_keep_origin_and_half_seconds(self):
        slots = self.replay(*self.fifo_fixture())
        self.assertEqual([(s['timestamp'], s['response']) for s in slots],
                         [('00:11', 'AB'), ('00:11.5', '<|silence|>')])

    def test_missing_or_reordered_FIFO_rejected(self):
        rows, raw = self.fifo_fixture()
        rows[-2]['chunks'].reverse()
        with self.assertRaises(AssertionError):
            self.replay(rows, raw)

    def test_original_strict_timing_logic_retained(self):
        module = load_scorer(ROOT / 'run_stream_eval.py')
        import yaml
        module.init_runtime_constants(yaml.safe_load((ROOT / 'config.yaml').read_text()))
        module.time_to_seconds = seconds
        class Judge:
            def judge(self, *args):
                return {'score': 4, 'explanation': 'fixed fixture'}
        for task in ('instant', 'backward'):
            sample = {'id': 1, 'sqa': [{'timestamp': '00:11', 'type': task, 'question': 'Q', 'response': 'Ref'}]}
            for time, score in [('00:11', 80), ('00:11.5', 0)]:
                sample['responses'] = [{'timestamp': time, 'response': 'An answer'}]
                self.assertEqual(evaluate(module, [sample], Judge())[0]['score'], score)


if __name__ == '__main__':
    unittest.main()
