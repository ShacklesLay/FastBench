"""Portable protocol contract. Gold annotations never control input scheduling."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = 'moss-native-framewise-v1'
CLOCK_ID = 'native-slot-input-event-fractional-v1'
PROMPT_SHA = '3e438e9c992bd290d70801ae43ebb8144642394d300ce6a8de626503b01d94d9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_text():
    path = ROOT / 'moss_native/realtime_system_prompt.txt'
    if sha(path) != PROMPT_SHA:
        raise ValueError('Packaged realtime system prompt changed')
    return path.read_text()


def describe():
    return {
        'protocol_id': PROTOCOL_ID, 'clock_id': CLOCK_ID,
        'system_prompt_sha256': PROMPT_SHA, 'sampler_fps': 2,
        'session': 'video_start_through_real_end_including_partial_tail',
        'question': 'original_time_before_first_causal_native_chunk',
        'visual_gate': 'fresh_generated_silence_only', 'visual_timeout': 60,
        'question_gate': 'first_native_decoded_text_or_generated_silence',
        'question_timeout': 30, 'timeout_result': 'execution_failure',
        'extra_Q_frames': False, 'forced_silence': False,
        'length_gate': False, 'deadline_advance': False,
        'output': 'all_generated_native_FIFO_chunks',
        'scoring': 'upstream_strict_rules_with_fractional_timestamp_parser',
        'adapter_source_sha256': {p.name: sha(p) for p in sorted((ROOT / 'moss_native').glob('*.py'))},
    }


def verify_upstream():
    contract = json.loads((ROOT / 'moss_native/upstream.json').read_text())
    for name, expected in contract['files'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Upstream {name} differs from audited commit; review before using this adapter')
    prompt_text()
    return ROOT


def clean_sample(sample):
    # Preserve the unmodified complete sample separately for scoring.
    clean = {k: sample[k] for k in ('id', 'uuid', 'video', 'stream_addr') if k in sample}
    info = sample['video_info']
    clean['video_info'] = {k: info[k] for k in ('duration', 'video') if k in info}
    clean['sqa'] = [{k: e[k] for k in ('timestamp', 'question', 'options') if k in e}
                    for e in sample['sqa'] if e.get('question')]
    if not clean['sqa'] or float(info['duration']) <= 0:
        raise ValueError('Need questions and measured video duration')
    return clean


def install_client(upstream, record):
    original = upstream.StreamingState

    class CausalState(original):
        def __init__(self, sample, **kwargs):
            clean = clean_sample(sample)
            super().__init__(sample=clean, **kwargs)
            self.last_event_time = self.duration
            for e in clean['sqa']:
                q = upstream.time_to_seconds(e['timestamp'])
                if not (0 <= q < self.duration and float(q).is_integer()):
                    raise ValueError('This released dataset uses integer question times strictly before real video end')
            record({'type': 'state', 'sample_id': self.sample_id,
                    'duration': self.duration, 'input_keys': sorted(clean), 'questions': clean['sqa']})

        def _collect_current_events(self):
            left = self.current_time - self.chunk_seconds
            out = []
            while self.events_queue and upstream.time_to_seconds(self.events_queue[0]['timestamp']) <= left:
                out.append(self.events_queue.popleft())
            return out

        def step(self):
            result = super().step()
            result['is_finished'] = self.current_time >= self.duration
            record({'type': 'step', 'sample_id': self.sample_id, 'time': self.current_time,
                    'chunk_start': result['chunk_start_seconds'], 'finished': result['is_finished'],
                    'has_chunk': bool(result.get('stream_chunk')), 'questions': result['new_events']})
            if result['is_finished'] and self.events_queue:
                raise ValueError('Question omitted at real video tail')
            return result

    upstream.StreamingState = CausalState
    upstream.UnifiedInferenceRunner._native_benchmark_answer_window = lambda self, previous, events: 1
