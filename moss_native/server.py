"""Every visual input has one image and requires a fresh generated silence.

Pure questions precede their causal chunk and use the historical question gate.
All FIFO chunks stay in the same native HTTP bucket; no Q-frame augmentation,
forced silence, length release, deadline advance, or text selection.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from .policy import verify_upstream, prompt_text, describe
from .transport import Trace
from .gate import QuestionGate, collect_snapshot_outputs
from .processor import InputProcessor, collect_silence_outputs
from .check_model import check


def groups(frames, prompts):
    return [([], [p], 'question') for p in prompts] + [([f], [], 'visual') for f in frames]


class StartupGate(QuestionGate):
    def __init__(self, *args):
        super().__init__(*args)
        self.startup_silence = False

    def observe(self, token_id):
        with self.cv:
            if self.current is None and token_id == self.silence_id:
                self.startup_silence = True
                self.trace.record({'type': 'startup_generated_silence', 'token_id': token_id})
            super().observe(token_id)
            self.cv.notify_all()


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--trace-dir', type=Path, required=True)
    p.add_argument('--model-path', type=Path, required=True)
    args, remaining = p.parse_known_args()
    root = verify_upstream()
    model_check = check(args.model_path)
    args.upstream_server = root / 'hf_openai_server.py'
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    args.trace_file = args.trace_dir / 'prefill.jsonl'
    if (args.trace_dir / 'handshake.jsonl').exists():
        raise FileExistsError('Use a fresh trace directory; preserve earlier attempts')
    (args.trace_dir / 'model_check.json').write_text(json.dumps(model_check, indent=2) + '\n')
    spec = importlib.util.spec_from_file_location('forward_upstream_server', args.upstream_server)
    upstream = importlib.util.module_from_spec(spec); sys.modules[spec.name] = upstream; spec.loader.exec_module(upstream)
    original = upstream.MossVLEngine
    trace = Trace(args.trace_dir / 'handshake.jsonl')

    class Engine(original):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            self.processor = InputProcessor(self.processor, trace, args.trace_file)
            self.gate = StartupGate(trace, 151671, self.processor.tokenizer)
            assert self.processor.tokenizer.convert_tokens_to_ids('<|silence|>') == 151671
            self.gate.install(self.model, self.processor)
            self.request_id = 0; self.failed = None

        def _generate_realtime(self, item, max_new_tokens, temperature):
            from PIL import Image
            if item['system_prompt'] != prompt_text():
                raise ValueError('This server requires the exact packaged realtime system prompt')
            if temperature != 0 or max_new_tokens != 1024:
                raise ValueError('Expected native deterministic generation with max_new_tokens=1024')
            sid = str(item['sample_id'])
            with self._lock:
                if self.failed: raise RuntimeError(self.failed)
                try:
                    if item.get('reset') or self._session is None or sid != self._session_sample_id:
                        with self.gate.cv:
                            assert not self.gate.pending
                            self.gate.current = None; self.gate.decoder.buffer.clear()
                            self.gate.startup_silence = False
                        self.processor.batch = None
                        self._reset_session(item['system_prompt'], max_new_tokens, temperature)
                        self._session_sample_id = sid
                        with self.gate.cv:
                            if not self.gate.cv.wait_for(lambda: self.gate.startup_silence, 60):
                                raise TimeoutError('No generated startup silence')
                        startup_chunks = collect_silence_outputs(self._session)
                        trace.record({'type': 'startup', 'sample_id': sid, 'chunks': startup_chunks,
                                      'system_prompt': item['system_prompt']})
                    self.request_id += 1; rid = self.request_id; parts = []
                    for index, (frames, prompts, role) in enumerate(groups(item.get('frames', []), item.get('prompts', []))):
                        decoded = []
                        for frame in frames:
                            with Image.open(frame['image']) as image:
                                decoded.append((image.convert('RGB'), float(frame['timestamp'])))
                        batch = self.gate.enqueue(self._session, decoded, prompts,
                            {'sample_id': sid, 'request_id': rid, 'active_index': index, 'role': role,
                             'question_batch': bool(prompts), 'requires_gate': True,
                             'input_pixels': [{'timestamp': f['timestamp'], 'image': f['image'],
                                               'image_sha256': hashlib.sha256(Path(f['image']).read_bytes()).hexdigest()} for f in frames],
                             'prompt_texts': prompts})
                        release = self.gate.wait(batch, timeout=30 if prompts else 60, allow_deadline=False)
                        assert release in ('answer', 'silence') and (prompts or release == 'silence')
                        chunks = collect_snapshot_outputs(self._session) if release == 'answer' else collect_silence_outputs(self._session)
                        parts.extend(chunks)
                        trace.record({'type': 'collection', 'batch_id': batch, 'source': release, 'chunks': chunks})
                    result = upstream.prepare_native_moss_output(''.join(parts))
                    trace.record({'type': 'request', 'sample_id': sid, 'request_id': rid,
                                  'frames': item.get('frames', []), 'prompts': item.get('prompts', []), 'response': result})
                    trace.flush()
                    return result
                except Exception as error:
                    self.failed = str(error); trace.flush(); raise

    upstream.MossVLEngine = Engine
    original_GET = upstream.AppHandler.do_GET
    def protocol_GET(handler):
        if handler.path == '/moss-native/protocol':
            from http import HTTPStatus
            upstream.write_json(handler, HTTPStatus.OK, describe())
        else:
            original_GET(handler)
    upstream.AppHandler.do_GET = protocol_GET
    sys.argv = [str(args.upstream_server), '--model-path', str(args.model_path)] + remaining
    upstream.main()


if __name__ == '__main__':
    main()
