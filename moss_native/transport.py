"""Audited native queue publication and consumption tracing; no length gate."""
import json
import queue
import threading
import time
import types

class Trace:

    def __init__(self, path):
        self.stream = path.open('a', buffering=1 << 20)
        self.lock = threading.Lock()

    def record(self, row):
        with self.lock:
            self.stream.write(json.dumps(row, ensure_ascii=False) + '\n')

    def flush(self):
        with self.lock:
            self.stream.flush()

class FrameGate:

    def __init__(self, trace, silence_id):
        self.trace, self.silence_id = (trace, silence_id)
        self.cv = threading.Condition(threading.RLock())
        self.pending = []
        self.next_batch = 0
        self.current = None
        self.states = {}
        self.generated_index = 0

    def enqueue(self, session, decoded, prompts, meta):
        with self.cv:
            assert not self.pending and session.pending_frames == 0 and session._prompt_queue.empty()
            self.next_batch += 1
            batch_id = self.next_batch
            event = dict(meta, batch_id=batch_id, frames=len(decoded), prompts=len(prompts), timestamps=[t for _, t in decoded])
            for image, timestamp in decoded:
                session.push_frame(image, timestamp=timestamp, drop_oldest=False)
            for prompt in prompts:
                session.push_prompt(str(prompt))
            self.pending.append(event)
            self.trace.record({'type': 'enqueue', **event})
            return batch_id

    def snapshot(self, model, kwargs):
        while True:
            with self.cv:
                frames, prompts = ([], [])
                for handle, target in [(kwargs['new_video_frames'], frames), (kwargs['new_prompts'], prompts)]:
                    while True:
                        try:
                            target.append(handle.get_nowait())
                        except queue.Empty:
                            break
                if frames or prompts:
                    assert len(self.pending) == 1, 'Never merge active frame inputs'
                    event = self.pending.pop()
                    assert len(frames) == event['frames'] and len(prompts) == event['prompts']
                    self.trace.record({'type': 'drain', **event})
                    return (frames, prompts, event)
                if not model.continue_generating or not kwargs['should_wait_for_new_input']:
                    state = self.states.get(self.current, {})
                    if not model.continue_generating or not state.get('requires_gate') or (not state.get('release')):
                        return ([], [], None)
            time.sleep(0.02)

    def consumed(self, event):
        with self.cv:
            self.current = event['batch_id']
            self.states[self.current] = {'generated': 0, 'silence_seen': False, 'release': None, 'requires_gate': event.get('requires_gate', False)}
            self.trace.record({'type': 'consumed', **event, 'token_index': self.generated_index})
            self.cv.notify_all()

    def install(self, model, processor):
        original = model._update_model_kwargs_for_real_time_generation

        def update(bound_model, *args, **kwargs):
            assert not args
            self.observe(int(kwargs['input_ids'][0, -1].item()))
            frames, prompts, event = self.snapshot(bound_model, kwargs)
            processor.batch = event
            local_frames, local_prompts = (queue.Queue(), queue.Queue())
            for frame in frames:
                local_frames.put_nowait(frame)
            for prompt in prompts:
                local_prompts.put_nowait(prompt)
            result = original(**dict(kwargs, new_video_frames=local_frames, new_prompts=local_prompts))
            if event:
                self.consumed(event)
            return result
        model._update_model_kwargs_for_real_time_generation = types.MethodType(update, model)
