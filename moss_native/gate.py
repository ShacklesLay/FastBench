"""Fresh generated silence for frames; first native text or silence for questions."""
import re
import time
from .transport import FrameGate
SPECIAL = re.compile(r"^<\|[^|]*\|>$")

class OutputDecoder:
    """Mirror the frozen native loop's multi-byte streaming decoder, not token names."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.buffer = []
        self.silence_id = tokenizer.convert_tokens_to_ids("<|silence|>")
        self.ellipsis_id = tokenizer.convert_tokens_to_ids(["<|...|>"])[0]
        self.invalid_id = tokenizer.convert_tokens_to_ids(["�"])[0]

    def feed(self, token_id):
        self.buffer.append(token_id)
        text = self.tokenizer.decode(self.buffer, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        complete = bool(text) and text[-1] != "�"
        invalid_complete = bool(text) and text[-1] == "�" and token_id == self.invalid_id
        if token_id in (self.silence_id, self.ellipsis_id) or complete or invalid_complete:
            self.buffer.clear()
            return text.replace("\ufffd", "")
        return ""

class QuestionGate(FrameGate):
    def __init__(self, trace, silence_id, tokenizer):
        super().__init__(trace, silence_id)
        self.decoder = OutputDecoder(tokenizer)

    def consumed(self, event):
        with self.cv:
            super().consumed(event)
            self.states[self.current]["question_batch"] = bool(event["question_batch"])
            if event["question_batch"]:
                self.decoder.buffer.clear()  # Frozen native update clears on new prompts.

    def observe(self, token_id):
        with self.cv:
            self.generated_index += 1
            text = self.decoder.feed(token_id)
            if self.current is None:
                return
            state = self.states[self.current]
            state["generated"] += 1
            source = None
            if token_id == self.silence_id:
                state["silence_seen"] = True
                source = "silence"
            elif state["question_batch"] and text and not SPECIAL.match(text.strip()):
                source = "answer"
            if source and state["release"] is None:
                state.update(release=source, release_generated=state["generated"],
                             release_token_index=self.generated_index)
            self.trace.record({"type": "generated", "batch_id": self.current,
                               "token_index": self.generated_index, "token_id": token_id,
                               "decoded_output_chunk": text})
            self.cv.notify_all()

    def wait(self, batch_id, timeout=30.0, allow_deadline=False):
        began = time.monotonic()
        with self.cv:
            ready = self.cv.wait_for(lambda: batch_id in self.states and
                                    self.states[batch_id]["release"] in ("silence", "answer"), timeout)
            state = self.states.get(batch_id, {})
            if allow_deadline:
                assert not state.get("question_batch"), "Question deadline is an execution failure"
            deadline = not ready and allow_deadline and batch_id in self.states
            if deadline:
                state.update(release="deadline", release_generated=state["generated"],
                             release_token_index=None)
            source = state["release"] if ready or deadline else "timeout"
            self.trace.record({"type": "gate", "batch_id": batch_id, "source": source,
                               "release_token_index": state.get("release_token_index"),
                               "generated": state.get("release_generated", 0), "timeout_seconds": timeout,
                               "elapsed_seconds": time.monotonic() - began})
        if source == "timeout":
            raise TimeoutError(f"Input {batch_id} has no fresh question answer/silence or visual silence within {timeout}s")
        return source

def collect_snapshot_outputs(session):
    # Capture a finite queue snapshot: waiting for another silence here would
    # undo the historical deadline advance. Never truncate strings, select
    # answers, or clear the remaining queue. Later chunks stay for next HTTP.
    queued = session._output_queue.qsize()
    parts = []
    for _ in range(queued):
        chunk = session.poll_output(timeout=0.0)
        if chunk is not None:
            parts.append(str(chunk))
    return parts
