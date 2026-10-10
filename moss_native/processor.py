"""Trace actual native text/image inputs and preserve FIFO collection."""
import json
import time

def collect_silence_outputs(session):
    parts = []
    last = time.monotonic()
    while True:
        chunk = session.poll_output(timeout=0.1)
        if chunk is not None:
            parts.append(str(chunk)); last = time.monotonic()
        elif time.monotonic() - last >= 0.3:
            return parts

class TokenizerTrace:
    def __init__(self, tokenizer, owner):
        self.tokenizer, self.owner = tokenizer, owner
    def __getattr__(self, name):
        return getattr(self.tokenizer, name)
    def __call__(self, *args, **kwargs):
        result = self.tokenizer(*args, **kwargs)
        event = self.owner.batch
        if event and event['question_batch']:
            assert event['frames'] == 0 and event['prompts'] == 1
            text = args[0] if args else kwargs['text']
            self.owner.record(text, result, 0)
        return result

class InputProcessor:
    def __init__(self, processor, trace, protocol_trace):
        self.processor, self.trace, self.protocol_trace = processor, trace, protocol_trace
        self.batch = None
        self.tokenizer = TokenizerTrace(processor.tokenizer, self)
    def __getattr__(self, name):
        return getattr(self.processor, name)
    def record(self, text, result, images):
        row = {'type':'prefill', **self.batch, 'original_text':text, 'actual_text':text,
               'input_ids':result['input_ids'].reshape(-1).tolist(), 'actual_images':images}
        assert row['input_ids'].count(151655) == images
        self.trace.record(row)
        with self.protocol_trace.open('a') as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    def __call__(self, *args, **kwargs):
        result = self.processor(*args, **kwargs)
        if kwargs.get('images') is not None:
            assert not args and self.batch and not self.batch['question_batch']
            assert len(kwargs['images']) == 1
            self.record(kwargs['text'],result,1)
        return result
