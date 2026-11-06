"""A stand-in VLM for tests and dry runs: answers from a function, a fixed dict, or a list (one per call, the last
repeated). Keeps every request it got in `requests`."""
import json

from .base import VLM


class MockVLM(VLM):
    def __init__(self, answer=None, name="mock"):
        self.answer, self.name, self.requests, self.price, self.max_workers = answer, name, [], (0.0, 0.0), 1

    def _call(self, req):
        self.requests.append(req)
        a = self.answer
        if callable(a):
            a = a(req)
        elif isinstance(a, list):
            a = a[min(len(self.requests), len(a)) - 1]
        text = a if isinstance(a, str) else json.dumps(a if a is not None else {})
        return dict(text=text, model=self.name, input_tokens=0, output_tokens=0, stop_reason="end_turn")
