"""Report generation progress without changing token selection or stopping rules."""
import threading
import time


class GenerationProgress:
    def __init__(self, prompt_tokens: int, batch_size: int, interval: float = 15.0):
        self.prompt_tokens = prompt_tokens
        self.batch_size = batch_size
        self.interval = interval
        self.steps = 0
        self.started = time.perf_counter()
        self.first_token_seconds = None
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def __call__(self, input_ids, scores, **kwargs):
        self.steps = input_ids.shape[1] - self.prompt_tokens
        if self.first_token_seconds is None:
            self.first_token_seconds = time.perf_counter() - self.started
            print(f'First token: {self.first_token_seconds:.2f}s', flush=True)
        return False

    def metrics(self) -> dict:
        elapsed = time.perf_counter() - self.started
        decode_seconds = elapsed - (self.first_token_seconds or elapsed)
        return {'generation_seconds': elapsed, 'first_token_seconds': self.first_token_seconds,
                'decode_steps': self.steps,
                'decode_steps_per_second': ((self.steps - 1) / decode_seconds
                                            if self.steps > 1 and decode_seconds > 0 else None)}

    def _heartbeat(self):
        while not self.finished.wait(self.interval):
            if self.steps == 0:
                status = 'waiting for first token (prefill / first-use kernel compilation)'
            else:
                rate = self.metrics()['decode_steps_per_second']
                status = f'{self.steps} decode steps, {rate:.2f} steps/s' if rate else f'{self.steps} decode steps'
            print(f'Generating: {time.perf_counter() - self.started:.1f}s elapsed; {status}', flush=True)

    def __enter__(self):
        self.started = time.perf_counter()
        print(f'Generating batch: size={self.batch_size}, padded_prompt_tokens={self.prompt_tokens}', flush=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.finished.set()
        self.thread.join()
        print(f'Generation {"interrupted" if exc_type else "finished"}: {self.metrics()}', flush=True)
