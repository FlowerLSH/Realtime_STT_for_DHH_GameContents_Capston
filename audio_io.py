import queue
import threading
import numpy as np
import sounddevice as sd
import time

class RingBuffer:
    def __init__(self, max_sec: float, sr: int):
        self.max_samples = int(max_sec * sr)
        self.buf = np.zeros(self.max_samples, dtype=np.float32)
        self.sr = sr
        self.write_idx = 0
        self.total_written = 0
        self.lock = threading.Lock()

    def write(self, x: np.ndarray):
        with self.lock:
            n = len(x)
            idx = self.write_idx
            end = idx + n
            if end <= self.max_samples:
                self.buf[idx:end] = x
            else:
                k = self.max_samples - idx
                self.buf[idx:] = x[:k]
                self.buf[:end % self.max_samples] = x[k:]
            self.write_idx = end % self.max_samples
            self.total_written += n

    def read_tail(self, sec: float) -> np.ndarray:
        with self.lock:
            n = int(sec * self.sr)
            n = min(n, self.max_samples)
            if n <= self.write_idx:
                out = self.buf[self.write_idx - n:self.write_idx].copy()
            else:
                k = n - self.write_idx
                out = np.concatenate([self.buf[-k:], self.buf[:self.write_idx]]).copy()
            return out

class AudioStream:
    def __init__(self, samplerate: int, window_sec: float, chunk_ms: int):
        self.sr = samplerate
        self.window_sec = window_sec
        self.blocksize = int(samplerate * (chunk_ms / 1000.0))
        self.q = queue.Queue()
        self.rb = RingBuffer(max_sec=max(window_sec + 1, 6), sr=samplerate)
        self.stream = sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
        )

    def _callback(self, indata, frames, t, status):
        if indata.ndim == 2:
            self.q.put(indata[:, 0].copy())
        else:
            self.q.put(indata.copy())

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def pump(self):
        while True:
            try:
                x = self.q.get_nowait()
                self.rb.write(x)
            except queue.Empty:
                break

    def get_window(self) -> np.ndarray:
        return self.rb.read_tail(self.window_sec)
