# -*- coding: utf-8 -*-
import os, sys, platform, datetime, threading, queue
import numpy as np
import soundfile as sf
from pynput import keyboard

OUTPUT_DIR = "recordings"
TARGET_SR = 16000
CHANNELS = 2
BLOCK_FRAMES = 4096
DEVICE_NAME_HINT = "CABLE Output"

class _COMApartment:
    def __init__(self):
        self._ole32 = None
        self._inited = False
    def __enter__(self):
        if platform.system() != "Windows":
            return self
        import ctypes
        self._ole32 = ctypes.windll.ole32
        if self._ole32.CoInitializeEx(None, 0) in (0,1):
            self._inited = True
            return self
        if self._ole32.CoInitializeEx(None, 2) in (0,1):
            self._inited = True
            return self
        return self
    def __exit__(self, a, b, c):
        if self._inited and self._ole32 is not None:
            try: self._ole32.CoUninitialize()
            except Exception: pass

class BaseBackend:
    def open(self): ...
    def close(self): ...
    def read_block(self, n): ...
    def capture_sr(self) -> int: ...

class SDLoopback(BaseBackend):
    def __init__(self, hint, samplerate, channels, block_frames):
        import sounddevice as sd
        self.sd = sd
        self.req_sr = int(samplerate)
        self.channels = int(channels)
        self.block_frames = int(block_frames)
        self.device = self._pick_device(hint)
        self.q = queue.Queue()
        self.stream = None
        self._opened = False
        self._sr = self.req_sr
    def _pick_device(self, hint):
        sd = self.sd
        wasapi = None
        for i, h in enumerate(sd.query_hostapis()):
            if "WASAPI" in h["name"].upper():
                wasapi = i; break
        if wasapi is None:
            raise RuntimeError("sounddevice: WASAPI 미지원")
        try:
            from inspect import signature
            if "loopback" in signature(sd.WasapiSettings).parameters:
                dev = sd.query_hostapis()[wasapi]["default_output_device"]
                if isinstance(dev, int) and dev >= 0:
                    return dev
        except Exception:
            pass
        devs = sd.query_devices()
        cands = []
        for i, d in enumerate(devs):
            if d["hostapi"] == wasapi and d["max_input_channels"] > 0 and "(loopback)" in d["name"].lower():
                cands.append((i, d))
        if hint:
            hlow = hint.lower()
            tmp = [(i, d) for (i, d) in cands if hlow in d["name"].lower()]
            if tmp: cands = tmp
        if cands: return cands[0][0]
        raise RuntimeError("sounddevice: loopback 장치를 찾지 못함")
    def _cb(self, indata, frames, time, status):
        if status: pass
        self.q.put(indata.copy())
    def open(self):
        sd = self.sd
        kw = {}
        try:
            from inspect import signature
            if "loopback" in signature(sd.WasapiSettings).parameters:
                kw["extra_settings"] = sd.WasapiSettings(loopback=True, exclusive=False)
        except Exception:
            pass
        self.stream = sd.InputStream(
            samplerate=self.req_sr, blocksize=self.block_frames, device=self.device,
            channels=self.channels, dtype="float32", latency="low", callback=self._cb, **kw
        )
        self.stream.start()
        self._opened = True
        self._sr = int(self.stream.samplerate)
    def close(self):
        if self.stream:
            self.stream.stop(); self.stream.close(); self.stream = None
        self._opened = False
        while not self.q.empty():
            try: self.q.get_nowait()
            except Exception: break
    def read_block(self, n):
        buf = []; need = n
        while need > 0:
            try: data = self.q.get(timeout=0.5)
            except queue.Empty:
                data = np.zeros((min(need, self.block_frames), self.channels), dtype=np.float32)
            buf.append(data); need -= len(data)
        return np.concatenate(buf, axis=0)
    def capture_sr(self) -> int:
        return self._sr if self._opened else self.req_sr

class SCLoopback(BaseBackend):
    def __init__(self, hint, samplerate, channels, block_frames):
        import soundcard as sc
        self.sc = sc
        self.req_sr = int(samplerate)
        self.channels = int(channels)
        self.block_frames = int(block_frames)
        self.spkr = self._pick_speaker(hint)
        self.mic = self.sc.get_microphone(self.spkr.name, include_loopback=True)
        self.rec = None
    def _pick_speaker(self, hint):
        spkrs = self.sc.all_speakers()
        if hint:
            h = hint.lower()
            for s in spkrs:
                if h in s.name.lower(): return s
        return self.sc.default_speaker()
    def open(self):
        self.rec = self.mic.recorder(samplerate=self.req_sr, channels=self.channels, blocksize=self.block_frames)
        self.rec.__enter__()
    def close(self):
        if self.rec:
            try: self.rec.__exit__(None, None, None)
            finally: self.rec = None
    def read_block(self, n):
        out = []; need = n
        while need > 0:
            blk = self.rec.record(min(self.block_frames, need))
            if blk.ndim == 1: blk = blk[:, None]
            if blk.shape[1] < self.channels: blk = np.repeat(blk, self.channels, axis=1)
            elif blk.shape[1] > self.channels: blk = blk[:, :self.channels]
            out.append(blk.astype(np.float32)); need -= len(blk)
        return np.concatenate(out, axis=0)
    def capture_sr(self) -> int:
        return self.req_sr

class Resampler:
    def __init__(self, in_sr, out_sr, channels):
        self.in_sr = int(in_sr); self.out_sr = int(out_sr); self.channels = int(channels)
        self.mode = "identity"; self._setup()
    def _setup(self):
        if self.in_sr == self.out_sr: self.mode = "identity"; return
        try:
            from scipy.signal import resample_poly  # noqa
            self._have_scipy = True
        except Exception:
            self._have_scipy = False
        if self._have_scipy:
            from math import gcd
            g = gcd(self.in_sr, self.out_sr)
            self.up = self.out_sr // g; self.down = self.in_sr // g; self.mode = "scipy_poly"
        else:
            if self.in_sr % self.out_sr == 0:
                self.factor = self.in_sr // self.out_sr; self.mode = "decimate_int"
            else:
                self.mode = "linear"
        print(f"[resample] {self.in_sr} -> {self.out_sr} mode={self.mode}")
    def process(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "identity": return x
        if self.mode == "scipy_poly":
            from scipy.signal import resample_poly
            ys = []
            for c in range(self.channels):
                ys.append(resample_poly(x[:, c], self.up, self.down).astype(np.float32))
            return np.stack(ys, axis=1)
        if self.mode == "decimate_int":
            return x[::self.factor]
        in_len = x.shape[0]
        out_len = int(round(in_len * self.out_sr / self.in_sr))
        idx = np.linspace(0, in_len - 1, num=out_len, endpoint=True, dtype=np.float32)
        i0 = np.floor(idx).astype(np.int32)
        i1 = np.clip(i0 + 1, 0, in_len - 1)
        t = idx - i0
        y = (1 - t)[:, None] * x[i0, :] + t[:, None] * x[i1, :]
        return y.astype(np.float32)

class Recorder:
    def __init__(self, root_outdir, target_sr, channels, block_frames, hint):
        if platform.system() != "Windows":
            raise RuntimeError("Windows only (WASAPI)")
        os.makedirs(root_outdir, exist_ok=True)
        self.root_outdir = root_outdir
        self.target_sr = int(target_sr)
        self.channels = int(channels)
        self.block_frames = int(block_frames)
        self.hint = hint
        self.backend = None
        self.capture_sr = None
        self.sf = None
        self.t = None
        self.stop_evt = threading.Event()
        self.recording = False
        self._subdir = ""
        self._lock = threading.Lock()
        self.backend, self.capture_sr = self._init_backend_chain()
        print(f"[backend] capture_sr={self.capture_sr}, target_sr={self.target_sr}")
        self.resampler = Resampler(self.capture_sr, self.target_sr, self.channels)
    def set_subdir(self, code: str):
        code = "".join(ch for ch in code if ch.isdigit())
        with self._lock:
            self._subdir = code
    def _current_outdir(self):
        with self._lock:
            sub = self._subdir
        return os.path.join(self.root_outdir, sub) if sub else self.root_outdir
    def _try_backend(self, backend_cls, sr):
        b = backend_cls(self.hint, sr, self.channels, self.block_frames)
        b.open(); cap_sr = b.capture_sr(); b.close()
        return backend_cls, cap_sr
    def _init_backend_chain(self):
        candidates = [(SDLoopback, self.target_sr),(SCLoopback, self.target_sr),(SDLoopback, 48000),(SCLoopback, 48000)]
        last_err = None
        for be, sr in candidates:
            try:
                cls, cap_sr = self._try_backend(be, sr)
                print(f"[probe] {be.__name__} {sr} OK (cap_sr={cap_sr})")
                return be(self.hint, cap_sr, self.channels, self.block_frames), cap_sr
            except Exception as e:
                print(f"[probe] {be.__name__} {sr} FAIL: {e}"); last_err = e
        raise RuntimeError(f"사용 가능한 오디오 백엔드가 없습니다: {last_err}")
    def _writer(self):
        with _COMApartment():
            try:
                self.backend.open()
                while not self.stop_evt.is_set():
                    data = self.backend.read_block(self.block_frames)
                    if data.size == 0: continue
                    y = self.resampler.process(data)
                    self.sf.write(y)
            except Exception as e:
                print(f"[writer] error: {e}", flush=True)
            finally:
                try: self.backend.close()
                except Exception: pass
    def start(self):
        if self.recording: return
        outdir = self._current_outdir()
        os.makedirs(outdir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(outdir, f"rec_{ts}.wav")
        self.sf = sf.SoundFile(path, mode="w", samplerate=self.target_sr, channels=self.channels, subtype="PCM_16")
        self.stop_evt.clear()
        self.t = threading.Thread(target=self._writer, daemon=True)
        self.t.start()
        self.recording = True
        print(f"● REC 시작 → {path}")
    def stop(self):
        if not self.recording: return
        self.stop_evt.set()
        self.t.join()
        saved = self.sf.name
        self.sf.close(); self.sf = None
        self.recording = False
        print(f"■ REC 종료. 저장됨: {saved}")

def main():
    try:
        rec = Recorder(OUTPUT_DIR, TARGET_SR, CHANNELS, BLOCK_FRAMES, DEVICE_NAME_HINT.strip())
    except Exception as e:
        print(f"초기화 실패: {e}"); sys.exit(1)
    print('대기 중... 숫자키로 폴더 지정, Backspace=지우기, c=전체지우기, "a"=시작/정지, ESC=종료.')
    a_down = {"state": False}
    folder_code = []
    def _print_folder():
        code = "".join(folder_code)
        rec.set_subdir(code)
        base = OUTPUT_DIR
        p = os.path.join(base, code) if code else base
        print(f"[folder] {p}")
    def on_press(key):
        try:
            ch = key.char
            if ch is None: return
            if ch.isdigit():
                folder_code.append(ch); _print_folder(); return
            if ch.lower() == "c":
                folder_code.clear(); _print_folder(); return
            if ch == "a":
                if not a_down["state"]:
                    a_down["state"] = True
                    if rec.recording: rec.stop()
                    else: rec.start()
        except AttributeError:
            from pynput.keyboard import Key
            if key == Key.backspace:
                if folder_code:
                    folder_code.pop(); _print_folder(); return
            if key == Key.esc:
                if rec.recording: rec.stop()
                return False
    def on_release(key):
        try:
            if key.char == "a": a_down["state"] = False
        except AttributeError:
            pass
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try: listener.join()
        except KeyboardInterrupt:
            if rec.recording: rec.stop()

if __name__ == "__main__":
    main()
