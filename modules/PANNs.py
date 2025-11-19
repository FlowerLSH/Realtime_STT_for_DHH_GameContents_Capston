# PANNs.py
# -*- coding: utf-8 -*-

import os, sys, time, math, queue
import numpy as np

CHECKPOINT_PATH = r"..\panns_data\Cnn14_mAP=0.431.pth"
DEVICE_INDEX = None
SR_IN = 32000
CHANNELS = 1
HOP_MS = 250
WIN_MS = 1000
TOPK = 5

ALPHA = 0.35
ON_THR = 0.28
OFF_THR = 0.18
VISUAL_FLOOR = 0.02

TAU_SPEECH = 2.0
BETA_CROWD = 1.5

PEAK_SHORT_MS = 200
PEAK_LONG_MS  = 1000
PEAK_SHORT_ADD = 0.10
PEAK_LONG_ADD  = 0.20

PRINT_SUM_DEBUG = True

from panns_inference import AudioTagging
_at = AudioTagging(checkpoint_path=CHECKPOINT_PATH, device='cpu')
labels = _at.labels

SPEECH_LIKE = {
    "Speech","Narration, monologue","Conversation","Babbling",
    "Speech synthesizer","Shout","Yell"
}
CROWD_SET = {
    "Crowd","Applause","Clapping","Cheering","Chant","Whoop","Yell","Shout","Gasp","Laughter"
}
NON_SPEECH_VOCAL = {
    "Chuckle, chortle","Laughter","Sigh","Gasp","Screaming",
}

idx_speech_like = np.array([i for i,l in enumerate(labels) if l in SPEECH_LIKE], dtype=np.int64)
idx_crowd_set   = np.array([i for i,l in enumerate(labels) if l in CROWD_SET], dtype=np.int64)
idx_non_speech  = np.array([i for i,l in enumerate(labels) if l in NON_SPEECH_VOCAL], dtype=np.int64)

WHITELIST_ONLY = True
WHITELIST_SET = CROWD_SET.union(NON_SPEECH_VOCAL)
idx_whitelist = np.array([i for i,l in enumerate(labels) if l in WHITELIST_SET], dtype=np.int64)

def to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)

def from_logit(z):
    return 1.0 / (1.0 + np.exp(-z))

class EWMA:
    def __init__(self, C, alpha=0.35):
        self.state = np.zeros(C, dtype=np.float32)
        self.alpha = alpha
    def step(self, p):
        self.state = self.alpha * p + (1 - self.alpha) * self.state
        return self.state

class Hysteresis:
    def __init__(self, C, on_thr=0.3, off_thr=0.2):
        self.on_thr = on_thr
        self.off_thr = off_thr
        self.on = np.zeros(C, dtype=bool)
    def step(self, p):
        turn_on  = (p >= self.on_thr) & (~self.on)
        turn_off = (p <= self.off_thr) & ( self.on)
        self.on[turn_on] = True
        self.on[turn_off] = False
        return self.on

class PeakHold:
    def __init__(self, C, sr_hz, hold_ms):
        n = max(1, int((hold_ms/1000.0) * (1000.0 / HOP_MS)))
        self.buf = np.zeros((n, C), dtype=np.float32)
        self.ptr = 0
    def step(self, p):
        self.buf[self.ptr] = p
        self.ptr = (self.ptr + 1) % len(self.buf)
        return np.max(self.buf, axis=0)

def max_of_index(p, idxs):
    if idxs.size == 0: return 0.0
    return float(p[idxs].max())

def apply_visual_floor(p, mask_on, floor):
    p2 = p.copy()
    on_idx = np.where(mask_on)[0]
    if on_idx.size:
        p2[on_idx] = np.maximum(p2[on_idx], floor)
    return p2

def apply_whitelist_for_ranking(p, idx_whitelist):
    r = np.full_like(p, -1.0)
    if idx_whitelist.size:
        r[idx_whitelist] = p[idx_whitelist]
    return r

def topk_nonnegative(p, k, labels):
    valid = np.where(p >= 0.0)[0]
    if valid.size == 0:
        return ""
    idx_sorted = valid[np.argsort(-p[valid])]
    idx_top = idx_sorted[:k]
    parts = [f"{labels[i]}:{p[i]:.02f}" for i in idx_top]
    return " | ".join(parts)

class DisplayPostProcessor:
    def __init__(self, C, alpha, on_thr, off_thr, visual_floor,
                 tau_speech, beta_crowd,
                 peak_short_ms, peak_long_ms):
        self.smo = EWMA(C, alpha=alpha)
        self.hys_crowd = Hysteresis(C, on_thr=on_thr, off_thr=off_thr)
        self.hold_short = PeakHold(C, sr_hz=SR_IN, hold_ms=peak_short_ms)
        self.hold_long  = PeakHold(C, sr_hz=SR_IN, hold_ms=peak_long_ms)
        self.visual_floor = visual_floor
        self.tau_speech = tau_speech
        self.beta_crowd = beta_crowd

    def process(self, probs_raw, speech_active_flag):
        p = probs_raw.astype(np.float32)
        p_s = self.smo.step(p)
        
        ps = max_of_index(p_s, idx_speech_like)
        z = to_logit(p_s)
        if self.tau_speech > 0 and idx_speech_like.size:
            z[idx_speech_like] -= self.tau_speech * ps
        p_def = from_logit(z)
        p_boost = p_def.copy()
        if idx_crowd_set.size:
            boost = (1.0 + self.beta_crowd * ps)
            p_boost[idx_crowd_set] = np.clip(p_boost[idx_crowd_set] * boost, 0.0, 1.0)
        p_short = self.hold_short.step(p_boost)
        p_long  = self.hold_long.step(p_boost)
        p_disp = p_boost.copy()
        p_disp = np.clip(p_disp + PEAK_SHORT_ADD * p_short + PEAK_LONG_ADD * p_long, 0.0, 1.0)
        crowd_on = self.hys_crowd.step(p_disp)
        p_disp = apply_visual_floor(p_disp, crowd_on & np.isin(np.arange(len(labels)), idx_crowd_set), self.visual_floor)
        if speech_active_flag and idx_speech_like.size:
            p_rank = p_disp.copy()
            p_rank[idx_speech_like] = 0.0
        else:
            p_rank = p_disp
        if WHITELIST_ONLY:
            p_rank = apply_whitelist_for_ranking(p_rank, idx_whitelist)
        return p_rank, p_disp, p_s, ps

post = DisplayPostProcessor(
    C=len(labels),
    alpha=ALPHA,
    on_thr=ON_THR,
    off_thr=OFF_THR,
    visual_floor=VISUAL_FLOOR,
    tau_speech=TAU_SPEECH,
    beta_crowd=BETA_CROWD,
    peak_short_ms=PEAK_SHORT_MS,
    peak_long_ms=PEAK_LONG_MS
)

try:
    import sounddevice as sd
except Exception:
    sd = None
    print("[warn] sounddevice 미탑재: 마이크 스트리밍 비활성")

buf_len = int(SR_IN * (WIN_MS / 1000.0))
hop_len = int(SR_IN * (HOP_MS / 1000.0))
ring = np.zeros(buf_len, dtype=np.float32)
ring_pos = 0
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        sys.stderr.write(str(status) + "\n")
    if CHANNELS > 1:
        x = indata[:, 0]
    else:
        x = indata.reshape(-1)
    audio_q.put(x.copy())

def pull_audio_and_fill_ring():
    global ring, ring_pos
    got = 0
    while got < hop_len:
        try:
            chunk = audio_q.get_nowait()
        except queue.Empty:
            break
        take = min(len(chunk), hop_len - got)
        chunk = chunk[:take]
        end = ring_pos + take
        if end <= len(ring):
            ring[ring_pos:end] = chunk
        else:
            n1 = len(ring) - ring_pos
            ring[ring_pos:] = chunk[:n1]
            ring[:take - n1] = chunk[n1:]
        ring_pos = (ring_pos + take) % len(ring)
        got += take
    if got < hop_len:
        return False
    return True

def get_current_window():
    end = ring_pos
    start = (ring_pos - buf_len) % len(ring)
    if start < end:
        return ring[start:end].copy()
    else:
        return np.concatenate([ring[start:], ring[:end]], axis=0)

def infer_probs(x_mono_32k):
    x = x_mono_32k[None, :]
    pred = _at.inference(x)
    if isinstance(pred, dict):
        clip = pred.get('clipwise_output', pred.get('output', None))
        if clip is None:
            raise RuntimeError("Unexpected dict keys from panns inference: " + str(list(pred.keys())))
        p = np.asarray(clip[0], dtype=np.float32)
    elif isinstance(pred, (tuple, list)):
        if len(pred) == 0:
            raise RuntimeError("Empty tuple/list returned from panns inference.")
        clip = pred[0]
        try:
            clip = clip.detach().cpu().numpy() if hasattr(clip, "detach") else np.asarray(clip)
        except Exception:
            clip = np.asarray(clip)
        if clip.ndim == 2 and clip.shape[0] == 1:
            p = clip[0].astype(np.float32)
        elif clip.ndim == 1:
            p = clip.astype(np.float32)
        else:
            raise RuntimeError(f"Unexpected clipwise_output shape from tuple: {clip.shape}")
    else:
        raise RuntimeError(f"Unsupported inference return type: {type(pred)}")
    p = np.clip(p, 0.0, 1.0)
    if p.shape[0] != len(labels):
        minC = min(p.shape[0], len(labels))
        p2 = np.zeros(len(labels), dtype=np.float32)
        p2[:minC] = p[:minC]
        p = p2
    return p

def estimate_speech_active(p):
    if idx_speech_like.size == 0:
        return False
    s = float(np.max(p[idx_speech_like]))
    return s >= 0.20

def format_top_adj_whitelist(p, k=5):
    return topk_nonnegative(p, k=k, labels=labels)

def main():
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    if sd is None:
        print("[error] sounddevice가 없어 마이크 스트리밍을 사용할 수 없습니다.")
        return
    with sd.InputStream(samplerate=SR_IN, channels=CHANNELS, dtype='float32',
                        blocksize=hop_len, callback=audio_callback, device=DEVICE_INDEX):
        while True:
            if not pull_audio_and_fill_ring():
                time.sleep(HOP_MS/1000.0 * 0.5)
                continue
            x = get_current_window()
            p = infer_probs(x)
            speech_flag = estimate_speech_active(p)
            p_rank, p_disp, p_ewma, ps = post.process(p, speech_flag)
            
            msg = "[panns][top* adj] " + format_top_adj_whitelist(p_rank, k=TOPK)
            print(msg)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
