from pathlib import Path
import os, sys, threading, queue, time, datetime, platform
import numpy as np
import torch, torchaudio
from speechbrain.pretrained import EncoderClassifier
import soundcard as sc
from pynput import keyboard

PROJECT_ROOT = r"C:\Nemo\ECAPA-TDNN"
GALLERY_DIR  = str(Path(PROJECT_ROOT) / "models" / "test")
MODEL_DIR    = str(Path(PROJECT_ROOT) / "models" / "ecapa_vox")
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

SR           = 16000
CHANNELS     = 1
BLOCK_FRAMES = 2048
DEVICE_NAME_HINT = "CABLE Output"

WIN_SEC      = 2.0
HOP_SEC      = 0.5
BASE_RMS_TH  = 0.003
DYN_KEEP_PCT = 70
TARGET_RMS   = 0.04
TOPK_FRAMES  = 5
COHORT_TOP   = 10
MIN_SEG_SEC  = 1.0
HANG_SEC     = 0.6
FORCE_SEG_SEC= 1.5
PRINT_TOP    = 10

class _COMApartment:
    def __init__(self):
        self._ole32=None; self._inited=False
    def __enter__(self):
        if platform.system()!="Windows": return self
        import ctypes
        self._ole32=ctypes.windll.ole32
        if self._ole32.CoInitializeEx(None,0) in (0,1):
            self._inited=True; return self
        if self._ole32.CoInitializeEx(None,2) in (0,1):
            self._inited=True; return self
        return self
    def __exit__(self,a,b,c):
        if self._inited and self._ole32 is not None:
            try: self._ole32.CoUninitialize()
            except Exception: pass

def ensure_gallery():
    g = Path(GALLERY_DIR)
    spks = [d for d in g.iterdir() if d.is_dir()]
    ok=[]
    for s in spks:
        if (s/"prototypes.npy").exists():
            ok.append(s)
    return ok

def frame_signal(x, win_samp, hop_samp):
    n = x.numel(); t=0; frames=[]
    while t + win_samp <= n:
        frames.append(x[t:t+win_samp]); t += hop_samp
    return frames

def rms(x): return float(torch.sqrt(torch.mean(x.float()**2) + 1e-12))

def rms_normalize(x, target=0.04):
    r = rms(x)
    if r < 1e-7: return x
    g = target / r
    y = x * g
    return torch.clamp(y, -1.0, 1.0)

def l2norm_np(v):
    v = np.asarray(v).reshape(-1)
    n = np.linalg.norm(v) + 1e-9
    return v / n

def encode_query_windows(clf, w):
    win = int(WIN_SEC*SR); hop = int(HOP_SEC*SR)
    frames = frame_signal(w, win, hop)
    if not frames:
        fv = w.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            e = clf.encode_batch(fv).squeeze().detach().cpu().numpy()
        return np.stack([l2norm_np(e)], axis=0)
    rms_vals = [rms(f) for f in frames]
    dyn_th = np.percentile(rms_vals, 100 - DYN_KEEP_PCT)
    th = max(dyn_th, BASE_RMS_TH)
    kept=[]
    for f, r in zip(frames, rms_vals):
        if r < th: continue
        fv = rms_normalize(f, TARGET_RMS).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            e = clf.encode_batch(fv).squeeze().detach().cpu().numpy()
        kept.append(l2norm_np(e))
    if not kept:
        fv = rms_normalize(frames[np.argmax(rms_vals)], TARGET_RMS).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            e = clf.encode_batch(fv).squeeze().detach().cpu().numpy()
        kept=[l2norm_np(e)]
    return np.stack(kept, axis=0)

def cosine_matrix(A, B):
    return A @ B.T

def bi_topk(C, k=5):
    if C.size==0: return 0.0
    na, nb = C.shape
    k = max(1, min(k, na, nb))
    top_a = np.sort(C.max(axis=1))[-k:]
    top_b = np.sort(C.max(axis=0))[-k:]
    return float((top_a.mean() + top_b.mean())/2.0)

def centroid_cos(Eq, P):
    cq = l2norm_np(Eq.mean(axis=0))
    cp = l2norm_np(P.mean(axis=0))
    return float(np.dot(cq, cp))

def raw_score(Eq, P, topk=5):
    C = cosine_matrix(Eq, P)
    s1 = bi_topk(C, k=topk)
    s2 = centroid_cos(Eq, P)
    return 0.5*s1 + 0.5*s2, s1, s2

def s_norm(scores, target_idx, cohort_top=10):
    arr = np.array(scores)
    idx = target_idx
    others = np.delete(arr, idx)
    if others.size==0: return float(arr[idx]), float("nan"), float("nan")
    k = min(cohort_top, others.size)
    cohort = np.sort(others)[-k:]
    mu = cohort.mean(); sd = cohort.std()+1e-6
    s = (arr[idx] - mu) / sd
    return float(s), float(mu), float(sd)

class LiveMatcher:
    def __init__(self, clf, gallery_dirs):
        self.clf = clf
        self.names=[]
        self.protos=[]
        for sdir in gallery_dirs:
            self.names.append(Path(sdir).name)
            self.protos.append(np.load(str(Path(sdir)/"prototypes.npy")))
        self.stop_evt = threading.Event()
        self.thread = None
        self.active = False
        self.spkr = self._pick_speaker(DEVICE_NAME_HINT)
        self.mic = sc.get_microphone(self.spkr.name, include_loopback=True)
    def _pick_speaker(self, hint):
        spks = sc.all_speakers()
        if hint:
            h = hint.lower()
            for s in spks:
                if h in s.name.lower():
                    return s
        return sc.default_speaker()
    def start(self):
        if self.active: return
        self.stop_evt.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.active = True
    def stop(self):
        if not self.active: return
        self.stop_evt.set()
        self.thread.join()
        self.active = False
    def _run(self):
        with _COMApartment():
            rec = self.mic.recorder(samplerate=SR, channels=max(1, CHANNELS), blocksize=BLOCK_FRAMES)
            rec.__enter__()
            try:
                seg = []
                in_speech = False
                speech_len = 0
                silence_len = 0
                min_len = int(MIN_SEG_SEC*SR)
                hang = int(HANG_SEC*SR)
                force = int(FORCE_SEG_SEC*SR)
                since_force = 0
                while not self.stop_evt.is_set():
                    blk = rec.record(BLOCK_FRAMES)
                    if blk.ndim==2 and blk.shape[1]>1:
                        x = blk.mean(axis=1).astype(np.float32)
                    else:
                        x = blk.reshape(-1).astype(np.float32)
                    r = float(np.sqrt(np.mean(x*x)+1e-12))
                    if r >= BASE_RMS_TH:
                        seg.append(x)
                        in_speech = True
                        speech_len += len(x)
                        silence_len = 0
                        since_force += len(x)
                    else:
                        if in_speech:
                            silence_len += len(x)
                            seg.append(x)
                            since_force += len(x)
                            if silence_len >= hang and speech_len >= min_len:
                                w = torch.from_numpy(np.concatenate(seg, axis=0)).float()
                                res = self._match_once(w)
                                self._print_result(res)
                                seg=[]; in_speech=False; speech_len=0; silence_len=0; since_force=0
                        else:
                            seg=[]; speech_len=0; silence_len=0; since_force=0
                    if in_speech and speech_len >= min_len and since_force >= force:
                        w = torch.from_numpy(np.concatenate(seg, axis=0)).float()
                        res = self._match_once(w)
                        self._print_result(res)
                        seg=[]; speech_len=0; silence_len=0; since_force=0; in_speech=False
            finally:
                try: rec.__exit__(None,None,None)
                except Exception: pass
    def _match_once(self, w):
        Eq = encode_query_windows(self.clf, w)
        scores=[]; parts=[]; ks=[]
        for P in self.protos:
            s, s1, s2 = raw_score(Eq, P, TOPK_FRAMES)
            scores.append(s); parts.append((s1,s2)); ks.append(P.shape[0])
        order = np.argsort(scores)[::-1]
        best_idx = order[0]
        if len(scores)>=2:
            sn, mu, sd = s_norm(scores, best_idx, COHORT_TOP)
        else:
            sn, mu, sd = float(scores[best_idx]), float("nan"), float("nan")
        return {
            "order": order,
            "scores": scores,
            "parts": parts,
            "ks": ks,
            "best_idx": best_idx,
            "s_norm": sn,
            "mu": mu,
            "sd": sd,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    def _print_result(self, res):
        order = res["order"]; scores = res["scores"]; parts = res["parts"]; ks=res["ks"]
        print("\n=== LIVE ranking @ {} ===".format(res["timestamp"]))
        for i, idx in enumerate(order[:PRINT_TOP]):
            print(f"{i+1:2d}. {self.names[idx]:15s}  raw={scores[idx]:.4f}  (topk={parts[idx][0]:.4f}, cent={parts[idx][1]:.4f}, K={ks[idx]})")
        bi = res["best_idx"]
        print("\n=== S-norm for best ===")
        print(f"best={self.names[bi]}  raw={scores[bi]:.4f}  s-norm={res['s_norm']:.4f}  (mu={res['mu']:.4f}, sd={res['sd']:.4f})")

def main():
    gallery = ensure_gallery()
    if not gallery:
        print(f"[error] empty gallery: {GALLERY_DIR}")
        return
    print(f"[info] device={DEVICE}, speakers={len(gallery)}")
    clf = EncoderClassifier.from_hparams(source=MODEL_SOURCE, savedir=MODEL_DIR, run_opts={"device": DEVICE})
    clf.eval()
    lm = LiveMatcher(clf, gallery)
    print('대기 중... "a" 시작/정지, ESC 종료.')
    a_state = {"down": False}
    def on_press(key):
        try:
            if key.char == "a":
                if not a_state["down"]:
                    a_state["down"]=True
                    if lm.active: lm.stop()
                    else: lm.start()
        except AttributeError:
            from pynput.keyboard import Key
            if key == Key.esc:
                if lm.active: lm.stop()
                return False
    def on_release(key):
        try:
            if key.char == "a":
                a_state["down"]=False
        except AttributeError:
            pass
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try: listener.join()
        except KeyboardInterrupt:
            if lm.active: lm.stop()

if __name__ == "__main__":
    main()
