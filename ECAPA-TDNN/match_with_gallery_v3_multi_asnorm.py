from pathlib import Path
import threading, datetime, platform, re
import numpy as np
import torch
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
COHORT_TOP   = 10
MIN_SEG_SEC  = 1.0
HANG_SEC     = 0.6
FORCE_SEG_SEC= 1.0
PRINT_TOP    = 10

# === 매핑 관련 설정 ===
MAP_DIR      = Path(PROJECT_ROOT) / "test"          # 여기서 매핑을 자동 탐색
MAP_FILE     = MAP_DIR / "casters.txt"              # (선택) 한 파일에 줄 단위 "숫자 - 이름"

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

def l2norm_rows_np(A):
    A = np.asarray(A, dtype=np.float32)
    n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    return A / n

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

# Adaptive S-norm
def asnorm_score(s_raw, enroll_vec, test_vec, cohort_cents, topk, exclude_idx=None):
    if cohort_cents.ndim == 1:
        cohort_cents = cohort_cents[None, :]
    if exclude_idx is not None and 0 <= exclude_idx < cohort_cents.shape[0] and cohort_cents.shape[0] > 1:
        cz = np.delete(cohort_cents, exclude_idx, axis=0)
    else:
        cz = cohort_cents
    z = cz @ enroll_vec
    t = cohort_cents @ test_vec
    kz = min(topk, z.shape[0]) if z.size else 0
    kt = min(topk, t.shape[0]) if t.size else 0
    if kz == 0 or kt == 0:
        return float(s_raw), float("nan"), float("nan"), float("nan"), float("nan")
    z_top = np.partition(z, -kz)[-kz:]
    t_top = np.partition(t, -kt)[-kt:]
    mu_z = float(z_top.mean()); sd_z = float(z_top.std() + 1e-6)
    mu_t = float(t_top.mean()); sd_t = float(t_top.std() + 1e-6)
    s_asn = 0.5 * ((s_raw - mu_z) / sd_z + (s_raw - mu_t) / sd_t)
    return float(s_asn), mu_z, sd_z, mu_t, sd_t

# =========================
# 숫자→이름 매핑 유틸 추가
# =========================
def _parse_line_to_map(line: str):
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(.+?)\s*$", line)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None

def load_id_map(map_dir: Path, map_file: Path):
    d = {}

    # 1) 줄 단위 매핑 파일(선택)
    if map_file.exists():
        for line in map_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line=line.replace("\ufeff","").strip()
            if not line or line.startswith("#"): 
                continue
            pair = _parse_line_to_map(line)
            if pair: d[pair[0]] = pair[1]

    # 2) 폴더 내 .txt 파일명/내용으로 추론 (예: "1 - LAURE.txt")
    if map_dir.exists():
        for p in map_dir.glob("*.txt"):
            # 파일명 기반
            m = re.match(r"^\s*(\d+)\s*[-_]\s*(.+?)\s*(?:\.\w+)?$", p.name)
            if m:
                d.setdefault(m.group(1).strip(), m.group(2).strip())
            # 내용 기반(우선순위 높음)
            try:
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line=line.replace("\ufeff","").strip()
                    pair = _parse_line_to_map(line)
                    if pair: d[pair[0]] = pair[1]
            except Exception:
                pass
    return d

def display_name(raw_label: str, id_map: dict):
    raw = str(raw_label).strip()
    # 완전 숫자면 그대로 매핑
    if raw.isdigit() and raw in id_map:
        return id_map[raw]
    # "1 - NAME" 혹은 "1_NAME" 같은 폴더명인 경우
    m = re.match(r"^\s*(\d+)\b", raw)
    if m and m.group(1) in id_map:
        return id_map[m.group(1)]
    # 그래도 못 찾으면 원본 반환
    return raw

class LiveMatcher:
    def __init__(self, clf, gallery_dirs):
        self.clf = clf
        # 매핑 로드
        self.id_map = load_id_map(MAP_DIR, MAP_FILE)

        self.names=[]           # 갤러리 폴더명(원본)
        self.display_names=[]   # 매핑 반영된 표시 이름
        self.protos=[]
        for sdir in gallery_dirs:
            raw_name = Path(sdir).name
            self.names.append(raw_name)
            self.display_names.append(display_name(raw_name, self.id_map))
            self.protos.append(np.load(str(Path(sdir)/"prototypes.npy")))
        # 코호트: 화자별 센트로이드 1개씩
        self.cents = []
        for P in self.protos:
            if P.ndim == 1:
                P = P[None, :]
            Pn = l2norm_rows_np(P)
            c = l2norm_np(Pn.mean(axis=0))
            self.cents.append(c)
        self.cents = np.stack(self.cents, axis=0)

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
        cq = l2norm_np(Eq.mean(axis=0))  # 테스트 센트로이드
        # 전 클래스에 대해 AS-norm 점수 계산
        asn_scores=[]
        for i in range(len(self.protos)):
            s_raw = float(np.dot(self.cents[i], cq))  # 코사인(raw)
            s_asn, _, _, _, _ = asnorm_score(
                s_raw=s_raw,
                enroll_vec=self.cents[i],
                test_vec=cq,
                cohort_cents=self.cents,
                topk=COHORT_TOP,
                exclude_idx=i
            )
            asn_scores.append(s_asn)
        order = np.argsort(asn_scores)[::-1]
        return {
            "order": order,
            "asn": asn_scores,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def _print_result(self, res):
        order = res["order"]; asn = res["asn"]
        print("\n=== LIVE S-norm ranking @ {} ===".format(res["timestamp"]))
        for i, idx in enumerate(order[:PRINT_TOP]):
            # 매핑 반영된 표시 이름 사용
            print(f"{i+1:2d}. {self.display_names[idx]:15s}  s-norm={asn[idx]:+.4f}")

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
