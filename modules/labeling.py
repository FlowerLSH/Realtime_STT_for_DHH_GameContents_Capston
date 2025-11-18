from pathlib import Path
import re
import numpy as np
import torch
from speechbrain.pretrained import EncoderClassifier
import torchaudio

PROJECT_ROOT = r"C:\Capston\ECAPA-TDNN"
GALLERY_DIR = str(Path(PROJECT_ROOT) / "models" / "test")
MODEL_DIR = str(Path(PROJECT_ROOT) / "models" / "ecapa_vox")
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SR = 16000
WIN_SEC = 2.0
HOP_SEC = 0.5
BASE_RMS_TH = 0.003
DYN_KEEP_PCT = 70
TARGET_RMS = 0.04
COHORT_TOP = 10

MAP_DIR = Path(PROJECT_ROOT) / "test"
MAP_FILE = MAP_DIR / "casters.txt"

_classifier = None
_names = None
_display_names = None
_cents = None


def ensure_gallery():
    g = Path(GALLERY_DIR)
    spks = [d for d in g.iterdir() if d.is_dir()]
    ok = []
    for s in spks:
        if (s / "prototypes.npy").exists():
            ok.append(s)
    return ok


def rms(x):
    return float(torch.sqrt(torch.mean(x.float() ** 2) + 1e-12))


def rms_normalize(x, target=0.04):
    r = rms(x)
    if r < 1e-7:
        return x
    g = target / r
    y = x * g
    return torch.clamp(y, -1.0, 1.0)


def frame_signal(x, win_samp, hop_samp):
    n = x.numel()
    t = 0
    frames = []
    while t + win_samp <= n:
        frames.append(x[t:t + win_samp])
        t += hop_samp
    return frames


def l2norm_np(v):
    v = np.asarray(v).reshape(-1)
    n = np.linalg.norm(v) + 1e-9
    return v / n


def l2norm_rows_np(A):
    A = np.asarray(A, dtype=np.float32)
    n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    return A / n


def _parse_line_to_map(line: str):
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(.+?)\s*$", line)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def load_id_map(map_dir: Path, map_file: Path):
    d = {}
    if map_file.exists():
        text = map_file.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.replace("\ufeff", "").strip()
            if not line or line.startswith("#"):
                continue
            pair = _parse_line_to_map(line)
            if pair:
                d[pair[0]] = pair[1]
    if map_dir.exists():
        for p in map_dir.glob("*.txt"):
            m = re.match(r"^\s*(\d+)\s*[-_]\s*(.+?)\s*(?:\.\w+)?$", p.name)
            if m:
                d.setdefault(m.group(1).strip(), m.group(2).strip())
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                for line in text.splitlines():
                    line = line.replace("\ufeff", "").strip()
                    pair = _parse_line_to_map(line)
                    if pair:
                        d[pair[0]] = pair[1]
            except Exception:
                pass
    return d


def display_name(raw_label: str, id_map: dict):
    raw = str(raw_label).strip()
    if raw.isdigit() and raw in id_map:
        return id_map[raw]
    m = re.match(r"^\s*(\d+)\b", raw)
    if m and m.group(1) in id_map:
        return id_map[m.group(1)]
    return raw


def encode_query_windows(clf, w):
    win = int(WIN_SEC * SR)
    hop = int(HOP_SEC * SR)
    frames = frame_signal(w, win, hop)
    if not frames:
        fv = w.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            e = clf.encode_batch(fv).squeeze().detach().cpu().numpy()
        return np.stack([l2norm_np(e)], axis=0)
    rms_vals = [rms(f) for f in frames]
    dyn_th = np.percentile(rms_vals, 100 - DYN_KEEP_PCT)
    th = max(dyn_th, BASE_RMS_TH)
    kept = []
    for f, r in zip(frames, rms_vals):
        if r < th:
            continue
        fv = rms_normalize(f, TARGET_RMS).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            e = clf.encode_batch(fv).squeeze().detach().cpu().numpy()
        kept.append(l2norm_np(e))
    if not kept:
        idx = int(np.argmax(rms_vals))
        fv = rms_normalize(frames[idx], TARGET_RMS).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            e = clf.encode_batch(fv).squeeze().detach().cpu().numpy()
        kept = [l2norm_np(e)]
    return np.stack(kept, axis=0)


def asnorm_score(s_raw, enroll_vec, test_vec, cohort_cents, topk, exclude_idx=None):
    if cohort_cents.ndim == 1:
        cohort_cents = cohort_cents[None, :]
    if (
        exclude_idx is not None
        and 0 <= exclude_idx < cohort_cents.shape[0]
        and cohort_cents.shape[0] > 1
    ):
        cz = np.delete(cohort_cents, exclude_idx, axis=0)
    else:
        cz = cohort_cents
    z = cz @ enroll_vec
    t = cohort_cents @ test_vec
    kz = min(topk, z.shape[0]) if z.size else 0
    kt = min(topk, t.shape[0]) if t.size else 0
    if kz == 0 or kt == 0:
        return float(s_raw)
    z_top = np.partition(z, -kz)[-kz:]
    t_top = np.partition(t, -kt)[-kt:]
    mu_z = float(z_top.mean())
    sd_z = float(z_top.std() + 1e-6)
    mu_t = float(t_top.mean())
    sd_t = float(t_top.std() + 1e-6)
    s_asn = 0.5 * ((s_raw - mu_z) / sd_z + (s_raw - mu_t) / sd_t)
    return float(s_asn)


def _init_model_if_needed():
    global _classifier, _names, _display_names, _cents
    if _classifier is not None:
        return
    gallery_dirs = ensure_gallery()
    if not gallery_dirs:
        raise RuntimeError(f"Empty gallery: {GALLERY_DIR}")
    clf = EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        savedir=MODEL_DIR,
        run_opts={"device": DEVICE},
    )
    clf.eval()
    id_map = load_id_map(MAP_DIR, MAP_FILE)
    names = []
    display_names = []
    protos = []
    for sdir in gallery_dirs:
        raw_name = Path(sdir).name
        names.append(raw_name)
        display_names.append(display_name(raw_name, id_map))
        protos.append(np.load(str(Path(sdir) / "prototypes.npy")))
    cents = []
    for P in protos:
        if P.ndim == 1:
            P = P[None, :]
        Pn = l2norm_rows_np(P)
        c = l2norm_np(Pn.mean(axis=0))
        cents.append(c)
    _classifier = clf
    _names = names
    _display_names = display_names
    _cents = np.stack(cents, axis=0)


def _load_and_preprocess_waveform(input_data, target_sr=SR):
    waveform = None
    sr = None
    if isinstance(input_data, (str, Path)):
        waveform, sr = torchaudio.load(str(input_data))
    elif isinstance(input_data, (torch.Tensor, np.ndarray)):
        waveform = input_data
        sr = target_sr
    else:
        raise TypeError("Input must be a file path (str) or a torch.Tensor/np.ndarray")
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    if sr is not None and sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    elif waveform.ndim > 2:
        waveform = waveform.reshape(-1, waveform.shape[-1]).mean(dim=0)
    return waveform


def GetSpeakerLabel(input_data, sr=SR):
    _init_model_if_needed()
    if isinstance(input_data, (str, Path)):
        x = _load_and_preprocess_waveform(input_data, target_sr=SR)
    else:
        if sr != SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SR)
            input_data = resampler(input_data)
        x = input_data
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.ndim == 2:
            x = x.mean(dim=0)
    Eq = encode_query_windows(_classifier, x)
    cq = l2norm_np(Eq.mean(axis=0))
    scores = []
    for i in range(_cents.shape[0]):
        s_raw = float(np.dot(_cents[i], cq))
        s_asn = asnorm_score(
            s_raw=s_raw,
            enroll_vec=_cents[i],
            test_vec=cq,
            cohort_cents=_cents,
            topk=COHORT_TOP,
            exclude_idx=i,
        )
        scores.append(s_asn)
    scores = np.asarray(scores)
    idx = int(np.argmax(scores))
    return _display_names[idx]


class Labeler:
    def __init__(self):
        pass

    def assign_labels(self, input_data, sr=SR, text=None, prosody_info=None):
        speaker = GetSpeakerLabel(input_data, sr=sr)
        return {"speaker": speaker}
