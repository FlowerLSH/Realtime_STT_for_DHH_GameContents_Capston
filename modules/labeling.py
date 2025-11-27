from pathlib import Path
import re
import numpy as np
import torch
from speechbrain.pretrained import EncoderClassifier
import torchaudio

import config

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
WORD_MIN_WIN_SEC = 1.0

MAP_DIR = Path(PROJECT_ROOT) / "test"
MAP_FILE = MAP_DIR / "casters.txt"

_classifier = None
_display_names = None
_cents = None

def smooth_speaker_labels(labels, confs):
    # 길이 N=6
    N = len(labels)
    
    # 결과 리스트는 원본을 복사하여 사용
    smoothed_labels = list(labels)
    
    # 규칙 4에 의해 영구적으로 Unknown이 된 위치를 기록 (추가 변경 방지)
    # 이 과정이 Rule 1, 2, 3에 영향을 주지 않도록 Rule 4는 별도로 처리
    locked_unknowns = [False] * N

    #########################################
    ## 1단계: Unknown 보간 및 짧은 오류 수정 (Rule 1, 2, 3)
    #########################################
    
    # 첫 번째 값(인덱스 0)은 건드리지 않으므로, 인덱스 1부터 N-1까지 확인
    for i in range(1, N):
        current_label = smoothed_labels[i]
        
        # 1. Unknown 처리 (Rule 1, 2)
        if current_label == 'Unknown':
            # 1-1. Unknown이 맨 뒤에 있는 경우 (i == N - 1)
            if i == N - 1:
                # Unknown을 바로 앞의 값과 같은 값으로 변경
                smoothed_labels[i] = smoothed_labels[i - 1]
                
            # 1-2. Unknown이 중간에 있는 경우 (i < N - 1)
            else:
                prev_label = smoothed_labels[i - 1]
                next_label = smoothed_labels[i + 1]
                prev_conf = confs[i - 1]
                next_conf = confs[i + 1]
                
                # 2-1. 앞의 값과 뒤의 값이 같다면
                if prev_label == next_label:
                    smoothed_labels[i] = prev_label
                    
                # 2-2. 앞의 값과 뒤의 값이 같지 않다면
                elif prev_label != next_label:
                    # 신뢰도가 더 큰 쪽의 값으로 변경
                    if prev_conf >= next_conf:
                        smoothed_labels[i] = prev_label
                    else:
                        smoothed_labels[i] = next_label
        
        # 2. 연속된 3개의 정보가 첫/세 번째가 동일하고 두 번째만 다른 경우 (Rule 3)
        # i는 중간 값(두 번째 값)을 가리킴. i-1 >= 0, i+1 < N 인 경우에만 확인
        if 1 <= i < N - 1:
            prev_label = smoothed_labels[i - 1]
            next_label = smoothed_labels[i + 1]
            current_label = smoothed_labels[i] # 업데이트되었을 수 있으므로 다시 가져옴
            
            # (X, Y, X) 패턴인지 확인
            if prev_label == next_label and prev_label != current_label:
                # 두 번째 값을 첫 번째 값(X)과 동일하게 변경
                smoothed_labels[i] = prev_label
                
    #########################################
    ## 2단계: 3개 모두 다를 경우 Unknown으로 변경 (Rule 4)
    #########################################
    
    # 윈도우(i-1, i, i+1)를 확인, 인덱스 1부터 N-2까지
    for i in range(1, N - 1):
        # 이미 Unknown이 아닌 값들 중에서 세 개가 모두 다른 경우를 찾습니다.
        # 이전 단계에서 Unknown은 이미 보간되어 다른 값일 가능성이 높습니다.
        prev = smoothed_labels[i - 1]
        curr = smoothed_labels[i]
        next = smoothed_labels[i + 1]
        
        # 4. 연속된 3개의 정보가 모두 다를 경우
        if prev != curr and curr != next and prev != next:
            # 3개를 모두 Unknown으로 변경
            smoothed_labels[i - 1] = 'Unknown'
            smoothed_labels[i] = 'Unknown'
            smoothed_labels[i + 1] = 'Unknown'
            
            # 이 Unknown은 다른 규칙에 의해 변경하지 않도록 잠금 설정
            locked_unknowns[i - 1] = True
            locked_unknowns[i] = True
            locked_unknowns[i + 1] = True
            
    # 최종 결과: locked_unknowns는 후속 처리를 막기 위한 메커니즘이지만, 
    # 요구사항에 따라 2단계까지만 실행 후 결과를 반환합니다.
    return smoothed_labels

def ensure_gallery():
    g = Path(GALLERY_DIR)
    spks = [d for d in g.iterdir() if d.is_dir()]
    ok = []
    for s in spks:
        if (s / "prototypes.npy").exists():
            ok.append(s)
    return ok


def rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.float() ** 2) + 1e-12))


def rms_normalize(x: torch.Tensor, target: float = TARGET_RMS) -> torch.Tensor:
    r = rms(x)
    if r < 1e-7:
        return x
    g = target / r
    y = x * g
    return torch.clamp(y, -1.0, 1.0)


def frame_signal(x: torch.Tensor, win_samp: int, hop_samp: int):
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


def encode_query_windows(clf: EncoderClassifier, w: torch.Tensor):
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
    global _classifier, _display_names, _cents
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
    display_names = []
    protos = []
    for sdir in gallery_dirs:
        raw_name = Path(sdir).name
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
    _display_names = display_names
    _cents = np.stack(cents, axis=0)


def _load_and_preprocess_waveform(input_data, target_sr=SR) -> torch.Tensor:
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


def _score_speaker_from_waveform(x: torch.Tensor):
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
    ex = np.exp(scores - scores.max())
    conf = float(ex[idx] / ex.sum()) if ex.sum() > 0 else 0.0
    return idx, conf


def GetSpeakerLabel(input_data, sr=SR, return_conf=False):
    _init_model_if_needed()
    if isinstance(input_data, (str, Path)):
        x = _load_and_preprocess_waveform(input_data, target_sr=SR)
    else:
        if isinstance(input_data, np.ndarray):
            x = torch.from_numpy(input_data).float()
        else:
            x = input_data
        if x.ndim == 2:
            x = x.mean(dim=0)
        if sr != SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SR)
            x = resampler(x.unsqueeze(0)).queeze(0)
    idx, conf = _score_speaker_from_waveform(x)
    name = _display_names[idx]
    if return_conf:
        return name, conf
    return name


class Labeler:
    def __init__(self, unknown_label: str = "Unknown", conf_threshold: float = None):
        self.unknown_label = unknown_label
        if conf_threshold is None:
            self.conf_threshold = float(getattr(config, "LABEL_THRESHOLD", 0.6))
        else:
            self.conf_threshold = float(conf_threshold)

        self.last_speaker = self.unknown_label
        self.last_conf = 0.0

    def assign_labels(self, input_data, sr=SR, text=None, prosody_info=None):
        _init_model_if_needed()
        if prosody_info is None or not isinstance(prosody_info, dict):
            return {}
        words = prosody_info.get("words") or []
        if not words:
            return {}

        x = _load_and_preprocess_waveform(input_data, target_sr=SR)
        n = x.numel()
        if n <= 0:
            return {}

        total_sec = float(n) / float(SR)
        win_sec = 1.0

        num_windows = max(1, int(np.ceil(total_sec / win_sec)))
        window_labels = [self.unknown_label] * num_windows
        window_confs = [0.0] * num_windows

        for idx in range(num_windows):
            win_start = idx * win_sec
            win_end = min((idx + 1) * win_sec, total_sec)

            start_idx = int(win_start * SR)
            end_idx = int(win_end * SR)
            start_idx = max(0, min(start_idx, n - 1))
            end_idx = max(start_idx + 1, min(end_idx, n))

            chunk = x[start_idx:end_idx]
            if chunk.numel() < int(0.1 * SR):
                continue

            w_idx, w_conf = _score_speaker_from_waveform(chunk)
            if w_conf < self.conf_threshold:
                continue

            window_labels[idx] = _display_names[w_idx]
            window_confs[idx] = w_conf

        ### 가드레일 추가 ###

        _id = [self.last_speaker]
        _id.extend(window_labels)

        _conf = [self.last_conf]
        _conf.extend(window_confs)

        window_labels = smooth_speaker_labels(_id, _conf)[1:]

        ###################


        #print(window_labels, window_confs)
        for w in words:
            ws = float(w.get("start", 0.0))
            we = float(w.get("end", 0.0))
            center = 0.5 * (ws + we)

            if center < 0.0:
                win_idx = 0
            else:
                win_idx = int(center // win_sec)

            if win_idx < 0 or win_idx >= num_windows:
                w["speaker"] = self.unknown_label
            else:
                w["speaker"] = window_labels[win_idx]

        # prosody_info["speaker_windows"] = []
        # for idx in range(num_windows):
        #     win_start = idx * win_sec
        #     win_end = min((idx + 1) * win_sec, total_sec)
        #     prosody_info["speaker_windows"].append(
        #         {
        #             "start": win_start,
        #             "end": win_end,
        #             "speaker": window_labels[idx],
        #             "conf": float(window_confs[idx]),
        #         }
        #     )

        self.last_speaker = window_labels[-1]
        self.last_conf = window_confs[-1]
        return {}

