from pathlib import Path
import json, math
import numpy as np
import torch, torchaudio
from sklearn.cluster import KMeans
from speechbrain.pretrained import EncoderClassifier

# ================= 설정 =================
PROJECT_ROOT = r".\ECAPA-TDNN"
ENROLL_ROOT  = str(Path(PROJECT_ROOT) / "test")   # 내부에 화자ID 폴더별 WAV
GALLERY_DIR  = str(Path(PROJECT_ROOT) / "models" / "test")
MODEL_DIR    = str(Path(PROJECT_ROOT) / "models" / "ecapa_vox")
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SR           = 16000
WIN_SEC      = 2.0
HOP_SEC      = 0.5
BASE_RMS_TH  = 0.003           # 절대 하한
DYN_KEEP_PCT = 70              # 상위 %만 유지(말소리 진한 프레임 우선)
TARGET_RMS   = 0.04            # 윈도 개별 RMS 정규화 목표
K_MAX        = 3               # 화자당 최대 프로토타입 수
OUTLIER_MAD  = 2.0             # MAD 기준 이상치 컷(2.0~3.0 권장)

# ============== 유틸 ==============
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def load_wav_16k_mono(p):
    wav, sr = torchaudio.load(p)
    if wav.dim()==2 and wav.size(0)>1: wav = wav.mean(dim=0, keepdim=True)
    if sr != SR: wav = torchaudio.transforms.Resample(sr, SR)(wav)
    return wav.squeeze(0)

def frame_signal(x, win_samp, hop_samp):
    n = x.numel(); t = 0; frames=[]
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

def encode_many(clf, wavs):
    win = int(WIN_SEC*SR); hop = int(HOP_SEC*SR)
    embs=[]; rms_list=[]
    for w in wavs:
        frames = frame_signal(w, win, hop)
        if not frames: continue
        rms_vals = [rms(f) for f in frames]
        rms_list.extend(rms_vals)
        # 동적 임계 계산
        dyn_th = np.percentile(rms_vals, 100 - DYN_KEEP_PCT)
        th = max(dyn_th, BASE_RMS_TH)
        for f, r in zip(frames, rms_vals):
            if r < th: continue
            fv = rms_normalize(f, TARGET_RMS).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                e = clf.encode_batch(fv).squeeze().detach().cpu().numpy()
            embs.append(l2norm_np(e))
    if len(embs)==0: return np.empty((0,192))
    return np.stack(embs,axis=0)  # [N,D]

def trimmed_by_mad(E, mad_k=2.0):
    # 중앙값 기준 L1 거리의 MAD로 이상치 컷
    med = np.median(E, axis=0)
    d = np.linalg.norm(E - med, axis=1)
    mad = np.median(np.abs(d - np.median(d))) + 1e-9
    keep = d <= (np.median(d) + mad_k * mad * 1.4826)  # 1.4826≈MAD→σ 스케일
    if keep.sum() == 0:
        # 모두 버려지면 전체 평균 하나
        return E
    return E[keep]

def choose_k(n, kmax=3):
    if n < 12: return 1
    return min(kmax, max(1, int(round(math.sqrt(n/6)))))  # n이 많을수록 2~3

def make_prototypes(E):
    E = trimmed_by_mad(E, OUTLIER_MAD)
    if len(E)==0: return np.empty((0,192))
    k = choose_k(len(E), K_MAX)
    if k==1:
        proto = l2norm_np(E.mean(axis=0, keepdims=False))
        return proto.reshape(1,-1)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(E)
    C=[]
    for ci in range(k):
        members = E[km.labels_==ci]
        if len(members)==0: continue
        C.append(l2norm_np(members.mean(axis=0)))
    if not C:
        C.append(l2norm_np(E.mean(axis=0)))
    return np.stack(C, axis=0)

def main():
    ensure_dir(GALLERY_DIR)
    clf = EncoderClassifier.from_hparams(source=MODEL_SOURCE, savedir=MODEL_DIR, run_opts={"device": DEVICE})
    clf.eval()

    enroll_root = Path(ENROLL_ROOT)
    speakers = [d for d in enroll_root.iterdir() if d.is_dir()]
    if not speakers:
        print(f"[error] no speaker dirs in {ENROLL_ROOT}")
        return

    for spk_dir in speakers:
        spk_id = spk_dir.name
        wavs = sorted(spk_dir.rglob("*.wav"))
        if not wavs:
            print(f"[warn] no wav for {spk_id}, skip")
            continue
        print(f"[enroll] {spk_id}: {len(wavs)} files")
        W = [load_wav_16k_mono(str(w)) for w in wavs]
        E = encode_many(clf, W)
        if E.shape[0]==0:
            print(f"[warn] {spk_id}: no frames kept, skip")
            continue
        P = make_prototypes(E)  # [K,D]

        out_dir = Path(GALLERY_DIR) / spk_id
        ensure_dir(out_dir)
        np.save(str(out_dir / "prototypes.npy"), P)
        meta = {
            "speaker_id": spk_id,
            "num_files": len(wavs),
            "num_frames": int(E.shape[0]),
            "num_prototypes": int(P.shape[0]),
            "win_sec": WIN_SEC, "hop_sec": HOP_SEC,
            "target_rms": TARGET_RMS, "dyn_keep_pct": DYN_KEEP_PCT,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[done] {spk_id}: prototypes={P.shape} -> {out_dir}")
    print("[all done]")
if __name__ == "__main__":
    main()