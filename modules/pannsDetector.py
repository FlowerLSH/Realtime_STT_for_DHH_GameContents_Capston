import time
from typing import List, Tuple, Optional

import numpy as np
from panns_inference import AudioTagging

try:
    import librosa
except Exception:
    librosa = None

DEBUG_PANNS = False


class PannsEventDetector:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        target_sr: int = 32000,
        alpha_baseline: float = 0.1,
        warmup_sec: float = 0.5,
    ):
        self.model = AudioTagging(checkpoint_path=checkpoint_path, device=device)
        self.labels = self.model.labels
        names = self.labels

        self.idx_sigh = np.array(
            [i for i, n in enumerate(names) if n in ("Sigh", "Gasp")],
            dtype=np.int64,
        )
        self.idx_laugh = np.array(
            [i for i, n in enumerate(names) if n in ("Laughter", "Chuckle, chortle")],
            dtype=np.int64,
        )
        self.idx_clap = np.array(
            [i for i, n in enumerate(names) if n in ("Clapping", "Applause")],
            dtype=np.int64,
        )
        self.idx_cheer = np.array(
            [
                i
                for i, n in enumerate(names)
                if n in ("Crowd", "Cheering", "Chant", "Whoop", "Yell", "Shout")
            ],
            dtype=np.int64,
        )
        self.idx_scream = np.array(
            [i for i, n in enumerate(names) if n in ("Screaming",)],
            dtype=np.int64,
        )

        self.group_names = ["sigh", "laugh", "clap", "cheer", "scream"]
        self.group_indices = {
            "sigh": self.idx_sigh,
            "laugh": self.idx_laugh,
            "clap": self.idx_clap,
            "cheer": self.idx_cheer,
            "scream": self.idx_scream,
        }

        all_target_indices = set()
        for indices in self.group_indices.values():
            all_target_indices.update(indices)
        
        # 2. 전체 레이블 인덱스 (0부터 len(labels)-1)를 정의합니다.
        all_indices = set(range(len(names)))
        
        # 3. 전체에서 타겟 인덱스를 제외한 나머지를 마스킹 대상으로 정의합니다.
        #    이 인덱스들에 해당하는 점수는 detect_events에서 0으로 설정됩니다.
        self.mask_indices = np.array(
            list(all_indices - all_target_indices), dtype=np.int64
        )

        self.baseline = {g: 0.0 for g in self.group_names}
        self.last_event_time = {g: -1e9 for g in self.group_names}
        self.alpha_baseline = alpha_baseline
        self.warmup_sec = warmup_sec
        self.start_time: Optional[float] = None
        self.target_sr = target_sr

        self.config = {
            "sigh":   {"min_abs": 0.007, "min_delta": 0.0005, "cooldown": 5.0},
            "laugh":  {"min_abs": 0.007, "min_delta": 0.0005, "cooldown": 5.0},
            "clap":   {"min_abs": 0.007, "min_delta": 0.0005, "cooldown": 5.0},
            "cheer":  {"min_abs": 0.007, "min_delta": 0.0005, "cooldown": 5.0},
            "scream": {"min_abs": 0.007, "min_delta": 0.0005, "cooldown": 5.0},
        }

    def _resample(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if sr == self.target_sr:
            return wav.astype(np.float32)
        if librosa is not None:
            return librosa.resample(
                wav.astype(np.float32),
                orig_sr=sr,
                target_sr=self.target_sr,
            )
        ratio = float(self.target_sr) / float(sr)
        n_out = int(len(wav) * ratio)
        if n_out <= 1:
            return wav.astype(np.float32)
        x_old = np.linspace(0.0, 1.0, len(wav), endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        return np.interp(x_new, x_old, wav).astype(np.float32)

    def _infer_probs(self, wav32k: np.ndarray) -> np.ndarray:
        if wav32k.ndim != 1:
            wav32k = wav32k.reshape(-1)

        target_len = int(self.target_sr * 2)
        n = wav32k.shape[0]

        if n <= 0:
            wav32k = np.zeros(target_len, dtype=np.float32)
        elif n < target_len:
            reps = int(np.ceil(target_len / float(n)))
            wav32k = np.tile(wav32k.astype(np.float32), reps)[:target_len]
        else:
            wav32k = wav32k.astype(np.float32)

        x = wav32k[None, :]
        pred = self.model.inference(x)

        if isinstance(pred, dict):
            clip = pred.get("clipwise_output", pred.get("output", None))
            if clip is None:
                raise RuntimeError("Unexpected dict keys from panns inference")
            p = np.asarray(clip[0], dtype=np.float32)
        elif isinstance(pred, (tuple, list)):
            if len(pred) == 0:
                raise RuntimeError("Empty tuple/list from panns inference")
            clip = pred[0]
            if hasattr(clip, "detach"):
                clip = clip.detach().cpu().numpy()
            else:
                clip = np.asarray(clip)
            if clip.ndim == 2 and clip.shape[0] == 1:
                p = clip[0].astype(np.float32)
            elif clip.ndim == 1:
                p = clip.astype(np.float32)
            else:
                raise RuntimeError("Unexpected clipwise_output shape")
        else:
            raise RuntimeError("Unsupported inference return type")

        p = np.clip(p, 0.0, 1.0)
        if p.shape[0] != len(self.labels):
            min_c = min(p.shape[0], len(self.labels))
            p2 = np.zeros(len(self.labels), dtype=np.float32)
            p2[:min_c] = p[:min_c]
            p = p2
        return p

    def detect_events(
        self,
        wav: np.ndarray,
        sr: int,
        t_now: Optional[float] = None,
    ) -> List[Tuple[str, float, float]]:
        if wav.size == 0:
            return []

        if t_now is None:
            t_now = time.time()
        if self.start_time is None:
            self.start_time = t_now

        warmup = (t_now - self.start_time) < self.warmup_sec

        wav32 = self._resample(wav, sr)
        p = self._infer_probs(wav32)

        if self.mask_indices.size > 0:
            p[self.mask_indices] = 0.0

        N=5
        if DEBUG_PANNS:
            top_N_indices = np.argsort(p)[::-1][:N]
            
            dbg_list = []
            for idx in top_N_indices:
                label = self.labels[idx]
                score = p[idx]
                dbg_list.append(f"{label}:{score:.3f}")
                    
            dbg_output = " | ".join(dbg_list)
            print(f"[panns DEBUG: Top {N}] {dbg_output}")

        events: List[Tuple[str, float, float]] = []

        for g in self.group_names:
            idx = self.group_indices[g]
            if idx.size == 0:
                continue

            score = float(p[idx].max())

            # ⭐️ 베이스라인 업데이트 및 디버그 출력 로직 수정 ⭐️
            
            # 1. 베이스라인 업데이트는 모든 스텝에서 진행합니다.
            self.baseline[g] = (1.0 - self.alpha_baseline) * self.baseline[g] + self.alpha_baseline * score
            b = self.baseline[g]
            cfg = self.config[g]
            
            if DEBUG_PANNS:
                # 2. 모든 스텝에서 베이스라인 및 점수를 출력합니다.
                #    워밍업 기간인지 아닌지를 표시해줍니다.
                prefix = "[panns] warmup" if warmup else "[panns] running"
                
                # 감지 조건 충족 여부 표시 (True이면 이벤트 발생)
                is_detected = (score >= cfg["min_abs"]) and (score - b >= cfg["min_delta"]) and (t_now - self.last_event_time[g] >= cfg["cooldown"])
                
                print(
                    f"{prefix} {g}: score={score:.4f} base={b:.4f} "
                    f"| min_abs={cfg['min_abs']:.4f} delta={score - b:.4f} min_delta={cfg['min_delta']:.4f} "
                    f"| Detected: {is_detected}"
                )
            
            # 3. 워밍업 기간일 경우, 이벤트 감지 로직(if/continue)을 건너뜁니다.
            if warmup:
                continue

            # 4. 워밍업이 끝나면 정상적인 감지 로직을 수행합니다.
            
            if score < cfg["min_abs"]:
                continue
            if score - b < cfg["min_delta"]:
                continue

            last_t = self.last_event_time[g]
            if t_now - last_t < cfg["cooldown"]:
                continue

            self.last_event_time[g] = t_now
            events.append((g, t_now, score))

            if DEBUG_PANNS:
                print(f"[panns] EVENT DETECTED: {g} score={score:.4f} base={b:.4f} t={t_now:.2f}")

        return events