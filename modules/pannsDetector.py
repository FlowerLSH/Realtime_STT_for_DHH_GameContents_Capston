import time
from typing import List, Tuple, Optional

import numpy as np
from panns_inference import AudioTagging

try:
    import librosa
except Exception:
    librosa = None

DEBUG_PANNS = True


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
            [i for i, n in enumerate(names) if n in ("Crowd", "Cheering", "Chant", "Whoop", "Yell", "Shout")],
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

        self.baseline = {g: 0.0 for g in self.group_names}
        self.last_event_time = {g: -1e9 for g in self.group_names}
        self.alpha_baseline = alpha_baseline
        self.warmup_sec = warmup_sec
        self.start_time: Optional[float] = None
        self.target_sr = target_sr

        self.config = {
            "sigh":   {"min_abs": 0.03, "min_delta": 0.01, "cooldown": 1.5},
            "laugh":  {"min_abs": 0.04, "min_delta": 0.02, "cooldown": 1.5},
            "clap":   {"min_abs": 0.05, "min_delta": 0.02, "cooldown": 1.5},
            "cheer":  {"min_abs": 0.05, "min_delta": 0.02, "cooldown": 2.0},
            "scream": {"min_abs": 0.05, "min_delta": 0.02, "cooldown": 2.0},
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
        events: List[Tuple[str, float, float]] = []

        for g in self.group_names:
            idx = self.group_indices[g]
            if idx.size == 0:
                score = 0.0
            else:
                score = float(p[idx].max())
            b = self.baseline[g]
            b = self.alpha_baseline * score + (1.0 - self.alpha_baseline) * b
            self.baseline[g] = b
            if warmup:
                continue
            cfg = self.config[g]
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
                print(f"[panns] event {g} score={score:.3f} base={b:.3f} t={t_now:.2f}")

        return events
