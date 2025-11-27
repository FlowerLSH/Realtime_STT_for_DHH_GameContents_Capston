import numpy as np
from typing import Any, List, Dict, Optional

import torch
import torch.nn as nn
import opensmile

from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

import config


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values: torch.Tensor):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


class ProsodyAnalyzer:
    def __init__(
        self,
        sample_rate: int = 16000,
        device: Optional[str] = None,
        min_segment_sec: float = 0.2,
        min_word_window_sec: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.min_segment_sec = float(min_segment_sec)
        self.min_word_window_sec = float(min_word_window_sec)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = EmotionModel.from_pretrained(model_name).to(self.device).eval()

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

    def _analyze_chunk_emotion(self, chunk: np.ndarray) -> Dict[str, float]:
        if chunk.ndim > 1:
            chunk = np.mean(chunk, axis=0)
        x = np.asarray(chunk, dtype=np.float32)

        proc = self.processor(x, sampling_rate=self.sample_rate)
        inp = proc["input_values"][0].astype(np.float32)
        inp_t = torch.from_numpy(inp).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            _, logits = self.model(inp_t)

        logits_np = logits.squeeze(0).detach().cpu().numpy()
        arousal = float(logits_np[0])
        valence = float(logits_np[2])

        return {
            "valence": valence,
            "arousal": arousal,
        }

    def _analyze_chunk_loudness(self, chunk: np.ndarray) -> float:
        if chunk.ndim > 1:
            chunk = np.mean(chunk, axis=0)
        chunk = np.asarray(chunk, dtype=np.float32)
        df = self.smile.process_signal(chunk, self.sample_rate)
        if "Loudness_sma3" not in df.columns:
            return float("nan")
        loudness = float(df["Loudness_sma3"].mean())
        return loudness

    def _collect_words(self, segments: List[Any]) -> List[Dict[str, Any]]:
        words: List[Dict[str, Any]] = []
        for seg in segments:
            seg_words = getattr(seg, "words", None)
            if not seg_words:
                continue
            for w in seg_words:
                start = getattr(w, "start", None)
                end = getattr(w, "end", None)
                text = getattr(w, "word", None) or getattr(w, "text", None)
                if start is None or end is None or text is None:
                    continue
                start = float(start)
                end = float(end)
                if end <= start:
                    continue
                words.append(
                    {
                        "start": start,
                        "end": end,
                        "text": text,
                    }
                )
        return words

    def _analyze_words(self, audio: np.ndarray, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        n = len(audio)
        sr = self.sample_rate
        min_win = self.min_word_window_sec

        for w in words:
            ws = float(w["start"])
            we = float(w["end"])
            text = w.get("text")

            center = 0.5 * (ws + we)
            dur = we - ws

            if dur >= min_win:
                win_start = ws
                win_end = we
            else:
                half = 0.5 * min_win
                win_start = center - half
                win_end = center + half

            if win_start < 0.0:
                shift = -win_start
                win_start += shift
                win_end += shift

            win_start_idx = int(win_start * sr)
            win_end_idx = int(win_end * sr)

            win_start_idx = max(0, min(win_start_idx, n))
            win_end_idx = max(win_start_idx + 1, min(win_end_idx, n))

            chunk = audio[win_start_idx:win_end_idx]

            loudness = self._analyze_chunk_loudness(chunk)
            emo_vals = self._analyze_chunk_emotion(chunk)

            results.append(
                {
                    "start": ws,
                    "end": we,
                    "text": text,
                    "loudness": min(max(loudness, 0.0), 1.0),
                    "valence": min(max(emo_vals["valence"], 0.0), 1.0),
                    "arousal": min(max(emo_vals["arousal"], 0.0), 1.0),
                }
            )

        return results

    def analyze(self, audio: np.ndarray, segments: List[Any]) -> Dict[str, Any]:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        word_list = self._collect_words(segments)
        word_level = self._analyze_words(audio, word_list) if word_list else []

        return {
            "words": word_level,
        }
