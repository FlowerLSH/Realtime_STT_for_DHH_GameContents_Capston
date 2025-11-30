import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
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
    """
    - 1단계: analyze()
        - word 단위로 prosody를 계산
        - fallback 기준으로 [0,1] 정규화
        - 동시에 *_raw 값도 같이 저장
    - 2단계: apply_presets_in_place()
        - Labeler가 word마다 speaker를 붙인 뒤 호출
        - preset JSON + speaker 이름을 이용해
          loudness/valence/arousal을 caster별로 다시 정규화해서 덮어씀
    """

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

        # preset 관련 캐시
        self._presets: Optional[Dict[str, Any]] = None  # JSON 그대로
        self._name_to_sid: Optional[Dict[str, int]] = None  # caster 이름 -> preset ID

    # ------------------------------------------------------------------
    # 내부 유틸: loudness / emotion fallback 정규화
    # ------------------------------------------------------------------
    def _fallback_normalize_loudness(self, loudness_raw: float) -> float:
        """
        opensmile Loudness_sma3 평균값 기준 대략적인 [0,1] 매핑.
        -60dB ~ 0dB 정도를 0~1로 맵핑한다고 가정.
        """
        if np.isnan(loudness_raw):
            return 0.5
        val = (loudness_raw + 60.0) / 60.0
        return float(np.clip(val, 0.0, 1.0))

    def _fallback_normalize_emotion(self, value_raw: float) -> float:
        """
        MSP-DIM 회귀 출력이 대략 [-1, 1] 근처라고 가정하고 [0,1]로 선형 매핑.
        """
        if np.isnan(value_raw):
            return 0.5
        val = (value_raw + 1.0) / 2.0
        return float(np.clip(val, 0.0, 1.0))

    # ------------------------------------------------------------------
    # preset JSON 로딩 + 이름 -> ID 매핑
    # ------------------------------------------------------------------
    def _load_presets(self) -> Dict[str, Any]:
        """
        config.CASTER_PRESET_JSON 에서 preset JSON을 한 번만 로딩.
        JSON 구조(예시):

        {
          "1": {
            "id": 1,
            "name": "DRAKOS",
            "samples": {"words": 1234},
            "loudness": {"p05": ..., "p50": ..., "p95": ...},
            "arousal": {"p05": ..., "p50": ..., "p95": ...},
            "valence": {"low": ..., "center": ..., "high": ...}
          },
          "2": { ... }
        }
        """
        if self._presets is not None:
            return self._presets

        path = getattr(config, "CASTER_PRESET_JSON", None)
        if not path:
            print("[prosody] CASTER_PRESET_JSON not set in config. Using fallback normalization only.")
            self._presets = {}
            self._name_to_sid = None
            return self._presets

        p = Path(path)
        if not p.is_file():
            print(f"[prosody] Preset JSON not found: {p}. Using fallback normalization only.")
            self._presets = {}
            self._name_to_sid = None
            return self._presets

        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[prosody] Failed to load presets from {p}: {e}. Using fallback only.")
            data = {}

        self._presets = data

        # caster 이름 -> speaker_id 매핑 생성
        name_to_sid: Dict[str, int] = {}
        for key, entry in data.items():
            try:
                sid = int(key)
            except Exception:
                continue
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not name:
                continue
            name_to_sid[str(name)] = sid
        self._name_to_sid = name_to_sid if name_to_sid else None

        print(f"[prosody] Loaded {len(data)} caster presets from {p}")
        return self._presets

    def _speaker_name_to_id(self, speaker_name: Optional[str]) -> Optional[int]:
        """
        word["speaker"] 에 들어있는 이름(DRAKOS 등)을 preset JSON과 매칭해서 ID 반환.
        """
        if not speaker_name:
            return None
        if self._presets is None:
            self._load_presets()
        if not self._name_to_sid:
            return None
        return self._name_to_sid.get(str(speaker_name))

    # ------------------------------------------------------------------
    # preset 기반 정규화 (ID 없으면 fallback 사용)
    # ------------------------------------------------------------------
    def _normalize_with_preset(
        self,
        loudness_raw: float,
        valence_raw: float,
        arousal_raw: float,
        speaker_id: Optional[int],
    ) -> (float, float, float):
        """
        - speaker_id 가 주어지고 preset이 있으면 preset 기반 정규화
        - 그렇지 않으면 fallback 정규화
        """
        presets = self._load_presets()
        if not presets or speaker_id is None:
            # preset 없거나 speaker_id 모르면 fallback
            return (
                self._fallback_normalize_loudness(loudness_raw),
                self._fallback_normalize_emotion(valence_raw),
                self._fallback_normalize_emotion(arousal_raw),
            )

        entry = presets.get(str(speaker_id))
        if not isinstance(entry, dict):
            return (
                self._fallback_normalize_loudness(loudness_raw),
                self._fallback_normalize_emotion(valence_raw),
                self._fallback_normalize_emotion(arousal_raw),
            )

        # --- loudness: p05 ~ p95 구간을 0~1로 매핑 ----------------------
        loud_cfg = entry.get("loudness", {})
        lp05 = float(loud_cfg.get("p05", -60.0))
        lp95 = float(loud_cfg.get("p95", 0.0))
        if lp95 <= lp05:
            loud_norm = self._fallback_normalize_loudness(loudness_raw)
        else:
            loud_norm = (loudness_raw - lp05) / (lp95 - lp05)
            loud_norm = float(np.clip(loud_norm, 0.0, 1.0))

        # --- arousal: p05 ~ p95 구간을 0~1로 매핑 ----------------------
        aro_cfg = entry.get("arousal", {})
        ap05 = float(aro_cfg.get("p05", -1.0))
        ap95 = float(aro_cfg.get("p95", 1.0))
        if ap95 <= ap05:
            aro_norm = self._fallback_normalize_emotion(arousal_raw)
        else:
            aro_norm = (arousal_raw - ap05) / (ap95 - ap05)
            aro_norm = float(np.clip(aro_norm, 0.0, 1.0))

        # --- valence: low/center/high 기준으로 -1~1 → 0~1 매핑 ----------
        val_cfg = entry.get("valence", {})
        v_low = float(val_cfg.get("low", -1.0))
        v_center = float(val_cfg.get("center", 0.0))
        v_high = float(val_cfg.get("high", 1.0))

        if not (v_low < v_center < v_high):
            val_norm = self._fallback_normalize_emotion(valence_raw)
        else:
            if valence_raw <= v_center:
                # low ~ center 구간을 0~0.5로
                val_norm = 0.0 + 0.5 * (valence_raw - v_low) / max(v_center - v_low, 1e-6)
            else:
                # center ~ high 구간을 0.5~1.0으로
                val_norm = 0.5 + 0.5 * (valence_raw - v_center) / max(v_high - v_center, 1e-6)
            val_norm = float(np.clip(val_norm, 0.0, 1.0))

        return loud_norm, val_norm, aro_norm

    # ------------------------------------------------------------------
    # chunk 단위 prosody 계산
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # STT segment에서 word 리스트 모으기
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # word 단위 prosody 분석 (1차: raw + fallback 정규화)
    # ------------------------------------------------------------------
    def _analyze_words(
        self,
        audio: np.ndarray,
        words: List[Dict[str, Any]],
        speaker_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
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

            loudness_raw = self._analyze_chunk_loudness(chunk)
            emo_vals = self._analyze_chunk_emotion(chunk)
            valence_raw = float(emo_vals["valence"])
            arousal_raw = float(emo_vals["arousal"])

            # 🔹 1차 분석 시점에는 speaker_id 정보가 없으므로
            #    speaker_id=None 을 넣고 -> fallback 기준으로만 정규화
            loud_norm, val_norm, aro_norm = self._normalize_with_preset(
                loudness_raw,
                valence_raw,
                arousal_raw,
                speaker_id=speaker_id,
            )

            results.append(
                {
                    "start": ws,
                    "end": we,
                    "text": text,
                    # 현재 overlay에서 바로 쓸 값 (fallback 기준 [0,1])
                    "loudness": float(loud_norm),
                    "valence": float(val_norm),
                    "arousal": float(aro_norm),
                    # 나중에 caster preset 적용할 때 쓸 raw 값
                    "loudness_raw": float(loudness_raw),
                    "valence_raw": float(valence_raw),
                    "arousal_raw": float(arousal_raw),
                }
            )

        return results

    # ------------------------------------------------------------------
    # public API: prosody 1차 분석
    # ------------------------------------------------------------------
    def analyze(
        self,
        audio: np.ndarray,
        segments: List[Any],
        speaker_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        - 실시간 파이프라인에서 사용되는 기본 엔트리.
        - speaker_id 는 (필요하다면) 전체 세그먼트를 한 명으로 가정할 때 쓸 수 있는 옵션.
          지금 구조에서는 대부분 None 으로 호출될 것임.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        word_list = self._collect_words(segments)
        word_level = self._analyze_words(audio, word_list, speaker_id=speaker_id) if word_list else []

        return {
            "words": word_level,
        }

    # ------------------------------------------------------------------
    # public API: Labeler 이후에 preset 기반 재정규화
    # ------------------------------------------------------------------
    def apply_presets_in_place(self, prosody_info: Dict[str, Any]) -> None:
        """
        Labeler.assign_labels() 이후에 호출해주는 함수.

        - 가정:
          prosody_info["words"][i] 에는 이미 다음 필드가 들어 있음
            - "loudness_raw", "valence_raw", "arousal_raw"  (analyze 단계에서 저장)
            - "speaker"  (Labeler가 붙인 caster 이름)
        - 동작:
          각 word의 speaker 이름을 preset JSON의 name과 매칭해서
          loudness/valence/arousal 을 caster별 preset 기준으로 다시 정규화하고 덮어씀.
        """
        if not isinstance(prosody_info, dict):
            return
        words = prosody_info.get("words")
        if not words:
            return

        presets = self._load_presets()
        if not presets or not self._name_to_sid:
            # preset 파일이 없으면 아무 것도 안 하고 종료
            return

        for w in words:
            speaker_name = w.get("speaker")
            speaker_id = self._speaker_name_to_id(speaker_name)

            loudness_raw = float(w.get("loudness_raw", w.get("loudness", 0.0)))
            valence_raw = float(w.get("valence_raw", w.get("valence", 0.0)))
            arousal_raw = float(w.get("arousal_raw", w.get("arousal", 0.0)))

            loud_norm, val_norm, aro_norm = self._normalize_with_preset(
                loudness_raw,
                valence_raw,
                arousal_raw,
                speaker_id=speaker_id,
            )

            # 최종적으로 overlay에서 사용할 값 덮어쓰기
            w["loudness"] = float(loud_norm)
            w["valence"] = float(val_norm)
            w["arousal"] = float(aro_norm)
