from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import soundfile as sf
import torch

import nemo.collections.asr as nemo_asr
from deepmultilingualpunctuation import PunctuationModel

from .base import STTBackend


@dataclass
class WordLike:
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class SegmentLike:
    id: int
    start: float
    end: float
    text: str
    words: List[WordLike] | None = None


class NemoHotwordPunctBackend(STTBackend):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        compute_type: str = "float16",
        default_language: Optional[str] = None,
        hotwords: Optional[Dict[str, float]] = None,
        use_punctuation: bool = True,
        punct_model_name: Optional[str] = None,
    ):
        # ASR model
        if Path(model_name).is_file():
            self.asr_model = nemo_asr.models.ASRModel.restore_from(
                restore_path=model_name
            )
        else:
            self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)

        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.asr_model.to(self.device)
        self.asr_model.eval()

        self.default_language = default_language
        self.hotwords: Dict[str, float] = hotwords or {}

        # dtype
        if compute_type in ("float16", "fp16"):
            self.autocast_dtype = torch.float16
        elif compute_type in ("bfloat16", "bf16"):
            self.autocast_dtype = torch.bfloat16
        else:
            self.autocast_dtype = torch.float32

        # HF punctuation model
        self.use_punctuation = use_punctuation
        self.punct_model: Optional[PunctuationModel] = None

        if self.use_punctuation:
            try:
                # HF repo id 형태("author/model")만 그대로 사용
                if punct_model_name and "/" in punct_model_name:
                    self.punct_model = PunctuationModel(model=punct_model_name)
                else:
                    # 기본 멀티랭 모델
                    self.punct_model = PunctuationModel()
            except Exception as e:
                print(
                    f"[NemoHotwordPunctBackend] WARNING: "
                    f"punctuation model load failed ({e}). Disabling punctuation."
                )
                self.punct_model = None
                self.use_punctuation = False

    def set_hotwords(self, hotwords: Dict[str, float]) -> None:
        self.hotwords = hotwords or {}

    def _run_nemo_transcribe(
        self,
        wav: np.ndarray,
        sampling_rate: int,
    ):
        """
        NeMo ASRModel.transcribe() wrapper.
        Returns list of Hypothesis.
        """
        tmp_path = Path("tmp_nemo_infer.wav")
        sf.write(tmp_path, wav, sampling_rate)

        cfg = getattr(self.asr_model, "cfg", None)
        if cfg is not None and hasattr(cfg, "decoding"):
            # hotwords (if supported)
            if self.hotwords:
                try:
                    cfg.decoding.hotwords = list(self.hotwords.keys())
                    cfg.decoding.hotword_weight = 2.0
                except Exception:
                    pass

            # timestamps (if supported)
            try:
                cfg.decoding.preserve_alignments = True
                cfg.decoding.compute_timestamps = True
            except Exception:
                pass

        try:
            hyps = self.asr_model.transcribe(
                [str(tmp_path)],
                return_hypotheses=True,
                timestamps=True,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        return hyps

    def _apply_punctuation(self, text: str) -> str:
        if not self.use_punctuation or self.punct_model is None:
            return text
        t = text.strip()
        if not t:
            return text
        try:
            return self.punct_model.restore_punctuation(t)
        except Exception:
            return text

    def transcribe_window(
        self,
        audio: np.ndarray,
        initial_prompt: str = "",
        language: Optional[str] = None,
    ) -> Tuple[str, List[SegmentLike]]:
        """
        WhisperFasterBackend와 동일한 인터페이스:
          - full_text: str
          - segs: List[SegmentLike] (Segment 비슷한 객체)
        """
        lang = language or self.default_language
        _ = lang  # 현재는 NeMo decoding에 직접 사용하지 않음

        sampling_rate = 16000

        with torch.autocast(
            device_type=self.device,
            dtype=self.autocast_dtype,
        ):
            hyps = self._run_nemo_transcribe(audio, sampling_rate)

        if not hyps:
            return "", []

        hyp = hyps[0]

        # 원본 텍스트
        raw_text = hyp.text.strip() if hasattr(hyp, "text") else str(hyp)

        # punctuation 적용 텍스트
        full_text = self._apply_punctuation(raw_text)

        # Hypothesis.timestamp -> Whisper-style segs
        ts = getattr(hyp, "timestamp", {}) or {}
        word_ts = ts.get("word", [])
        seg_ts = ts.get("segment", [])

        if seg_ts:
            seg_start = float(seg_ts[0]["start"])
            seg_end = float(seg_ts[0]["end"])
        elif word_ts:
            seg_start = float(word_ts[0]["start"])
            seg_end = float(word_ts[-1]["end"])
        else:
            seg_start = 0.0
            seg_end = 0.0

        words: List[WordLike] = []
        for w in word_ts:
            words.append(
                WordLike(
                    word=" " + w["word"],
                    start=float(w["start"]),
                    end=float(w["end"]),
                    probability=1.0,
                )
            )

        seg = SegmentLike(
            id=0,
            start=seg_start,
            end=seg_end,
            text=full_text,
            words=words,
        )

        segs: List[SegmentLike] = [seg]

        return full_text, segs
