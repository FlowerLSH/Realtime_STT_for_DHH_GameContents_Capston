from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import soundfile as sf
import torch

import nemo.collections.asr as nemo_asr
from nemo.collections.nlp.models import PunctuationCapitalizationModel

from .base import STTBackend


class NemoHotwordPunctBackend(STTBackend):
    def __init__(
        self,
        model_name: str,                 # [설정 포인트] 사용할 NeMo ASR 모델 이름 또는 .nemo 경로
        device: str = "cuda",                # [설정 포인트] "cuda" / "cpu"
        compute_type: str = "float16",       # [설정 포인트] "float16" / "bfloat16" / "float32"
        default_language: Optional[str] = None,
        hotwords: Optional[Dict[str, float]] = None,  # [설정 포인트] {"단어": weight} 형태로 넘겨줄 수 있음
        use_punctuation: bool = True,        # [설정 포인트] 문장부호/대문자 복원 사용할지 여부
        punct_model_name: Optional[str] = None,  # [설정 포인트] NeMo punctuation 모델 이름
    ):
        # ASR 모델 로드 (파일 경로 or from_pretrained 이름 둘 다 허용)
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

        # [설정 포인트] 연산 dtype (성능/안정성 트레이드오프)
        if compute_type in ("float16", "fp16"):
            self.autocast_dtype = torch.float16
        elif compute_type in ("bfloat16", "bf16"):
            self.autocast_dtype = torch.bfloat16
        else:
            self.autocast_dtype = torch.float32

        self.use_punctuation = use_punctuation
        self.punct_model = None
        if self.use_punctuation and punct_model_name:
            self.punct_model = PunctuationCapitalizationModel.from_pretrained(
                 punct_model_name
            )
            self.punct_model.to(self.device)
            self.punct_model.eval()

    def set_hotwords(self, hotwords: Dict[str, float]) -> None:
        # [설정 포인트] 런타임에 hotword dict를 교체할 때 사용
        self.hotwords = hotwords or {}

    def _run_nemo_transcribe(
        self,
        wav: np.ndarray,
        sampling_rate: int,
    ):
        # [설정 포인트] 필요하면 sampling_rate를 config.SAMPLE_RATE에 맞게 조정
        tmp_path = Path("tmp_nemo_infer.wav")
        sf.write(tmp_path, wav, sampling_rate)

        # ↓↓↓ NeMo 디코더 관련 설정을 조정하는 구간 ↓↓↓
        decoding_cfg = getattr(self.asr_model, "cfg", None)
        if decoding_cfg is not None and hasattr(decoding_cfg, "decoding"):
            decoding_cfg = decoding_cfg.decoding

            # [설정 포인트] hotword 적용 / weight 튜닝
            if self.hotwords:
                try:
                    decoding_cfg.hotwords = list(self.hotwords.keys())
                    decoding_cfg.hotword_weight = 2.0  # hotword boost 강도
                except Exception:
                    pass

            # [설정 포인트] 단어 단위 timestamp가 필요하면 True 유지
            try:
                decoding_cfg.preserve_alignments = True
            except Exception:
                pass

            try:
                with self.asr_model.decoding.override(decoding_cfg):
                    hyps = self.asr_model.transcribe(
                        [str(tmp_path)],
                        return_hypotheses=True,
                    )
            finally:
                tmp_path.unlink(missing_ok=True)
                return hyps

        # decoding cfg를 못 쓸 때: 기본 transcribe (fallback)
        hyps = self.asr_model.transcribe(
            [str(tmp_path)],
            return_hypotheses=True,
        )
        tmp_path.unlink(missing_ok=True)
        return hyps

    def _apply_punctuation(self, text: str) -> str:
        # [설정 포인트] punctuation을 끄고 싶으면 USE_PUNCTUATION=False 또는 use_punctuation=False
        if not self.use_punctuation or self.punct_model is None:
            return text
        t = text.strip()
        if not t:
            return text
        try:
            out = self.punct_model.add_punctuation_capitalization([t])
            if out and isinstance(out[0], str):
                return out[0]
        except Exception:
            return text
        return text

    def transcribe_window(
        self,
        audio: np.ndarray,
        initial_prompt: str = "",
        language: Optional[str] = None,
    ) -> Tuple[str, List[Any]]:
        # [설정 포인트] default_language / LANGUAGE 설정에 따라 lang을 쓸 수 있음
        lang = language or self.default_language
        _ = lang  # 필요하면 나중에 decoding 설정에서 사용

        sampling_rate = 16000  # [설정 포인트] 전체 파이프라인 SAMPLE_RATE와 맞추기

        with torch.autocast(
            device_type=self.device,
            dtype=self.autocast_dtype,
        ):
            hyps = self._run_nemo_transcribe(audio, sampling_rate)

        if not hyps:
            return "", []

        hyp = hyps[0]
        full_text = hyp.text.strip() if hasattr(hyp, "text") else str(hyp)

        # [설정 포인트] 여기서 바로 punctuation 적용 여부 선택
        full_text = self._apply_punctuation(full_text)

        segments: List[Any] = hyps  # [설정 포인트] 필요시 여기서 segments 구조를 가공 가능
        return full_text, segments
