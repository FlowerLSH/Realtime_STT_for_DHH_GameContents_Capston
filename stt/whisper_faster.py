from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel
from .base import STTBackend

class WhisperFasterBackend(STTBackend):
    def __init__(
        self,
        model_size: str,
        device: str = "cuda",
        compute_type: str = "float16",
        default_language: str | None = None,
    ):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.default_language = default_language

    def transcribe_window(
        self,
        audio: np.ndarray,
        initial_prompt: str = "",
        language: str | None = None,
    ):
        lang = language or self.default_language
        segments, info = self.model.transcribe(
            audio,
            language=lang,
            beam_size=1,
            temperature=0.0,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt if len(initial_prompt) < 200 else initial_prompt[-200:],
        )
        segs = list(segments)
        full_text = "".join(s.text for s in segs).strip()
        return full_text, segs
