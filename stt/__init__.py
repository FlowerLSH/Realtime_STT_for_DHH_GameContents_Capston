# stt/__init__.py

from .base import STTBackend
from .whisper_faster import WhisperFasterBackend
from .nemo_hotword_punct import NemoHotwordPunctBackend


def create_stt_backend(
    model_name: str,
    backend_type: str,
    device: str = "cuda",
    compute_type: str = "float16",
    language: str | None = None,
) -> STTBackend:
    if backend_type == "whisper_faster":
        return WhisperFasterBackend(
            model_size=model_name,
            device=device,
            compute_type=compute_type,
            default_language=language,
        )

    if backend_type == "nemo_hotword_punct":
        # config를 여기서 import해서 순환참조 피함
        from config import USE_PUNCTUATION, PUNCT_MODEL_NAME

        return NemoHotwordPunctBackend(
            model_name=model_name,
            device=device,
            default_language=language,
            use_punctuation=USE_PUNCTUATION,
            punct_model_name=PUNCT_MODEL_NAME,
        )

    raise ValueError(f"Unknown STT backend type: {backend_type}")
