from .whisper_faster import WhisperFasterBackend
from .base import STTBackend

def create_stt_backend(
    model_name: str,
    backend_type: str = "whisper_faster",
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
    raise ValueError(f"Unknown STT backend: {backend_type}")
