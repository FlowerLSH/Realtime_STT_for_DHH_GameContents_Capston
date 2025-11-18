from abc import ABC, abstractmethod
import numpy as np
from typing import List, Any, Tuple

class STTBackend(ABC):
    @abstractmethod
    def transcribe_window(
        self,
        audio: np.ndarray,
        initial_prompt: str = "",
        language: str | None = None,
    ) -> Tuple[str, List[Any]]:
        pass
