from pathlib import Path
from typing import Any, Dict
import sys
import time

class OutputSink:
    def __init__(self, out_path: str):
        self.path = Path(out_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w", encoding="utf-8")
        self.start_time = time.time()

    def close(self):
        self.f.close()

    def _timestamp(self) -> str:
        elapsed = time.time() - self.start_time
        m = int(elapsed // 60)
        s = elapsed % 60
        return f"[{m:02d}:{s:05.2f}]"

    def write_line(self, text: str, labels: Dict[str, Any] | None = None, prosody_info: Dict[str, Any] | None = None):
        ts = self._timestamp()
        line = f"{ts} {text}"
        print(line, flush=True)
        sys.stdout.flush()
        self.f.write(line + "\n")
        self.f.flush()
