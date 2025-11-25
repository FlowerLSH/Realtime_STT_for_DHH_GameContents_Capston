from pathlib import Path
from typing import Any, Dict, Optional
import json
import sys
import time
import config


class OutputSink:
    def __init__(self, out_path: str, json_mode: bool = True):
        self.dir = Path(out_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.json_mode = json_mode

    def close(self):
        return

    def _elapsed(self) -> float:
        return time.time() - self.start_time

    def _timestamp(self) -> str:
        elapsed = self._elapsed()
        m = int(elapsed // 60)
        s = elapsed % 60
        return f"[{m:02d}:{s:05.2f}]"

    def _strip_time_in_prosody(self, prosody_info: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(prosody_info, dict):
            return prosody_info
        out: Dict[str, Any] = {}
        for key, value in prosody_info.items():
            if key in ("segments", "words") and isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, dict):
                        new_item = {k: v for k, v in item.items() if k not in ("start", "end")}
                        new_list.append(new_item)
                    else:
                        new_list.append(item)
                out[key] = new_list
            else:
                out[key] = value
        return out

    def _make_file_path(self) -> Path:
        t = time.time()
        base = time.strftime("%Y%m%d_%H%M%S", time.localtime(t))
        ext = ".json" if self.json_mode else ".txt"
        path = self.dir / f"{base}{ext}"
        idx = 1
        while path.exists():
            path = self.dir / f"{base}_{idx:03d}{ext}"
            idx += 1
        return path

    def write_line(
        self,
        text: str,
        prosody_info: Optional[Dict[str, Any]] = None,
    ):
        include_time = getattr(config, "INCLUDE_TIME_DATA", True)

        if prosody_info is None:
            prosody_info = {}

        if not include_time:
            prosody = self._strip_time_in_prosody(prosody_info)
        else:
            prosody = prosody_info

        ts = self._timestamp()

        if self.json_mode:
            record: Dict[str, Any] = {
                "text": text,
                "prosody": prosody,
            }
            if include_time:
                record["time"] = time.time()
                record["elapsed"] = self._elapsed()
            line = json.dumps(record, ensure_ascii=False)
        else:
            line = f"{ts} {text}"

        print(f"{ts} {text}", flush=True)
        sys.stdout.flush()

        file_path = self._make_file_path()
        with file_path.open("w", encoding="utf-8") as f:
            f.write(line + "\n")
            print(file_path)
