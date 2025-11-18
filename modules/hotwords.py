import os
from pathlib import Path

from config import DEFAULT_HOTWORDS_TXT, USE_SCREEN_OCR_HOTWORDS, SCREEN_OCR_INCLUDE_SKILLS

try:
    from . import screenocr

    get_ocr_results = 0
except ImportError:
    get_ocr_results = None


def _load_default_hotwords(path: str) -> list[str]:
    p = Path(path)
    if not p.is_file():
        return []
    words = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = str(line).strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            words.append(s)
    return words


def _load_screen_ocr_hotwords() -> list[str]:
    if not USE_SCREEN_OCR_HOTWORDS:
        return []
    if get_ocr_results is None:
        return []
    try:
        return screenocr.get_ocr_results(flag_skills=SCREEN_OCR_INCLUDE_SKILLS)
    except Exception:
        return []


def build_hotword_list() -> list[str]:
    base = _load_default_hotwords(DEFAULT_HOTWORDS_TXT)
    ocr = _load_screen_ocr_hotwords()
    merged = []
    seen = set()
    for w in list(base) + list(ocr):
        s = str(w).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(s)
    return merged
