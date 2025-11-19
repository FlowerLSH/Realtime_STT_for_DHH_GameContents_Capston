from config import (
    MODEL_NAME,
    STT_BACKEND_TYPE,
    DEVICE,
    COMPUTE_TYPE,
    LANGUAGE,
    SAMPLE_RATE,
    WINDOW_SEC,
    STEP_SEC,
    CHUNK_MS,
    OUT_TXT,
)

from audio_io import AudioStream
from stt import create_stt_backend
from modules.prosody import ProsodyAnalyzer
from modules.labeling import Labeler
from modules.output import OutputSink
from modules.hotwords import build_hotword_list

import time
import re


def strip_common_prefix(prev: str, cur: str) -> str:
    prev = prev.strip()
    cur = cur.strip()
    if not prev:
        return cur
    if cur.startswith(prev):
        incremental_text = cur[len(prev):].strip()
        if re.match(r"^[\s.,!?]*$", incremental_text):
            return ""
        return incremental_text
    prev_words = prev.split()
    cur_words = cur.split()
    match_count = 0
    normalize = lambda text: re.sub(r"\s+", "", re.sub(r"[.,!?]", "", text.lower()))
    for p_word, c_word in zip(prev_words, cur_words):
        if normalize(p_word) == normalize(c_word):
            match_count += 1
        else:
            break
    if match_count < len(cur_words):
        return " ".join(cur_words[match_count:]).strip()
    return ""


def run_pipeline():
    hotwords = build_hotword_list()
    print(f"[hotwords] loaded {len(hotwords)} items")

    stt_engine = create_stt_backend(
        model_name=MODEL_NAME,
        backend_type=STT_BACKEND_TYPE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        language=LANGUAGE,
    )

    audio = AudioStream(
        samplerate=SAMPLE_RATE,
        window_sec=WINDOW_SEC,
        chunk_ms=CHUNK_MS,
    )

    prosody = ProsodyAnalyzer(sample_rate=SAMPLE_RATE)
    labeler = Labeler()
    sink = OutputSink(OUT_TXT)

    committed_text = ""
    last_step_time = time.time()

    audio.start()
    print(f"[ready] streaming... window={WINDOW_SEC}s step={STEP_SEC}s")

    try:
        while True:
            audio.pump()
            now = time.time()
            if now - last_step_time < STEP_SEC:
                time.sleep(0.01)
                continue
            last_step_time = now

            wav = audio.get_window()
            if wav.size == 0:
                continue

            full_text, segments = stt_engine.transcribe_window(
                wav,
                initial_prompt=committed_text,
                language=LANGUAGE,
            )

            if not segments or not full_text:
                continue

            inc = strip_common_prefix(committed_text, full_text)
            if not inc:
                continue

            prosody_info = prosody.analyze(wav, segments)

            labeler.assign_labels(wav, sr=SAMPLE_RATE, prosody_info=prosody_info)

            sink.write_line(inc, prosody_info=prosody_info)

            committed_text += " " + inc

    except KeyboardInterrupt:
        pass
    finally:
        audio.stop()
        sink.close()


if __name__ == "__main__":
    run_pipeline()
