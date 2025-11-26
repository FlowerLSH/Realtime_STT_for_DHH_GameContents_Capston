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
from modules.pannsDetector import PannsEventDetector

import time
from typing import List, Tuple


PANN_CHECKPOINT = r".\panns_data\Cnn14_mAP=0.431.pth"

DEBUG_MAIN = False
DEBUG_ATTACH = False


def attach_events_to_words(
    segments,
    window_start_time: float,
    events: List[Tuple[str, float, float]],
    last_committed_time: float,
):
    new_last_time = last_committed_time

    if not segments or not hasattr(segments[0], "words") or not segments[0].words:
        full_text = ""
        if segments and hasattr(segments[0], "text"):
            full_text = str(segments[0].text)
        return full_text, segments, new_last_time

    seg_infos: List[Tuple[float, int]] = []
    for idx, w in enumerate(segments[0].words):
        start_local = getattr(w, "start", 0.0)
        start_global = window_start_time + float(start_local)
        seg_infos.append((start_global, idx))

    seg_infos.sort(key=lambda x: x[0])

    events_sorted = sorted(events, key=lambda e: e[1])
    events_filtered = [e for e in events_sorted if e[1] >= last_committed_time]

    if seg_infos:
        last_word_start = seg_infos[-1][0]
        if last_word_start > new_last_time:
            new_last_time = last_word_start

    if events_filtered:
        if DEBUG_ATTACH:
            dbg = ", ".join(f"{e[0]}@{e[1]:.2f}" for e in events_filtered)
            print(f"[attach] events_filtered: {dbg}")

        for etype, t_event, score in events_filtered:
            loc = 0
            for start_global, idx in seg_infos:
                if t_event >= start_global:
                    loc += 1
                    continue
                else:
                    break

            tag_str = f"({etype})"

            if loc == len(seg_infos):
                last_idx = seg_infos[-1][1]
                wobj = segments[0].words[last_idx]
                cur = str(getattr(wobj, "word", ""))
                setattr(wobj, "word", cur + " " + tag_str)
            else:
                target_idx = seg_infos[loc][1]
                wobj = segments[0].words[target_idx]
                cur = str(getattr(wobj, "word", ""))
                setattr(wobj, "word", tag_str + " " + cur)

    words_list = [str(w.word) for w in segments[0].words]
    full_text = " ".join(words_list).strip()

    return full_text, segments, new_last_time


def run_pipeline():
    hotwords = build_hotword_list()
    print(f"[hotwords] loaded {len(hotwords)} items")
    print(hotwords)

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
    panns = PannsEventDetector(checkpoint_path=PANN_CHECKPOINT, device="cpu")

    committed_text = ""
    last_step_time = time.time()

    stream_time = 0.0
    last_committed_time = 0.0

    audio.start()
    print(f"[ready] streaming... window={WINDOW_SEC}s step={STEP_SEC}s")

    try:
        while True:
            audio.pump()
            now_wall = time.time()
            if now_wall - last_step_time < STEP_SEC:
                time.sleep(0.01)
                continue
            last_step_time = now_wall

            wav = audio.get_window()
            if wav.size == 0:
                continue

            stream_time += STEP_SEC
            window_start_time = max(0.0, stream_time - WINDOW_SEC)

            seg_len = int(SAMPLE_RATE * 1.0)
            if wav.size <= seg_len:
                seg_wav = wav
                seg_dur = wav.size / float(SAMPLE_RATE)
                t_center = stream_time - seg_dur * 0.5
            else:
                seg_wav = wav[-seg_len:]
                t_center = stream_time - 0.5

            events = panns.detect_events(seg_wav, sr=SAMPLE_RATE, t_now=t_center)

            if DEBUG_MAIN and events:
                dbg = ", ".join(f"{e[0]}:{e[2]:.3f}@{e[1]:.2f}" for e in events)
                print(f"[main] panns events: {dbg}")

            full_text, segments = stt_engine.transcribe_window(
                wav,
                initial_prompt=committed_text,
                language=LANGUAGE,
            )

            if not segments:
                if events:
                    tags = []
                    for etype, t_event, score in events:
                        if etype and etype not in tags:
                            tags.append(f"({etype})")
                    if tags:
                        text_out = " ".join(tags)
                        if DEBUG_MAIN:
                            print(f"[main] event-only output(no speech): '{text_out}'")
                        sink.write_line(text_out, prosody_info=None)
                continue

            full_text, segments, last_committed_time = attach_events_to_words(
                segments=segments,
                window_start_time=window_start_time,
                events=events,
                last_committed_time=last_committed_time,
            )

            prosody_info = prosody.analyze(wav, segments)

            text_out = full_text

            labeler.assign_labels(wav, sr=SAMPLE_RATE, prosody_info=prosody_info)

            sink.write_line(text_out, prosody_info=prosody_info)

            committed_text += " " + full_text

    except KeyboardInterrupt:
        pass
    finally:
        audio.stop()
        sink.close()


if __name__ == "__main__":
    run_pipeline()
