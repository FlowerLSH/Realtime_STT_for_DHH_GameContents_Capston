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
    LABEL_THRESHOLD,
)

from audio_io import AudioStream
from stt import create_stt_backend
from modules.prosody import ProsodyAnalyzer
from modules.labeling import Labeler
from modules.output import OutputSink
from modules.hotwords import build_hotword_list
from modules.pannsDetector import PannsEventDetector

import time
import webbrowser
import os
import re
from collections import Counter
from typing import List, Tuple

from modules.subtitle import start_subtitle_server, send_subtitle


PANN_CHECKPOINT = r".\panns_data\Cnn14_mAP=0.431.pth"

DEBUG_MAIN = False
DEBUG_ATTACH = False

def clean_and_split_text(text: str) -> List[str]:
    # 1. 특수 기호(。, 쉼표, 온점 등)를 제거하고 공백으로 대체
    # 'It's' 같은 축약어는 유지하기 위해 \w+ 패턴은 그대로 둡니다.
    cleaned_text = re.sub(r'[^\w\s\']', ' ', text)
    
    # 2. 공백을 기준으로 단어를 분리하고 빈 문자열 제거
    words = [word for word in cleaned_text.lower().split() if word]
    
    return words

def is_error(full_text: str, top_k: int = 2, max_ratio: float = 0.8) -> bool:
    """
    텍스트 내에서 상위 K개의 토큰이 전체 텍스트의 일정 비율(max_ratio) 이상을 
    차지하는 경우 (즉, 반복 루프 또는 오염)를 감지합니다.
    """
    words = clean_and_split_text(full_text)
    if not words:
        return False
        
    total_words = len(words)
    
    # 1. 단어 빈도수 계산
    word_counts = Counter(words)
    
    # 2. 가장 흔한 상위 K개 단어의 빈도 합산
    most_common = word_counts.most_common(top_k)
    top_k_count = sum(count for word, count in most_common)
    
    # 3. 비율 계산
    repetition_ratio = top_k_count / total_words
    
    # 4. 비율이 임계값을 초과하면 오염(반복 루프)으로 판단
    if repetition_ratio >= max_ratio:
        print(f"[Warning] High Repetition Ratio ({repetition_ratio:.2f}): Top {top_k} words dominate the text.")
        return True
        
    return False

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
    start_subtitle_server(host="localhost", port=8765)
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
    labeler = Labeler(conf_threshold = LABEL_THRESHOLD)
    sink = OutputSink(OUT_TXT)
    panns = PannsEventDetector(checkpoint_path=PANN_CHECKPOINT, device="cpu")

    committed_text = ""
    last_step_time = time.time()

    stream_time = 0.0
    last_committed_time = 0.0

    audio.start()

    print(f"[ready] streaming... window={WINDOW_SEC}s step={STEP_SEC}s")
    
    try:

        jid = 0
        while True:
            jid += 1
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



            if is_error(full_text):
                record = {"id": jid, "time": time.time(), "text": "STT Error Occured. Skip this Window.", "prosody":{'words' : []}}
                print("STT 텍스트 오염 감지")
                print(full_text)
                print("Dummy 자막을 전송합니다.")
                try:
                    send_subtitle(record)
                except Exception as e:
                    # 자막 전송 실패해도 메인 파이프라인은 죽지 않도록
                    print(f"[ws] send_subtitle error: {e}", file=sys.stderr)
                committed_text = " "
                continue
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

            committed_text = " " + full_text

            record = {"id": jid, "time": time.time(), "text": text_out, "prosody":prosody_info}
            try:
                send_subtitle(record)
            except Exception as e:
                # 자막 전송 실패해도 메인 파이프라인은 죽지 않도록
                print(f"[ws] send_subtitle error: {e}", file=sys.stderr)

    except KeyboardInterrupt:
        pass
    finally:
        audio.stop()
        sink.close()


if __name__ == "__main__":
    html_file_name = "overlay_words.html"
    html_path = os.path.abspath(html_file_name)

    browser_process = None
    try:
        print(f"[overlay] Opening browser: {html_path}")
        webbrowser.open_new_tab(f"file:///{html_path}")

        run_pipeline()

    except KeyboardInterrupt:
        print("\n[stop] Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n[error] An unexpected error occurred: {e}")
    finally:
        # 3. 🛑 브라우저 자동 종료 (⚠️ 제약 사항 있음)
        print("[cleanup] Stopping services...")
