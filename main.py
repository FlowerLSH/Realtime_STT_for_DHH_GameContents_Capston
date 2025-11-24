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
    EVENT_TAGS,
)

from audio_io import AudioStream
from stt import create_stt_backend
from modules.prosody import ProsodyAnalyzer
from modules.labeling import Labeler
from modules.output import OutputSink
from modules.hotwords import build_hotword_list
from modules.pannsDetector import PannsEventDetector

import time
import re
from typing import List, Tuple


PANN_CHECKPOINT = r".\panns_data\Cnn14_mAP=0.431.pth"

# 이벤트를 어느 정도까지 단어와 "가깝다고" 볼 것인지 (초 단위)
EVENT_MAX_DIST = 3.0

# 너무 오래된 이벤트는 버리는 기준 (last_committed_time 기준)
EVENT_EXPIRE_SEC = 5.0

DEBUG_MAIN = True
DEBUG_ATTACH = True


def get_seg_time(seg, key: str) -> float:
    if isinstance(seg, dict):
        return float(seg.get(key, 0.0))
    return float(getattr(seg, key, 0.0))


def get_seg_text(seg) -> str:
    if isinstance(seg, dict):
        return str(seg.get("text", ""))
    return str(getattr(seg, "text", ""))


def attach_events_to_words(
    segments,
    window_start_time: float,
    pending_events: List[Tuple[str, float, float]],
    last_committed_time: float,
    now_audio_time: float,
):
    tokens: List[str] = []
    new_last_time = last_committed_time
    used_event = False

    # 1) 이벤트 유효 기간 필터링
    #    - t_event, last_committed_time, now_audio_time 모두 "오디오 타임라인" 기준
    filtered = []
    for (etype, t_event, score) in pending_events:
        # 너무 과거(마지막 커밋 시점보다 EVENT_EXPIRE_SEC 초 이전)면 버림
        if t_event < last_committed_time - EVENT_EXPIRE_SEC:
            continue
        # 오디오 타임라인 상으로 너무 먼 미래일 일은 없겠지만 안전하게
        if now_audio_time - t_event > 10.0:
            continue
        filtered.append((etype, t_event, score))
    pending_events[:] = filtered

    # 2) 이번 윈도에서 실제로 출력 대상이 되는 단어들만 모으기
    #    (이미 commit한 시점보다 뒤에 있는 단어들)
    seg_infos = []  # (index, start_global, end_global, word)
    for i, seg in enumerate(segments):
        local_start = get_seg_time(seg, "start")
        local_end = get_seg_time(seg, "end")
        word = get_seg_text(seg).strip()
        if not word:
            continue

        start_global = window_start_time + local_start
        end_global = window_start_time + local_end

        if end_global <= last_committed_time:
            continue

        seg_infos.append((i, start_global, end_global, word))

    if not seg_infos:
        # 이번 윈도에서는 새로 commit할 단어가 없음
        return "", new_last_time, False

    # 3) 각 이벤트를 "가장 가까운 단어"에 매칭
    #    word_tags[i] = 이 단어 앞에 붙일 태그 리스트
    word_tags = {i: [] for (i, _, _, _) in seg_infos}
    remaining_events: List[Tuple[str, float, float]] = []

    for etype, t_event, score in pending_events:
        best_idx = None
        best_dt = None
        best_start = None

        for i, start_global, end_global, word in seg_infos:
            dt = abs(t_event - start_global)
            if (best_dt is None) or (dt < best_dt):
                best_dt = dt
                best_idx = i
                best_start = start_global

        if best_idx is not None and best_dt is not None and best_dt <= EVENT_MAX_DIST:
            tag = EVENT_TAGS.get(etype, "")
            if tag:
                word_tags[best_idx].append((tag, etype, t_event, score, best_dt, best_start))
                used_event = True
        else:
            # 너무 멀어서 매칭 안 하는 이벤트는 다음 윈도우에서도 쓸 수 있게 남겨둠
            remaining_events.append((etype, t_event, score))

    pending_events[:] = remaining_events

    # 4) 단어들을 순회하면서 태그 붙인 토큰 문자열 생성
    for i, start_global, end_global, word in seg_infos:
        if end_global > new_last_time:
            new_last_time = end_global

        tags_info = word_tags.get(i, [])
        if tags_info:
            tags_str = " ".join(t[0] for t in tags_info)  # t[0] = tag 텍스트
            tokens.append(f"{tags_str} {word}")
            if DEBUG_ATTACH:
                for tag, etype, t_event, score, dt, wstart in tags_info:
                    print(
                        f"[attach] {etype} -> '{word}' "
                        f"t_event={t_event:.2f} word_start={wstart:.2f} "
                        f"dt={dt:.2f} score={score:.3f}"
                    )
        else:
            tokens.append(word)

    inc = " ".join(tokens).strip()
    return inc, new_last_time, used_event


def pop_event_only_tag(
    pending_events: List[Tuple[str, float, float]],
    last_committed_time: float,
    now_audio_time: float,
):
    if not pending_events:
        return ""

    # 오디오 타임라인 기준으로 필터링
    filtered = []
    for (etype, t_event, score) in pending_events:
        if t_event < last_committed_time - EVENT_EXPIRE_SEC:
            continue
        if now_audio_time - t_event > 10.0:
            continue
        filtered.append((etype, t_event, score))
    pending_events[:] = filtered

    if not pending_events:
        return ""

    # 가장 최근 이벤트 하나만 사용
    best_i = max(range(len(pending_events)), key=lambda i: pending_events[i][1])
    etype, t_event, score = pending_events.pop(best_i)
    tag = EVENT_TAGS.get(etype, "")
    if DEBUG_ATTACH:
        print(f"[attach-only] {etype} t={t_event:.2f} score={score:.3f} -> '{tag}'")
    return tag


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

    # 오디오 타임라인(초 단위) – 실제 녹화된 길이라고 생각하면 됨
    stream_time = 0.0
    last_committed_time = 0.0

    # 윈도 사이에서 이벤트를 유지하기 위한 버퍼
    pending_events: List[Tuple[str, float, float]] = []

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

            # 오디오 타임라인 진행
            stream_time += STEP_SEC
            window_start_time = max(0.0, stream_time - WINDOW_SEC)

            # 1) PANNs 이벤트 감지 (t_now=오디오 타임라인)
            events = panns.detect_events(wav, sr=SAMPLE_RATE, t_now=stream_time)
            if events:
                pending_events.extend(events)
            if DEBUG_MAIN and events:
                dbg = ", ".join(f"{e[0]}:{e[2]:.3f}" for e in events)
                print(f"[main] new_events: {dbg} (pending={len(pending_events)})")

            # 2) STT
            full_text, segments = stt_engine.transcribe_window(
                wav,
                initial_prompt=committed_text,
                language=LANGUAGE,
            )

            # 단어가 하나도 없을 때 → 이벤트만 출력
            if not segments:
                tag_only = pop_event_only_tag(
                    pending_events,
                    last_committed_time=last_committed_time,
                    now_audio_time=stream_time,
                )
                if tag_only:
                    text_out = tag_only
                    if DEBUG_MAIN:
                        print(f"[main] event-only output: '{text_out}' (no speech)")
                    sink.write_line(text_out, prosody_info=None)
                continue

            # 3) prosody
            prosody_info = prosody.analyze(wav, segments)

            # 4) 이벤트를 단어에 붙이기
            inc, last_committed_time, used_event = attach_events_to_words(
                segments,
                window_start_time=window_start_time,
                pending_events=pending_events,
                last_committed_time=last_committed_time,
                now_audio_time=stream_time,
            )

            # 새로 commit할 단어가 없지만, 이벤트만 있는 경우
            if not inc:
                tag_only = pop_event_only_tag(
                    pending_events,
                    last_committed_time=last_committed_time,
                    now_audio_time=stream_time,
                )
                if tag_only:
                    text_out = tag_only
                    if DEBUG_MAIN:
                        print(f"[main] event-only output: '{text_out}' (no new words)")
                    sink.write_line(text_out, prosody_info=prosody_info)
                continue

            text_out = inc

            if DEBUG_MAIN:
                print(
                    f"[main] text_out='{text_out}' len_events={len(pending_events)} "
                    f"last_time={last_committed_time:.2f}"
                )

            # 원래 있던 prosody 기반 라벨링
            labeler.assign_labels(wav, sr=SAMPLE_RATE, prosody_info=prosody_info)

            sink.write_line(text_out, prosody_info=prosody_info)

            committed_text += " " + inc

    except KeyboardInterrupt:
        pass
    finally:
        audio.stop()
        sink.close()


if __name__ == "__main__":
    run_pipeline()
