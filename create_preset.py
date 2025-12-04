import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import librosa
import torch
from faster_whisper import WhisperModel

from modules.prosody import ProsodyAnalyzer

# ===== 설정 =====

# recordings 루트: 1,2,...,10 폴더가 있는 위치
ROOT_DIR = r".\ECAPA-TDNN\test"

# caster id ↔ 이름 매핑 파일
CASTERS_TXT = r".\ECAPA-TDNN\casters.txt"

# 결과 preset json 저장 위치
OUTPUT_JSON = r".\ECAPA-TDNN\caster_presets_word_prosody.json"

# Whisper 모델 이름 (다운로드/캐시 사용)
# faster-whisper가 지원하는 이름 중 하나: large-v3-turbo
WHISPER_MODEL_NAME = "large-v3-turbo"

# Prosody/Whisper 공통 샘플레이트
SAMPLE_RATE = 16000

# 캐스터 한 명당 최소 몇 개의 단어 prosody가 있어야 preset을 만들지
MIN_WORDS_PER_SPEAKER = 80


# ===== 유틸 함수들 =====

def load_caster_map(path: str) -> Dict[int, str]:
    """
    casters.txt를 읽어서 {id: name} 딕셔너리로 반환.
    예: "1 - CasterA" 이런 형식이라고 가정.
    """
    mapping: Dict[int, str] = {}
    p = Path(path)
    if not p.is_file():
        print(f"[warn] casters.txt not found: {path}")
        return mapping

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # "1 - 이름" 또는 "1-이름" 같은 형태 가정
            parts = line.split("-", maxsplit=1)
            if len(parts) != 2:
                continue
            sid_str, name = parts
            sid_str = sid_str.strip()
            name = name.strip()
            if not sid_str.isdigit():
                continue
            mapping[int(sid_str)] = name

    return mapping


def list_speaker_dirs(root_dir: str) -> List[Tuple[int, Path]]:
    """
    recordings 루트에서 숫자 이름의 폴더(1,2,3,...)들을 speaker id로 간주.
    """
    root = Path(root_dir)
    result: List[Tuple[int, Path]] = []
    if not root.is_dir():
        print(f"[error] ROOT_DIR is not a directory: {root_dir}")
        return result

    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue
        result.append((int(child.name), child))

    return result


def list_audio_files(folder: Path) -> List[Path]:
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    files: List[Path] = []
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            files.append(p)
    return files


def create_models() -> Tuple[ProsodyAnalyzer, WhisperModel]:
    """
    ProsodyAnalyzer + WhisperModel 생성.
    WhisperModel은 모델 이름(large-v3-turbo)만 넘겨서
    faster-whisper가 알아서 다운로드/캐시를 사용하게 함.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] using device: {device_str}")

    prosody = ProsodyAnalyzer(sample_rate=SAMPLE_RATE, device=device_str)

    whisper = WhisperModel(
        WHISPER_MODEL_NAME,          # <=== 경로 대신 모델 이름
        device=device_str,
        compute_type="float16" if device_str == "cuda" else "int8",
    )

    return prosody, whisper


def analyze_file_with_prosody(
    wav_path: Path,
    prosody: ProsodyAnalyzer,
    whisper: WhisperModel,
) -> List[Dict]:
    """
    한 wav 파일에 대해:
      1) Whisper로 word timestamps 포함한 segments 생성
      2) ProsodyAnalyzer로 word-level prosody 계산
    """
    # ProsodyAnalyzer.analyze()는 audio array와 segments 둘 다 필요
    audio, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)

    # faster-whisper의 transcribe: word_timestamps=True
    segments, info = whisper.transcribe(
        str(wav_path),
        beam_size=5,
        word_timestamps=True,
    )
    segments = list(segments)
    if not segments:
        return []

    prosody_info = prosody.analyze(audio, segments)
    words = prosody_info.get("words", [])
    return words


def aggregate_speaker_values(
    root_dir: str,
    prosody: ProsodyAnalyzer,
    whisper: WhisperModel,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    각 speaker 폴더(1~10)를 돌면서
    word-level loudness/valence/arousal 값을 speaker별로 모음.
    """
    speaker_values: Dict[int, Dict[str, List[float]]] = {}

    for sid, folder in list_speaker_dirs(root_dir):
        loud_list: List[float] = []
        val_list: List[float] = []
        aro_list: List[float] = []

        audio_files = list_audio_files(folder)
        print(f"[info] speaker {sid} ({folder}): {len(audio_files)} files")

        for wav_path in audio_files:
            try:
                word_infos = analyze_file_with_prosody(wav_path, prosody, whisper)
            except Exception as e:
                print(f"[warn] skip file due to error: {wav_path} ({e})")
                continue

            for w in word_infos:
                loud = w.get("loudness")
                val = w.get("valence")
                aro = w.get("arousal")
                if loud is None or val is None or aro is None:
                    continue
                if np.isnan(loud) or np.isnan(val) or np.isnan(aro):
                    continue
                loud_list.append(float(loud))
                val_list.append(float(val))
                aro_list.append(float(aro))

        loud_arr = np.array(loud_list, dtype=np.float32) if loud_list else np.array([], dtype=np.float32)
        val_arr = np.array(val_list, dtype=np.float32) if val_list else np.array([], dtype=np.float32)
        aro_arr = np.array(aro_list, dtype=np.float32) if aro_list else np.array([], dtype=np.float32)

        speaker_values[sid] = {
            "loudness": loud_arr,
            "valence": val_arr,
            "arousal": aro_arr,
        }

        print(
            f"[info] speaker {sid} collected: "
            f"words={loud_arr.size}, val={val_arr.size}, aro={aro_arr.size}"
        )

    return speaker_values


def compute_percentiles(values: np.ndarray, ps: List[float]) -> Dict[float, float]:
    if values.size == 0:
        return {p: float("nan") for p in ps}
    return {p: float(np.percentile(values, p)) for p in ps}


def build_presets(
    speaker_values: Dict[int, Dict[str, np.ndarray]],
    caster_map: Dict[int, str],
    min_words: int = MIN_WORDS_PER_SPEAKER,
) -> Dict[int, dict]:
    """
    speaker별로 퍼센타일 기반 preset 계산.
    - loudness: 5 / 50 / 95 퍼센타일
    - arousal: 5 / 50 / 95 퍼센타일
    - valence: 10 / 50 / 90 퍼센타일 → low / center / high
    """
    presets: Dict[int, dict] = {}

    for sid, data in speaker_values.items():
        loud = data.get("loudness", np.array([], dtype=np.float32))
        val = data.get("valence", np.array([], dtype=np.float32))
        aro = data.get("arousal", np.array([], dtype=np.float32))

        if loud.size < min_words:
            print(f"[warn] skip speaker {sid}: not enough words ({loud.size} < {min_words})")
            continue

        loud_p = compute_percentiles(loud, [5, 50, 95])
        aro_p = compute_percentiles(aro, [5, 50, 95]) if aro.size > 0 else {5: float("nan"), 50: float("nan"), 95: float("nan")}

        if val.size > 0:
            v_low = float(np.percentile(val, 10))
            v_center = float(np.percentile(val, 50))
            v_high = float(np.percentile(val, 90))
        else:
            v_low = float("nan")
            v_center = float("nan")
            v_high = float("nan")

        presets[sid] = {
            "id": sid,
            "name": caster_map.get(sid, ""),
            "samples": {
                "words": int(loud.size),
            },
            "loudness": {
                "p05": float(loud_p[5]),
                "p50": float(loud_p[50]),
                "p95": float(loud_p[95]),
            },
            "arousal": {
                "p05": float(aro_p[5]),
                "p50": float(aro_p[50]),
                "p95": float(aro_p[95]),
            },
            "valence": {
                "low": v_low,
                "center": v_center,
                "high": v_high,
            },
        }

        print(f"[info] built preset for speaker {sid}")

    return presets


def save_presets(presets: Dict[int, dict], output_path: str) -> None:
    obj = {str(k): v for k, v in presets.items()}
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[info] saved presets to {output_path}")


def main() -> None:
    caster_map = load_caster_map(CASTERS_TXT)
    print(f"[info] caster map: {caster_map}")

    prosody, whisper = create_models()

    speaker_values = aggregate_speaker_values(
        ROOT_DIR,
        prosody=prosody,
        whisper=whisper,
    )

    presets = build_presets(
        speaker_values,
        caster_map=caster_map,
        min_words=MIN_WORDS_PER_SPEAKER,
    )

    save_presets(presets, OUTPUT_JSON)


if __name__ == "__main__":
    main()
