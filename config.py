MODEL_NAME = "large-v3-turbo"
STT_BACKEND_TYPE = "whisper_faster"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
LANGUAGE = "en"

SAMPLE_RATE = 16000
CHUNK_MS = 500
WINDOW_SEC = 5.0
STEP_SEC = 5.0

OUT_TXT = "logs/output.txt"

DEFAULT_HOTWORDS_TXT = r"C:\Capston\default_hotwords.txt"
USE_SCREEN_OCR_HOTWORDS = True
SCREEN_OCR_INCLUDE_SKILLS = False