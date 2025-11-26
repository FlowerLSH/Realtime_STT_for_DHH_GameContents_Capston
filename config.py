MODEL_NAME = "large-v3-turbo"
STT_BACKEND_TYPE = "whisper_faster"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
LANGUAGE = "en"

INCLUDE_TIME_DATA = True

SAMPLE_RATE = 16000
CHUNK_MS = 500
WINDOW_SEC = 5.0
STEP_SEC = 1.0

OUT_TXT = "./logs"

LABEL_THRESHOLD = 0.4

CHAMP_THRESHOLD = 0.9
RETRY_MAX = 1
RETRY_DELAY = 0.3
MIN_CLS_PROB = 0.55
MIN_OCR_SCORE = 0.65
CHAMP_RETRY_WAIT = 5.0

DEFAULT_HOTWORDS_TXT = r"C:\Capston\default_hotwords.txt"
USE_SCREEN_OCR_HOTWORDS = False
SCREEN_OCR_INCLUDE_SKILLS = False

EVENT_PRIORITY = {
    "scream": 5,
    "cheer": 4,
    "clap": 3,
    "laugh": 2,
    "sigh": 1,
}

# EVENT_TAGS = {
#     "sigh": "(한숨) ",
#     "laugh": "(웃음) ",
#     "clap": "(박수) ",
#     "cheer": "(환호) ",
#     "scream": "(비명) ",
# }
