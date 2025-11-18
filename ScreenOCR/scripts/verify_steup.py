import os, glob
import cv2
import numpy as np

# ===== 사용자 설정(필요 시 조정) =====
PROJECT_ROOT = r"C:\Nemo\ScreenOCR"
TARGET_W, TARGET_H = 1920, 1080   # 표준화 해상도

# 상단바/하단패널 대략적 위치(퍼센트, 0~1)
# 방송 스킨마다 다르므로 Step 2에서 정밀 조정할 예정. 지금은 대략만.
TOP_BAR = dict(x=0.05, y=0.02, w=0.90, h=0.08)
BOTTOM_PANEL = dict(x=0.02, y=0.80, w=0.96, h=0.18)  # h가 화면 아래로 살짝 넘칠 수 있어 Step 2에서 미세조정 예정

# 하단 패널 10칸 분할(좌 5, 우 5) 가정
PLAYER_SLOTS = 10

# ===== 경로 =====
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "data", "samples")
OUT_DIR = os.path.join(PROJECT_ROOT, "out")
os.makedirs(OUT_DIR, exist_ok=True)

def load_any_image(samples_dir):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(samples_dir, ext)))
    if not files:
        raise FileNotFoundError(f"No image found in {samples_dir}. Put one screenshot like frame1.png")
    return files[0]

def resize_to_target(img, tw, th):
    # 단순 리사이즈 (Step 2에서 필요 시 letterbox/크롭 교체)
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

def pct_rect_to_xyxy(W, H, box):
    x1 = int(W * box["x"])
    y1 = int(H * box["y"])
    x2 = int(x1 + W * box["w"])
    y2 = int(y1 + H * box["h"])
    # 화면 밖으로 나가지 않게 클램프
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W-1, x2), min(H-1, y2)
    return x1, y1, x2, y2

def draw_rect(img, xyxy, color=(0,255,0), thickness=2, label=None):
    x1,y1,x2,y2 = xyxy
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, max(20, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def main():
    src_path = load_any_image(SAMPLES_DIR)
    img = cv2.imread(src_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {src_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 표준화 해상도 리사이즈
    std = resize_to_target(img, TARGET_W, TARGET_H)

    # 사본 저장
    bgr_std = cv2.cvtColor(std, cv2.COLOR_RGB2BGR)
    out_std = os.path.join(OUT_DIR, "frame1_1080.png")
    cv2.imwrite(out_std, bgr_std)

    # 오버레이 이미지
    overlay = bgr_std.copy()

    # 상단바/하단패널 박스 표시
    top_xyxy = pct_rect_to_xyxy(TARGET_W, TARGET_H, TOP_BAR)
    bot_xyxy = pct_rect_to_xyxy(TARGET_W, TARGET_H, BOTTOM_PANEL)
    draw_rect(overlay, top_xyxy, color=(0, 200, 255), thickness=3, label="TOP_BAR(rough)")
    draw_rect(overlay, bot_xyxy, color=(0, 255, 0), thickness=3, label="BOTTOM_PANEL(rough)")

    # 하단 패널 10칸 등분선(대략)
    x1,y1,x2,y2 = bot_xyxy
    panel_w = x2 - x1
    slot_w = panel_w / PLAYER_SLOTS
    for i in range(1, PLAYER_SLOTS):
        xi = int(x1 + i*slot_w)
        cv2.line(overlay, (xi, y1), (xi, y2), (255, 200, 0), 1)
    cv2.putText(overlay, "Slots: L1..L5 | R1..R5", (x1, min(TARGET_H-10, y2+25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2, cv2.LINE_AA)

    out_overlay = os.path.join(OUT_DIR, "frame1_overlay.png")
    cv2.imwrite(out_overlay, overlay)

    print("===== Step 1 Result =====")
    print(f"Input:  {src_path}")
    print(f"Saved:  {out_std}")
    print(f"Saved:  {out_overlay}")
    print("Open the overlay image and check if the TOP_BAR and BOTTOM_PANEL roughly align with your broadcast frame.")

if __name__ == "__main__":
    main()
