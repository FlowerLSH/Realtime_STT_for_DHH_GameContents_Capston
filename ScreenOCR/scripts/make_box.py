import os, glob, json
import cv2

# ===== 사용자 설정 =====
PROJECT_ROOT = r"C:\Nemo\ScreenOCR"
TARGET_W, TARGET_H = 1920, 1080   # 표준화 해상도
PROFILE_PATH = os.path.join(PROJECT_ROOT, "profiles", "default.json")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "data", "samples")
OUT_DIR = os.path.join(PROJECT_ROOT, "out")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)

# ===== 유틸 =====
def load_any_image(samples_dir):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(samples_dir, ext)))
    if not files:
        raise FileNotFoundError(f"No image found in {samples_dir}. Put one screenshot like frame1.png")
    return files[0]

def pct_rect_to_xyxy(W, H, box):
    x1 = int(W * box["x"])
    y1 = int(H * box["y"])
    x2 = int(x1 + W * box["w"])
    y2 = int(y1 + H * box["h"])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W-1, x2), min(H-1, y2)
    return x1, y1, x2, y2

def xyxy_to_pct(W, H, rect):
    x1,y1,x2,y2 = rect
    x = max(0, min(x1, x2)) / W
    y = max(0, min(y1, y2)) / H
    w = abs(x2 - x1) / W
    h = abs(y2 - y1) / H
    # 화면 밖 방지
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0 - x, w))
    h = max(0.0, min(1.0 - y, h))
    return dict(x=x, y=y, w=w, h=h)

def draw_rect(img, xyxy, color, label):
    x1,y1,x2,y2 = xyxy
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    cv2.putText(img, label, (x1, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def overlay_slots(img, bottom_xyxy, slots=10):
    x1,y1,x2,y2 = bottom_xyxy
    panel_w = x2 - x1
    slot_w = panel_w / max(1, slots)
    for i in range(1, slots):
        xi = int(x1 + i*slot_w)
        cv2.line(img, (xi, y1), (xi, y2), (255, 200, 0), 1)
    cv2.putText(img, "Slots: L1..L5 | R1..R5", (x1, min(img.shape[0]-10, y2+25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2, cv2.LINE_AA)

# ===== 메인 =====
def main():
    src_path = load_any_image(SAMPLES_DIR)
    bgr = cv2.imread(src_path)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {src_path}")

    # 표준화 리사이즈
    bgr = cv2.resize(bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    window = "Calibrate ROI (Left drag to draw) | Keys: T=Top, B=Bottom, S=Save, R=Reset, Q=Quit"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    # 상태
    last_rect = None
    top_rect_xyxy = None
    bottom_rect_xyxy = None
    dragging = False
    ix = iy = 0

    def mouse_cb(event, x, y, flags, param):
        nonlocal dragging, ix, iy, last_rect
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            ix, iy = x, y
            last_rect = (ix, iy, ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            last_rect = (ix, iy, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            last_rect = (ix, iy, x, y)

    cv2.setMouseCallback(window, mouse_cb)

    print("===== Calibrate ROI =====")
    print(f"Image: {src_path}")
    print("How to use:")
    print(" 1) 드래그로 사각형을 그립니다.")
    print(" 2) 'T' 키: 방금 그린 사각형을 TOP_BAR로 지정")
    print(" 3) 'B' 키: 방금 그린 사각형을 BOTTOM_PANEL로 지정")
    print(" 4) 'S' 키: 저장(JSON) 및 오버레이 PNG 저장")
    print(" 5) 'R' 키: 초기화")
    print(" 6) 'Q' 키: 종료")

    while True:
        canvas = bgr.copy()

        # 현재 그리는 사각형
        if last_rect is not None:
            x1,y1,x2,y2 = last_rect
            cv2.rectangle(canvas, (x1,y1), (x2,y2), (180,180,180), 1)

        # 고정된 사각형
        if top_rect_xyxy is not None:
            draw_rect(canvas, top_rect_xyxy, (0, 200, 255), "TOP_BAR")
        if bottom_rect_xyxy is not None:
            draw_rect(canvas, bottom_rect_xyxy, (0, 255, 0), "BOTTOM_PANEL")
            overlay_slots(canvas, bottom_rect_xyxy, slots=10)

        cv2.imshow(window, canvas)
        key = cv2.waitKey(16) & 0xFF

        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord('t'):
            if last_rect is not None:
                # 정렬
                x1,y1,x2,y2 = last_rect
                top_rect_xyxy = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
                print("TOP_BAR set.")
        elif key == ord('b'):
            if last_rect is not None:
                x1,y1,x2,y2 = last_rect
                bottom_rect_xyxy = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
                print("BOTTOM_PANEL set.")
        elif key == ord('r'):
            last_rect = None
            top_rect_xyxy = None
            bottom_rect_xyxy = None
            print("Reset.")
        elif key == ord('s'):
            if top_rect_xyxy is None or bottom_rect_xyxy is None:
                print("Both TOP and BOTTOM must be set before saving.")
                continue
            # 퍼센트로 저장
            top_pct = xyxy_to_pct(TARGET_W, TARGET_H, top_rect_xyxy)
            bottom_pct = xyxy_to_pct(TARGET_W, TARGET_H, bottom_rect_xyxy)
            profile = {
                "target_wh": [TARGET_W, TARGET_H],
                "top_bar": top_pct,
                "bottom_panel": bottom_pct,
                "player_slots": 10
            }
            with open(PROFILE_PATH, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
            print(f"Saved profile -> {PROFILE_PATH}")

            # 오버레이 저장
            ov = bgr.copy()
            draw_rect(ov, (int(top_rect_xyxy[0]), int(top_rect_xyxy[1]), int(top_rect_xyxy[2]), int(top_rect_xyxy[3])), (0,200,255), "TOP_BAR")
            draw_rect(ov, (int(bottom_rect_xyxy[0]), int(bottom_rect_xyxy[1]), int(bottom_rect_xyxy[2]), int(bottom_rect_xyxy[3])), (0,255,0), "BOTTOM_PANEL")
            overlay_slots(ov, bottom_rect_xyxy, slots=10)
            out_path = os.path.join(OUT_DIR, "frame1_overlay_from_profile.png")
            cv2.imwrite(out_path, ov)
            print(f"Saved overlay -> {out_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
