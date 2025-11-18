import os, glob, json, time
from typing import List, Tuple
import numpy as np
import cv2
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = r"C:\Nemo\ScreenOCR"
SAMPLES_DIR  = os.path.join(PROJECT_ROOT, "data", "samples")
PROFILE_5x2  = os.path.join(PROJECT_ROOT, "profiles", "layout_5x2.json")
PROFILE_ROI  = os.path.join(PROJECT_ROOT, "profiles", "default.json")
TARGET_W, TARGET_H = 1920, 1080
TOPK = 3

CKPT_PATH   = r"C:\Nemo\ScreenOCR\runs\champion_cls_v2\model_e20.pth"
LABELS_JSON = r"C:\Nemo\ScreenOCR\runs\champion_cls_v2\labels.json"
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_any_image(samples_dir):
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.webp")
    files = []
    for e in exts: files.extend(glob.glob(os.path.join(samples_dir, e)))
    if not files: raise FileNotFoundError(f"No image found in {samples_dir}")
    return files[0]

def resize_to_target(bgr, w=TARGET_W, h=TARGET_H):
    return cv2.resize(bgr, (w,h), interpolation=cv2.INTER_AREA)

def clamp_rect(rc, parent):
    x1,y1,x2,y2 = rc
    px1,py1,px2,py2 = parent
    x1 = max(px1, min(px2-1, x1)); y1 = max(py1, min(py2-1, y1))
    x2 = max(px1+1, min(px2, x2));  y2 = max(py1+1, min(py2, y2))
    if x2<=x1: x2=x1+1
    if y2<=y1: y2=y1+1
    return (int(x1),int(y1),int(x2),int(y2))

def pct_to_xyxy(W, H, pct):
    x1 = int(W*pct["x"]); y1 = int(H*pct["y"])
    x2 = x1 + int(W*pct["w"]); y2 = y1 + int(H*pct["h"])
    return clamp_rect((x1,y1,x2,y2),(0,0,W,H))

def pct_to_rect(parent_xyxy, pct):
    px1,py1,px2,py2 = parent_xyxy
    pw = max(1, px2-px1); ph = max(1, py2-py1)
    x1 = px1 + int(pct.get("x",0.0)*pw)
    y1 = py1 + int(pct.get("y",0.0)*ph)
    x2 = x1 + int(pct.get("w",0.0)*pw)
    y2 = y1 + int(pct.get("h",0.0)*ph)
    return clamp_rect((x1,y1,x2,y2), parent_xyxy)

def split_rows_2cols(bottom_xyxy, n_rows, center_ratio, center_gap_ratio, row_pad_y_ratio, lr_pad_x_ratio):
    bx1,by1,bx2,by2 = bottom_xyxy
    W = bx2 - bx1; H = by2 - by1
    row_h = H / n_rows
    rows = []
    for r in range(n_rows):
        ry1 = by1 + int(r*row_h)
        ry2 = by1 + int((r+1)*row_h)
        pad_y = int((ry2-ry1)*row_pad_y_ratio)
        ry1 += pad_y; ry2 -= pad_y
        cx = bx1 + int(W*center_ratio)
        gap = int(W*center_gap_ratio)
        pad_x = int(W*lr_pad_x_ratio)
        left  = (bx1+pad_x, ry1, max(bx1+pad_x, cx-gap), ry2)
        right = (min(cx+gap, bx2-pad_x), ry1, bx2-pad_x, ry2)
        rows.append((clamp_rect(left,  (bx1,by1,bx2,by2)),
                     clamp_rect(right, (bx1,by1,bx2,by2))))
    return rows

def to_square(bgr):
    h, w = bgr.shape[:2]
    s = max(h, w)
    pad_top = (s - h)//2
    pad_bottom = s - h - pad_top
    pad_left = (s - w)//2
    pad_right = s - w - pad_left
    return cv2.copyMakeBorder(bgr, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))

class TorchClassifier:
    def __init__(self, ckpt_path, labels_json, img_size=128, device="cpu"):
        with open(labels_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.id2cls = {int(k): v for k,v in meta["id2cls"].items()}
        num_classes = len(self.id2cls)
        self.tf = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.device = device
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        ck = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(ck["model"], strict=True)
        m.eval().to(device)
        self.m = m

    @torch.no_grad()
    def predict_topk(self, bgr, topk=3):
        sq = to_square(bgr)
        sq = cv2.resize(sq, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
        x = self.tf(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        prob = torch.softmax(self.m(x), dim=1)[0]
        vals, idxs = torch.topk(prob, k=min(topk, prob.numel()))
        out = [(self.id2cls[int(i.item())], float(v.item())) for v,i in zip(vals, idxs)]
        return out

def main():
    src = load_any_image(SAMPLES_DIR)
    bgr = cv2.imread(src)
    if bgr is None: raise RuntimeError(f"fail read {src}")
    bgr = resize_to_target(bgr, TARGET_W, TARGET_H)
    H,W = bgr.shape[:2]

    with open(PROFILE_ROI, "r", encoding="utf-8") as f:
        prof_roi = json.load(f)
    with open(PROFILE_5x2, "r", encoding="utf-8") as f:
        prof = json.load(f)

    top_xyxy = pct_to_xyxy(W,H, prof_roi["top_bar"])
    bot_xyxy = pct_to_xyxy(W,H, prof_roi["bottom_panel"])
    n_rows = prof["n_rows"]
    center_ratio     = prof["center_ratio"]
    center_gap_ratio = prof["center_gap_ratio"]
    row_pad_y_ratio  = prof["row_pad_y_ratio"]
    lr_pad_x_ratio   = prof["lr_pad_x_ratio"]
    layout           = prof["layout"]

    clf = TorchClassifier(CKPT_PATH, LABELS_JSON, IMG_SIZE, DEVICE)

    rows = split_rows_2cols(bot_xyxy, n_rows, center_ratio, center_gap_ratio, row_pad_y_ratio, lr_pad_x_ratio)
    results = []
    for r, (left_base, right_base) in enumerate(rows):
        for side, base in (("BLUE", left_base), ("RED", right_base)):
            lay = layout["L"] if side=="BLUE" else layout["R"]
            slot_inner = pct_to_rect(base, lay["slot_inner_pct"])
            portrait_rc= pct_to_rect(slot_inner, lay["portrait_pct"])
            x1,y1,x2,y2 = portrait_rc
            crop = bgr[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                results.append((r+1, side, "EMPTY", 0.0, []))
                continue
            topk = clf.predict_topk(crop, topk=TOPK)
            name, prob = topk[0]
            results.append((r+1, side, name, prob, topk))

    def sort_key(t):
        row, side = t[0], t[1]
        side_ord = 0 if side=="BLUE" else 1
        return (side_ord, row)
    results_sorted = sorted(results, key=sort_key)

    print("\n=== Champion Recognition (Classifier) ===")
    print(f"Image: {os.path.basename(src)}")
    print(f"Model: {os.path.basename(CKPT_PATH)} | Labels: {os.path.basename(LABELS_JSON)}\n")
    header = f"{'Side':<6} {'Row':<3} {'Top-1':<18} {'Prob':>6}    Top-K candidates"
    print(header)
    print("-"*len(header))
    for row, side, name, prob, topk in results_sorted:
        tk = ", ".join([f"{n}({p:.2f})" for n, p in topk])
        print(f"{side:<6} {row:<3} {name:<18} {prob:>6.2f}    {tk}")
    print("\nDone.")

if __name__ == "__main__":
    main()
