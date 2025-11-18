import os, glob, json, re
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
LEX_TXT      = os.path.join(PROJECT_ROOT, "assets", "players.txt")

TARGET_W, TARGET_H = 1920, 1080
TOPK = 3
CKPT_PATH   = r"C:\Nemo\ScreenOCR\runs\champion_cls_v2\model_e20.pth"
LABELS_JSON = r"C:\Nemo\ScreenOCR\runs\champion_cls_v2\labels.json"
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STRICT_LEXICON = True

ALLOWED_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- "
WS_RE = re.compile(r"\s+")
KEEP_RE = re.compile(rf"[{re.escape(ALLOWED_CHARS)}]+")

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
    x1 = max(px1, min(px2-1, int(x1))); y1 = max(py1, min(py2-1, int(y1)))
    x2 = max(px1+1, min(px2,   int(x2))); y2 = max(py1+1, min(py2,   int(y2)))
    if x2<=x1: x2=x1+1
    if y2<=y1: y2=y1+1
    return (x1,y1,x2,y2)

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

class OCRBackend:
    def __init__(self, use_gpu=False):
        try:
            import easyocr
        except Exception as e:
            raise RuntimeError("easyocr is required. Install with: pip install easyocr") from e
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        self.kind = "easyocr"

    def _variants(self, img):
        outs=[]
        h,w=img.shape[:2]
        s = 2 if max(h,w)<220 else 1
        up=cv2.resize(img,(w*s,h*s),interpolation=cv2.INTER_CUBIC)
        outs.append(up)
        gray=cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        outs.append(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))
        thr=cv2.threshold(clahe,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        outs.append(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))
        inv=cv2.bitwise_not(thr)
        outs.append(cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR))
        sharp=cv2.GaussianBlur(up,(0,0),1.0)
        sharp=cv2.addWeighted(up,1.8,sharp,-0.8,0)
        outs.append(sharp)
        return outs

    def _clean(self, s):
        s = "".join(KEEP_RE.findall(s))
        return WS_RE.sub(" ", s).strip()

    def _easy(self, img):
        res=self.reader.readtext(img, detail=1, paragraph=False)
        if not res: return "", 0.0
        best=max(res, key=lambda x: float(x[2]) if len(x)>=3 else 0.0)
        return str(best[1]), float(best[2]) if len(best)>=3 else 0.0

    def candidates(self, img):
        outs=[]
        for v in self._variants(img):
            t,c = self._easy(v)
            t=self._clean(t)
            if t: outs.append((t,c))
        return outs

def load_lexicon(path):
    if not os.path.isfile(path): return None
    with open(path,"r",encoding="utf-8") as f:
        items=[line.strip() for line in f if line.strip()]
    return items or None

def fuzzy_match_best(lex, s):
    try:
        from rapidfuzz import process, fuzz
    except Exception:
        return None, 0.0
    if not lex or not s: return None, 0.0
    m=process.extractOne(s, lex, scorer=fuzz.WRatio)
    if not m: return None, 0.0
    return m[0], float(m[1])/100.0

def rerank_with_lexicon(ocr_candidates, lex, strict=False):
    if not ocr_candidates:
        return "", 0.0
    if strict and lex:
        best_name=""; best_score=-1.0
        try:
            from rapidfuzz import process, fuzz
        except Exception:
            return max((t for t,_ in ocr_candidates), key=len, default=""), 0.0
        for t,_ in ocr_candidates:
            if not t: continue
            m=process.extractOne(t, lex, scorer=fuzz.WRatio)
            if m and m[1]>best_score:
                best_name, best_score = m[0], float(m[1])/100.0
        return best_name or "", best_score if best_name else 0.0
    best_txt=""; best_score=-1.0
    for t,c in ocr_candidates:
        name, l = fuzzy_match_best(lex, t) if lex else (None,0.0)
        ln = len(name) if name else len(t)
        lp = 1.0 - abs(ln - len(t))/max(ln if ln>0 else 1, 1)
        s = 0.6*c + 0.35*l + 0.05*lp
        out = name if name and (l>=0.70 or (c<=0.85 and l>=0.60)) else t
        if s>best_score:
            best_score=s; best_txt=out
    return best_txt, best_score

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

    bot_xyxy = pct_to_xyxy(W,H, prof_roi["bottom_panel"])
    n_rows = prof["n_rows"]
    center_ratio     = prof["center_ratio"]
    center_gap_ratio = prof["center_gap_ratio"]
    row_pad_y_ratio  = prof["row_pad_y_ratio"]
    lr_pad_x_ratio   = prof["lr_pad_x_ratio"]
    layout           = prof["layout"]

    clf = TorchClassifier(CKPT_PATH, LABELS_JSON, IMG_SIZE, DEVICE)
    ocr = OCRBackend(use_gpu=(DEVICE=="cuda"))
    lex = load_lexicon(LEX_TXT)

    rows = split_rows_2cols(bot_xyxy, n_rows, center_ratio, center_gap_ratio, row_pad_y_ratio, lr_pad_x_ratio)
    results = []
    for r, (left_base, right_base) in enumerate(rows):
        for side, base in (("BLUE", left_base), ("RED", right_base)):
            lay = layout["L"] if side=="BLUE" else layout["R"]
            slot_inner = pct_to_rect(base, lay["slot_inner_pct"])
            portrait_rc= pct_to_rect(slot_inner, lay["portrait_pct"])
            nickname_rc= pct_to_rect(base, lay["nickname_pct"])
            x1,y1,x2,y2 = portrait_rc
            crop = bgr[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                results.append((side, r+1, "", "EMPTY", 0.0, []))
                continue
            topk = clf.predict_topk(crop, topk=TOPK)
            name, prob = topk[0]
            nx1,ny1,nx2,ny2 = nickname_rc
            nick_crop = bgr[ny1:ny2, nx1:nx2]
            cands = ocr.candidates(nick_crop)
            nick_txt, _ = rerank_with_lexicon(cands, lex, strict=STRICT_LEXICON)
            results.append((side, r+1, nick_txt, name, prob, topk))

    def sort_key(t):
        side = t[0]; row=t[1]
        side_ord = 0 if side in ("BLUE","L") else 1
        return (side_ord, row)
    results_sorted = sorted(results, key=sort_key)

    print("\n=== Side  NickName  Top-1 ===")
    print(f"Image: {os.path.basename(src)} | Model: {os.path.basename(CKPT_PATH)} | OCR: easyocr | Lexicon: {os.path.basename(LEX_TXT) if lex else 'None'} | StrictLex: {STRICT_LEXICON}\n")
    print(f"{'Side':<6} {'NickName':<20} {'Top-1':<18}")
    print("-"*50)
    for side, row, nick, name, prob, topk in results_sorted:
        print(f"{side:<6} {nick[:20]:<20} {name:<18}")

if __name__ == "__main__":
    main()
