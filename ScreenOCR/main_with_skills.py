import os, glob, json, re
import cv2
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = r"C:\Nemo\ScreenOCR"
SAMPLES_DIR  = os.path.join(PROJECT_ROOT, "data", "samples")
PROFILE_5x2  = os.path.join(PROJECT_ROOT, "profiles", "layout_5x2.json")
PROFILE_ROI  = os.path.join(PROJECT_ROOT, "profiles", "default.json")
TOP_PLAN     = os.path.join(PROJECT_ROOT, "out", "top_plan.json")
LEX_PLAYERS  = os.path.join(PROJECT_ROOT, "assets", "players.txt")
LEX_TEAMS    = os.path.join(PROJECT_ROOT, "assets", "teams.txt")
LEX_LEAGUES  = os.path.join(PROJECT_ROOT, "assets", "leagues.txt")

TARGET_W, TARGET_H = 1920, 1080
TOPK = 3
CKPT_PATH   = r"C:\Nemo\ScreenOCR\runs\champion_cls_v2\model_e20.pth"
LABELS_JSON = r"C:\Nemo\ScreenOCR\runs\champion_cls_v2\labels.json"
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STRICT_PLAYERS = True
STRICT_TEAMS = True
STRICT_LEAGUE = True

ALLOWED_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- &"
WS_RE = re.compile(r"\s+")
KEEP_RE = re.compile(rf"[{re.escape(ALLOWED_CHARS)}]+")

TEAMS_ALLOW   = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.&-"
LEAGUES_ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.&-"
TEAM_PAD_PCT  = 0.18
LEAG_PAD_PCT  = 0.14
SCAN_OFFSETS  = [-0.04, 0.0, 0.04]

def load_any_image(samples_dir):
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.webp")
    files = []
    for e in exts: files.extend(glob.glob(os.path.join(samples_dir, e)))
    if not files: raise FileNotFoundError(f"No image found in {samples_dir}")
    return files[0]

def resize_to_target(bgr, w=TARGET_W, h=TARGET_H): 
    return cv2.resize(bgr, (w,h), interpolation=cv2.INTER_AREA)

def clamp_rect(rc, parent):
    x1,y1,x2,y2 = rc; px1,py1,px2,py2 = parent
    x1=max(px1,min(px2-1,int(x1))); y1=max(py1,min(py2-1,int(y1)))
    x2=max(px1+1,min(px2,int(x2)));  y2=max(py1+1,min(py2,int(y2)))
    if x2<=x1: x2=x1+1
    if y2<=y1: y2=y1+1
    return (x1,y1,x2,y2)

def expand_rect(rc, pad_pct, bounds):
    x1,y1,x2,y2 = rc; w=x2-x1; h=y2-y1
    px=int(w*pad_pct); py=int(h*pad_pct)
    return clamp_rect((x1-px, y1-py, x2+px, y2+py), bounds)

def offset_rect(rc, dx_pct, dy_pct, bounds):
    x1,y1,x2,y2 = rc; w=x2-x1; h=y2-y1
    dx=int(w*dx_pct); dy=int(h*dy_pct)
    return clamp_rect((x1+dx,y1+dy,x2+dx,y2+dy), bounds)

def pct_to_xyxy(W, H, pct):
    x1 = int(W*pct["x"]); y1 = int(H*pct["y"])
    x2 = x1 + int(W*pct["w"]); y2 = y1 + int(H*pct["h"])
    return clamp_rect((x1,y1,x2,y2),(0,0,W,H))

def pct_to_rect(parent_xyxy, pct):
    px1,py1,px2,py2 = parent_xyxy; pw=max(1,px2-px1); ph=max(1,py2-py1)
    x1=px1+int(pct.get("x",0.0)*pw); y1=py1+int(pct.get("y",0.0)*ph)
    x2=x1+int(pct.get("w",0.0)*pw);  y2=y1+int(pct.get("h",0.0)*ph)
    return clamp_rect((x1,y1,x2,y2), parent_xyxy)

def split_rows_2cols(bottom_xyxy, n_rows, center_ratio, center_gap_ratio, row_pad_y_ratio, lr_pad_x_ratio):
    bx1,by1,bx2,by2 = bottom_xyxy; W=bx2-bx1; H=by2-by1; row_h=H/n_rows; rows=[]
    for r in range(n_rows):
        ry1=by1+int(r*row_h); ry2=by1+int((r+1)*row_h); pad=int((ry2-ry1)*row_pad_y_ratio)
        ry1+=pad; ry2-=pad; cx=bx1+int(W*center_ratio); gap=int(W*center_gap_ratio); pad_x=int(W*lr_pad_x_ratio)
        left  = (bx1+pad_x, ry1, max(bx1+pad_x,cx-gap), ry2)
        right = (min(cx+gap,bx2-pad_x), ry1, bx2-pad_x, ry2)
        rows.append((clamp_rect(left,(bx1,by1,bx2,by2)), clamp_rect(right,(bx1,by1,bx2,by2))))
    return rows

def to_square(bgr):
    h,w=bgr.shape[:2]; s=max(h,w)
    pt=(s-h)//2; pb=s-h-pt; pl=(s-w)//2; pr=s-w-pl
    return cv2.copyMakeBorder(bgr, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(0,0,0))

class TorchClassifier:
    def __init__(self, ckpt_path, labels_json, img_size=128, device="cpu"):
        with open(labels_json,"r",encoding="utf-8") as f: meta=json.load(f)
        self.id2cls={int(k):v for k,v in meta["id2cls"].items()}
        m=models.resnet18(weights=None); m.fc=nn.Linear(m.fc.in_features, len(self.id2cls))
        ck=torch.load(ckpt_path, map_location=device); m.load_state_dict(ck["model"], strict=True)
        m.eval().to(device); self.m=m; self.device=device
        self.tf=transforms.Compose([
            transforms.Resize((img_size,img_size)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    @torch.no_grad()
    def predict_topk(self, bgr, topk=3):
        sq=to_square(bgr); sq=cv2.resize(sq,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_AREA)
        rgb=cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
        x=self.tf(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        prob=torch.softmax(self.m(x), dim=1)[0]; vals, idxs=torch.topk(prob, k=min(topk, prob.numel()))
        return [(self.id2cls[int(i.item())], float(v.item())) for v,i in zip(vals, idxs)]

class OCRBackend:
    def __init__(self, use_gpu=False):
        import easyocr
        self.reader=easyocr.Reader(['en'], gpu=use_gpu)
    def _clean(self, s):
        s="".join(KEEP_RE.findall(s)); return WS_RE.sub(" ", s).strip()
    def _pre_variants(self, img):
        outs=[]; h,w=img.shape[:2]; s=3 if max(h,w)<180 else 2
        up=cv2.resize(img,(w*s,h*s),interpolation=cv2.INTER_CUBIC); outs.append(up)
        gray=cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        tophat=cv2.morphologyEx(clahe, cv2.MORPH_TOPHAT, kernel, iterations=1)
        outs.append(cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR))
        thr=cv2.threshold(clahe,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        outs.append(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))
        inv=cv2.bitwise_not(thr); outs.append(cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR))
        blur=cv2.GaussianBlur(up,(0,0),1.0); sharp=cv2.addWeighted(up,1.8,blur,-0.8,0); outs.append(sharp)
        return outs
    def _easy(self, img, allow=None):
        res=self.reader.readtext(img, detail=1, paragraph=False, allowlist=allow,
                                 text_threshold=0.4, low_text=0.2, link_threshold=0.4,
                                 mag_ratio=2.0, slope_ths=0.9, ycenter_ths=0.7, height_ths=0.5, width_ths=0.1)
        if not res: return "", 0.0
        best=max(res, key=lambda x: float(x[2]) if len(x)>=3 else 0.0)
        return str(best[1]), float(best[2]) if len(best)>=3 else 0.0
    def candidates(self, img, allow=None, force_upper=False):
        outs=[]
        for v in self._pre_variants(img):
            t,c=self._easy(v, allow=allow); t=self._clean(t)
            if t: outs.append(((t.upper() if force_upper else t), c))
        return outs

def load_lexicon(path):
    if not os.path.isfile(path): return None
    with open(path,"r",encoding="utf-8") as f: it=[line.strip() for line in f if line.strip()]
    return it or None

def fuzzy_match_best(lex, s):
    try:
        from rapidfuzz import process, fuzz
    except: return None, 0.0
    if not lex or not s: return None, 0.0
    m=process.extractOne(s, lex, scorer=fuzz.WRatio)
    return (m[0], float(m[1])/100.0) if m else (None, 0.0)

def rerank_with_lexicon(ocr_candidates, lex, strict=False):
    if not ocr_candidates: return "", 0.0
    if strict and lex:
        try:
            from rapidfuzz import process, fuzz
        except:
            return max((t for t,_ in ocr_candidates), key=len, default=""), 0.0
        best_name=""; best_score=-1.0
        for t,_ in ocr_candidates:
            if not t: continue
            m=process.extractOne(t, lex, scorer=fuzz.WRatio)
            if m and m[1]>best_score: best_name, best_score=m[0], float(m[1])/100.0
        return best_name or "", best_score if best_name else 0.0
    best_txt=""; best_score=-1.0
    for t,c in ocr_candidates:
        name,l = fuzzy_match_best(lex, t) if lex else (None,0.0)
        ln=len(name) if name else len(t)
        lp=1.0 - abs(ln - len(t))/max(ln if ln>0 else 1, 1)
        s=0.6*c + 0.35*l + 0.05*lp
        out=name if name and (l>=0.70 or (c<=0.85 and l>=0.60)) else t
        if s>best_score: best_score=s; best_txt=out
    return best_txt, best_score

def rect_or_default(parent_xyxy, pct, default_pct): 
    return pct_to_rect(parent_xyxy, pct if pct else default_pct)

def maybe_rect(parent_xyxy, pct): 
    return None if not pct else pct_to_rect(parent_xyxy, pct)

def get_top_rects(bgr, top_xyxy, prof_layout):
    tl = prof_layout.get("top_layout", {})
    fb_teamL = rect_or_default(top_xyxy, tl.get("teamL_pct"), {"x":0.02,"y":0.10,"w":0.22,"h":0.80})
    fb_teamR = rect_or_default(top_xyxy, tl.get("teamR_pct"), {"x":0.76,"y":0.10,"w":0.22,"h":0.80})
    fb_lgL   = maybe_rect(top_xyxy, tl.get("leagueL_pct"))
    fb_lgR   = maybe_rect(top_xyxy, tl.get("leagueR_pct"))
    if os.path.isfile(TOP_PLAN):
        with open(TOP_PLAN,"r",encoding="utf-8") as f: tp=json.load(f)
        t=tp.get("top",{})
        def xy(k):
            v=t.get(k)
            if not v: return None
            x1,y1,x2,y2=map(int,v)
            return clamp_rect((x1,y1,x2,y2),(0,0,bgr.shape[1],bgr.shape[0]))
        teamL = xy("teamL_xyxy") or fb_teamL
        teamR = xy("teamR_xyxy") or fb_teamR
        lgL   = xy("leagueL_xyxy") or fb_lgL
        lgR   = xy("leagueR_xyxy") or fb_lgR
        return teamL, teamR, lgL, lgR, "top_plan.json(+fallback)"
    return fb_teamL, fb_teamR, fb_lgL, fb_lgR, "layout_5x2.json"

def ocr_box_with_scans(ocr, bgr, rc, kind, lex, strict, pad_pct):
    H,W=bgr.shape[:2]; base=expand_rect(rc, pad_pct, (0,0,W,H))
    best_txt=""; best_s=-1.0
    allow = TEAMS_ALLOW if kind=="teams" else (LEAGUES_ALLOW if kind=="leagues" else None)
    force_upper = kind in ("teams","leagues")
    def cand_fn(img, use_allow=True):
        return ocr.candidates(img, allow=(allow if use_allow else None), force_upper=force_upper)
    for dx in SCAN_OFFSETS:
        for dy in SCAN_OFFSETS:
            sub=offset_rect(base, dx, dy, (0,0,W,H)); crop=bgr[sub[1]:sub[3], sub[0]:sub[2]]
            cands=cand_fn(crop, use_allow=True)
            if not cands and allow is not None:
                cands=cand_fn(crop, use_allow=False)
            txt,s=rerank_with_lexicon(cands, lex, strict=strict)
            if s>best_s and txt: best_s=s; best_txt=txt
    if strict and (not best_txt):
        for dx in SCAN_OFFSETS:
            for dy in SCAN_OFFSETS:
                sub=offset_rect(base, dx, dy, (0,0,W,H)); crop=bgr[sub[1]:sub[3], sub[0]:sub[2]]
                cands=cand_fn(crop, use_allow=False)
                txt,s=rerank_with_lexicon(cands, lex, strict=False)
                if s>best_s and txt: best_s=s; best_txt=txt
    return best_txt, best_s


def _strip_bom(s: str) -> str:
    return s.lstrip("\ufeff").strip() if isinstance(s, str) else s

def load_champion_skills(paths):
    import csv, os
    m = {}
    for path in paths:
        if not path:
            continue
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    champ = _strip_bom(row[0])
                    skills = [x.strip() for x in row[1:] if x and x.strip()]
                    if champ:
                        m[champ] = skills
                        m[champ.lower()] = skills
                        alias = champ.replace(" ", "").replace("'", "").lower()
                        m[alias] = skills
            break
    return m


def main():
    __printed_champions = []
    src=load_any_image(SAMPLES_DIR); bgr=cv2.imread(src)
    if bgr is None: raise RuntimeError(f"fail read {src}")
    bgr=resize_to_target(bgr, TARGET_W, TARGET_H); H,W=bgr.shape[:2]
    with open(PROFILE_ROI,"r",encoding="utf-8") as f: prof_roi=json.load(f)
    with open(PROFILE_5x2,"r",encoding="utf-8") as f: prof=json.load(f)

    bot_xyxy=pct_to_xyxy(W,H, prof_roi["bottom_panel"])
    top_xyxy=pct_to_xyxy(W,H, prof_roi["top_panel"] if "top_panel" in prof_roi else {"x":0.0,"y":0.0,"w":1.0,"h":0.12})

    n_rows=prof["n_rows"]; center_ratio=prof["center_ratio"]; center_gap_ratio=prof["center_gap_ratio"]
    row_pad_y_ratio=prof["row_pad_y_ratio"]; lr_pad_x_ratio=prof["lr_pad_x_ratio"]; layout=prof["layout"]

    clf=TorchClassifier(CKPT_PATH, LABELS_JSON, IMG_SIZE, DEVICE)
    try: 
        ocr=OCRBackend(use_gpu=(DEVICE=="cuda"))
    except: 
        ocr=OCRBackend(use_gpu=False)
    lex_players=load_lexicon(LEX_PLAYERS); lex_teams=load_lexicon(LEX_TEAMS); lex_leagues=load_lexicon(LEX_LEAGUES)

    tL_rc, tR_rc, lgL_rc, lgR_rc, src_tag = get_top_rects(bgr, top_xyxy, prof)

    tL_txt,_ = ocr_box_with_scans(ocr, bgr, tL_rc, "teams",   lex_teams,   strict=STRICT_TEAMS,   pad_pct=TEAM_PAD_PCT) if tL_rc else ("",0.0)
    tR_txt,_ = ocr_box_with_scans(ocr, bgr, tR_rc, "teams",   lex_teams,   strict=STRICT_TEAMS,   pad_pct=TEAM_PAD_PCT) if tR_rc else ("",0.0)
    lgL_txt,_= ocr_box_with_scans(ocr, bgr, lgL_rc, "leagues",lex_leagues, strict=STRICT_LEAGUE,  pad_pct=LEAG_PAD_PCT) if lgL_rc else ("",0.0)
    lgR_txt,_= ocr_box_with_scans(ocr, bgr, lgR_rc, "leagues",lex_leagues, strict=STRICT_LEAGUE,  pad_pct=LEAG_PAD_PCT) if lgR_rc else ("",0.0)

    rows=split_rows_2cols(bot_xyxy, n_rows, center_ratio, center_gap_ratio, row_pad_y_ratio, lr_pad_x_ratio)
    results=[]
    for r,(left_base,right_base) in enumerate(rows):
        for side, base in (("BLUE", left_base), ("RED", right_base)):
            lay=layout["L"] if side=="BLUE" else layout["R"]
            slot_inner=pct_to_rect(base, lay["slot_inner_pct"])
            portrait_rc=pct_to_rect(slot_inner, lay["portrait_pct"])
            nickname_rc=pct_to_rect(base, lay["nickname_pct"])
            x1,y1,x2,y2=portrait_rc; crop=bgr[y1:y2, x1:x2]
            if crop is None or crop.size==0: results.append((side,r+1,"","EMPTY",0.0,[])); continue
            topk=clf.predict_topk(crop, topk=TOPK); name,prob=topk[0]
            nx1,ny1,nx2,ny2=nickname_rc; nick_crop=bgr[ny1:ny2, nx1:nx2]
            cands=ocr.candidates(nick_crop, allow=None, force_upper=False)
            nick_txt,_=rerank_with_lexicon(cands, lex_players, strict=STRICT_PLAYERS)
            results.append((side,r+1,nick_txt,name,prob,topk))

    results_sorted=sorted(results, key=lambda t:(0 if t[0] in ("BLUE","L") else 1, t[1]))

    print("\n=== League / Teams ===")
    print(f"CoordSource: {src_tag}")
    print(f"Teams: L={tL_txt or '-'} | R={tR_txt or '-'}")
    print(f"Leagues: L={lgL_txt or '-'} | R={lgR_txt or '-'}")

    print("\n=== Side  NickName  Top-1 ===")
    print(f"Image: {os.path.basename(src)} | Model: {os.path.basename(CKPT_PATH)} | OCR: easyocr")
    print(f"{'Side':<6} {'NickName':<20} {'Top-1':<18}")
    print("-"*50)
    for side, row, nick, name, prob, topk in results_sorted:
        print(f"{side:<6} {nick[:20]:<20} {name:<18}")
        __printed_champions.append(name)


    # === Append champion skills from CSV at the end of stdout ===
    skills_map = load_champion_skills([
        os.path.join(PROJECT_ROOT, "assets", "champion_skills.csv"),
        os.path.join(PROJECT_ROOT, "champion_skills.csv"),
        os.path.join(os.path.dirname(__file__), "champion_skills.csv"),
    ])
    if skills_map:
        seen = set()
        ordered = []
        for c in __printed_champions:
            if not c or c in ("-", "EMPTY"):
                continue
            if c not in seen:
                seen.add(c); ordered.append(c)
        if ordered:
            print("해당 챔피언의 스킬들")
            for champ in ordered:
                kv = [champ, champ.lower(), champ.replace(" ", "").replace("'", "").lower()]
                skills = None
                for k in kv:
                    if k in skills_map:
                        skills = skills_map[k]; break
                if skills:
                    print(f"{champ}: " + ", ".join(skills))
                else:
                    print(f"{champ}: (skills not found)")
if __name__ == "__main__":
    main()
