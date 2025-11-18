import os, json, glob, cv2

PROJECT_ROOT = r"C:\Nemo\ScreenOCR"
PROFILE_ROI   = os.path.join(PROJECT_ROOT, "profiles", "default.json")
PROFILE_LAYOUT= os.path.join(PROJECT_ROOT, "profiles", "layout_5x2.json")
SAMPLES_DIR   = os.path.join(PROJECT_ROOT, "data", "samples")
OUT_DIR       = os.path.join(PROJECT_ROOT, "out")
TARGET_W, TARGET_H = 1920, 1080

TOP_DEFAULTS = {
    "teamL_pct": {"x":0.02,"y":0.10,"w":0.22,"h":0.80},
    "teamR_pct": {"x":0.76,"y":0.10,"w":0.22,"h":0.80},
    "league_pct":{"x":0.44,"y":0.00,"w":0.12,"h":1.00},
    "leagueL_pct":{"x":0.02,"y":0.00,"w":0.12,"h":1.00},
    "leagueR_pct":{"x":0.86,"y":0.00,"w":0.12,"h":1.00}
}

def load_any_image(d):
    exts=("*.png","*.jpg","*.jpeg","*.bmp","*.webp")
    files=[]
    for e in exts: files.extend(glob.glob(os.path.join(d,e)))
    if not files: raise FileNotFoundError(f"No image in {d}")
    return files[0]

def clamp_rect(rc, parent):
    x1,y1,x2,y2 = rc
    px1,py1,px2,py2 = parent
    x1=max(px1,min(px2-1,int(x1))); y1=max(py1,min(py2-1,int(y1)))
    x2=max(px1+1,min(px2,int(x2)));  y2=max(py1+1,min(py2,int(y2)))
    if x2<=x1: x2=x1+1
    if y2<=y1: y2=y1+1
    return (int(x1),int(y1),int(x2),int(y2))

def pct_to_rect(parent_xyxy, pct):
    px1,py1,px2,py2=parent_xyxy
    pw=max(1,px2-px1); ph=max(1,py2-py1)
    x1=px1+int(pct.get("x",0.0)*pw); y1=py1+int(pct.get("y",0.0)*ph)
    x2=x1+int(pct.get("w",0.0)*pw);  y2=y1+int(pct.get("h",0.0)*ph)
    return clamp_rect((x1,y1,x2,y2), parent_xyxy)

def rect_to_pct(parent_xyxy, child_xyxy):
    px1,py1,px2,py2=parent_xyxy
    x1,y1,x2,y2=clamp_rect(child_xyxy, parent_xyxy)
    pw=max(1,px2-px1); ph=max(1,py2-py1)
    x=(x1-px1)/pw; y=(y1-py1)/ph; w=(x2-x1)/pw; h=(y2-y1)/ph
    return {"x":max(0,min(1,x)),"y":max(0,min(1,y)),"w":max(0,min(1-x,w)),"h":max(0,min(1-y,h))}

def mirror_pct(p):
    return {"x": max(0.0, min(1.0, 1.0 - (p["x"]+p["w"]))), "y": p["y"], "w": p["w"], "h": p["h"]}

def draw_box(img, rc, color, label=None, thick=2):
    x1,y1,x2,y2=map(int,rc)
    cv2.rectangle(img,(x1,y1),(x2,y2),color,thick)
    if label:
        cv2.putText(img,label,(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2,cv2.LINE_AA)

def main():
    src=load_any_image(SAMPLES_DIR)
    bgr=cv2.imread(src); bgr=cv2.resize(bgr,(TARGET_W,TARGET_H))
    H,W=bgr.shape[:2]
    with open(PROFILE_ROI,"r",encoding="utf-8") as f: prof=json.load(f)
    def pct2xyxy(b):
        x1=int(W*b["x"]); y1=int(H*b["y"]); x2=int(x1+W*b["w"]); y2=int(y1+H*b["h"])
        return clamp_rect((x1,y1,x2,y2),(0,0,W,H))
    top_key="top_panel" if "top_panel" in prof else ("top_bar" if "top_bar" in prof else None)
    if top_key is None: raise KeyError("default.json must include top_panel or top_bar")
    top_rc=pct2xyxy(prof[top_key])

    if os.path.isfile(PROFILE_LAYOUT):
        with open(PROFILE_LAYOUT,"r",encoding="utf-8") as f:
            layout=json.load(f)
            top_layout=layout.get("top_layout", TOP_DEFAULTS)
    else:
        layout={}
        top_layout=dict(TOP_DEFAULTS)

    last_rect=None; dragging=False; ix=iy=0
    win="TopBar Calib"
    def mouse_cb(e,x,y,f,p):
        nonlocal dragging, ix, iy, last_rect
        if e==cv2.EVENT_LBUTTONDOWN: dragging=True; ix,iy=x,y; last_rect=(x,y,x,y)
        elif e==cv2.EVENT_MOUSEMOVE and dragging: last_rect=(ix,iy,x,y)
        elif e==cv2.EVENT_LBUTTONUP: dragging=False; last_rect=(ix,iy,x,y)
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE); cv2.setMouseCallback(win, mouse_cb)

    def build():
        canvas=bgr.copy()
        draw_box(canvas, top_rc,(0,200,255),"TOP_BAR")
        tL=pct_to_rect(top_rc, top_layout["teamL_pct"])
        tR=pct_to_rect(top_rc, top_layout["teamR_pct"])
        draw_box(canvas, tL,(200,220,255),"teamL")
        draw_box(canvas, tR,(200,220,255),"teamR")
        if "league_pct" in top_layout:
            lg=pct_to_rect(top_rc, top_layout["league_pct"])
            draw_box(canvas, lg,(200,180,255),"league")
        if "leagueL_pct" in top_layout:
            ll=pct_to_rect(top_rc, top_layout["leagueL_pct"])
            draw_box(canvas, ll,(180,255,180),"leagueL")
        if "leagueR_pct" in top_layout:
            lr=pct_to_rect(top_rc, top_layout["leagueR_pct"])
            draw_box(canvas, lr,(180,255,180),"leagueR")
        if last_rect is not None: draw_box(canvas,last_rect,(180,180,180),"last",1)
        y=22
        txt="Drag | Y:teamL U:teamR L:leagueL R:leagueR I:league M:mirror L->R P:mirror teamL->teamR S:save Q:exit"
        cv2.putText(canvas,txt,(10,y),cv2.FONT_HERSHEY_PLAIN,1.0,(240,240,240),1,cv2.LINE_AA)
        return canvas

    while True:
        cv2.imshow(win, build())
        k=cv2.waitKey(16)&0xFF
        if k in (27, ord('q')): break
        elif k==ord('y') and last_rect is not None:
            top_layout["teamL_pct"]=rect_to_pct(top_rc, last_rect)
        elif k==ord('u') and last_rect is not None:
            top_layout["teamR_pct"]=rect_to_pct(top_rc, last_rect)
        elif k==ord('l') and last_rect is not None:
            top_layout["leagueL_pct"]=rect_to_pct(top_rc, last_rect)
        elif k==ord('r') and last_rect is not None:
            top_layout["leagueR_pct"]=rect_to_pct(top_rc, last_rect)
        elif k==ord('i') and last_rect is not None:
            top_layout["league_pct"]=rect_to_pct(top_rc, last_rect)
        elif k==ord('m'):
            if "leagueL_pct" in top_layout:
                top_layout["leagueR_pct"]=mirror_pct(top_layout["leagueL_pct"])
        elif k==ord('p'):
            top_layout["teamR_pct"]=mirror_pct(top_layout["teamL_pct"])
        elif k==ord('s'):
            os.makedirs(OUT_DIR, exist_ok=True)
            layout["top_layout"]=top_layout
            with open(PROFILE_LAYOUT,"w",encoding="utf-8") as f:
                json.dump(layout, f, ensure_ascii=False, indent=2)
            ov=bgr.copy()
            draw_box(ov, top_rc,(0,200,255),"TOP_BAR")
            tL=pct_to_rect(top_rc, top_layout["teamL_pct"]); draw_box(ov, tL,(200,220,255),"teamL")
            tR=pct_to_rect(top_rc, top_layout["teamR_pct"]); draw_box(ov, tR,(200,220,255),"teamR")
            plan={"image":os.path.basename(src),"target_wh":[W,H],"top_bar_xyxy":list(map(int,top_rc)),"top":{}}
            if "league_pct" in top_layout:
                lg=pct_to_rect(top_rc, top_layout["league_pct"])
                draw_box(ov, lg,(200,180,255),"league")
                plan["top"]["league_xyxy"]=list(map(int,lg))
            if "leagueL_pct" in top_layout:
                ll=pct_to_rect(top_rc, top_layout["leagueL_pct"])
                draw_box(ov, ll,(180,255,180),"leagueL")
                plan["top"]["leagueL_xyxy"]=list(map(int,ll))
            if "leagueR_pct" in top_layout:
                lr=pct_to_rect(top_rc, top_layout["leagueR_pct"])
                draw_box(ov, lr,(180,255,180),"leagueR")
                plan["top"]["leagueR_xyxy"]=list(map(int,lr))
            cv2.imwrite(os.path.join(OUT_DIR,"top_overlay.png"), ov)
            with open(os.path.join(OUT_DIR,"top_plan.json"),"w",encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
            print("Saved:", PROFILE_LAYOUT, os.path.join(OUT_DIR,"top_overlay.png"), os.path.join(OUT_DIR,"top_plan.json"))
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
