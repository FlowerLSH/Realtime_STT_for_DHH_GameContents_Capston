import os, json, glob, cv2

PROJECT_ROOT = r"C:\Nemo\ScreenOCR"
PROFILE_ROI   = os.path.join(PROJECT_ROOT, "profiles", "default.json")
PROFILE_LAYOUT= os.path.join(PROJECT_ROOT, "profiles", "layout_5x2.json")
SAMPLES_DIR   = os.path.join(PROJECT_ROOT, "data", "samples")
OUT_DIR       = os.path.join(PROJECT_ROOT, "out")
TARGET_W, TARGET_H = 1920, 1080

N_ROWS = 5
CENTER_RATIO_INIT = 0.50
CENTER_GAP_RATIO  = 0.004
ROW_PAD_Y_RATIO   = 0.04
LEFT_PAD_X_RATIO  = 0.01

DEFAULTS_L = {
    "slot_inner_pct": {"x":0.02, "y":0.10, "w":0.96, "h":0.80},
    "portrait_pct"  : {"x":0.00, "y":0.05, "w":0.18, "h":0.90},
    "nickname_pct"  : {"x":0.20, "y":0.00, "w":0.35, "h":0.45},
    "items_area_pct": {"x":0.56, "y":0.05, "w":0.42, "h":0.90},
    "n_item_slots"  : 6,
    "has_trinket"   : True,
    "item_gap_ratio": 0.010
}

TOP_DEFAULTS = {
    "teamL_pct": {"x":0.02,"y":0.10,"w":0.22,"h":0.80},
    "teamR_pct": {"x":0.76,"y":0.10,"w":0.22,"h":0.80},
    "league_pct":{"x":0.44,"y":0.00,"w":0.12,"h":1.00}
}

def mirror_pct(p):
    return {"x": max(0.0, min(1.0, 1.0 - (p["x"]+p["w"]))), "y": p["y"], "w": p["w"], "h": p["h"]}

def load_any_image(d):
    exts=("*.png","*.jpg","*.jpeg","*.bmp","*.webp")
    files=[]; [files.extend(glob.glob(os.path.join(d,e))) for e in exts]
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

def draw_box(img, rc, color, label=None, thick=2):
    x1,y1,x2,y2=map(int,rc)
    cv2.rectangle(img,(x1,y1),(x2,y2),color,thick)
    if label:
        cv2.putText(img,label,(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2,cv2.LINE_AA)

def auto_items(items_rc, n_items, has_trinket, gap_ratio):
    x1,y1,x2,y2=items_rc; W=x2-x1
    n=n_items+(1 if has_trinket else 0)
    if n<=0: return []
    gap=int(W*max(0.0,min(0.2,gap_ratio)))
    tot_gap=gap*(n-1); cw=max(1,(W-tot_gap)//n)
    out=[]
    for k in range(n):
        cx1=x1+k*(cw+gap); cx2=min(x1+(k+1)*cw+k*gap,x2)
        out.append((cx1,y1,cx2,y2))
    return out

def split_rows_2cols(bottom_xyxy, n_rows, center_ratio, center_gap_ratio, row_pad_y_ratio, lr_pad_x_ratio):
    bx1,by1,bx2,by2=bottom_xyxy
    W=bx2-bx1; H=by2-by1
    row_h=H/n_rows
    rows=[]
    for r in range(n_rows):
        ry1=by1+int(r*row_h); ry2=by1+int((r+1)*row_h)
        pad_y=int((ry2-ry1)*row_pad_y_ratio)
        ry1+=pad_y; ry2-=pad_y
        cx=bx1+int(W*center_ratio)
        gap=int(W*center_gap_ratio)
        pad_x=int(W*lr_pad_x_ratio)
        left=(bx1+pad_x, ry1, max(bx1+pad_x, cx-gap), ry2)
        right=(min(cx+gap, bx2-pad_x), ry1, bx2-pad_x, ry2)
        rows.append((clamp_rect(left,  (bx1,by1,bx2,by2)),
                     clamp_rect(right, (bx1,by1,bx2,by2))))
    return rows

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
    top_rc=pct2xyxy(prof[top_key]); bot_rc=pct2xyxy(prof["bottom_panel"])

    layout={"L":json.loads(json.dumps(DEFAULTS_L)),"R":{}}
    layout["R"]={
        "slot_inner_pct": mirror_pct(layout["L"]["slot_inner_pct"]),
        "portrait_pct"  : mirror_pct(layout["L"]["portrait_pct"]),
        "nickname_pct"  : mirror_pct(layout["L"]["nickname_pct"]),
        "items_area_pct": mirror_pct(layout["L"]["items_area_pct"]),
        "n_item_slots"  : layout["L"]["n_item_slots"],
        "has_trinket"   : layout["L"]["has_trinket"],
        "item_gap_ratio": layout["L"]["item_gap_ratio"]
    }

    top_layout=json.loads(json.dumps(TOP_DEFAULTS))

    center_ratio=CENTER_RATIO_INIT
    selected_row=0
    selected_side="L"
    last_rect=None; dragging=False; ix=iy=0

    def build_overlay():
        rows = split_rows_2cols(bot_rc, N_ROWS, center_ratio, CENTER_GAP_RATIO, ROW_PAD_Y_RATIO, LEFT_PAD_X_RATIO)
        canvas=bgr.copy()
        draw_box(canvas, top_rc,(0,200,255),"TOP_BAR")
        tL=pct_to_rect(top_rc, top_layout["teamL_pct"])
        tR=pct_to_rect(top_rc, top_layout["teamR_pct"])
        lg=pct_to_rect(top_rc, top_layout["league_pct"])
        draw_box(canvas, tL,(200,220,255),"teamL")
        draw_box(canvas, tR,(200,220,255),"teamR")
        draw_box(canvas, lg,(200,180,255),"league")
        draw_box(canvas, bot_rc,(0,255,0),"BOTTOM_PANEL")
        for r,(left_base,right_base) in enumerate(rows):
            for side,base in (("L",left_base),("R",right_base)):
                slot_inner=pct_to_rect(base, layout[side]["slot_inner_pct"])
                portrait  =pct_to_rect(slot_inner, layout[side]["portrait_pct"])
                nickname  =pct_to_rect(slot_inner, layout[side]["nickname_pct"])
                items_ar  =pct_to_rect(slot_inner, layout[side]["items_area_pct"])
                items_list=auto_items(items_ar, layout[side]["n_item_slots"], layout[side]["has_trinket"], layout[side]["item_gap_ratio"])
                col=(255,200,0) if side=="L" else (255,120,0)
                draw_box(canvas, slot_inner, col, f"ROW {r+1} {side}")
                draw_box(canvas, portrait,(0,255,255),"portrait")
                draw_box(canvas, nickname,(180,255,180),"nickname")
                draw_box(canvas, items_ar,(180,180,255),"items_area")
                for k,rc in enumerate(items_list):
                    lab=f"item{k+1}" if (not layout[side]["has_trinket"] or k<layout[side]["n_item_slots"]) else "trinket"
                    draw_box(canvas, rc,(200,150,255) if "trinket" in lab else (200,150,200), lab)
        rows_now = split_rows_2cols(bot_rc, N_ROWS, center_ratio, CENTER_GAP_RATIO, ROW_PAD_Y_RATIO, LEFT_PAD_X_RATIO)
        base = rows_now[selected_row][0 if selected_side=="L" else 1]
        sel_inner=pct_to_rect(base, layout[selected_side]["slot_inner_pct"])
        draw_box(canvas, sel_inner,(0,0,255), f"SELECT {selected_side} r{selected_row+1}", 3)
        if last_rect is not None: draw_box(canvas,last_rect,(180,180,180),"last_rect",1)
        y=20
        for t in [
            "Drag: box | 1-5 row, Tab: L<->R, J/K: center -/+",
            "L: last->SLOT, O: ->PORTRAIT, N: ->NICK, M: ->ITEMS",
            "[/]: item slots -/+, T: trinket, -/=: gap -/+",
            "Y: last->teamL, U: ->teamR, I: ->league",
            "F: mirror L->R, G: copy L->R, H: copy R->L",
            "P: mirror teamL->teamR, S: save, Q/ESC: exit"
        ]:
            cv2.putText(canvas,t,(10,y),cv2.FONT_HERSHEY_PLAIN,1.0,(240,240,240),1,cv2.LINE_AA); y+=16
        inf=f"center={center_ratio:.3f}  L(items={layout['L']['n_item_slots']} trk={layout['L']['has_trinket']} gap={layout['L']['item_gap_ratio']:.3f})  R(items={layout['R']['n_item_slots']} trk={layout['R']['has_trinket']} gap={layout['R']['item_gap_ratio']:.3f})"
        cv2.putText(canvas,inf,(10,y+6),cv2.FONT_HERSHEY_PLAIN,1.0,(255,255,150),1,cv2.LINE_AA)
        return canvas

    def set_from_last(which):
        nonlocal last_rect
        if last_rect is None: return
        rows = split_rows_2cols(bot_rc, N_ROWS, center_ratio, CENTER_GAP_RATIO, ROW_PAD_Y_RATIO, LEFT_PAD_X_RATIO)
        base = rows[selected_row][0 if selected_side=="L" else 1]
        slot_inner = pct_to_rect(base, layout[selected_side]["slot_inner_pct"])
        if which=="slot_inner":
            layout[selected_side]["slot_inner_pct"]=rect_to_pct(base, last_rect)
        elif which=="portrait":
            layout[selected_side]["portrait_pct"]=rect_to_pct(slot_inner, last_rect)
        elif which=="nickname":
            layout[selected_side]["nickname_pct"]=rect_to_pct(slot_inner, last_rect)
        elif which=="items_area":
            layout[selected_side]["items_area_pct"]=rect_to_pct(slot_inner, last_rect)

    def set_top_from_last(which):
        nonlocal last_rect
        if last_rect is None: return
        top_layout[which]=rect_to_pct(top_rc, last_rect)

    def copy_L_to_R(mirror=False):
        if mirror:
            layout["R"]["slot_inner_pct"]=mirror_pct(layout["L"]["slot_inner_pct"])
            layout["R"]["portrait_pct"]=mirror_pct(layout["L"]["portrait_pct"])
            layout["R"]["nickname_pct"]=mirror_pct(layout["L"]["nickname_pct"])
            layout["R"]["items_area_pct"]=mirror_pct(layout["L"]["items_area_pct"])
        else:
            for k in ("slot_inner_pct","portrait_pct","nickname_pct","items_area_pct"):
                layout["R"][k]=dict(layout["L"][k])
        for k in ("n_item_slots","has_trinket","item_gap_ratio"):
            layout["R"][k]=layout["L"][k]

    def copy_R_to_L():
        for k in ("slot_inner_pct","portrait_pct","nickname_pct","items_area_pct"):
            layout["L"][k]=dict(layout["R"][k])
        for k in ("n_item_slots","has_trinket","item_gap_ratio"):
            layout["L"][k]=layout["R"][k]

    def mouse_cb(e,x,y,f,p):
        nonlocal dragging, ix, iy, last_rect
        if e==cv2.EVENT_LBUTTONDOWN: dragging=True; ix,iy=x,y; last_rect=(x,y,x,y)
        elif e==cv2.EVENT_MOUSEMOVE and dragging: last_rect=(ix,iy,x,y)
        elif e==cv2.EVENT_LBUTTONUP: dragging=False; last_rect=(ix,iy,x,y)

    win="LOL 5x2 Layout Calib"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE); cv2.setMouseCallback(win, mouse_cb)

    while True:
        cv2.imshow(win, build_overlay())
        k=cv2.waitKey(16)&0xFF
        if k in (27, ord('q')): break
        elif k in (ord('1'),ord('2'),ord('3'),ord('4'),ord('5')):
            selected_row=int(chr(k))-1
        elif k==9:
            selected_side="R" if selected_side=="L" else "L"
        elif k==ord('j'):
            center_ratio=max(0.40, center_ratio-0.002)
        elif k==ord('k'):
            center_ratio=min(0.60, center_ratio+0.002)
        elif k==ord('l'): set_from_last("slot_inner")
        elif k==ord('o'): set_from_last("portrait")
        elif k==ord('n'): set_from_last("nickname")
        elif k==ord('m'): set_from_last("items_area")
        elif k==ord('t'):
            layout[selected_side]["has_trinket"]=not layout[selected_side]["has_trinket"]
        elif k==ord('['):
            layout[selected_side]["n_item_slots"]=max(0, layout[selected_side]["n_item_slots"]-1)
        elif k==ord(']'):
            layout[selected_side]["n_item_slots"]=min(8, layout[selected_side]["n_item_slots"]+1)
        elif k==ord('-'):
            layout[selected_side]["item_gap_ratio"]=max(0.0, layout[selected_side]["item_gap_ratio"]-0.002)
        elif k==ord('='):
            layout[selected_side]["item_gap_ratio"]=min(0.2, layout[selected_side]["item_gap_ratio"]+0.002)
        elif k==ord('f'): copy_L_to_R(mirror=True)
        elif k==ord('g'): copy_L_to_R(mirror=False)
        elif k==ord('h'): copy_R_to_L()
        elif k==ord('y'): set_top_from_last("teamL_pct")
        elif k==ord('u'): set_top_from_last("teamR_pct")
        elif k==ord('i'): set_top_from_last("league_pct")
        elif k==ord('p'):
            top_layout["teamR_pct"]=mirror_pct(top_layout["teamL_pct"])
        elif k==ord('s'):
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(PROFILE_LAYOUT,"w",encoding="utf-8") as f:
                json.dump({
                    "target_wh":[W,H],
                    "n_rows": N_ROWS,
                    "center_ratio": center_ratio,
                    "center_gap_ratio": CENTER_GAP_RATIO,
                    "row_pad_y_ratio": ROW_PAD_Y_RATIO,
                    "lr_pad_x_ratio": LEFT_PAD_X_RATIO,
                    "layout": layout,
                    "top_layout": top_layout
                }, f, ensure_ascii=False, indent=2)
            plan={"image":os.path.basename(src),"target_wh":[W,H],
                  "top_bar_xyxy":list(map(int,top_rc)),
                  "bottom_panel_xyxy":list(map(int,bot_rc)),
                  "top":{
                      "teamL_xyxy": list(map(int, pct_to_rect(top_rc, top_layout["teamL_pct"]))),
                      "teamR_xyxy": list(map(int, pct_to_rect(top_rc, top_layout["teamR_pct"]))),
                      "league_xyxy": list(map(int, pct_to_rect(top_rc, top_layout["league_pct"])))
                  },
                  "rows":[]}
            rows = split_rows_2cols(bot_rc, N_ROWS, center_ratio, CENTER_GAP_RATIO, ROW_PAD_Y_RATIO, LEFT_PAD_X_RATIO)
            ov=bgr.copy(); draw_box(ov,top_rc,(0,200,255),"TOP_BAR"); draw_box(ov,bot_rc,(0,255,0),"BOTTOM_PANEL")
            tL=pct_to_rect(top_rc, top_layout["teamL_pct"]); tR=pct_to_rect(top_rc, top_layout["teamR_pct"]); lg=pct_to_rect(top_rc, top_layout["league_pct"])
            draw_box(ov, tL,(200,220,255),"teamL"); draw_box(ov, tR,(200,220,255),"teamR"); draw_box(ov, lg,(200,180,255),"league")
            for r,(left_base,right_base) in enumerate(rows):
                for side,base in (("L",left_base),("R",right_base)):
                    slot_inner=pct_to_rect(base, layout[side]["slot_inner_pct"])
                    portrait  =pct_to_rect(slot_inner, layout[side]["portrait_pct"])
                    nickname  =pct_to_rect(slot_inner, layout[side]["nickname_pct"])
                    items_ar  =pct_to_rect(slot_inner, layout[side]["items_area_pct"])
                    items_list=auto_items(items_ar, layout[side]["n_item_slots"], layout[side]["has_trinket"], layout[side]["item_gap_ratio"])
                    col=(255,200,0) if side=="L" else (255,120,0)
                    draw_box(ov, slot_inner,col, f"ROW {r+1} {side}")
                    draw_box(ov, portrait,(0,255,255),"portrait")
                    draw_box(ov, nickname,(180,255,180),"nickname")
                    draw_box(ov, items_ar,(180,180,255),"items_area")
                    for k2,rc in enumerate(items_list):
                        lab=f"item{k2+1}" if (not layout[side]["has_trinket"] or k2<layout[side]["n_item_slots"]) else "trinket"
                        draw_box(ov, rc,(200,150,255) if "trinket" in lab else (200,150,200), lab)
                    plan["rows"].append({
                        "row": r+1, "side": ("BLUE" if side=="L" else "RED"),
                        "row_base_xyxy": list(map(int, base)),
                        "slot_inner_xyxy": list(map(int, slot_inner)),
                        "portrait_xyxy": list(map(int, portrait)),
                        "nickname_xyxy": list(map(int, nickname)),
                        "items_area_xyxy": list(map(int, items_ar)),
                        "items_xyxy_list": [list(map(int,b)) for b in items_list],
                        "n_item_slots": layout[side]["n_item_slots"],
                        "has_trinket": layout[side]["has_trinket"]
                    })
            ov_path=os.path.join(OUT_DIR,"step2d_5x2_overlay.png")
            plan_path=os.path.join(OUT_DIR,"step2d_5x2_plan.json")
            cv2.imwrite(ov_path, ov)
            with open(plan_path,"w",encoding="utf-8") as f: json.dump(plan,f,ensure_ascii=False,indent=2)
            print(f"Saved:\n - {PROFILE_LAYOUT}\n - {ov_path}\n - {plan_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
