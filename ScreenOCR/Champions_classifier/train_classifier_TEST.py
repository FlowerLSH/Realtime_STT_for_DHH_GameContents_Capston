# train_classifier_TEST.py
import os, json, random, time
from collections import defaultdict, Counter
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from augments import find_images, ChampionsAugDataset

# ====== 경로/하이퍼파라미터 ======
SRC_DIR_TRAIN = r"C:\Nemo\ScreenOCR\assets\champions"
# 완전히 분리된 검증셋이 있다면 여기에 폴더 경로 지정. 없으면 None로 두면 층화 분할 수행
SRC_DIR_VAL   = None

SAVE_DIR      = r"C:\Nemo\ScreenOCR\runs\champion_cls_v3"

IMG_SIZE      = 128
REPEATS_TRAIN = 400
REPEATS_VAL   = 1
BATCH_SIZE    = 64
EPOCHS        = 20
LR            = 1e-3
WD            = 1e-4
LABEL_SMOOTH  = 0.1
VAL_RATIO     = 0.2
SEED          = 42

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# 문제가 생기면 0으로 두고 에러 원인 먼저 확인하세요.
NUM_WORKERS   = 4 if DEVICE == "cuda" else 0
PIN_MEMORY    = (DEVICE == "cuda")

# ====== 유틸 ======
def seed_all(s):
    random.seed(s); os.environ["PYTHONHASHSEED"]=str(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def make_scaler():
    # torch 2.x / 1.x 호환
    try:
        return torch.amp.GradScaler('cuda', enabled=(DEVICE=='cuda'))
    except TypeError:
        try:
            return torch.amp.GradScaler(enabled=(DEVICE=='cuda'))
        except AttributeError:
            return torch.cuda.amp.GradScaler(enabled=(DEVICE=='cuda'))

def amp_ctx():
    # torch 2.x / 1.x 호환
    try:
        return torch.amp.autocast('cuda', enabled=(DEVICE=='cuda'))
    except TypeError:
        try:
            return torch.amp.autocast(device_type='cuda', enabled=(DEVICE=='cuda'))
        except AttributeError:
            return torch.cuda.amp.autocast(enabled=(DEVICE=='cuda'))

def stratified_split(paths, labels, val_ratio, seed):
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for i, y in enumerate(labels):
        by_cls[y].append(i)
    trn_idx, val_idx = [], []
    for y, idxs in by_cls.items():
        rng.shuffle(idxs)
        if len(idxs) <= 1:
            trn_idx += idxs
        else:
            n_val = max(1, int(len(idxs)*val_ratio))
            val_idx += idxs[:n_val]
            trn_idx += idxs[n_val:]
    return trn_idx, val_idx

def subset(lst, idxs):
    return [lst[i] for i in idxs]

def create_resnet18(num_classes: int):
    """
    torchvision 버전별 가중치 API 차이를 흡수.
    - 최신: models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    - 구버전: models.resnet18(pretrained=True)
    """
    m = None
    try:
        # 최신 torchvision
        from torchvision.models import ResNet18_Weights
        m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        # 구버전 호환
        m = models.resnet18(pretrained=True)
    # 분류 헤드 교체
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# ====== 학습 스크립트 ======
def main():
    seed_all(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    tr_paths, tr_names = find_images(SRC_DIR_TRAIN)
    assert len(tr_paths) > 0, f"No images found in {SRC_DIR_TRAIN}"

    if SRC_DIR_VAL:
        va_paths, va_names = find_images(SRC_DIR_VAL)
        classes = sorted(list(set(tr_names + va_names)))
        cls2id = {c:i for i,c in enumerate(classes)}
        id2cls = {v:k for k,v in cls2id.items()}
        tr_labels = [cls2id[n] for n in tr_names]
        va_labels = [cls2id[n] for n in va_names]
        leak_warn = False
    else:
        classes = sorted(list(set(tr_names)))
        cls2id = {c:i for i,c in enumerate(classes)}
        id2cls = {v:k for k,v in cls2id.items()}
        tr_labels = [cls2id[n] for n in tr_names]
        # 클래스당 샘플 수 점검 후 층화 분할
        from collections import Counter
        cnt = Counter(tr_labels)
        if min(cnt.values()) >= 2:
            tr_idx, va_idx = stratified_split(tr_paths, tr_labels, VAL_RATIO, SEED)
            va_paths, va_names = subset(tr_paths, va_idx), subset(tr_names, va_idx)
            va_labels = subset(tr_labels, va_idx)
            tr_paths, tr_names, tr_labels = subset(tr_paths, tr_idx), subset(tr_names, tr_idx), subset(tr_labels, tr_idx)
            leak_warn = False
        else:
            # 클래스당 1장뿐이면 누수 경고 (train/val 동일)
            va_paths, va_names, va_labels = tr_paths, tr_names, tr_labels
            leak_warn = True

    with open(os.path.join(SAVE_DIR,"labels.json"),"w",encoding="utf-8") as f:
        json.dump({"cls2id":cls2id,"id2cls":id2cls}, f, ensure_ascii=False, indent=2)

    train_ds = ChampionsAugDataset(tr_paths, tr_labels, IMG_SIZE, REPEATS_TRAIN, train=True)
    val_ds   = ChampionsAugDataset(va_paths, va_labels, IMG_SIZE, REPEATS_VAL,   train=False)

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS>0)
    )
    val_dl   = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=max(0, NUM_WORKERS//2), pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS>0)
    )

    if leak_warn:
        print("WARNING: Each class has 1 image. Validation uses the same sources as training and will be optimistic.")
    print(f"train: {len(tr_paths)} imgs | val: {len(va_paths)} imgs | classes: {len(classes)}")

    m = create_resnet18(num_classes=len(cls2id))
    m = m.to(DEVICE)

    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = make_scaler()

    def eval_loader(dloader):
        m.eval(); correct=0; total=0; loss_sum=0.0
        with torch.no_grad(), amp_ctx():
            for xb, yb in dloader:
                xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
                logits = m(xb); loss = crit(logits, yb)
                pred = logits.argmax(1); correct += (pred==yb).sum().item(); total += yb.numel()
                loss_sum += loss.item()*yb.size(0)
        return correct/max(1,total), loss_sum/max(1,total)

    best_acc=0.0; best_path=os.path.join(SAVE_DIR,"model_best.pth")
    for epoch in range(1, EPOCHS+1):
        m.train(); t0=time.time(); run_loss=0.0; seen=0; correct=0; total=0
        for xb, yb in train_dl:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                logits = m(xb); loss = crit(logits, yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run_loss += loss.item()*yb.size(0); seen += yb.size(0)
            correct += (logits.argmax(1)==yb).sum().item(); total += yb.numel()
        trn_loss = run_loss/max(1,seen); trn_acc = correct/max(1,total)
        val_acc, val_loss = eval_loader(val_dl); sched.step(); dt=time.time()-t0
        print(f"Epoch {epoch:02d}/{EPOCHS} | train {trn_loss:.4f}/{trn_acc*100:.2f}% | val {val_loss:.4f}/{val_acc*100:.2f}% | {dt:.1f}s")
        ck={"model":m.state_dict(),"epoch":epoch,"val_acc":val_acc,"cfg":{"IMG_SIZE":IMG_SIZE,"classes":{int(k):v for k,v in id2cls.items()}}}
        torch.save(ck, os.path.join(SAVE_DIR,f"model_e{epoch:02d}.pth"))
        if val_acc>best_acc:
            best_acc=val_acc; torch.save(ck, best_path)
    print(f"Best val acc: {best_acc*100:.2f}% | saved: {best_path}")

if __name__ == "__main__":
    main()
