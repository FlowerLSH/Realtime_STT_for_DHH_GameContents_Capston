import os, json, random, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from augments import find_images, ChampionsAugDataset

SRC_DIR   = r"C:\Nemo\ScreenOCR\assets\champions"
SAVE_DIR  = r"C:\Nemo\ScreenOCR\runs\champion_cls_v2"
IMG_SIZE  = 128
REPEATS_TRAIN = 400
REPEATS_VAL   = 60
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 1e-3
WD         = 1e-4
LABEL_SMOOTH = 0.1
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

def seed_all(s):
    random.seed(s); os.environ["PYTHONHASHSEED"]=str(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def make_scaler():
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler('cuda', enabled=(DEVICE=='cuda'))
        except TypeError:
            return torch.amp.GradScaler(enabled=(DEVICE=='cuda'))
    else:
        return torch.cuda.amp.GradScaler(enabled=(DEVICE=='cuda'))

def amp_ctx():
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast('cuda', enabled=(DEVICE=='cuda'))
        except TypeError:
            return torch.amp.autocast(device_type='cuda', enabled=(DEVICE=='cuda'))
    else:
        return torch.cuda.amp.autocast(enabled=(DEVICE=='cuda'))

def main():
    seed_all(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    paths, names = find_images(SRC_DIR)
    assert len(paths)>0
    uniq = sorted(list(set(names)))
    cls2id = {c:i for i,c in enumerate(uniq)}
    id2cls = {v:k for k,v in cls2id.items()}
    labels = [cls2id[n] for n in names]
    with open(os.path.join(SAVE_DIR,"labels.json"),"w",encoding="utf-8") as f:
        json.dump({"cls2id":cls2id,"id2cls":id2cls}, f, ensure_ascii=False, indent=2)

    train_ds = ChampionsAugDataset(paths, labels, IMG_SIZE, REPEATS_TRAIN, train=True)
    val_ds   = ChampionsAugDataset(paths, labels, IMG_SIZE, REPEATS_VAL,   train=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS>0))
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=max(1,NUM_WORKERS//2), pin_memory=True, persistent_workers=(NUM_WORKERS>0))

    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, len(cls2id))
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
        ck={"model":m.state_dict(),"epoch":epoch,"val_acc":val_acc,"cfg":{"IMG_SIZE":IMG_SIZE,"classes":id2cls}}
        torch.save(ck, os.path.join(SAVE_DIR,f"model_e{epoch:02d}.pth"))
        if val_acc>best_acc:
            best_acc=val_acc; torch.save(ck, best_path)
    print(f"Best val acc: {best_acc*100:.2f}% | saved: {best_path}")

if __name__ == "__main__":
    main()
