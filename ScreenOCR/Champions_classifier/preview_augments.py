import os, json, math
from PIL import Image
import torch
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from augments import find_images, get_train_transform

SRC_DIR  = r"C:\Nemo\ScreenOCR\assets\champions"
OUT_DIR  = r"C:\Nemo\ScreenOCR\previews\aug_v1"
IMG_SIZE = 128
SAMPLES_PER_CLASS = 16
GRID_COLS = 8

os.makedirs(OUT_DIR, exist_ok=True)

paths, names = find_images(SRC_DIR)
uniq = sorted(list(set(names)))
cls2id = {c:i for i,c in enumerate(uniq)}
id2cls = {v:k for k,v in cls2id.items()}

tf = get_train_transform(IMG_SIZE)

mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)

def denorm(x):
    return (x*std + mean).clamp(0,1)

for p, n in zip(paths, names):
    im = Image.open(p).convert("RGB")
    xs = []
    for _ in range(SAMPLES_PER_CLASS):
        x = tf(im)
        xs.append(x.unsqueeze(0))
    x = torch.cat(xs, dim=0)
    x = denorm(x)
    rows = math.ceil(SAMPLES_PER_CLASS/GRID_COLS)
    grid = make_grid(x, nrow=GRID_COLS, padding=2)
    save_image(grid, os.path.join(OUT_DIR, f"{n}_grid.png"))

with open(os.path.join(OUT_DIR,"labels.json"),"w",encoding="utf-8") as f:
    json.dump({"cls2id":cls2id,"id2cls":id2cls}, f, ensure_ascii=False, indent=2)
print(f"saved to {OUT_DIR}")
