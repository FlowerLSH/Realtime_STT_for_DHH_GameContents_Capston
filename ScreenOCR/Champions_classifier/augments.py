import os, glob, random
from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms
from torch.utils.data import Dataset

def find_images(src_dir):
    exts = ("*.png","*.jpg","*.jpeg","*.webp","*.bmp")
    paths = []
    for e in exts: paths += glob.glob(os.path.join(src_dir, e))
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return paths, names

class PadToSquareResize:
    def __init__(self, size, pad_color=(0,0,0)):
        self.size=size; self.pad_color=pad_color
    def __call__(self, img):
        w,h = img.size
        s=max(w,h)
        canvas = Image.new("RGB",(s,s),self.pad_color)
        canvas.paste(img, ((s-w)//2,(s-h)//2))
        return canvas.resize((self.size,self.size), Image.BILINEAR)

class RandomPurpleRectFrame:
    def __init__(self, p=0.45, thickness=(0.05,0.12), glow=True):
        self.p=p; self.thickness=thickness; self.glow=glow
    def __call__(self, img):
        if random.random()>=self.p: return img
        w,h = img.size
        t = max(2, int(min(w,h)*random.uniform(*self.thickness)))
        inset = max(1, int(t*0.4))
        r = int(min(w,h)*random.uniform(0.0,0.12))
        col = (170,0,255,255)
        im = img.convert("RGBA")
        d = ImageDraw.Draw(im, "RGBA")
        try:
            d.rounded_rectangle([inset,inset,w-inset-1,h-inset-1], radius=r, outline=col, width=t)
        except:
            for k in range(t):
                d.rectangle([inset+k,inset+k,w-inset-1-k,h-inset-1-k], outline=col)
        if random.random()<0.5:
            inner_inset = inset + max(1,t//2)
            inner_t = max(1,t//2)
            col2 = (210,120,255,255)
            try:
                d.rounded_rectangle([inner_inset,inner_inset,w-inner_inset-1,h-inner_inset-1], radius=max(0,r-int(t*0.6)), outline=col2, width=inner_t)
            except:
                for k in range(inner_t):
                    d.rectangle([inner_inset+k,inner_inset+k,w-inner_inset-1-k,h-inner_inset-1-k], outline=col2)
        if self.glow:
            ov = Image.new("RGBA", im.size, (0,0,0,0))
            dd = ImageDraw.Draw(ov, "RGBA")
            try:
                dd.rounded_rectangle([inset,inset,w-inset-1,h-inset-1], radius=r, outline=(170,0,255,140), width=t)
            except:
                for k in range(t):
                    dd.rectangle([inset+k,inset+k,w-inset-1-k,h-inset-1-k], outline=(170,0,255,140))
            ov = ov.filter(ImageFilter.GaussianBlur(radius=max(1,t//2)))
            im = Image.alpha_composite(im, ov)
        return im.convert("RGB")

class RandomCornerBanner:
    def __init__(self, p=0.45): self.p=p
    def __call__(self, img):
        if random.random() >= self.p: return img
        w,h = img.size
        d = ImageDraw.Draw(img, "RGBA")
        side = int(min(w,h)*random.uniform(0.25,0.38))
        mode = random.choice(["top","bottom","left","right"])
        col = (240,180,40, random.randint(140,220)) if random.random()<0.5 else (230,230,230, random.randint(120,200))
        if mode=="top": d.rectangle([0,0,w,side], fill=col)
        elif mode=="bottom": d.rectangle([0,h-side,w,h], fill=col)
        elif mode=="left": d.rectangle([0,0,side,h], fill=col)
        else: d.rectangle([w-side,0,w,h], fill=col)
        return img

class RandomRectOcclusion:
    def __init__(self, p=0.7, n_rect=(1,3)):
        self.p=p; self.n_rect=n_rect
    def __call__(self, img):
        if random.random() >= self.p: return img
        w,h = img.size
        d = ImageDraw.Draw(img, "RGBA")
        k = random.randint(self.n_rect[0], self.n_rect[1])
        for _ in range(k):
            rw = int(w*random.uniform(0.12,0.28))
            rh = int(h*random.uniform(0.12,0.28))
            x0 = random.randint(0, max(1,w-rw))
            y0 = random.randint(0, max(1,h-rh))
            if random.random()<0.5:
                col=(0,0,0, random.randint(160,230))
            else:
                v = random.randint(0,255)
                col=(v,v,v, random.randint(120,220))
            d.rectangle([x0,y0,x0+rw,y0+rh], fill=col)
        return img

def get_train_transform(image_size):
    return transforms.Compose([
        PadToSquareResize(image_size),
        RandomPurpleRectFrame(p=0.45),
        RandomCornerBanner(p=0.45),
        RandomRectOcclusion(p=0.7),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.35, contrast=0.4, saturation=0.35, hue=0.05)], p=0.9),
        transforms.RandomAdjustSharpness(sharpness_factor=random.uniform(0.5,2.0), p=0.5),
        transforms.RandomAutocontrast(p=0.4),
        transforms.RandomEqualize(p=0.25),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1,2.0))], p=0.6),
        transforms.RandomRotation(degrees=8, fill=(0,0,0)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.RandomResizedCrop(image_size, scale=(0.85,1.0), ratio=(0.85,1.15)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.6, scale=(0.02,0.2), ratio=(0.3,3.3), value="random")
    ])

def get_val_transform(image_size):
    return transforms.Compose([
        PadToSquareResize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

class ChampionsAugDataset(Dataset):
    def __init__(self, paths, labels, image_size, repeats, train=True):
        self.paths=paths
        self.labels=labels
        self.repeats=repeats
        self.train=train
        self.tf_train=get_train_transform(image_size)
        self.tf_val=get_val_transform(image_size)
        self.images=[Image.open(p).convert("RGB") for p in paths]
    def __len__(self):
        return len(self.images)*self.repeats
    def __getitem__(self, idx):
        j = idx % len(self.images)
        img = self.images[j].copy()
        y = self.labels[j]
        x = self.tf_train(img) if self.train else self.tf_val(img)
        return x, y
