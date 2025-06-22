"""
Train MobileNet-V2 fuel-load classifier on the enlarged
data/interim/patches.h5 (balanced pos/neg tiles).

Outputs: best weights →  models/fuel_cnn.pt
"""
import collections          # ← add this line
import h5py, torch, numpy as np, random, math, itertools
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path
from tqdm import tqdm
from src.models.fuel_cnn import FuelCNN

PATCH_H5   = Path("data/interim/patches.h5")
OUT_MODEL  = Path("models/fuel_cnn.pt")
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

# ─────────────────────────────────────────────────────────────
class TileSet(Dataset):
    def __init__(self, idxs, h5_path, training: bool):
        self.idxs = idxs
        self.h5   = h5py.File(h5_path, "r")
        self.training = training

    def __len__(self):  return len(self.idxs)

    def __getitem__(self, i):
        j   = self.idxs[i]
        img = self.h5["images"][j][:].astype("float32") / 10000.0   # (4,256,256)
        lbl = float(self.h5["labels"][j])

        # simple aug
        if self.training:
            if random.random() < .5:
                img = img[:, :, ::-1]
            k = random.randint(0, 3)
            img = np.rot90(img, k, axes=(1, 2))
        img = np.ascontiguousarray(img)        # positive strides

        return torch.from_numpy(img), torch.tensor(lbl, dtype=torch.float32)

# ─── load & split ─────────────────────────────────────────────
with h5py.File(PATCH_H5) as f:
    N    = len(f["labels"])
    posN = int((f["labels"][:] == 1).sum())
print(f"patches.h5  →  total {N}  |  pos {posN}  neg {N-posN}")

idxs = list(range(N))
random.shuffle(idxs)
train_len = math.floor(0.8 * N)
train_set = TileSet(idxs[:train_len],  PATCH_H5, True)
val_set   = TileSet(idxs[train_len:],  PATCH_H5, False)

print("train",  collections.Counter([train_set[i][1].item() for i in range(len(train_set))]))
print("val  ",  collections.Counter([val_set[i][1].item()   for i in range(len(val_set))]))

train_dl = DataLoader(train_set, 64, shuffle=True, drop_last=True)
val_dl   = DataLoader(val_set,   64, shuffle=False)

# ─── model / optimiser / loss ────────────────────────────────
model = FuelCNN().to(DEVICE)
crit  = nn.BCEWithLogitsLoss()
opt   = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 3e-4)

best_f1, patience, wait = 0, 5, 0

# ─── helper to run one epoch ─────────────────────────────────
def run(dl, train: bool):
    model.train(train)
    all_logits, all_lbls = [], []
    epoch_loss = 0.0

    for imgs, lbls in dl:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        logits = model(imgs)
        loss   = crit(logits, lbls)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()

        epoch_loss += loss.item() * len(lbls)
        all_logits.append(logits.detach().cpu())
        all_lbls.append(lbls.cpu())

    logits = torch.cat(all_logits).numpy()
    lbls   = torch.cat(all_lbls).numpy()
    # threshold search per epoch
    best_thr, best_f1 = 0.5, 0
    for thr in np.linspace(0.1, 0.9, 17):
        preds = (logits >= thr).astype(int)
        f1    = f1_score(lbls, preds)
        if f1 > best_f1: best_f1, best_thr = f1, thr
    cm = confusion_matrix(lbls, (logits >= best_thr).astype(int))
    return epoch_loss/len(dl.dataset), best_f1, best_thr, cm

# ─── training loop ───────────────────────────────────────────
for ep in itertools.count():
    tl, tr_f1, tr_thr, _   = run(train_dl, True)
    vl, va_f1, va_thr, cm  = run(val_dl,   False)
    print(f"E{ep:02d}  trF1 {tr_f1:.2f}  vaF1 {va_f1:.2f}  thr {va_thr:.2f}  cm {cm}")

    if va_f1 > best_f1:
        best_f1, wait = va_f1, 0
        OUT_MODEL.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), OUT_MODEL)
        print("  ✓ saved")
    else:
        wait += 1
        if wait >= patience:
            print("early stop"); break

print(f"best F1 {best_f1:.2f} → {OUT_MODEL}")
