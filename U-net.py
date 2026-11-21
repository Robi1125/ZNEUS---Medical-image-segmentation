"""Simple U-Net for BUSI breast ultrasound segmentation.

Install:
    pip install segmentation-models-pytorch opencv-python wandb

Run:
    python train.py --root ./Dataset_BUSI_with_GT --epochs 40 --run-name busi-unet-r34
    python train.py --root ./Dataset_BUSI_with_GT --epochs 40 --weights none --run-name busi-scratch

Output:
    best_unet.pt  - best model (selected on the validation split)
    results/      - every test image with its prediction coloured on it:
                    green = tumour pixels the model found
                    red   = tumour pixels the model missed
                    file name starts with the image's Dice score (e.g. 0.83_benign (12).png)
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except ImportError:
    wandb = None

SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------------------------------------------- data
def find_pairs(root):
    """(image, [mask files]) pairs. 'normal' images have an empty mask."""
    pairs = []
    for img in sorted(Path(root).rglob("*.png")):
        if "_mask" in img.name:
            continue
        masks = sorted(img.parent.glob(img.stem + "_mask*.png"))
        if masks:
            pairs.append((img, masks))
    return pairs


class BusiDataset(Dataset):
    def __init__(self, pairs, train):
        self.pairs = pairs
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, mask_paths = self.pairs[i]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = np.zeros(img.shape[:2], np.uint8)
        for p in mask_paths:  # some images have several masks -> union
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, (m > 127).astype(np.uint8))

        img = cv2.resize(img, (SIZE, SIZE))
        mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        if self.train:  # simple augmentation: random flips
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                img, mask = img[::-1], mask[::-1]
        img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)

        x = torch.from_numpy(((img / 255.0 - MEAN) / STD).transpose(2, 0, 1)).float()
        y = torch.from_numpy(mask).float().unsqueeze(0)
        return x, y


# -------------------------------------------------------------- training
def dice_per_image(logits, y, eps=1e-7):
    pred = (logits > 0).float()  # logit > 0  <=>  sigmoid > 0.5
    inter = (pred * y).sum((1, 2, 3))
    total = pred.sum((1, 2, 3)) + y.sum((1, 2, 3))
    return (2 * inter + eps) / (total + eps)  # empty GT + empty prediction -> 1.0


def evaluate(model, loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for x, y in loader:
            scores.append(dice_per_image(model(x.to(device)), y.to(device)).cpu())
    return torch.cat(scores)


# --------------------------------------------------------------- results
def save_results(model, ds, device, out_dir):
    """Save every image of ds with green (hit) and red (missed) tumour pixels."""
    out_dir.mkdir(exist_ok=True)
    model.eval()
    saved = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            pred = (model(x[None].to(device))[0, 0] > 0).cpu().numpy()
            gt = y[0].numpy() > 0.5

            total = pred.sum() + gt.sum()
            dice = 1.0 if total == 0 else 2 * (pred & gt).sum() / total

            img = ((x.numpy().transpose(1, 2, 0) * STD + MEAN) * 255).clip(0, 255)
            for region, color in ((gt & pred, (0, 255, 0)),    # found   -> green
                                  (gt & ~pred, (255, 0, 0))):  # missed  -> red
                img[region] = 0.5 * img[region] + 0.5 * np.array(color)
            img = img.astype(np.uint8)
            cv2.putText(img, f"Dice {dice:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)

            path = out_dir / f"{dice:.2f}_{ds.pairs[i][0].stem}.png"
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            saved.append((dice, path))
    return saved


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset_BUSI_with_GT folder")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weights", default="imagenet", help="imagenet or none")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    use_wandb = wandb is not None and not args.no_wandb
    if use_wandb:
        wandb.init(project="medical-segmentation", name=args.run_name, config=vars(args))

    # 70 / 15 / 15 split
    pairs = find_pairs(args.root)
    random.Random(42).shuffle(pairs)
    n_test, n_val = int(len(pairs) * 0.15), int(len(pairs) * 0.15)
    test_p, val_p, train_p = pairs[:n_test], pairs[n_test:n_test + n_val], pairs[n_test + n_val:]
    print(f"train/val/test: {len(train_p)}/{len(val_p)}/{len(test_p)}")

    train_ds = BusiDataset(train_p, train=True)
    val_ds = BusiDataset(val_p, train=False)
    test_ds = BusiDataset(test_p, train=False)
    train_dl = DataLoader(train_ds, args.bs, shuffle=True)
    val_dl = DataLoader(val_ds, args.bs)
    test_dl = DataLoader(test_ds, args.bs)

    model = smp.Unet("resnet34",
                     encoder_weights=None if args.weights == "none" else args.weights,
                     in_channels=3, classes=1).to(device)
    dice_loss = smp.losses.DiceLoss("binary", from_logits=True)
    bce_loss = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = bce_loss(logits, y) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(x)

        train_loss = total_loss / len(train_ds)
        val_dice = evaluate(model, val_dl, device).mean().item()
        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} | val Dice {val_dice:.4f}")
        if use_wandb:
            wandb.log({"epoch": epoch, "train/loss": train_loss, "val/dice": val_dice})
        if val_dice > best_dice:  # pick the best epoch on VALIDATION, never on test
            best_dice = val_dice
            torch.save(model.state_dict(), "best_unet.pt")

    # final test evaluation with the best model
    model.load_state_dict(torch.load("best_unet.pt", map_location=device))
    test_dice = evaluate(model, test_dl, device)
    print(f"TEST Dice: {test_dice.mean():.4f}")

    saved = save_results(model, test_ds, device, Path("results"))
    print(f"{len(saved)} images saved to the 'results' folder")

    if use_wandb:
        wandb.summary["test/dice"] = test_dice.mean().item()
        worst = sorted(saved)[:8]
        wandb.log({"test/worst_8": [wandb.Image(str(p), caption=f"Dice {d:.2f}") for d, p in worst]})
        wandb.finish()


if __name__ == "__main__":
    main()