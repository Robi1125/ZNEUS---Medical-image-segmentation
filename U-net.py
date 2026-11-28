"""1-channel U-Net for LGG MRI segmentation (Kaggle: mateuszbuda/lgg-mri-segmentation).

Install:
    pip install segmentation-models-pytorch opencv-python wandb

Run:
    # baseline (no augmentation, no oversampling)
    python U-net.py --root ./lgg-mri-segmentation --run-name baseline

    # baseline + augmentation (scale +-10 %, flip with 50 % probability, rotation +-15 degrees)
    python U-net.py --root ./lgg-mri-segmentation --augment --run-name aug

    # lesion oversampling (lesion crop + 50/50 tumour / tumour-free sampling)
    python U-net.py --root ./lgg-mri-segmentation --lesion-oversampling --run-name oversample

    # any combination (do not use '+' in the run name)
    python U-net.py --root ./lgg-mri-segmentation --augment --lesion-oversampling --run-name aug-oversample

Split: by PATIENT, 70 % train / 15 % val / 15 % test (a patient's slices are never shared between parts).
    train: used for fitting the weights
    val:   used ONLY to pick the best epoch
    test:  never used for training or tuning; evaluated once at the end with the best-val checkpoint

Experiment tracking (Weights & Biases):
    pip install wandb && wandb login
    Every run logs to W&B (enabled by default): loss, val Dice, train tumour fraction, hyperparameters,
    final TEST metrics, failure-category counts, per-image table, best/worst overlays, checkpoint.
    Options: --wandb-project, --wandb-entity, --wandb-group, --wandb-host, --wandb-mode, --no-wandb

Sharing the results publicly:
    A W&B team can forbid public projects (Team settings -> Privacy -> "public sharing not allowed"); no
    code can override that. Three ways to get a public link anyway:
      1) log to a public-capable server/account, then set the project to Public (Settings -> Privacy):
           python U-net.py ... --wandb-host https://api.wandb.ai --wandb-entity <your-username>
      2) run offline now and upload later from any account:
           python U-net.py ... --wandb-mode offline
           python -m wandb sync --entity <your-username> --project lgg-mri-segmentation wandb/offline-run-*
      3) share the local files (no W&B needed): every run also writes results_<run>/history.csv
         (per-epoch curves) and results_<run>/metrics.json (config, split sizes, val/test metrics,
         failure counts) next to the overlays.

Output:
    best_unet_<run>.pt          - best model (selected on val Dice)
    results_<run>/              - every TEST image with its prediction overlaid:
                                  green = tumour found, red = tumour missed, yellow = false positive
                                  the file name starts with the image's Dice score
    results_<run>/per_image.csv - per test image: Dice, tumour area, predicted area, size bin, failure category
    results_<run>/grid.png      - best (top row) and worst (bottom row) test tumour slices
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import wandb
except ImportError:
    wandb = None

SIZE = 256

# failure categories, assigned per test slice (see categorize())
CATEGORIES = {
    "good": "good (tumour slice, Dice >= 0.7)",
    "correct_empty": "correct (tumour-free slice, empty prediction)",
    "false_positive": "false positive (tumour-free slice, non-empty prediction)",
    "missed": "missed tumour (empty prediction)",
    "wrong_location": "wrong location (prediction does not overlap the tumour)",
    "under": "under-segmentation (prediction < 60 % of the tumour area)",
    "over": "over-segmentation (prediction > 160 % of the tumour area)",
    "boundary": "boundary / shape error (similar area, Dice < 0.7)",
}


# ----------------------------------------------------------------- data
def find_items(root):
    """[(image_path, mask_path, patient, has_tumor)]. The Kaggle download often contains the folder
    twice (kaggle_3m + a nested copy), so we de-duplicate by (patient, file name)."""
    items, seen = [], set()
    for img in sorted(Path(root).rglob("*.tif")):
        if img.stem.endswith("_mask"):
            continue
        mask = img.with_name(img.stem + "_mask.tif")
        patient = img.parent.name
        if not mask.exists() or (patient, img.name) in seen:
            continue
        seen.add((patient, img.name))
        has_tumor = bool((cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE) > 127).any())
        items.append((img, mask, patient, has_tumor))
    return items


def split_by_patient(items, val_frac=0.15, test_frac=0.15, seed=42):
    """Patient-level train / val / test split. Returns ({'train','val','test'} -> items, patient counts)."""
    by_patient = defaultdict(list)
    for it in items:
        by_patient[it[2]].append(it)
    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)
    n_test = int(round(len(patients) * test_frac))
    n_val = int(round(len(patients) * val_frac))
    groups = {
        "test": patients[:n_test],
        "val": patients[n_test:n_test + n_val],
        "train": patients[n_test + n_val:],
    }
    assert not (set(groups["train"]) & set(groups["val"]))
    assert not (set(groups["train"]) & set(groups["test"]))
    assert not (set(groups["val"]) & set(groups["test"]))
    splits = {k: [it for p in v for it in by_patient[p]] for k, v in groups.items()}
    return splits, {k: len(v) for k, v in groups.items()}


def random_affine(img, mask):
    """Scale +-10 %, rotation +-15 degrees."""
    h, w = img.shape
    angle = random.uniform(-15, 15)
    scale = random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, mask


def random_crop(img, mask, size):
    """For tumour slices the crop always contains tumour pixels; for tumour-free slices it is random."""
    h, w = img.shape
    if mask.any():
        ys, xs = np.nonzero(mask)
        k = random.randrange(len(ys))
        y0 = ys[k] - random.randint(0, size - 1)
        x0 = xs[k] - random.randint(0, size - 1)
    else:
        y0 = random.randint(0, h - size)
        x0 = random.randint(0, w - size)
    y0 = int(np.clip(y0, 0, h - size))
    x0 = int(np.clip(x0, 0, w - size))
    return img[y0:y0 + size, x0:x0 + size], mask[y0:y0 + size, x0:x0 + size]


class LggDataset(Dataset):
    def __init__(self, items, channel, train=False, augment=False, crop_size=0):
        self.items = items
        self.channel = channel
        self.train = train
        self.augment = augment
        self.crop_size = crop_size  # 0 = no crop

    def __len__(self):
        return len(self.items)

    def load(self, i):
        """Raw (uint8 HxW image, uint8 HxW mask) using the selected channel."""
        img_path, mask_path, _, _ = self.items[i]
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[:, :, self.channel]  # file channel order
        mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if img.shape != (SIZE, SIZE):
            img = cv2.resize(img, (SIZE, SIZE))
            mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        return img, mask

    def __getitem__(self, i):
        img, mask = self.load(i)
        if self.train and self.augment:
            if random.random() < 0.5:  # flip with 50 % probability
                img, mask = img[:, ::-1], mask[:, ::-1]
            img, mask = random_affine(np.ascontiguousarray(img), np.ascontiguousarray(mask))

        img = img.astype(np.float32) / 255.0
        img = (img - img.mean()) / (img.std() + 1e-6)  # per-image z-score (computed on the full image)

        if self.train and self.crop_size:
            img, mask = random_crop(img, mask, self.crop_size)

        x = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0)  # (1, H, W)
        y = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        return x, y


def balanced_sampler(items):
    """Each epoch contains ~50 % tumour and ~50 % tumour-free images (sampling with replacement)."""
    n_pos = sum(it[3] for it in items)
    n_neg = len(items) - n_pos
    w = [0.5 / n_pos if it[3] else 0.5 / n_neg for it in items]
    return WeightedRandomSampler(w, num_samples=len(items), replacement=True)


# -------------------------------------------------------------- training
def dice_per_image(logits, y, eps=1e-7):
    pred = (logits > 0).float()
    inter = (pred * y).sum((1, 2, 3))
    total = pred.sum((1, 2, 3)) + y.sum((1, 2, 3))
    return (2 * inter + eps) / (total + eps)  # empty GT + empty prediction -> 1.0


def evaluate(model, loader, device):
    """Returns (Dice for every image, whether the image contains a tumour)."""
    model.eval()
    scores, has_tumor = [], []
    with torch.no_grad():
        for x, y in loader:
            scores.append(dice_per_image(model(x.to(device)), y.to(device)).cpu())
            has_tumor.append(y.sum((1, 2, 3)) > 0)
    return torch.cat(scores), torch.cat(has_tumor)


# ------------------------------------------------- test analysis / results
def categorize(gt_area, pred_area, inter, dice):
    """Failure category of one slice (keys of CATEGORIES)."""
    if gt_area == 0:
        return "false_positive" if pred_area > 0 else "correct_empty"
    if dice >= 0.7:
        return "good"
    if pred_area == 0:
        return "missed"
    if inter == 0:
        return "wrong_location"
    if pred_area < 0.6 * gt_area:
        return "under"
    if pred_area > 1.6 * gt_area:
        return "over"
    return "boundary"


def analyze_and_save(model, ds, device, out_dir):
    """Predict every slice of ds, save the colour overlays, return one record per slice."""
    out_dir.mkdir(exist_ok=True)
    model.eval()
    records = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, _ = ds[i]
            raw, gt = ds.load(i)
            pred = (model(x[None].to(device))[0, 0] > 0).cpu().numpy()
            gt = gt > 0

            gt_area, pred_area = int(gt.sum()), int(pred.sum())
            inter = int((gt & pred).sum())
            total = gt_area + pred_area
            dice = 1.0 if total == 0 else 2 * inter / total

            img = np.repeat(raw[:, :, None], 3, axis=2).astype(np.float32)
            for region, color in ((gt & pred, (0, 255, 0)),      # found          -> green
                                  (gt & ~pred, (255, 0, 0)),     # missed         -> red
                                  (pred & ~gt, (255, 255, 0))):  # false positive -> yellow
                img[region] = 0.5 * img[region] + 0.5 * np.array(color)
            img = img.astype(np.uint8)
            cv2.putText(img, f"Dice {dice:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)

            img_path, _, patient, _ = ds.items[i]
            path = out_dir / f"{dice:.2f}_{img_path.stem}.png"
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            records.append(dict(name=img_path.name, patient=patient, has_tumor=gt_area > 0,
                                gt_area=gt_area, pred_area=pred_area, dice=float(dice),
                                category=categorize(gt_area, pred_area, inter, dice), path=path))

    # tumour size bins = tertiles of the ground-truth area over the tumour slices
    areas = [r["gt_area"] for r in records if r["has_tumor"]]
    lo, hi = np.percentile(areas, [100 / 3, 200 / 3])
    for r in records:
        r["size"] = "-" if not r["has_tumor"] else (
            "small" if r["gt_area"] <= lo else "medium" if r["gt_area"] <= hi else "large")

    with open(out_dir / "per_image.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "patient", "has_tumor", "size", "gt_area", "pred_area", "dice", "category"])
        for r in records:
            w.writerow([r["name"], r["patient"], r["has_tumor"], r["size"], r["gt_area"],
                        r["pred_area"], f"{r['dice']:.4f}", r["category"]])
    return records


def save_grid(records, path, k=6):
    """Best k (top row) and worst k (bottom row) TEST tumour slices as one picture."""
    tumour = sorted((r for r in records if r["has_tumor"]), key=lambda r: r["dice"])
    k = min(k, len(tumour) // 2)
    worst, best = tumour[:k], tumour[-k:][::-1]
    row = lambda rs: np.hstack([cv2.imread(str(r["path"])) for r in rs])
    cv2.imwrite(str(path), np.vstack([row(best), row(worst)]))
    return best, worst


def mean_dice(records, pick):
    d = [r["dice"] for r in records if pick(r)]
    return float(np.mean(d)) if d else float("nan")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LGG dataset folder (contains kaggle_3m)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weights", default="imagenet", help="imagenet or none")
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of patients used for validation")
    ap.add_argument("--test-frac", type=float, default=0.15, help="fraction of patients used for testing")
    ap.add_argument("--channel", type=int, default=1, choices=[0, 1, 2],
                    help="which channel to use: 0=pre-contrast, 1=FLAIR (default), 2=post-contrast")
    ap.add_argument("--augment", action="store_true",
                    help="augmentation: scale +-10%%, flip with 50%% probability, rotation +-15 degrees")
    ap.add_argument("--lesion-oversampling", action="store_true",
                    help="lesion crop + 50/50 tumour / tumour-free sampling in the train set")
    ap.add_argument("--crop-size", type=int, default=128, help="crop size used with oversampling")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true", help="disable W&B logging")
    ap.add_argument("--wandb-project", default="lgg-mri-segmentation",
                    help="W&B project; it is created automatically by the first run if it does not exist")
    ap.add_argument("--wandb-entity", default="segmentation-zneus", help="W&B team / username")
    ap.add_argument("--wandb-group", default=None, help="group runs together (e.g. 'final')")
    ap.add_argument("--wandb-host", default=None,
                    help="W&B server to log to, e.g. https://api.wandb.ai (default: the one you are logged in to)")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline"],
                    help="offline = log locally and upload later with 'python -m wandb sync'")
    args = ap.parse_args()

    run = args.run_name or ("baseline" + ("-aug" if args.augment else "") +
                            ("-oversample" if args.lesion_oversampling else ""))

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| run:", run)

    use_wandb = not args.no_wandb
    if use_wandb:
        if wandb is None:
            raise SystemExit("wandb is not installed: pip install wandb && wandb login (or use --no-wandb)")
        if args.wandb_host and args.wandb_mode == "online":
            wandb.login(host=args.wandb_host)  # asks for an API key of that server if not stored yet
        wandb.init(
            mode=args.wandb_mode,
            project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group, name=run,
            tags=["unet-resnet34", "1ch", "augment" if args.augment else "no-aug",
                  "oversample" if args.lesion_oversampling else "no-oversample"],
            config={**vars(args), "run": run, "dataset": "LGG-MRI", "in_channels": 1, "encoder": "resnet34"},
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    # patient-level 70 / 15 / 15 split
    items = find_items(args.root)
    splits, n_pat = split_by_patient(items, args.val_frac, args.test_frac)
    train_i, val_i, test_i = splits["train"], splits["val"], splits["test"]
    frac = lambda xs: sum(it[3] for it in xs) / len(xs)
    print(f"patients train/val/test: {n_pat['train']}/{n_pat['val']}/{n_pat['test']} | "
          f"slices train/val/test: {len(train_i)}/{len(val_i)}/{len(test_i)}")
    print(f"tumour fraction  train: {frac(train_i):.1%}  val: {frac(val_i):.1%}  test: {frac(test_i):.1%}")
    if use_wandb:
        wandb.config.update({
            "n_patients_train": n_pat["train"], "n_patients_val": n_pat["val"], "n_patients_test": n_pat["test"],
            "n_slices_train": len(train_i), "n_slices_val": len(val_i), "n_slices_test": len(test_i),
            "tumor_fraction_train_natural": frac(train_i),
            "tumor_fraction_val": frac(val_i), "tumor_fraction_test": frac(test_i),
        })

    crop = args.crop_size if args.lesion_oversampling else 0
    train_ds = LggDataset(train_i, args.channel, train=True, augment=args.augment, crop_size=crop)
    val_ds = LggDataset(val_i, args.channel)
    test_ds = LggDataset(test_i, args.channel)

    if args.lesion_oversampling:
        train_dl = DataLoader(train_ds, args.bs, sampler=balanced_sampler(train_i), num_workers=args.workers)
    else:
        train_dl = DataLoader(train_ds, args.bs, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, args.bs, num_workers=args.workers)

    model = smp.Unet("resnet34",
                     encoder_weights=None if args.weights == "none" else args.weights,
                     in_channels=1, classes=1).to(device)
    dice_loss = smp.losses.DiceLoss("binary", from_logits=True)
    bce_loss = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt = f"best_unet_{run}.pt"
    best_dice = -1.0
    history = []  # per-epoch numbers, also written to results_<run>/history.csv
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, seen, seen_pos = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = bce_loss(logits, y) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(x)
            seen += len(x)
            seen_pos += (y.sum((1, 2, 3)) > 0).sum().item()

        train_loss = total_loss / seen
        dice, has_t = evaluate(model, val_dl, device)
        val_dice = dice.mean().item()
        val_dice_tumor = dice[has_t].mean().item()
        print(f"epoch {epoch:3d} | loss {train_loss:.4f} | train tumour fraction {seen_pos / seen:.1%} | "
              f"val Dice {val_dice:.4f} | val Dice (tumour only) {val_dice_tumor:.4f}")
        history.append((epoch, train_loss, seen_pos / seen, val_dice, val_dice_tumor))
        if use_wandb:
            wandb.log({"epoch": epoch, "train/loss": train_loss, "train/tumor_fraction": seen_pos / seen,
                       "val/dice": val_dice, "val/dice_tumor_only": val_dice_tumor})
        if val_dice > best_dice:  # the best epoch is picked on VALIDATION only
            best_dice = val_dice
            torch.save(model.state_dict(), ckpt)

    # ------------- final evaluation: best-val checkpoint, evaluated ONCE on the untouched test split
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"best VAL Dice: {best_dice:.4f}")

    out_dir = Path(f"results_{run}")
    records = analyze_and_save(model, test_ds, device, out_dir)
    t_all = mean_dice(records, lambda r: True)
    t_tum = mean_dice(records, lambda r: r["has_tumor"])
    t_free = mean_dice(records, lambda r: not r["has_tumor"])
    print(f"TEST Dice: {t_all:.4f} | tumour only: {t_tum:.4f} | "
          f"tumour-free (1.0 = no false positives): {t_free:.4f}")

    by_size = {s: mean_dice(records, lambda r, s=s: r["size"] == s) for s in ("small", "medium", "large")}
    print("TEST Dice by tumour size (tertiles of the ground-truth area): " +
          " | ".join(f"{s} {v:.4f}" for s, v in by_size.items()))

    counts = Counter(r["category"] for r in records)
    print("TEST failure categories:")
    for code, n in counts.most_common():
        print(f"  {n:4d}  {CATEGORIES[code]}")

    best, worst = save_grid(records, out_dir / "grid.png")
    print(f"{len(records)} test overlays + per_image.csv + grid.png saved to {out_dir}/")

    # local, W&B-independent copy of the numbers, so they can be shared even without a public W&B project
    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_tumor_fraction", "val_dice", "val_dice_tumor_only"])
        w.writerows(history)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "run": run, "args": vars(args),
            "patients": n_pat, "slices": {k: len(v) for k, v in splits.items()},
            "best_val_dice": best_dice,
            "test": {"dice": t_all, "dice_tumor_only": t_tum, "dice_tumor_free": t_free,
                     "dice_by_tumor_size": by_size, "failure_categories": dict(counts)},
        }, f, indent=2)
    print(f"history.csv + metrics.json saved to {out_dir}/")

    if use_wandb:
        wandb.summary["val/dice_best"] = best_dice
        wandb.summary["test/dice"] = t_all
        wandb.summary["test/dice_tumor_only"] = t_tum
        wandb.summary["test/dice_no_tumor_slices"] = t_free
        for s, v in by_size.items():
            wandb.summary[f"test/dice_{s}_tumors"] = v
        for code in CATEGORIES:
            wandb.summary[f"test/failures/{code}"] = counts.get(code, 0)

        cap = lambda r: f"Dice {r['dice']:.2f} | {r['category']} | {r['size']} tumour"
        wandb.log({
            "test/best_tumor_slices": [wandb.Image(str(r["path"]), caption=cap(r)) for r in best],
            "test/worst_tumor_slices": [wandb.Image(str(r["path"]), caption=cap(r)) for r in worst],
            "test/grid_best_top_worst_bottom": wandb.Image(str(out_dir / "grid.png")),
        })

        table = wandb.Table(columns=["image", "patient", "has_tumor", "size", "gt_area", "pred_area",
                                     "dice", "category"])
        for r in records:
            table.add_data(r["name"], r["patient"], r["has_tumor"], r["size"], r["gt_area"],
                           r["pred_area"], r["dice"], r["category"])
        wandb.log({"test/per_image": table})

        # best checkpoint as an artifact (artifact names may only contain letters, digits, '-', '_', '.')
        art_name = "unet-" + "".join(c if c.isalnum() or c in "-_." else "-" for c in run)
        art = wandb.Artifact(art_name, type="model", metadata={"val_dice": best_dice, "test_dice": t_all})
        art.add_file(ckpt)
        wandb.log_artifact(art)

        print("W&B run:", wandb.run.url)
        wandb.finish()


if __name__ == "__main__":
    main()