"""Train or evaluate the official EGE-UNet on a prepared ISIC2017 dataset."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from vendor.egeunet import EGEUNet


THRESHOLD = 0.5
AUX_WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5)
PAPER = {"dice": 0.8877, "iou": 0.7981, "seed_runs": 5, "epochs": 300}
UPSTREAM = "f52ba30c6bf7d0ca479c2c9d4a3cbda999f49d3a"


class ISIC(Dataset):
    def __init__(self, root, split, size, rows, limit=None):
        self.rows = [row for row in rows if row["split"] == split][:limit]
        self.size = size
        self.training = split == "train"
        self.directory = root / split

    def __len__(self):
        return len(self.rows)

    def raw(self, index):
        row = self.rows[index]
        isic_id = row["isic_id"]
        suffix = ".jpg" if "/thumbnails/" in row["image_source"] else ".full.jpg"
        with Image.open(self.directory / (isic_id + suffix)) as source:
            image = source.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        with Image.open(self.directory / f"{isic_id}_Segmentation.png") as source:
            mask = source.convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
        return np.asarray(image), np.asarray(mask) >= 128, isic_id

    def __getitem__(self, index):
        image, mask, isic_id = self.raw(index)
        if self.training:
            if random.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                image, mask = image[::-1], mask[::-1]
            if random.random() < 0.5:
                turns = random.randrange(1, 4)
                image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        rgb = torch.from_numpy(image.transpose(2, 0, 1).copy())
        target = torch.from_numpy(mask[None].copy())
        return rgb, target, isic_id


def read_manifest(root):
    path = root / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run prepare_data.py first.")
    content = path.read_bytes()
    rows = list(csv.DictReader(content.decode("utf-8").splitlines()))
    ids = [row["isic_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Image IDs must be unique across train, val and test")
    if {row["split"] for row in rows} != {"train", "val", "test"}:
        raise ValueError("The manifest must contain train, val and test")
    return rows, hashlib.sha256(content).hexdigest()


def loader(dataset, args, device, shuffle=False):
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )


def inputs(rgb, device):
    return rgb.to(
        device=device, dtype=torch.float32, memory_format=torch.channels_last, non_blocking=True,
    ).div_(255)


def bce_dice(prediction, target):
    dims = (1, 2, 3)
    intersection = (prediction * target).sum(dims)
    dice = (2 * intersection + 1) / (prediction.sum(dims) + target.sum(dims) + 1)
    return F.binary_cross_entropy(prediction, target) + 1 - dice.mean()


def train_epoch(model, batches, optimizer, device):
    model.train()
    total = torch.zeros((), device=device)
    for images, masks, _ in batches:
        images = inputs(images, device)
        masks = masks.to(device=device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        auxiliary, prediction = model(images)
        loss = bce_dice(prediction, masks)
        for weight, intermediate in zip(AUX_WEIGHTS, auxiliary):
            loss = loss + weight * bce_dice(intermediate, masks)
        loss.backward()
        optimizer.step()
        total += loss.detach() * images.shape[0]
    return total.item() / len(batches.dataset)


def scores(tp, fp, fn):
    return {
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
    }


@torch.inference_mode()
def evaluate(model, batches, device):
    model.eval()
    image_scores = []
    for images, masks, ids in batches:
        _, probability = model(inputs(images, device))
        prediction = probability >= THRESHOLD
        target = masks.to(device, non_blocking=True)
        dims = (1, 2, 3)
        tp = (prediction & target).sum(dims)
        fp = (prediction & ~target).sum(dims)
        fn = (~prediction & target).sum(dims)
        tn = (~prediction & ~target).sum(dims)
        counts = torch.stack((tp, fp, fn, tn), dim=1).cpu().tolist()
        for isic_id, (tp, fp, fn, tn) in zip(ids, counts):
            image_scores.append({
                "isic_id": isic_id, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                **scores(tp, fp, fn),
            })
    totals = {key: sum(row[key] for row in image_scores) for key in ("tp", "fp", "fn", "tn")}
    metrics = scores(totals["tp"], totals["fp"], totals["fn"])
    return {**metrics, **totals, "images": len(image_scores)}, image_scores


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def train(model, args, device, datasets, metadata):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.00001,
    )
    batches = loader(datasets["train"], args, device, shuffle=True)
    validation = loader(datasets["val"], args, device)
    checkpoint_path = args.output / "best.pt"
    history = []
    best_dice = -1.0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        loss = train_epoch(model, batches, optimizer, device)
        metrics, _ = evaluate(model, validation, device)
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save({
                "model": model.state_dict(), "epoch": epoch, "size": args.size,
                "validation": metrics, "manifest_sha256": metadata["manifest_sha256"],
            }, checkpoint_path)
        elapsed = time.perf_counter() - epoch_start
        history.append({
            "epoch": epoch, "train_loss": loss, "val_dice": metrics["dice"],
            "val_iou": metrics["iou"], "lr": lr, "seconds": elapsed,
        })
        save_csv(args.output / "history.csv", history)
        print(
            f"Epoch {epoch:02d}/{args.epochs}  loss {loss:.4f}  "
            f"val Dice {metrics['dice']:.2%}  val IoU {metrics['iou']:.2%}  {elapsed:.1f}s",
            flush=True,
        )
        schedule.step()
    metadata["training_seconds"] = time.perf_counter() - started
    return checkpoint_path, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/isic2017"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Evaluate existing weights instead of training")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit-per-split", type=int, help="Use a few examples per split for a smoke run")
    parser.add_argument("--no-plots", action="store_true", help="Write CSV and JSON only")
    args = parser.parse_args()
    if args.size < 32 or args.size % 32 or args.epochs < 1 or args.batch_size < 1:
        parser.error("size must be a positive multiple of 32, epochs and batch size must be positive")
    if args.workers < 0 or (args.limit_per_split is not None and args.limit_per_split < 1):
        parser.error("workers must be nonnegative and any split limit must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch installation")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    rows, digest = read_manifest(args.data)
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        args.size = checkpoint["size"]
        if checkpoint["manifest_sha256"] != digest:
            parser.error("the checkpoint and dataset manifest have different SHA-256 hashes")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("choose a new or empty output directory")
    sets = {
        split: ISIC(args.data, split, args.size, rows, args.limit_per_split)
        for split in ("train", "val", "test")
    }
    model = EGEUNet().to(device=device, memory_format=torch.channels_last)
    metadata = {
        "dataset": "ISIC2017 Task 1", "splits": {key: len(value) for key, value in sets.items()},
        "upstream_commit": UPSTREAM, "manifest_sha256": digest, "seed": args.seed,
        "size": args.size, "batch_size": args.batch_size, "workers": args.workers,
        "epochs": args.epochs if checkpoint is None else None,
        "threshold": THRESHOLD, "parameters": sum(p.numel() for p in model.parameters()),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "optimizer": "AdamW", "initial_lr": 0.001, "weight_decay": 0.01,
        "schedule": "CosineAnnealingLR", "min_lr": 0.00001, "aux_weights": AUX_WEIGHTS,
        "paper_reference": PAPER,
    }
    print(f"{device}  |  {metadata['parameters']:,} parameters  |  {metadata['splits']}", flush=True)
    history = None
    if checkpoint is None:
        checkpoint_path, history = train(model, args, device, sets, metadata)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    metadata["best_epoch"] = checkpoint["epoch"]
    metadata["validation_at_best_epoch"] = checkpoint["validation"]
    metrics, per_image = evaluate(model, loader(sets["test"], args, device), device)
    metadata["test"] = metrics
    save_json(args.output / "metrics.json", metadata)
    save_csv(args.output / "test_per_image.csv", per_image)
    print(
        f"Best epoch {checkpoint['epoch']}  |  ISIC2017 test n={metrics['images']}  |  "
        f"Dice {metrics['dice']:.2%}  |  IoU {metrics['iou']:.2%}",
        flush=True,
    )
    if not args.no_plots:
        from figures import save_figures

        def predict(rgb):
            with torch.inference_mode():
                image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0)
                _, probability = model(inputs(image, device))
                return (probability[0, 0] >= THRESHOLD).cpu().numpy()

        save_figures(args.output, metadata, history, per_image, sets["test"], predict)


if __name__ == "__main__":
    main()
