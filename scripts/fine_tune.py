"""Fine-tune 3-class UNet++ EfficientNet-B0 with paired normalized PNGs / class-ID masks.

This is a reusable *fine-tuning example*, not an exact reimplementation of the
original from-scratch training notebook. Patches are 512 px by default; training
and validation must be split by SOURCE IMAGE, not patches from the same image.
"""
import argparse
import csv
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from run_inference import build_model, compute_pad_to_multiple, generate_patch_grid, load_checkpoint, get_device


class SegmentationPatches(Dataset):
    def __init__(self, ids, images_dir, masks_dir, patch_size=512, stride=256, augment=False):
        self.ids = ids
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.entries = []
        self.pads = {}
        for image_id in ids:
            img_path = self.images_dir / f"{image_id}.png"
            mask_path = self.masks_dir / f"{image_id}.png"
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if image is None or mask is None:
                raise FileNotFoundError(f"Missing or unreadable image/mask pair: {img_path} / {mask_path}")
            if mask.ndim != 2 or image.shape[:2] != mask.shape[:2]:
                raise ValueError(f"Expected same-size RGB image and 2D ID mask for {image_id}")
            vals = np.unique(mask)
            if not np.all(np.isin(vals, [0, 1, 2])):
                raise ValueError(f"Unexpected class IDs in {mask_path}: {vals.tolist()} (expected 0,1,2)")
            h, w = mask.shape
            pad, (hp, wp) = compute_pad_to_multiple(h, w, patch_size)
            self.pads[image_id] = pad
            # Do not materialize every whole-slide image in memory; only store catalogue.
            padded_mask = cv2.copyMakeBorder(mask, *pad, cv2.BORDER_CONSTANT, value=0)
            for x, y in generate_patch_grid(hp, wp, patch_size, stride):
                has_soma = bool(np.any(padded_mask[y:y+patch_size, x:x+patch_size] == 2))
                self.entries.append((image_id, x, y, has_soma))
        if not self.entries:
            raise ValueError("No patches found. Check images and settings.")

    @lru_cache(maxsize=2)
    def _read(self, image_id):
        rgb = cv2.cvtColor(cv2.imread(str(self.images_dir / f"{image_id}.png")), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks_dir / f"{image_id}.png"), cv2.IMREAD_UNCHANGED)
        top, bottom, left, right = self.pads[image_id]
        rgb = cv2.copyMakeBorder(rgb, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=0)
        return rgb, mask

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        name, x, y, _ = self.entries[idx]
        image, mask = self._read(name)
        sz = self.patch_size
        image = image[y:y+sz, x:x+sz].copy()
        mask = mask[y:y+sz, x:x+sz].copy()
        if self.augment:
            if random.random() < 0.5:
                image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
            if random.random() < 0.5:
                image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)
            k = random.randrange(4)
            image, mask = np.rot90(image, k).copy(), np.rot90(mask, k).copy()
            # Mild intensity jitter only: not equivalent to re-normalizing slides.
            if random.random() < 0.3:
                image = np.clip(image.astype(np.float32) * random.uniform(0.9, 1.1), 0, 255).astype(np.uint8)
        features = torch.from_numpy(np.ascontiguousarray(image)).permute(2,0,1).float() / 255.0
        target = torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64)))
        return features, target


def read_ids(path):
    ids = [line.strip() for line in Path(path).read_text().splitlines()
           if line.strip() and not line.strip().startswith('#')]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Need unique nonempty image IDs in {path}")
    return ids


def dice_by_class(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp, fn = cm.sum(0)-tp, cm.sum(1)-tp
    return (2*tp + 1e-7) / (2*tp + fp + fn + 1e-7)


def make_loader(dataset, batch_size, train=False, target_soma_fraction=0.2):
    if not train:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    positive = np.array([bool(e[3]) for e in dataset.entries])
    p = positive.mean()
    if 0 < p < 1:
        soma_weight = max(1., min(100., target_soma_fraction*(1-p) / (p*(1-target_soma_fraction))))
        weights = torch.tensor(np.where(positive, soma_weight, 1), dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def train_epoch(model, loader, opt, device, freeze_encoder=False):
    model.train()
    if freeze_encoder:
        model.encoder.eval()
    total, n = 0., 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(images), masks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()*images.shape[0]
        n += images.shape[0]
    return total/max(n,1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    cm = np.zeros((3,3), dtype=np.int64)
    total, n = 0., 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, masks)
        preds = logits.argmax(1)
        bins = torch.bincount((masks.reshape(-1)*3+preds.reshape(-1)).long(), minlength=9)
        cm += bins.reshape(3,3).cpu().numpy()
        total += loss.item()*images.shape[0]
        n += images.shape[0]
    dice = dice_by_class(cm)
    return total/max(n,1), dice, float(0.3*dice[1]+0.7*dice[2])


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained 3-class UNet++ on histology PNGs")
    parser.add_argument("--images", required=True, help="Folder of normalized RGB PNGs")
    parser.add_argument("--masks", required=True, help="Folder of grayscale 0/1/2 PNG masks")
    parser.add_argument("--train-ids", required=True, help="TXT with one training image basename per line")
    parser.add_argument("--val-ids", required=True, help="TXT with one validation image basename per line")
    parser.add_argument("--checkpoint", required=True, help="Pretrained weights (.pt), matching architecture")
    parser.add_argument("--output", default="outputs/finetune")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze the pretrained encoder weights")
    parser.add_argument("--device", choices=['auto','cpu','cuda','mps'], default='auto')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, and learning rate must be positive")
    train_ids, val_ids = read_ids(args.train_ids), read_ids(args.val_ids)
    if set(train_ids) & set(val_ids):
        parser.error(f"Training and validation IDs overlap: {sorted(set(train_ids)&set(val_ids))}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device() if args.device == 'auto' else torch.device(args.device)
    training = SegmentationPatches(train_ids, args.images, args.masks, augment=True)
    validation = SegmentationPatches(val_ids, args.images, args.masks, augment=False)
    train_loader = make_loader(training, args.batch_size, train=True)
    val_loader = make_loader(validation, args.batch_size)
    model = load_checkpoint(build_model(), args.checkpoint, device)
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    best = -float('inf')
    fields = ['epoch', 'train_loss','val_loss','dice_bg','dice_neu','dice_soma','foreground_score']
    with (outdir / 'training_history.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs+1):
            train_loss = train_epoch(model, train_loader, opt, device, args.freeze_encoder)
            val_loss, dice, score = eval_epoch(model, val_loader, device)
            row = dict(zip(fields,[epoch,train_loss,val_loss,*map(float,dice),score]))
            writer.writerow(row)
            fh.flush()
            torch.save(model.state_dict(), outdir/'last.pt')
            if score > best:
                best = score
                torch.save(model.state_dict(), outdir/'best_finetuned.pt')
            print(f"Epoch {epoch:02d}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"dice_neu={dice[1]:.3f} dice_soma={dice[2]:.3f} fg_score={score:.3f}")
    print(f"Saved best checkpoint: {outdir/'best_finetuned.pt'}")
    print('Important: validation Dice here aggregates *overlapping patches*, not independent whole images.')


if __name__ == '__main__':
    main()
