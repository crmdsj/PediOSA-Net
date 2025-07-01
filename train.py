#!/usr/bin/env python3
# train.py

import os
import sys
import argparse
import logging
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# make sure we can import your preprocessing module
sys.path.append(os.path.abspath(os.path.join(__file__, "..")))
from preprocessing import create_dataloader
from my_resnet import ResNet3DWithCBAM

def parse_args():
    parser = argparse.ArgumentParser(description="PediOSA-Net Training Script")
    # I/O paths
    parser.add_argument("--images_dir",    required=True,
                        help="Directory containing CBCT .npy files")
    parser.add_argument("--label_path",    required=True,
                        help="CSV file with labels")
    parser.add_argument("--pretrained",    default=None,
                        help="Path to pretrained weights (optional)")
    parser.add_argument("--checkpoint_dir", default="checkpoints/",
                        help="Where to save best models")
    # DataLoader settings
    parser.add_argument("--batch_size",   type=int, default=8,  help="Batch size")
    parser.add_argument("--num_workers",  type=int, default=4,  help="DataLoader workers")
    parser.add_argument("--cache_dir",     default="./cache", help="PersistentDataset cache directory")
    # Training hyperparameters
    parser.add_argument("--epochs",       type=int,   default=150,  help="Number of epochs")
    parser.add_argument("--lr",           type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    # Logging & TensorBoard
    parser.add_argument("--log_file", default="training.log",   help="Path for log file")
    parser.add_argument("--tb_dir",   default="./tb_logs",     help="TensorBoard log directory")
    return parser.parse_args()

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_model(device, pretrained_path=None):
    model = ResNet3DWithCBAM(
        block="basic",
        layers=[2,2,2,2],
        block_inplanes=[64,128,256,512],
        spatial_dims=3,
        n_input_channels=1,
        num_classes=1
    ).to(device)
    if pretrained_path:
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state, strict=False)
        logging.info(f"Loaded pretrained weights from {pretrained_path}")
    return model

def train_folds(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the full dataset (PersistentDataset) once
    full_loader = create_dataloader(
        images_dir  = args.images_dir,
        label_path  = args.label_path,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        mode        = "train",
        cache_dir   = args.cache_dir
    )
    dataset = full_loader.dataset  # underlying PersistentDataset
    indices = np.arange(len(dataset))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_overall_f1 = 0.0

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
        logging.info(f"--- Starting fold {fold_idx} ---")

        train_ds = Subset(dataset, train_idx)
        val_ds   = Subset(dataset, val_idx)

        # reuse the same collate_fn
        collate_fn = full_loader.collate_fn

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn
        )

        model = load_model(device, args.pretrained)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=10)
        scaler = GradScaler()
        criterion = nn.BCEWithLogitsLoss()

        best_fold_f1 = 0.0

        for epoch in range(1, args.epochs+1):
            # — Train —
            model.train()
            train_losses = []
            y_true, y_pred = [], []

            for batch in train_loader:
                imgs = batch["image"].to(device)
                lbs  = batch["label"].float().unsqueeze(1).to(device)

                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, lbs)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                y_true .extend(lbs.cpu().numpy().flatten().tolist())
                y_pred .extend(preds.flatten().tolist())

            avg_train_loss = np.mean(train_losses)
            train_f1 = f1_score(y_true, y_pred, zero_division=0)
            logging.info(f"[Fold {fold_idx}][Epoch {epoch}] "
                         f"Train loss={avg_train_loss:.4f}  F1={train_f1:.3f}")
            tb_step = (fold_idx-1)*args.epochs + epoch
            writer.add_scalar(f"fold{fold_idx}/train/f1", train_f1, tb_step)

            # — Validate —
            model.eval()
            val_losses = []
            v_true, v_pred = [], []

            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(device)
                    lbs  = batch["label"].float().unsqueeze(1).to(device)
                    logits = model(imgs)
                    loss = criterion(logits, lbs)
                    val_losses.append(loss.item())

                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    v_true .extend(lbs.cpu().numpy().flatten().tolist())
                    v_pred .extend(preds.flatten().tolist())

            avg_val_loss = np.mean(val_losses)
            val_f1 = f1_score(v_true, v_pred, zero_division=0)
            logging.info(f"[Fold {fold_idx}][Epoch {epoch}] "
                         f"Val   loss={avg_val_loss:.4f}  F1={val_f1:.3f}")
            writer.add_scalar(f"fold{fold_idx}/val/f1", val_f1, tb_step)
            scheduler.step(val_f1)

            if val_f1 > best_fold_f1:
                best_fold_f1 = val_f1
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(args.checkpoint_dir, f"fold{fold_idx}_best.pth")
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f"Saved best model of fold {fold_idx} to {ckpt_path}")

        best_overall_f1 = max(best_overall_f1, best_fold_f1)
        logging.info(f"--- Fold {fold_idx} done: best F1 = {best_fold_f1:.3f} ---")

    logging.info(f"=== Training complete, best overall F1 = {best_overall_f1:.3f} ===")

def main():
    args = parse_args()
    setup_logging(args.log_file)
    os.makedirs(args.tb_dir, exist_ok=True)

    global writer
    writer = SummaryWriter(log_dir=args.tb_dir)

    train_folds(args)

    writer.close()

if __name__ == "__main__":
    main()
