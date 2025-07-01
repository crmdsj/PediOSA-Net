#!/usr/bin/env python3
# infer.py

import os
import sys
import argparse
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# add project root so we can import preprocessing.py
sys.path.append(os.path.abspath(os.path.join(__file__, "..")))
from preprocessing import create_dataloader
from my_resnet import ResNet3DWithCBAM

def parse_args():
    parser = argparse.ArgumentParser(description="PediOSA-Net Inference Script")
    parser.add_argument(
        "--model_dir", required=True,
        help="Directory containing fold checkpoints named fold0.pth … fold4.pth"
    )
    parser.add_argument(
        "--images_dir", required=True,
        help="Directory of test CBCT .npy files"
    )
    parser.add_argument(
        "--label_path", required=True,
        help="CSV file of test labels"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Inference batch size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader num_workers"
    )
    parser.add_argument(
        "--cache_dir", default="./cache_test",
        help="PersistentDataset cache dir"
    )
    parser.add_argument(
        "--output_path", default="ensemble_preds.npy",
        help="Where to save the ensemble predictions (.npy)"
    )
    parser.add_argument(
        "--log_file", default="inference.log",
        help="Where to save the inference log"
    )
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

def load_models(device, model_dir, num_folds=5):
    models = []
    for fold in range(num_folds):
        ckpt = os.path.join(model_dir, f"fold{fold}.pth")
        if not os.path.exists(ckpt):
            logging.warning(f"Checkpoint not found: {ckpt}")
            continue

        model = ResNet3DWithCBAM(
            block="basic",
            layers=[2,2,2,2],
            block_inplanes=[64,128,256,512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1
        ).to(device)

        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        logging.info(f"Loaded fold {fold} from {ckpt}")

    if not models:
        logging.error("No models loaded; check --model_dir")
        sys.exit(1)
    return models

def main():
    args = parse_args()
    setup_logging(args.log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # DataLoader
    test_loader = create_dataloader(
        images_dir  = args.images_dir,
        label_path  = args.label_path,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        mode        = "test",
        cache_dir   = args.cache_dir
    )
    logging.info(f"Loaded test set: {len(test_loader.dataset)} samples")

    # Load all fold models
    models = load_models(device, args.model_dir)

    all_labels     = []
    fold_predictions = [ [] for _ in models ]
    ensemble_preds   = []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            labels = batch.get("label", None)
            if labels is not None:
                all_labels.extend(labels.numpy().tolist())

            # collect each fold's outputs
            outs = []
            for i, m in enumerate(models):
                logits = m(imgs)
                probs  = torch.sigmoid(logits).cpu().numpy().flatten()
                fold_predictions[i].extend(probs.tolist())
                outs.append(torch.from_numpy(probs))

            # average across folds
            avg = torch.stack(outs, dim=0).mean(dim=0)
            ensemble_preds.extend(avg.numpy().tolist())

    # save ensemble predictions
    np.save(args.output_path, np.array(ensemble_preds))
    logging.info(f"Ensemble predictions saved to {args.output_path}")

    # if labels exist, compute metrics
    if all_labels:
        y_true = np.array(all_labels)
        # per-fold metrics
        for i, preds in enumerate(fold_predictions):
            y_prob = np.array(preds)
            y_pred = (y_prob > 0.5).astype(int)
            logging.info(
                f"Fold {i} | Acc: {accuracy_score(y_true,y_pred):.3f} "
                f"Prec: {precision_score(y_true,y_pred):.3f} "
                f"Rec: {recall_score(y_true,y_pred):.3f} "
                f"F1: {f1_score(y_true,y_pred):.3f} "
                f"AUC: {roc_auc_score(y_true,y_prob):.3f}"
            )
        # ensemble metrics
        y_prob = np.array(ensemble_preds)
        y_pred = (y_prob > 0.5).astype(int)
        logging.info(
            f"Ensemble | Acc: {accuracy_score(y_true,y_pred):.3f} "
            f"Prec: {precision_score(y_true,y_pred):.3f} "
            f"Rec: {recall_score(y_true,y_pred):.3f} "
            f"F1: {f1_score(y_true,y_pred):.3f} "
            f"AUC: {roc_auc_score(y_true,y_prob):.3f}"
        )

if __name__ == "__main__":
    main()
