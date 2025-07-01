#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from config.preprocessingcopy import create_dataloader
import torch.nn as nn
from my_resnet import ResNet3DWithCBAM

class ResNetFeatureExtractor(ResNet3DWithCBAM):
    """Extracts global pooled features from a pre-trained ResNet3DWithCBAM."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = None  # remove final classification layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

class MLP(nn.Module):
    """Simple three-layer MLP for binary classification."""
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MLP on multimodal pediatric OSA features"
    )
    parser.add_argument(
        "--images-dir", required=True,
        help="Directory of CBCT .npy files"
    )
    parser.add_argument(
        "--label-path", required=True,
        help="CSV file mapping sample IDs to labels"
    )
    parser.add_argument(
        "--radiomics", required=True,
        help="CSV file of radiomics features"
    )
    parser.add_argument(
        "--clinical", required=True,
        help="CSV file of clinical features"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Pretrained ResNet3DWithCBAM weights (.pth)"
    )
    parser.add_argument(
        "--splits", type=int, default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Proportion of data held out for testing"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Extract deep features from CBCT
    loader = create_dataloader(
        args.images_dir, args.label_path,
        batch_size=8, num_workers=0, mode="test"
    )
    extractor = ResNetFeatureExtractor(
        block="basic", layers=[2,2,2,2],
        block_inplanes=[64,128,256,512],
        spatial_dims=3, n_input_channels=1, num_classes=1
    ).to(device)
    extractor.load_state_dict(
        torch.load(args.checkpoint, map_location=device)
    )
    extractor.eval()

    features = {}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            ids = batch["id"]
            out = extractor(images).cpu().numpy()
            for sid, vec in zip(ids, out):
                features[str(sid)] = vec

    deep_df = pd.DataFrame.from_dict(features, orient="index")
    deep_df.index.name = "no"

    # 2) Load radiomics, clinical, and labels; merge
    rad_df = pd.read_csv(args.radiomics, dtype={"no":str}).set_index("no")
    cli_df = pd.read_csv(args.clinical,  dtype={"no":str}).set_index("no")
    lbl_df = pd.read_csv(args.label_path, dtype={"no":str}).set_index("no")[["label"]]

    df = deep_df.join(rad_df, how="inner") \
                .join(cli_df, how="inner") \
                .join(lbl_df, how="inner")
    print(f"Merged dataset shape: {df.shape}")

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(int)

    # 3) Train/test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size,
        stratify=y, random_state=args.seed
    )
    print(f"Train+Val size: {len(y_trainval)}, Test size: {len(y_test)}")

    # 4) Cross-validated training
    skf = StratifiedKFold(
        n_splits=args.splits,
        shuffle=True, random_state=args.seed
    )
    test_preds = []
    fold_aucs, fold_accs, fold_f1s = [], [], []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X_trainval, y_trainval), start=1
    ):
        scaler = StandardScaler().fit(X_trainval[train_idx])
        Xt = scaler.transform(X_trainval[train_idx])
        Xv = scaler.transform(X_trainval[val_idx])
        yt = y_trainval[train_idx]
        yv = y_trainval[val_idx]

        model = MLP(Xt.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(50):
            model.train()
            logits = model(torch.from_numpy(Xt).to(device)).cpu()
            loss = criterion(logits, torch.from_numpy(yt).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(
                model(torch.from_numpy(Xv).to(device))
            ).cpu().numpy()

        auc = roc_auc_score(yv, vp)
        acc = accuracy_score(yv, vp > 0.5)
        f1 = f1_score(yv, vp > 0.5)
        print(f"Fold {fold}: Val AUC={auc:.3f}, Acc={acc:.3f}, F1={f1:.3f}")

        fold_aucs.append(auc)
        fold_accs.append(acc)
        fold_f1s.append(f1)

        with torch.no_grad():
            tp = torch.sigmoid(
                model(torch.from_numpy(scaler.transform(X_test)).to(device))
            ).cpu().numpy()
        test_preds.append(tp)

    print(f"\nCV AUC: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")

    # 5) Test set ensemble
    test_prob = np.mean(test_preds, axis=0)
    test_auc = roc_auc_score(y_test, test_prob)
    test_acc = accuracy_score(y_test, test_prob > 0.5)
    test_f1 = f1_score(y_test, test_prob > 0.5)
    print(f"Test AUC={test_auc:.3f}, Acc={test_acc:.3f}, F1={test_f1:.3f}")

if __name__ == "__main__":
    main()
