import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit

sys.path.append(os.path.dirname(__file__))
from net import WinProbNet, FEATURE_NAMES

DATA_PATH  = os.path.join(os.path.dirname(__file__), "../data/game_states.parquet")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")
MEAN_PATH  = os.path.join(os.path.dirname(__file__), "feature_mean.npy")
STD_PATH   = os.path.join(os.path.dirname(__file__), "feature_std.npy")

EPOCHS     = 50
BATCH_SIZE = 4096
LR         = 1e-3
SEED       = 42


def load_data():
    df = pd.read_parquet(DATA_PATH)
    df = df.dropna(subset=FEATURE_NAMES + ["win"])
    X      = df[FEATURE_NAMES].values.astype(np.float32)
    y      = df["win"].values.astype(np.float32)
    groups = df["game_id"].values
    return X, y, groups


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading data...")
    X, y, groups = load_data()
    print(f"  {len(X)} rows | {X.shape[1]} features | home win rate: {y.mean():.3f}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std

    np.save(MEAN_PATH, mean)
    np.save(STD_PATH,  std)
    print(f"  Train: {len(X_train)} rows | Val: {len(X_val)} rows")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = WinProbNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_epoch    = 0

    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        correct  = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds     = model(xb)
                val_loss += criterion(preds, yb).item() * len(xb)
                correct  += ((preds > 0.5) == yb.bool()).sum().item()
        val_loss /= len(val_ds)
        val_acc   = correct / len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), MODEL_PATH)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} | acc={val_acc:.3f}")

    print(f"\nBest epoch {best_epoch} | val_loss={best_val_loss:.4f} → {MODEL_PATH}")


if __name__ == "__main__":
    main()