import copy
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from modelo_mortalidad import MortalityTransformer
from dataloader_physionet import make_dataloaders




CONFIG = {
    "csv_path": "data/processed/physionet2012_transformed.csv",
    "checkpoint": "models/checkpoints/best_model.pt",
    "seed": 42,
    # datos
    "batch_size": 64,
    "max_len": 880,
    "num_workers": 0,
    # modelo
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "d_ff": 256,
    "dropout": 0.1,
    # entrenamiento
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 50,
    "patience": 10,          # early stopping sobre aucpr de validación
    "use_pos_weight": True,  
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_prob, threshold=0.5):
    """AUROC, AUPRC, min(Se, P+) y F1"""
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    min_se_ppv = float(np.max(np.minimum(precision, recall)))

    return {"auroc": auroc, "auprc": auprc, "min_se_ppv": min_se_ppv}

def to_device(batch, device):
    return [b.to(device, non_blocking=True) for b in batch]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0

    for batch in loader:
        t, v, x, static, pad, y = to_device(batch, device)

        optimizer.zero_grad()
        logit, _ = model(t, v, x, static, pad)
        loss = criterion(logit, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_probs, all_labels = [], []

    for batch in loader:
        t, v, x, static, pad, y = to_device(batch, device)

        logit, _ = model(t, v, x, static, pad)
        loss = criterion(logit, y)

        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        all_probs.append(torch.sigmoid(logit).cpu())
        all_labels.append(y.cpu())

    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()

    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / n
    return metrics


def format_metrics(m):
    return (f"loss {m['loss']:.4f} | AUROC {m['auroc']:.4f} | AUPRC {m['auprc']:.4f} | "
            f"min(Se,P+) {m['min_se_ppv']:.4f}")

def training_module(cfg):
    set_seed(cfg["seed"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Datos
    loaders, var_names, n_static = make_dataloaders(
        cfg["csv_path"], batch_size=cfg["batch_size"],
        max_len=cfg["max_len"], num_workers=cfg["num_workers"],
    )

    # Modelo
    model = MortalityTransformer(
        var_names=var_names, n_static=n_static, d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"], dropout=cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}")

    # Pérdida
    pos_weight = None
    if cfg["use_pos_weight"]:
        y_train = torch.stack([s[4] for s in loaders["train"].dataset.samples])
        pos_weight = ((1 - y_train).sum() / y_train.sum()).to(device)
        print(f"pos_weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # Entrenamiento con early stopping
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate(model, loaders["val"], criterion, device)

        print(f"Época {epoch:3d} | train loss {train_loss:.4f} | val {format_metrics(val_metrics)} ")

        if val_metrics["loss"] < best_score:
            best_score = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            torch.save(best_state, cfg["checkpoint"])
            wait = 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"Early stopping en la época {epoch}")
                break

    # Evaluación final con el mejor modelo
    print(f"\nMejor época: {best_epoch} (val loss {best_score:.4f})")
    model.load_state_dict(best_state)

    val_metrics = evaluate(model, loaders["val"], criterion, device)
    test_metrics = evaluate(model, loaders["test"], criterion, device)
    print(f"Val  | {format_metrics(val_metrics)}")
    print(f"Test | {format_metrics(test_metrics)}")

    return model, test_metrics


if __name__ == "__main__":
    training_module(CONFIG)