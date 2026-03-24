import os
import yaml
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from model import CLIPVisionRegressor
from dataset import TactileCoordinateDataset
from utils import set_seed, save_checkpoint, mae_metric, rmse_metric, euclidean_distance_metric, TrainLogger


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        preds = model(images)
        loss = criterion(preds, targets)

        running_loss += loss.item()
        all_preds.append(preds)
        all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    val_loss = running_loss / len(loader)
    mae = mae_metric(all_preds, all_targets)
    rmse = rmse_metric(all_preds, all_targets)
    euc = euclidean_distance_metric(all_preds, all_targets)

    return val_loss, mae, rmse, euc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = cfg["model"]
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"]["num_workers"]
    save_dir = cfg["train"]["save_dir"]

    os.makedirs(save_dir, exist_ok=True)

    # CLIP image processor
    image_processor = AutoProcessor.from_pretrained(
        model_cfg["pretrained_model_name"]
    ).image_processor

    print(image_processor)

    train_dataset = TactileCoordinateDataset(
        csv_path=cfg["data"]["train_csv"],
        image_dir=cfg["data"]["train_image_dir"],
        output_dim=model_cfg["output_dim"],
        image_processor=image_processor,
    )

    val_dataset = TactileCoordinateDataset(
        csv_path=cfg["data"]["val_csv"],
        image_dir=cfg["data"]["val_image_dir"],
        output_dim=model_cfg["output_dim"],
        image_processor=image_processor,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = CLIPVisionRegressor(
        pretrained_model_name=model_cfg["pretrained_model_name"],
        output_dim=model_cfg["output_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
        freeze_strategy=model_cfg.get("freeze_strategy", "all"),
        unfreeze_layers=model_cfg.get("unfreeze_layers", 2),
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    best_val_loss = float("inf")

    logger = TrainLogger(
        log_dir=os.path.join(save_dir, "logs"),
        experiment_name=cfg.get("experiment_name", "regression"),
        config=cfg,
    )

    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, mae, rmse, euc = validate(model, val_loader, criterion, device)

        print(
            f"[Epoch {epoch+1}/{cfg['train']['epochs']}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"MAE={mae:.4f} "
            f"RMSE={rmse:.4f} "
            f"Euclidean={euc:.4f}"
        )

        logger.log({
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "mae": f"{mae:.6f}",
            "rmse": f"{rmse:.6f}",
            "euclidean": f"{euc:.6f}",
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(save_dir, "best.pt")
            save_checkpoint(model, optimizer, epoch, best_val_loss, ckpt_path)
            print(f"Best model saved to {ckpt_path}")

    logger.close()


if __name__ == "__main__":
    main()
