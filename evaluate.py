import os
import json
import yaml
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from model import CLIPVisionRegressor
from dataset import TactileCoordinateDataset
from utils import mae_metric, rmse_metric, euclidean_distance_metric


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_target_cols(output_dim):
    if output_dim == 2:
        return ["x", "y"]
    elif output_dim == 3:
        return ["x", "y", "z"]
    elif output_dim == 6:
        return ["dX", "dY", "dZ", "Fx", "Fy", "Fz"]
    else:
        raise ValueError(f"Unsupported output_dim: {output_dim}")


def load_label_stats(stats_path, target_cols, device):
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    mean = torch.tensor(
        [stats["mean"][col] for col in target_cols], dtype=torch.float32, device=device
    )
    std = torch.tensor(
        [stats["std"][col] for col in target_cols], dtype=torch.float32, device=device
    )
    return mean, std


def inverse_transform(preds, targets, mean, std):
    preds_original = preds * std.unsqueeze(0) + mean.unsqueeze(0)
    targets_original = targets * std.unsqueeze(0) + mean.unsqueeze(0)
    return preds_original, targets_original


def columnwise_mae(preds, targets, target_cols):
    abs_err = torch.abs(preds - targets)
    mae_per_col = abs_err.mean(dim=0)
    return {col: mae_per_col[i].item() for i, col in enumerate(target_cols)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--label_stats", type=str, default="data/processed/label_stats.json"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = cfg["model"]
    output_dim = model_cfg["output_dim"]
    target_cols = get_target_cols(output_dim)

    # CLIP image processor
    image_processor = AutoProcessor.from_pretrained(
        model_cfg["pretrained_model_name"]
    ).image_processor

    # 이미지 mean/std 로드
    image_stats_path = cfg["data"].get("image_stats")
    image_mean, image_std = None, None
    if image_stats_path and os.path.exists(image_stats_path):
        with open(image_stats_path, "r") as f:
            img_stats = json.load(f)
        image_mean = img_stats["mean"]
        image_std = img_stats["std"]

    test_dataset = TactileCoordinateDataset(
        csv_path=cfg["data"]["test_csv"],
        image_dir=cfg["data"]["test_image_dir"],
        output_dim=output_dim,
        image_processor=image_processor,
        image_mean=image_mean,
        image_std=image_std,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    model = CLIPVisionRegressor(
        pretrained_model_name=model_cfg["pretrained_model_name"],
        output_dim=model_cfg["output_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
        freeze_strategy=model_cfg.get("freeze_strategy", "all"),
        unfreeze_layers=model_cfg.get("unfreeze_layers", 2),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = nn.SmoothL1Loss()

    all_preds = []
    all_targets = []
    running_loss = 0.0

    for batch in test_loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        preds = model(images)
        loss = criterion(preds, targets)

        running_loss += loss.item()
        all_preds.append(preds)
        all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    test_loss = running_loss / len(test_loader)
    test_mae = mae_metric(all_preds, all_targets)
    test_rmse = rmse_metric(all_preds, all_targets)
    test_euc = euclidean_distance_metric(all_preds, all_targets)
    mae_per_col = columnwise_mae(all_preds, all_targets, target_cols)

    print("\n========== [Current Input Scale Metrics] ==========")
    print(f"Test Loss               : {test_loss:.6f}")
    print(f"Test MAE                : {test_mae:.6f}")
    print(f"Test RMSE               : {test_rmse:.6f}")
    print(f"Test Euclidean Distance : {test_euc:.6f}")

    print("\n[Column-wise MAE]")
    for col in target_cols:
        print(f"{col:>4} : {mae_per_col[col]:.6f}")

    if output_dim == 6:
        try:
            mean, std = load_label_stats(args.label_stats, target_cols, device)
            preds_original, targets_original = inverse_transform(
                all_preds, all_targets, mean, std
            )

            orig_mae = mae_metric(preds_original, targets_original)
            orig_rmse = rmse_metric(preds_original, targets_original)
            orig_euc = euclidean_distance_metric(preds_original, targets_original)
            orig_mae_per_col = columnwise_mae(
                preds_original, targets_original, target_cols
            )

            disp_cols = ["dX", "dY", "dZ"]
            force_cols = ["Fx", "Fy", "Fz"]

            disp_mae = sum(orig_mae_per_col[c] for c in disp_cols) / len(disp_cols)
            force_mae = sum(orig_mae_per_col[c] for c in force_cols) / len(force_cols)

            print(
                "\n========== [Original Scale Metrics via Inverse Transform] =========="
            )
            print(f"Original MAE            : {orig_mae:.6f}")
            print(f"Original RMSE           : {orig_rmse:.6f}")
            print(f"Original Euclidean Dist.: {orig_euc:.6f}")

            print("\n[Original Column-wise MAE]")
            for col in target_cols:
                print(f"{col:>4} : {orig_mae_per_col[col]:.6f}")

            print(f"\nOriginal Displacement MAE (dX,dY,dZ): {disp_mae:.6f}")
            print(f"Original Force MAE (Fx,Fy,Fz)       : {force_mae:.6f}")

        except FileNotFoundError:
            print("\n[label_stats.json not found]")
            print("Inverse transform 결과는 출력하지 않습니다.")
        except KeyError as e:
            print(f"\n[label_stats.json key mismatch] Missing key: {e}")
            print("Inverse transform 결과는 출력하지 않습니다.")


if __name__ == "__main__":
    main()
