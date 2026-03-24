"""
visualize.py

테스트 이미지 N개에 대해 GT vs Predicted 값을 시각화.
Regression(A) / Contrastive(B) 모두 지원.

사용법:
  # Regression
  python visualize.py --mode regression \
      --config configs/regression.yaml \
      --checkpoint outputs/clip_vision/best.pt \
      --num_samples 5

  # Contrastive
  python visualize.py --mode contrastive \
      --config configs/contrastive.yaml \
      --checkpoint outputs/clip_contrastive/best.pt \
      --num_samples 5
"""

import os
import json
import yaml
import argparse
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, CLIPProcessor

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
plt.rcParams["font.size"] = 10


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_label_stats(stats_path, target_cols):
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    mean = {col: stats["mean"][col] for col in target_cols}
    std = {col: stats["std"][col] for col in target_cols}
    return mean, std


def inverse_transform_dict(values, mean, std, cols):
    """normalized values -> original scale"""
    return [values[i] * std[col] + mean[col] for i, col in enumerate(cols)]


# Regression prediction
def predict_regression(cfg, checkpoint_path, num_samples, device):
    from model import CLIPVisionRegressor
    from dataset import TactileCoordinateDataset

    model_cfg = cfg["model"]
    output_dim = model_cfg["output_dim"]

    if output_dim == 6:
        target_cols = ["dX", "dY", "dZ", "Fx", "Fy", "Fz"]
    elif output_dim == 3:
        target_cols = ["x", "y", "z"]
    else:
        target_cols = ["x", "y"]

    image_processor = AutoProcessor.from_pretrained(
        model_cfg["pretrained_model_name"]
    ).image_processor

    test_dataset = TactileCoordinateDataset(
        csv_path=cfg["data"]["test_csv"],
        image_dir=cfg["data"]["test_image_dir"],
        output_dim=output_dim,
        image_processor=image_processor,
    )

    model = CLIPVisionRegressor(
        pretrained_model_name=model_cfg["pretrained_model_name"],
        output_dim=model_cfg["output_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
        freeze_strategy=model_cfg.get("freeze_strategy", "all"),
        unfreeze_layers=model_cfg.get("unfreeze_layers", 2),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    results = []
    with torch.no_grad():
        for idx in range(min(num_samples, len(test_dataset))):
            sample = test_dataset[idx]
            image_tensor = sample["image"].unsqueeze(0).to(device)
            gt = sample["target"].numpy()
            pred = model(image_tensor).squeeze(0).cpu().numpy()
            image_name = sample["image_name"]

            results.append({
                "image_name": image_name,
                "gt": gt,
                "pred": pred,
            })

    return results, target_cols


# Contrastive prediction (retrieval)
def predict_contrastive(cfg, checkpoint_path, num_samples, device):
    from contrastive_model import CLIPContrastive
    from contrastive_dataset import TactileContrastiveDataset, label_to_text

    import pandas as pd

    model_cfg = cfg["model"]
    label_cols = cfg["data"]["label_cols"]

    processor = CLIPProcessor.from_pretrained(model_cfg["pretrained_model_name"])
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer

    model = CLIPContrastive(
        pretrained_model_name=model_cfg["pretrained_model_name"],
        freeze_image_encoder=model_cfg["freeze_image_encoder"],
        freeze_text_encoder=model_cfg["freeze_text_encoder"],
        learnable_temperature=model_cfg["learnable_temperature"],
        init_temperature=model_cfg["init_temperature"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build candidate pool
    csv_paths = [
        cfg["data"]["train_csv"],
        cfg["data"]["val_csv"],
        cfg["data"]["test_csv"],
    ]
    all_texts, all_values = [], []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for i in range(len(df)):
            row = df.iloc[i]
            text = label_to_text(row, label_cols)
            values = [row[col] for col in label_cols]
            all_texts.append(text)
            all_values.append(values)

    all_values = torch.tensor(all_values, dtype=torch.float32)
    unique_map = {}
    for i, text in enumerate(all_texts):
        if text not in unique_map:
            unique_map[text] = i
    unique_indices = list(unique_map.values())
    unique_texts = [all_texts[i] for i in unique_indices]
    unique_values = all_values[unique_indices]

    # Encode candidates
    cand_embeds_list = []
    with torch.no_grad():
        for start in range(0, len(unique_texts), 256):
            batch_texts = unique_texts[start : start + 256]
            tok = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            )
            embeds = model.encode_text(
                tok["input_ids"].to(device),
                tok["attention_mask"].to(device),
            )
            cand_embeds_list.append(embeds.cpu())
    candidate_embeds = torch.cat(cand_embeds_list, dim=0).to(device)
    candidate_values = unique_values.to(device)

    # Test dataset
    test_dataset = TactileContrastiveDataset(
        csv_path=cfg["data"]["test_csv"],
        image_dir=cfg["data"]["test_image_dir"],
        label_cols=label_cols,
        image_processor=image_processor,
        tokenizer=tokenizer,
    )

    results = []
    with torch.no_grad():
        for idx in range(min(num_samples, len(test_dataset))):
            sample = test_dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            gt = sample["label_values"].numpy()

            img_embed = model.encode_image(pixel_values)
            sim = img_embed @ candidate_embeds.t()
            top1_idx = sim.argmax(dim=1).item()
            pred = candidate_values[top1_idx].cpu().numpy()
            image_name = sample["image_name"]

            results.append({
                "image_name": image_name,
                "gt": gt,
                "pred": pred,
            })

    return results, label_cols


# Visualization
def visualize_results(results, target_cols, image_dir, label_stats_path, save_path):
    n = len(results)

    # Load label stats for inverse transform
    mean, std = None, None
    try:
        mean, std = load_label_stats(label_stats_path, target_cols)
    except (FileNotFoundError, KeyError):
        pass

    disp_cols = [c for c in ["dX", "dY", "dZ"] if c in target_cols]
    force_cols = [c for c in ["Fx", "Fy", "Fz"] if c in target_cols]

    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, res in enumerate(results):
        image_path = os.path.join(image_dir, res["image_name"])
        img = Image.open(image_path).convert("RGB")

        gt_norm = res["gt"]
        pred_norm = res["pred"]

        #  Column 1: Original image 
        ax_img = axes[i, 0]
        ax_img.imshow(img)
        ax_img.set_title(f"[{i}] {res['image_name']}", fontsize=12, fontweight="bold")
        ax_img.axis("off")

        #  Column 2: Normalized scale bar chart 
        ax_norm = axes[i, 1]
        x_pos = np.arange(len(target_cols))
        width = 0.35

        bars_gt = ax_norm.bar(x_pos - width / 2, gt_norm, width, label="GT", color="#2196F3", alpha=0.8)
        bars_pred = ax_norm.bar(x_pos + width / 2, pred_norm, width, label="Pred", color="#FF5722", alpha=0.8)

        ax_norm.set_xticks(x_pos)
        ax_norm.set_xticklabels(target_cols)
        ax_norm.set_ylabel("Normalized Value")
        ax_norm.set_title("GT vs Pred (Normalized)", fontsize=11)
        ax_norm.legend(loc="upper right")
        ax_norm.axhline(y=0, color="gray", linewidth=0.5)
        ax_norm.grid(axis="y", alpha=0.3)

        #  Column 3: Original scale or error 
        ax_orig = axes[i, 2]

        if mean is not None and std is not None:
            gt_orig = inverse_transform_dict(gt_norm, mean, std, target_cols)
            pred_orig = inverse_transform_dict(pred_norm, mean, std, target_cols)
            error = [abs(g - p) for g, p in zip(gt_orig, pred_orig)]

            bars_gt2 = ax_orig.bar(x_pos - width / 2, gt_orig, width, label="GT", color="#2196F3", alpha=0.8)
            bars_pred2 = ax_orig.bar(x_pos + width / 2, pred_orig, width, label="Pred", color="#FF5722", alpha=0.8)

            ax_orig.set_xticks(x_pos)
            ax_orig.set_xticklabels(target_cols)
            ax_orig.set_ylabel("Original Scale Value")
            ax_orig.set_title("GT vs Pred (Original Scale)", fontsize=11)
            ax_orig.legend(loc="upper right")
            ax_orig.axhline(y=0, color="gray", linewidth=0.5)
            ax_orig.grid(axis="y", alpha=0.3)

            # Error text at bottom
            err_text = "  |  ".join(f"{col}: {e:.4f}" for col, e in zip(target_cols, error))
            ax_orig.set_xlabel(f"Error: {err_text}", fontsize=8)
        else:
            # No label stats: show error bar chart
            error_norm = np.abs(gt_norm - pred_norm)
            colors = ["#4CAF50" if e < 0.3 else "#FF9800" if e < 0.6 else "#F44336" for e in error_norm]
            ax_orig.bar(x_pos, error_norm, color=colors, alpha=0.8)
            ax_orig.set_xticks(x_pos)
            ax_orig.set_xticklabels(target_cols)
            ax_orig.set_ylabel("Absolute Error")
            ax_orig.set_title("Per-column Error (Normalized)", fontsize=11)
            ax_orig.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["regression", "contrastive"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--label_stats", type=str, default="data/processed/label_stats.json")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "regression":
        results, target_cols = predict_regression(cfg, args.checkpoint, args.num_samples, device)
        image_dir = cfg["data"]["test_image_dir"]
        default_save = f"outputs/{cfg['experiment_name']}_vis.png"
    else:
        results, target_cols = predict_contrastive(cfg, args.checkpoint, args.num_samples, device)
        image_dir = cfg["data"]["test_image_dir"]
        default_save = f"outputs/{cfg['experiment_name']}_vis.png"

    save_path = args.save_path or default_save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    visualize_results(results, target_cols, image_dir, args.label_stats, save_path)

    # Print summary
    print(f"\n{'='*50}")
    print(f"  Mode: {args.mode}")
    print(f"  Samples: {len(results)}")
    print(f"  Saved: {save_path}")
    print(f"{'='*50}")

    for i, res in enumerate(results):
        gt = res["gt"]
        pred = res["pred"]
        err = np.abs(gt - pred)
        print(f"\n  [{i}] {res['image_name']}")
        for j, col in enumerate(target_cols):
            print(f"       {col:>3}: GT={gt[j]:+.4f}  Pred={pred[j]:+.4f}  Err={err[j]:.4f}")


if __name__ == "__main__":
    main()
