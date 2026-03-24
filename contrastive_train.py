"""
contrastive_train.py

CLIP contrastive fine-tuning:
  tactile image <-> 6-axis force/torque text
"""

import os
import yaml
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from contrastive_model import CLIPContrastive, clip_contrastive_loss
from contrastive_dataset import TactileContrastiveDataset
from utils import set_seed, save_checkpoint, TrainLogger


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        logits_per_image, logits_per_text = model(
            pixel_values, input_ids, attention_mask
        )
        loss = clip_contrastive_loss(logits_per_image, logits_per_text)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    total_correct_i2t = 0
    total_correct_t2i = 0
    total_samples = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        B = pixel_values.size(0)

        logits_per_image, logits_per_text = model(
            pixel_values, input_ids, attention_mask
        )
        loss = clip_contrastive_loss(logits_per_image, logits_per_text)
        running_loss += loss.item()

        # batch 내 top-1 accuracy
        preds_i2t = logits_per_image.argmax(dim=1)
        preds_t2i = logits_per_text.argmax(dim=1)
        labels = torch.arange(B, device=device)

        total_correct_i2t += (preds_i2t == labels).sum().item()
        total_correct_t2i += (preds_t2i == labels).sum().item()
        total_samples += B

    val_loss = running_loss / len(loader)
    acc_i2t = total_correct_i2t / total_samples
    acc_t2i = total_correct_t2i / total_samples

    return val_loss, acc_i2t, acc_t2i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = cfg["model"]
    save_dir = cfg["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Processor (image + text)
    processor = CLIPProcessor.from_pretrained(model_cfg["pretrained_model_name"])
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer

    label_cols = cfg["data"]["label_cols"]

    # Datasets
    train_dataset = TactileContrastiveDataset(
        csv_path=cfg["data"]["train_csv"],
        image_dir=cfg["data"]["train_image_dir"],
        label_cols=label_cols,
        image_processor=image_processor,
        tokenizer=tokenizer,
    )
    val_dataset = TactileContrastiveDataset(
        csv_path=cfg["data"]["val_csv"],
        image_dir=cfg["data"]["val_image_dir"],
        label_cols=label_cols,
        image_processor=image_processor,
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        drop_last=True,  # contrastive loss needs consistent batch size
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        drop_last=True,
    )

    # Model
    model = CLIPContrastive(
        pretrained_model_name=model_cfg["pretrained_model_name"],
        freeze_image_encoder=model_cfg["freeze_image_encoder"],
        freeze_text_encoder=model_cfg["freeze_text_encoder"],
        learnable_temperature=model_cfg["learnable_temperature"],
        init_temperature=model_cfg["init_temperature"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    best_val_loss = float("inf")

    logger = TrainLogger(
        log_dir=os.path.join(save_dir, "logs"),
        experiment_name=cfg.get("experiment_name", "contrastive"),
        config=cfg,
    )

    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, acc_i2t, acc_t2i = validate(model, val_loader, device)

        temp = model.temperature.item()

        print(
            f"[Epoch {epoch+1}/{cfg['train']['epochs']}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"i2t_acc={acc_i2t:.4f}  "
            f"t2i_acc={acc_t2i:.4f}  "
            f"temp={temp:.4f}"
        )

        logger.log({
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "i2t_acc": f"{acc_i2t:.6f}",
            "t2i_acc": f"{acc_t2i:.6f}",
            "temperature": f"{temp:.6f}",
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(save_dir, "best.pt")
            save_checkpoint(model, optimizer, epoch, best_val_loss, ckpt_path)
            print(f"  -> Best model saved to {ckpt_path}")

    logger.close()


if __name__ == "__main__":
    main()
