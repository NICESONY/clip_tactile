"""
contrastive_dataset.py

6축 force/torque 값을 문자열로 변환하여 CLIP text encoder 입력으로 사용.
예: "dX 0.12, dY -0.31, dZ 1.45, Fx 0.02, Fy -0.07, Fz 0.11"
"""

import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


def label_to_text(row, label_cols):
    """6축 값을 CLIP text encoder용 문자열로 변환."""
    parts = [f"{col} {row[col]:.4f}" for col in label_cols]
    return ", ".join(parts)


class TactileContrastiveDataset(Dataset):
    def __init__(self, csv_path, image_dir, label_cols, image_processor, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.label_cols = label_cols
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        for col in ["image_name"] + label_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        # 미리 text 문자열 생성
        self.texts = [
            label_to_text(self.df.iloc[i], label_cols) for i in range(len(self.df))
        ]

    def __len__(self):
        return len(self.df)

    def get_label_values(self, idx):
        """idx번째 샘플의 6축 float 값 반환."""
        row = self.df.iloc[idx]
        return [row[col] for col in self.label_cols]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["image_name"])
        image = Image.open(image_path).convert("RGB")

        # image processing
        pixel_values = self.image_processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

        # text tokenization
        text = self.texts[idx]
        text_inputs = self.tokenizer(
            text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=77,
        )

        label_values = torch.tensor(
            [row[col] for col in self.label_cols], dtype=torch.float32
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "label_values": label_values,
            "text": text,
            "image_name": row["image_name"],
        }
