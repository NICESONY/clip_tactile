"""
원본 이미지를 letterbox padding (비율 유지 + 검정 패딩) 하여 224x224로 저장.
패딩 이미지 기준 mean/std도 계산하여 저장.

Usage:
    python prepare_padded_images.py
"""

import os
import glob
import json
import numpy as np
from PIL import Image
from tqdm import tqdm


TARGET_SIZE = 224


def letterbox(img, target_size=TARGET_SIZE, fill_color=(0, 0, 0)):
    """비율 유지 resize + 패딩으로 정사각형 만들기."""
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (target_size, target_size), fill_color)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas.paste(img_resized, (pad_x, pad_y))
    return canvas


def process_split(src_dir, dst_dir):
    """한 split의 이미지를 letterbox 패딩하여 저장."""
    os.makedirs(dst_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(src_dir, "*.png")))
    print(f"  {src_dir} -> {dst_dir} ({len(image_paths)} images)")

    for img_path in tqdm(image_paths, desc=f"  {os.path.basename(os.path.dirname(src_dir))}"):
        img = Image.open(img_path).convert("RGB")
        padded = letterbox(img)
        padded.save(os.path.join(dst_dir, os.path.basename(img_path)))


def compute_mean_std(image_dir):
    """디렉토리 내 모든 이미지의 채널별 mean/std 계산 ([0,1] 스케일)."""
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    count = 0

    for img_path in tqdm(image_paths, desc=f"  stats {os.path.basename(os.path.dirname(image_dir))}"):
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float64) / 255.0
        pixel_sum += img.reshape(-1, 3).sum(axis=0)
        pixel_sq_sum += (img.reshape(-1, 3) ** 2).sum(axis=0)
        count += img.shape[0] * img.shape[1]

    mean = pixel_sum / count
    std = np.sqrt(pixel_sq_sum / count - mean ** 2)
    return mean, std


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    splits = ["train", "val", "test"]

    # 1) Letterbox 패딩 이미지 생성
    print("=== Letterbox padding ===")
    for split in splits:
        src = os.path.join(base, "data", split, "images")
        dst = os.path.join(base, "data", split, "images_padded")
        if not os.path.exists(src):
            print(f"  [SKIP] {src} not found")
            continue
        process_split(src, dst)

    # 2) train 셋 기준 mean/std 계산
    print("\n=== Computing mean/std (train padded) ===")
    train_padded = os.path.join(base, "data", "train", "images_padded")
    mean, std = compute_mean_std(train_padded)

    stats = {
        "mean": [float(m) for m in mean],
        "std": [float(s) for s in std],
    }

    stats_path = os.path.join(base, "data", "processed", "image_stats_padded.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Mean: R={mean[0]:.6f}, G={mean[1]:.6f}, B={mean[2]:.6f}")
    print(f"  Std:  R={std[0]:.6f}, G={std[1]:.6f}, B={std[2]:.6f}")
    print(f"  Saved to {stats_path}")

    # 참고: 기존 원본 이미지 mean/std
    print("\n  (참고) 기존 원본 이미지 mean/std:")
    print(f"  Mean: R=0.416133, G=0.343241, B=0.326122")
    print(f"  Std:  R=0.344964, G=0.323882, B=0.323514")


if __name__ == "__main__":
    main()
