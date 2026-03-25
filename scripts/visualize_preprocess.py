"""
원본 이미지 vs CLIP crop vs Letterbox padding 시각화.

Usage:
    python visualize_preprocess.py --image_dir data/test/images --num_samples 5
    python visualize_preprocess.py --image_dir data/test/images --padded_dir data/test/images_padded --padded_stats data/processed/image_stats_padded.json --num_samples 5
"""

import argparse
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor


def denormalize(tensor, mean, std):
    """[C,H,W] normalized tensor → [H,W,C] numpy uint8 for display."""
    img = tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="data/test/images")
    parser.add_argument("--padded_dir", type=str, default=None,
                        help="패딩 이미지 디렉토리 (미지정시 image_dir 기반 자동 탐색)")
    parser.add_argument("--padded_stats", type=str, default="data/processed/image_stats_padded.json")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="outputs/preprocess_comparison.png")
    args = parser.parse_args()

    # 패딩 디렉토리 자동 탐색
    if args.padded_dir is None:
        auto_padded = args.image_dir.replace("/images", "/images_padded")
        if os.path.exists(auto_padded):
            args.padded_dir = auto_padded

    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    image_paths = image_paths[: args.num_samples]

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = processor.image_processor

    # 기존 CLIP crop용 mean/std (원본 이미지 통계)
    orig_mean = [0.41613302234013877, 0.34324077486038207, 0.3261217144838969]
    orig_std = [0.3449644943991341, 0.3238820460207181, 0.32351404629653335]

    # 패딩 이미지용 mean/std
    pad_mean, pad_std = orig_mean, orig_std  # fallback
    if os.path.exists(args.padded_stats):
        with open(args.padded_stats, "r") as f:
            stats = json.load(f)
        pad_mean = stats["mean"]
        pad_std = stats["std"]
        print(f"Padded stats: mean={pad_mean}, std={pad_std}")

    n = len(image_paths)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Original (640x480)",
        "CLIP Crop (224x224)\nresize+center crop",
        "Letterbox Padding (224x224)\n비율 유지+검정 패딩",
    ]

    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")
        fname = os.path.basename(img_path)

        # 1) 원본
        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title(col_titles[0], fontsize=11, fontweight="bold")
        axes[i, 0].set_ylabel(fname, fontsize=9, rotation=0, labelpad=60, va="center")

        # 2) CLIP Crop (resize shortest→224, center crop 224x224)
        crop_out = image_processor(
            images=img,
            return_tensors="pt",
            image_mean=orig_mean,
            image_std=orig_std,
        )["pixel_values"].squeeze(0)
        axes[i, 1].imshow(denormalize(crop_out, orig_mean, orig_std))
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 1].set_title(col_titles[1], fontsize=11, fontweight="bold")

        # 3) Letterbox Padding
        if args.padded_dir:
            padded_path = os.path.join(args.padded_dir, fname)
            img_padded = Image.open(padded_path).convert("RGB")
        else:
            # padded 디렉토리 없으면 직접 패딩
            from prepare_padded_images import letterbox
            img_padded = letterbox(img, target_size=224)

        pad_out = image_processor(
            images=img_padded,
            return_tensors="pt",
            do_center_crop=False,
            do_resize=False,
            image_mean=pad_mean,
            image_std=pad_std,
        )["pixel_values"].squeeze(0)
        axes[i, 2].imshow(denormalize(pad_out, pad_mean, pad_std))
        axes[i, 2].axis("off")
        if i == 0:
            axes[i, 2].set_title(col_titles[2], fontsize=11, fontweight="bold")

    plt.suptitle("CLIP Crop vs Letterbox Padding", fontsize=14, y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    plt.savefig(args.save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.save_path}")
    plt.show()


if __name__ == "__main__":
    main()
