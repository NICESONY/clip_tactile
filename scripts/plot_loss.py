"""
plot_loss.py

학습 로그 CSV에서 loss / accuracy 그래프 생성.

사용법:
  python plot_loss.py --log outputs/clip_contrastive/logs/clip_contrastive_20260324_102955.csv
  python plot_loss.py --log outputs/clip_vision/logs/clip_vision_20260324_102946.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="Path to log CSV")
    parser.add_argument("--save", type=str, default=None, help="Save path (default: auto)")
    args = parser.parse_args()

    # ── 1. CSV 불러오기 (# 주석 행 건너뛰기) ──
    df = pd.read_csv(args.log, comment="#")
    print(f"Loaded {len(df)} epochs from {args.log}")
    print(f"Columns: {list(df.columns)}")

    # contrastive vs regression 자동 판별
    is_contrastive = "i2t_acc" in df.columns

    if is_contrastive:
        # ── Contrastive: 3개 subplot ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # (1) Loss
        ax = axes[0]
        ax.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#2196F3", linewidth=1.5)
        ax.plot(df["epoch"], df["val_loss"], label="Val Loss", color="#FF5722", linewidth=1.5)
        best_idx = df["val_loss"].idxmin()
        best_epoch = df.loc[best_idx, "epoch"]
        best_val = df.loc[best_idx, "val_loss"]
        ax.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5)
        ax.scatter([best_epoch], [best_val], color="red", zorder=5, s=50)
        ax.annotate(f"best: ep{best_epoch}\nloss={best_val:.4f}",
                    xy=(best_epoch, best_val), fontsize=8,
                    xytext=(10, 10), textcoords="offset points")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train / Val Loss")
        ax.legend()
        ax.grid(alpha=0.3)

        # (2) Accuracy
        ax = axes[1]
        ax.plot(df["epoch"], df["i2t_acc"], label="i2t acc", color="#4CAF50", linewidth=1.5)
        ax.plot(df["epoch"], df["t2i_acc"], label="t2i acc", color="#9C27B0", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("i2t / t2i Accuracy (batch 내)")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

        # (3) Temperature
        ax = axes[2]
        ax.plot(df["epoch"], df["temperature"], color="#FF9800", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Temperature")
        ax.set_title("Learnable Temperature")
        ax.grid(alpha=0.3)

    else:
        # ── Regression: 2개 subplot ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # (1) Loss
        ax = axes[0]
        ax.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#2196F3", linewidth=1.5)
        ax.plot(df["epoch"], df["val_loss"], label="Val Loss", color="#FF5722", linewidth=1.5)
        best_idx = df["val_loss"].idxmin()
        best_epoch = df.loc[best_idx, "epoch"]
        best_val = df.loc[best_idx, "val_loss"]
        ax.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5)
        ax.scatter([best_epoch], [best_val], color="red", zorder=5, s=50)
        ax.annotate(f"best: ep{best_epoch}\nloss={best_val:.4f}",
                    xy=(best_epoch, best_val), fontsize=8,
                    xytext=(10, 10), textcoords="offset points")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train / Val Loss")
        ax.legend()
        ax.grid(alpha=0.3)

        # (2) Metrics
        ax = axes[1]
        metric_cols = [c for c in ["val_mae", "val_rmse", "val_euclidean"] if c in df.columns]
        colors = ["#4CAF50", "#9C27B0", "#FF9800"]
        for col, color in zip(metric_cols, colors):
            ax.plot(df["epoch"], df[col], label=col, color=color, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Validation Metrics")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(args.log.split("/")[-1].replace(".csv", ""), fontsize=13, fontweight="bold")
    plt.tight_layout()

    save_path = args.save or args.log.replace(".csv", ".png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
