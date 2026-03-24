"""
benchmark.py

Regression vs Contrastive 추론 속도 비교.

사용법:
  # 둘 다 비교
  python benchmark.py

  # Regression만
  python benchmark.py --mode regression --config configs/regression_full.yaml --checkpoint outputs/clip_vision_full/best.pt

  # Contrastive만
  python benchmark.py --mode contrastive --config configs/contrastive.yaml --checkpoint outputs/clip_contrastive/best.pt

  # 측정 횟수 변경
  python benchmark.py --num_samples 200
"""

import time
import argparse
import yaml
import pandas as pd

import torch
from transformers import AutoProcessor, CLIPProcessor


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def benchmark_regression(config_path, checkpoint_path, num_samples, device):
    from model import CLIPVisionRegressor
    from dataset import TactileCoordinateDataset

    cfg = load_config(config_path)
    model_cfg = cfg["model"]

    image_processor = AutoProcessor.from_pretrained(
        model_cfg["pretrained_model_name"]
    ).image_processor

    dataset = TactileCoordinateDataset(
        csv_path=cfg["data"]["test_csv"],
        image_dir=cfg["data"]["test_image_dir"],
        output_dim=model_cfg["output_dim"],
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

    # Warm up
    with torch.no_grad():
        sample = dataset[0]
        _ = model(sample["image"].unsqueeze(0).to(device))
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure
    times = []
    n = min(num_samples, len(dataset))
    with torch.no_grad():
        for i in range(n):
            img = dataset[i]["image"].unsqueeze(0).to(device)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            pred = model(img)

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            times.append(t1 - t0)

    return times, None


def benchmark_contrastive(config_path, checkpoint_path, num_samples, device):
    from contrastive_model import CLIPContrastive
    from contrastive_dataset import TactileContrastiveDataset, label_to_text

    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    label_cols = cfg["data"]["label_cols"]

    processor = CLIPProcessor.from_pretrained(model_cfg["pretrained_model_name"])
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor

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

    # Build candidate pool (measure setup time)
    t_setup_start = time.perf_counter()

    csv_paths = [cfg["data"]["train_csv"], cfg["data"]["val_csv"], cfg["data"]["test_csv"]]
    all_texts, all_values = [], []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for i in range(len(df)):
            row = df.iloc[i]
            text = label_to_text(row, label_cols)
            values = [row[col] for col in label_cols]
            all_texts.append(text)
            all_values.append(values)

    all_values_t = torch.tensor(all_values, dtype=torch.float32)
    unique_map = {}
    for i, text in enumerate(all_texts):
        if text not in unique_map:
            unique_map[text] = i
    unique_indices = list(unique_map.values())
    unique_texts = [all_texts[i] for i in unique_indices]
    unique_values = all_values_t[unique_indices]

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
                tok["input_ids"].to(device), tok["attention_mask"].to(device)
            )
            cand_embeds_list.append(embeds.cpu())

    candidate_embeds = torch.cat(cand_embeds_list, 0).to(device)
    candidate_values = unique_values.to(device)

    if device == "cuda":
        torch.cuda.synchronize()
    t_setup_end = time.perf_counter()
    setup_time = (t_setup_end - t_setup_start) * 1000

    # Test dataset
    dataset = TactileContrastiveDataset(
        csv_path=cfg["data"]["test_csv"],
        image_dir=cfg["data"]["test_image_dir"],
        label_cols=label_cols,
        image_processor=image_processor,
        tokenizer=tokenizer,
    )

    # Warm up
    with torch.no_grad():
        s = dataset[0]
        _ = model.encode_image(s["pixel_values"].unsqueeze(0).to(device))
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure
    times = []
    n = min(num_samples, len(dataset))
    with torch.no_grad():
        for i in range(n):
            pv = dataset[i]["pixel_values"].unsqueeze(0).to(device)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            img_embed = model.encode_image(pv)
            sim = img_embed @ candidate_embeds.t()
            top1_idx = sim.argmax(dim=1).item()
            pred = candidate_values[top1_idx]

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            times.append(t1 - t0)

    return times, setup_time


def print_results(name, times, setup_time=None):
    avg = sum(times) / len(times) * 1000
    min_t = min(times) * 1000
    max_t = max(times) * 1000
    fps = 1000 / avg

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    if setup_time is not None:
        print(f"  Setup (one-time)  : {setup_time:.0f} ms")
    print(f"  Samples measured  : {len(times)}")
    print(f"  Avg inference     : {avg:.2f} ms/image")
    print(f"  Min               : {min_t:.2f} ms")
    print(f"  Max               : {max_t:.2f} ms")
    print(f"  FPS               : {fps:.1f}")

    return avg, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["regression", "contrastive", "both"],
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Num samples: {args.num_samples}")

    results = {}

    if args.mode in ("regression", "both"):
        cfg_path = args.config or "configs/regression_full.yaml"
        ckpt_path = args.checkpoint or "outputs/clip_vision_full/best.pt"
        times, _ = benchmark_regression(cfg_path, ckpt_path, args.num_samples, device)
        avg, fps = print_results("Regression (A-3 Full)", times)
        results["regression"] = {"avg": avg, "fps": fps}

    if args.mode in ("contrastive", "both"):
        cfg_path = args.config if args.mode == "contrastive" else "configs/contrastive.yaml"
        ckpt_path = (
            args.checkpoint if args.mode == "contrastive"
            else "outputs/clip_contrastive/best.pt"
        )
        times, setup_time = benchmark_contrastive(
            cfg_path, ckpt_path, args.num_samples, device
        )
        avg, fps = print_results("Contrastive (B)", times, setup_time)
        results["contrastive"] = {"avg": avg, "fps": fps, "setup": setup_time}

    if args.mode == "both" and len(results) == 2:
        r = results["regression"]
        c = results["contrastive"]
        ratio = c["avg"] / r["avg"]

        print(f"\n{'='*50}")
        print(f"  Comparison")
        print(f"{'='*50}")
        print(f"  Regression  : {r['avg']:.2f} ms ({r['fps']:.1f} FPS)")
        print(f"  Contrastive : {c['avg']:.2f} ms ({c['fps']:.1f} FPS)")
        print(f"  Speed ratio : Contrastive is {ratio:.2f}x vs Regression")
        print(f"  Contrastive setup (one-time): {c['setup']:.0f} ms")


if __name__ == "__main__":
    main()
