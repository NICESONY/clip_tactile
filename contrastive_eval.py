"""
contrastive_eval.py

Retrieval evaluation:
  1. 후보 라벨 풀(~5000개)의 text embedding을 미리 계산
  2. 테스트 이미지마다 전체 후보와 similarity 계산
  3. Top-1, Top-5 retrieval accuracy
  4. 검색된 라벨 vs 실제 라벨 간 MAE / RMSE
"""

import os
import json
import yaml
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from contrastive_model import CLIPContrastive
from contrastive_dataset import TactileContrastiveDataset, label_to_text
from utils import mae_metric, rmse_metric, euclidean_distance_metric

import pandas as pd


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_candidate_pool(cfg, label_cols, tokenizer, device, model):
    """
    후보 라벨 풀 구성: text embedding + label values.
    candidate_pool == "all" 이면 train+val+test 전부 사용.
    """
    pool_mode = cfg["eval"].get("candidate_pool", "all")

    csv_paths = []
    if pool_mode == "all":
        csv_paths = [
            cfg["data"]["train_csv"],
            cfg["data"]["val_csv"],
            cfg["data"]["test_csv"],
        ]
    elif pool_mode == "test":
        csv_paths = [cfg["data"]["test_csv"]]
    else:
        csv_paths = [pool_mode]  # custom path

    all_texts = []
    all_values = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for i in range(len(df)):
            row = df.iloc[i]
            text = label_to_text(row, label_cols)
            values = [row[col] for col in label_cols]
            all_texts.append(text)
            all_values.append(values)

    all_values = torch.tensor(all_values, dtype=torch.float32)  # [N, 6]

    # 중복 제거 (같은 라벨 문자열)
    unique_map = {}
    for i, text in enumerate(all_texts):
        if text not in unique_map:
            unique_map[text] = i

    unique_indices = list(unique_map.values())
    unique_texts = [all_texts[i] for i in unique_indices]
    unique_values = all_values[unique_indices]

    print(f"Candidate pool: {len(all_texts)} total -> {len(unique_texts)} unique labels")

    # Encode all candidate texts
    model.eval()
    text_embeds_list = []
    batch_size = 256

    with torch.no_grad():
        for start in tqdm(range(0, len(unique_texts), batch_size), desc="Encoding candidates"):
            batch_texts = unique_texts[start : start + batch_size]
            tok = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            )
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok["attention_mask"].to(device)

            embeds = model.encode_text(input_ids, attention_mask)  # [B, D]
            text_embeds_list.append(embeds.cpu())

    candidate_embeds = torch.cat(text_embeds_list, dim=0)  # [N_unique, D]

    return candidate_embeds, unique_values, unique_texts


def load_label_stats(stats_path, label_cols, device):
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    mean = torch.tensor([stats["mean"][c] for c in label_cols], dtype=torch.float32, device=device)
    std = torch.tensor([stats["std"][c] for c in label_cols], dtype=torch.float32, device=device)
    return mean, std


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--label_stats", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = cfg["model"]
    label_cols = cfg["data"]["label_cols"]
    top_k_list = cfg["eval"]["top_k"]

    if args.label_stats is None:
        args.label_stats = cfg["data"].get("label_stats", "data/processed/label_stats.json")

    # Processor
    processor = CLIPProcessor.from_pretrained(model_cfg["pretrained_model_name"])
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer

    # Model
    model = CLIPContrastive(
        pretrained_model_name=model_cfg["pretrained_model_name"],
        freeze_image_encoder=model_cfg["freeze_image_encoder"],
        freeze_text_encoder=model_cfg["freeze_text_encoder"],
        learnable_temperature=model_cfg["learnable_temperature"],
        init_temperature=model_cfg["init_temperature"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from {args.checkpoint} (epoch {ckpt['epoch']})")

    # ── 1. 후보 라벨 풀 구축 ──
    candidate_embeds, candidate_values, candidate_texts = build_candidate_pool(
        cfg, label_cols, tokenizer, device, model
    )
    candidate_embeds = candidate_embeds.to(device)  # [N, D]
    candidate_values = candidate_values.to(device)  # [N, 6]

    # 이미지 mean/std 로드
    image_stats_path = cfg["data"].get("image_stats")
    image_mean, image_std = None, None
    if image_stats_path and os.path.exists(image_stats_path):
        with open(image_stats_path, "r") as f:
            img_stats = json.load(f)
        image_mean = img_stats["mean"]
        image_std = img_stats["std"]

    # ── 2. 테스트 이미지 임베딩 & 검색 ──
    test_dataset = TactileContrastiveDataset(
        csv_path=cfg["data"]["test_csv"],
        image_dir=cfg["data"]["test_image_dir"],
        label_cols=label_cols,
        image_processor=image_processor,
        tokenizer=tokenizer,
        image_mean=image_mean,
        image_std=image_std,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    all_gt_values = []      # ground truth [N_test, 6]
    all_gt_texts = []       # ground truth text
    all_retrieved_values = {k: [] for k in top_k_list}  # top-k retrieved values
    all_top1_texts = []
    topk_correct = {k: 0 for k in top_k_list}
    total = 0

    for batch in tqdm(test_loader, desc="Retrieval"):
        pixel_values = batch["pixel_values"].to(device)
        gt_values = batch["label_values"]  # [B, 6]
        gt_texts = batch["text"]           # list of str

        image_embeds = model.encode_image(pixel_values)  # [B, D]

        # similarity: [B, N_candidates]
        sim = image_embeds @ candidate_embeds.t()

        max_k = max(top_k_list)
        topk_indices = sim.topk(max_k, dim=1).indices  # [B, max_k]

        for i in range(pixel_values.size(0)):
            gt_text = gt_texts[i]
            all_gt_values.append(gt_values[i])
            all_gt_texts.append(gt_text)

            retrieved_texts_topk = [candidate_texts[topk_indices[i, j].item()] for j in range(max_k)]
            all_top1_texts.append(retrieved_texts_topk[0])

            for k in top_k_list:
                # exact match: 같은 text string이 top-k 안에 있는지
                if gt_text in retrieved_texts_topk[:k]:
                    topk_correct[k] += 1

                # top-k에서 가장 가까운 값 (top-1 값 사용 for MAE/RMSE)
                if k not in all_retrieved_values:
                    all_retrieved_values[k] = []

            # top-1 retrieved values for MAE/RMSE
            top1_idx = topk_indices[i, 0].item()
            all_retrieved_values[1].append(candidate_values[top1_idx].cpu())

            # top-5: 가장 유사도 높은 것의 값
            for k in top_k_list:
                if k > 1:
                    topk_idx = topk_indices[i, 0].item()  # still use top-1 for value
                    all_retrieved_values[k].append(candidate_values[topk_idx].cpu())

        total += pixel_values.size(0)

    # ── 3. 결과 출력 ──
    print(f"\n{'='*60}")
    print(f"  CLIP Contrastive Retrieval Evaluation")
    print(f"  Test samples: {total}")
    print(f"  Candidate pool: {candidate_embeds.size(0)} unique labels")
    print(f"{'='*60}")

    # Top-K Retrieval Accuracy (exact match)
    print("\n[Retrieval Accuracy - Exact Match]")
    for k in top_k_list:
        acc = topk_correct[k] / total
        print(f"  Top-{k} Accuracy: {acc:.4f} ({topk_correct[k]}/{total})")

    # MAE / RMSE (normalized scale)
    all_gt_values = torch.stack(all_gt_values, dim=0)  # [N_test, 6]
    top1_retrieved = torch.stack(all_retrieved_values[1], dim=0)  # [N_test, 6]

    mae = mae_metric(top1_retrieved, all_gt_values)
    rmse = rmse_metric(top1_retrieved, all_gt_values)
    euc = euclidean_distance_metric(top1_retrieved, all_gt_values)

    print("\n[Top-1 Retrieved vs GT - Normalized Scale]")
    print(f"  MAE               : {mae:.6f}")
    print(f"  RMSE              : {rmse:.6f}")
    print(f"  Euclidean Distance: {euc:.6f}")

    # Column-wise MAE
    abs_err = torch.abs(top1_retrieved - all_gt_values)
    col_mae = abs_err.mean(dim=0)

    print("\n  [Column-wise MAE]")
    for i, col in enumerate(label_cols):
        print(f"    {col:>4} : {col_mae[i].item():.6f}")

    # ── 4. Original scale (inverse transform) ──
    try:
        mean, std = load_label_stats(args.label_stats, label_cols, device)
        gt_orig = all_gt_values.to(device) * std.unsqueeze(0) + mean.unsqueeze(0)
        ret_orig = top1_retrieved.to(device) * std.unsqueeze(0) + mean.unsqueeze(0)

        orig_mae = mae_metric(ret_orig, gt_orig)
        orig_rmse = rmse_metric(ret_orig, gt_orig)
        orig_euc = euclidean_distance_metric(ret_orig, gt_orig)

        abs_err_orig = torch.abs(ret_orig - gt_orig)
        col_mae_orig = abs_err_orig.mean(dim=0)

        disp_cols = ["dX", "dY", "dZ"]
        force_cols = ["Fx", "Fy", "Fz"]

        disp_mae = sum(col_mae_orig[label_cols.index(c)].item() for c in disp_cols) / 3
        force_mae = sum(col_mae_orig[label_cols.index(c)].item() for c in force_cols) / 3

        print("\n[Top-1 Retrieved vs GT - Original Scale]")
        print(f"  MAE               : {orig_mae:.6f}")
        print(f"  RMSE              : {orig_rmse:.6f}")
        print(f"  Euclidean Distance: {orig_euc:.6f}")

        print("\n  [Column-wise MAE]")
        for i, col in enumerate(label_cols):
            print(f"    {col:>4} : {col_mae_orig[i].item():.6f}")

        print(f"\n  Displacement MAE (dX,dY,dZ): {disp_mae:.6f}")
        print(f"  Force MAE (Fx,Fy,Fz)       : {force_mae:.6f}")

    except FileNotFoundError:
        print(f"\n[label_stats.json not found at {args.label_stats}]")
        print("Original scale 결과는 출력하지 않습니다.")
    except KeyError as e:
        print(f"\n[label_stats.json key error: {e}]")

    # ── 5. 샘플 출력 ──
    print(f"\n{'='*60}")
    print("  Sample Retrieval Results (first 5)")
    print(f"{'='*60}")
    for i in range(min(5, total)):
        print(f"\n  [{i}] GT:        {all_gt_texts[i]}")
        print(f"       Retrieved: {all_top1_texts[i]}")
        gt_vals = all_gt_values[i].tolist()
        ret_vals = top1_retrieved[i].tolist()
        err = [abs(g - r) for g, r in zip(gt_vals, ret_vals)]
        print(f"       Error:     {', '.join(f'{e:.4f}' for e in err)}")


if __name__ == "__main__":
    main()
