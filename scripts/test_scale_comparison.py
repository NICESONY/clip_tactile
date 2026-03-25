"""
test_scale_comparison.py

정규화 텍스트 vs 원본 스케일 텍스트로 candidate pool을 구성했을 때
Contrastive / Regression 모델 성능 비교 실험.

사용법:
  python test_scale_comparison.py
"""

import json
import yaml
import numpy as np
import torch
import pandas as pd
from transformers import AutoProcessor, CLIPProcessor


def load_label_stats(path, cols):
    with open(path, "r") as f:
        stats = json.load(f)
    mean = {c: stats["mean"][c] for c in cols}
    std = {c: stats["std"][c] for c in cols}
    return mean, std


def denormalize_df(df, label_cols, mean, std):
    """정규화된 DataFrame을 원본 스케일로 역변환."""
    df_orig = df.copy()
    for col in label_cols:
        df_orig[col] = df[col] * std[col] + mean[col]
    return df_orig


def label_to_text(row, label_cols):
    parts = [f"{col} {row[col]:.4f}" for col in label_cols]
    return ", ".join(parts)


# ============================================================
#  CONTRASTIVE 비교 실험
# ============================================================
def test_contrastive():
    print("=" * 60)
    print("  [Contrastive] 정규화 vs 원본 스케일 텍스트 비교")
    print("=" * 60)

    with open("configs/contrastive.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    label_cols = cfg["data"]["label_cols"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    from contrastive_model import CLIPContrastive
    from contrastive_dataset import TactileContrastiveDataset

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

    ckpt = torch.load("outputs/clip_contrastive/best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load label stats
    mean, std = load_label_stats("data/processed/label_stats.json", label_cols)

    # Build candidate pools: normalized vs original scale
    csv_paths = [cfg["data"]["train_csv"], cfg["data"]["val_csv"], cfg["data"]["test_csv"]]

    def build_candidates(use_original_scale=False):
        all_texts, all_values = [], []
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            if use_original_scale:
                df = denormalize_df(df, label_cols, mean, std)
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
        return unique_texts, unique_values

    def encode_candidates(texts):
        embeds_list = []
        with torch.no_grad():
            for start in range(0, len(texts), 256):
                batch = texts[start:start + 256]
                tok = tokenizer(
                    batch, return_tensors="pt",
                    padding="max_length", truncation=True, max_length=77,
                )
                emb = model.encode_text(
                    tok["input_ids"].to(device),
                    tok["attention_mask"].to(device),
                )
                embeds_list.append(emb.cpu())
        return torch.cat(embeds_list, dim=0).to(device)

    # 1) Normalized candidates (정상)
    norm_texts, norm_values = build_candidates(use_original_scale=False)
    norm_embeds = encode_candidates(norm_texts)

    # 2) Original scale candidates (실험)
    orig_texts, orig_values = build_candidates(use_original_scale=True)
    orig_embeds = encode_candidates(orig_texts)

    print(f"\n  Candidate pool 크기: {len(norm_texts)} (normalized), {len(orig_texts)} (original)")
    print(f"\n  텍스트 예시 비교:")
    print(f"    정규화: {norm_texts[0]}")
    print(f"    원본:   {orig_texts[0]}")

    # Test dataset (항상 정규화된 CSV 기반)
    test_dataset = TactileContrastiveDataset(
        csv_path=cfg["data"]["test_csv"],
        image_dir=cfg["data"]["test_image_dir"],
        label_cols=label_cols,
        image_processor=image_processor,
        tokenizer=tokenizer,
    )

    # Evaluate both
    num_test = len(test_dataset)
    correct_norm, correct_orig = 0, 0
    mae_norm_list, mae_orig_list = [], []

    print(f"\n  테스트 샘플 수: {num_test}")
    print(f"\n  {'idx':>5} | {'Normalized MAE':>15} | {'Original MAE':>15} | {'Norm correct':>13} | {'Orig correct':>13}")
    print(f"  {'-'*5}-+-{'-'*15}-+-{'-'*15}-+-{'-'*13}-+-{'-'*13}")

    with torch.no_grad():
        for idx in range(num_test):
            sample = test_dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            gt_norm = sample["label_values"].numpy()

            img_embed = model.encode_image(pixel_values)

            # Normalized retrieval
            sim_norm = img_embed @ norm_embeds.t()
            top1_norm = sim_norm.argmax(dim=1).item()
            pred_norm = norm_values[top1_norm].numpy()
            mae_n = np.abs(gt_norm - pred_norm).mean()
            mae_norm_list.append(mae_n)
            if np.allclose(gt_norm, pred_norm, atol=1e-3):
                correct_norm += 1

            # Original scale retrieval
            sim_orig = img_embed @ orig_embeds.t()
            top1_orig = sim_orig.argmax(dim=1).item()
            # orig_values는 원본 스케일이므로, 비교를 위해 gt도 역변환
            gt_orig = np.array([gt_norm[i] * std[col] + mean[col] for i, col in enumerate(label_cols)])
            pred_orig = orig_values[top1_orig].numpy()
            mae_o = np.abs(gt_orig - pred_orig).mean()
            mae_orig_list.append(mae_o)
            if np.allclose(gt_orig, pred_orig, atol=1e-3):
                correct_orig += 1

            if idx < 10 or idx % 100 == 0:
                print(f"  {idx:>5} | {mae_n:>15.6f} | {mae_o:>15.6f} | {'O' if np.allclose(gt_norm, pred_norm, atol=1e-3) else 'X':>13} | {'O' if np.allclose(gt_orig, pred_orig, atol=1e-3) else 'X':>13}")

    print(f"\n  {'='*70}")
    print(f"  [결과 요약]")
    print(f"  {'':>20} | {'정규화 텍스트':>15} | {'원본 스케일 텍스트':>18}")
    print(f"  {'-'*20}-+-{'-'*15}-+-{'-'*18}")
    print(f"  {'정확 매칭 (atol=1e-3)':>20} | {correct_norm:>10}/{num_test:<4} | {correct_orig:>13}/{num_test:<4}")
    print(f"  {'정확도':>20} | {100*correct_norm/num_test:>14.2f}% | {100*correct_orig/num_test:>17.2f}%")
    print(f"  {'평균 MAE':>20} | {np.mean(mae_norm_list):>15.6f} | {np.mean(mae_orig_list):>18.6f}")
    print(f"  {'='*70}")


# ============================================================
#  REGRESSION 비교 실험
# ============================================================
def test_regression():
    print("\n\n" + "=" * 60)
    print("  [Regression] 정규화 예측 vs 원본 스케일 비교")
    print("=" * 60)

    with open("configs/regression.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    label_cols = ["dX", "dY", "dZ", "Fx", "Fy", "Fz"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from model import CLIPVisionRegressor
    from dataset import TactileCoordinateDataset

    image_processor = AutoProcessor.from_pretrained(
        model_cfg["pretrained_model_name"]
    ).image_processor

    # Load label stats
    mean, std = load_label_stats("data/processed/label_stats.json", label_cols)

    # 1) 정규화된 테스트셋 (정상 — 학습과 동일)
    test_norm = TactileCoordinateDataset(
        csv_path=cfg["data"]["test_csv"],
        image_dir=cfg["data"]["test_image_dir"],
        output_dim=model_cfg["output_dim"],
        image_processor=image_processor,
    )

    # 2) 원본 스케일 테스트셋 — 역변환된 CSV 생성
    df_test = pd.read_csv(cfg["data"]["test_csv"])
    df_test_orig = denormalize_df(df_test, label_cols, mean, std)
    orig_csv_path = "data/processed/_test_labels_original_scale.csv"
    df_test_orig.to_csv(orig_csv_path, index=False)

    test_orig = TactileCoordinateDataset(
        csv_path=orig_csv_path,
        image_dir=cfg["data"]["test_image_dir"],
        output_dim=model_cfg["output_dim"],
        image_processor=image_processor,
    )

    # Load model
    model = CLIPVisionRegressor(
        pretrained_model_name=model_cfg["pretrained_model_name"],
        output_dim=model_cfg["output_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
        freeze_strategy=model_cfg.get("freeze_strategy", "all"),
        unfreeze_layers=model_cfg.get("unfreeze_layers", 2),
    ).to(device)

    ckpt = torch.load("outputs/clip_vision/best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    num_test = len(test_norm)
    mae_norm_per_col = {c: [] for c in label_cols}
    mae_orig_per_col = {c: [] for c in label_cols}
    mae_denorm_per_col = {c: [] for c in label_cols}  # pred도 역변환해서 비교

    print(f"\n  테스트 샘플 수: {num_test}")
    print(f"\n  실험 A: 모델(정규화 출력) vs GT(정규화) — 정상")
    print(f"  실험 B: 모델(정규화 출력) vs GT(원본 스케일) — 스케일 불일치")
    print(f"  실험 C: 모델(역변환 출력) vs GT(원본 스케일) — 둘 다 원본 스케일")

    print(f"\n  {'idx':>5} | {'A: norm MAE':>12} | {'B: mismatch MAE':>16} | {'C: both orig MAE':>17}")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*16}-+-{'-'*17}")

    with torch.no_grad():
        for idx in range(num_test):
            sample_n = test_norm[idx]
            sample_o = test_orig[idx]

            image_tensor = sample_n["image"].unsqueeze(0).to(device)
            pred = model(image_tensor).squeeze(0).cpu().numpy()  # 항상 정규화 스케일 출력

            gt_norm = sample_n["target"].numpy()
            gt_orig = sample_o["target"].numpy()

            # A: 정규화 pred vs 정규화 GT (정상)
            mae_a = np.abs(pred - gt_norm).mean()

            # B: 정규화 pred vs 원본 GT (스케일 불일치)
            mae_b = np.abs(pred - gt_orig).mean()

            # C: 역변환 pred vs 원본 GT (둘 다 원본 스케일)
            pred_denorm = np.array([pred[i] * std[col] + mean[col] for i, col in enumerate(label_cols)])
            mae_c = np.abs(pred_denorm - gt_orig).mean()

            for j, col in enumerate(label_cols):
                mae_norm_per_col[col].append(abs(pred[j] - gt_norm[j]))
                mae_orig_per_col[col].append(abs(pred[j] - gt_orig[j]))
                mae_denorm_per_col[col].append(abs(pred_denorm[j] - gt_orig[j]))

            if idx < 10 or idx % 100 == 0:
                print(f"  {idx:>5} | {mae_a:>12.6f} | {mae_b:>16.6f} | {mae_c:>17.6f}")

    print(f"\n  {'='*70}")
    print(f"  [축별 평균 MAE 비교]")
    print(f"  {'Col':>5} | {'A (정상)':>12} | {'B (불일치)':>14} | {'C (둘다 원본)':>16}")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*14}-+-{'-'*16}")
    for col in label_cols:
        a = np.mean(mae_norm_per_col[col])
        b = np.mean(mae_orig_per_col[col])
        c = np.mean(mae_denorm_per_col[col])
        print(f"  {col:>5} | {a:>12.6f} | {b:>14.6f} | {c:>16.6f}")

    a_total = np.mean([np.mean(v) for v in mae_norm_per_col.values()])
    b_total = np.mean([np.mean(v) for v in mae_orig_per_col.values()])
    c_total = np.mean([np.mean(v) for v in mae_denorm_per_col.values()])
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*14}-+-{'-'*16}")
    print(f"  {'전체':>5} | {a_total:>12.6f} | {b_total:>14.6f} | {c_total:>16.6f}")
    print(f"  {'='*70}")

    print(f"\n  해석:")
    print(f"  A: 학습과 동일한 조건 → 정상 성능")
    print(f"  B: 모델은 정규화 출력인데 GT가 원본 스케일 → 스케일 불일치로 MAE 폭증")
    print(f"  C: 양쪽 다 원본 스케일로 맞춰줌 → A와 동일한 '성능' (선형 변환)")

    # cleanup
    import os
    os.remove(orig_csv_path)


if __name__ == "__main__":
    test_contrastive()
    test_regression()
