"""
prepare_grid_data.py

x-y 좌표 공간을 4x4 격자(16개 영역)로 나누고,
같은 셀에 속하는 샘플들은 동일한 라벨(셀 내 평균값)을 부여.

출력:
  data/processed/grid4x4_train.csv
  data/processed/grid4x4_val.csv
  data/processed/grid4x4_test.csv
  data/processed/grid4x4_train_normalized.csv
  data/processed/grid4x4_val_normalized.csv
  data/processed/grid4x4_test_normalized.csv
  data/processed/grid4x4_label_stats.json
"""

import json
import numpy as np
import pandas as pd


LABEL_COLS = ["dX", "dY", "dZ", "Fx", "Fy", "Fz"]
GRID_N = 4  # 4x4


def assign_grid_cell(df, x_edges, y_edges):
    """각 샘플에 grid cell (row, col) 할당."""
    # np.digitize: 1-based, 경계 밖은 0 or len(edges)
    x_bin = np.clip(np.digitize(df["x"].values, x_edges) - 1, 0, GRID_N - 1)
    y_bin = np.clip(np.digitize(df["y"].values, y_edges) - 1, 0, GRID_N - 1)
    df = df.copy()
    df["grid_row"] = x_bin
    df["grid_col"] = y_bin
    df["grid_id"] = x_bin * GRID_N + y_bin
    return df


def compute_cell_means(df):
    """각 grid cell의 라벨 평균값 계산."""
    cell_means = df.groupby("grid_id")[LABEL_COLS].mean()
    return cell_means.to_dict("index")


def replace_with_cell_mean(df, cell_means):
    """각 샘플의 라벨을 해당 셀의 평균값으로 교체."""
    df = df.copy()
    for idx in df.index:
        gid = df.loc[idx, "grid_id"]
        if gid in cell_means:
            for col in LABEL_COLS:
                df.loc[idx, col] = cell_means[gid][col]
        # 셀에 train 데이터가 없는 경우 원본 유지
    return df


def main():
    # 전체 데이터 로드
    df_train = pd.read_csv("data/train/labels.csv")
    df_val = pd.read_csv("data/val/labels.csv")
    df_test = pd.read_csv("data/test/labels.csv")

    # train 기준으로 x, y 범위 결정 (전체 데이터 커버하도록 약간 여유)
    all_x = pd.concat([df_train["x"], df_val["x"], df_test["x"]])
    all_y = pd.concat([df_train["y"], df_val["y"], df_test["y"]])

    x_min, x_max = all_x.min() - 0.01, all_x.max() + 0.01
    y_min, y_max = all_y.min() - 0.01, all_y.max() + 0.01

    x_edges = np.linspace(x_min, x_max, GRID_N + 1)[1:-1]  # 내부 경계만
    y_edges = np.linspace(y_min, y_max, GRID_N + 1)[1:-1]

    print(f"x 범위: [{x_min:.2f}, {x_max:.2f}]  edges: {x_edges}")
    print(f"y 범위: [{y_min:.2f}, {y_max:.2f}]  edges: {y_edges}")

    # Grid cell 할당
    df_train = assign_grid_cell(df_train, x_edges, y_edges)
    df_val = assign_grid_cell(df_val, x_edges, y_edges)
    df_test = assign_grid_cell(df_test, x_edges, y_edges)

    # Train 기준으로 셀 평균 계산
    cell_means = compute_cell_means(df_train)

    print(f"\n4x4 격자 = {GRID_N * GRID_N}개 셀")
    print(f"Train 데이터가 존재하는 셀: {len(cell_means)}개")

    # 셀별 샘플 수
    print(f"\n[셀별 Train 샘플 수]")
    counts = df_train.groupby("grid_id").size()
    for gid in sorted(counts.index):
        row, col = gid // GRID_N, gid % GRID_N
        print(f"  셀 ({row},{col}) id={gid:>2}: {counts[gid]:>4}개  "
              f"평균 dX={cell_means[gid]['dX']:+.4f}, dY={cell_means[gid]['dY']:+.4f}, "
              f"dZ={cell_means[gid]['dZ']:+.4f}")

    # 라벨을 셀 평균으로 교체
    df_train = replace_with_cell_mean(df_train, cell_means)
    df_val = replace_with_cell_mean(df_val, cell_means)
    df_test = replace_with_cell_mean(df_test, cell_means)

    # 고유 라벨 수 확인
    unique_labels = df_train.drop_duplicates(subset=LABEL_COLS)[LABEL_COLS]
    print(f"\n고유 라벨 조합 수: {len(unique_labels)} (원래 4000개 → {len(unique_labels)}개로 축소)")

    # 원본 스케일 CSV 저장
    cols_to_save = ["text_instruction", "image_name", "x", "y", "z"] + LABEL_COLS + ["grid_id"]
    df_train[cols_to_save].to_csv("data/processed/grid4x4_train.csv", index=False)
    df_val[cols_to_save].to_csv("data/processed/grid4x4_val.csv", index=False)
    df_test[cols_to_save].to_csv("data/processed/grid4x4_test.csv", index=False)

    # 정규화 (train 기준 mean/std)
    mean = {col: df_train[col].mean() for col in LABEL_COLS}
    std = {col: df_train[col].std() for col in LABEL_COLS}

    # std가 0인 경우 방지
    for col in LABEL_COLS:
        if std[col] < 1e-8:
            std[col] = 1.0

    print(f"\n[정규화 통계 (train 기준)]")
    for col in LABEL_COLS:
        print(f"  {col}: mean={mean[col]:+.6f}, std={std[col]:.6f}")

    # 정규화 적용
    for df, name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
        df_norm = df.copy()
        for col in LABEL_COLS:
            df_norm[col] = (df[col] - mean[col]) / std[col]
        df_norm[cols_to_save].to_csv(f"data/processed/grid4x4_{name}_normalized.csv", index=False)

    # label_stats 저장
    stats = {
        "target_cols": LABEL_COLS,
        "mean": mean,
        "std": std,
        "grid": {
            "n": GRID_N,
            "x_edges": x_edges.tolist(),
            "y_edges": y_edges.tolist(),
            "x_range": [x_min, x_max],
            "y_range": [y_min, y_max],
            "cell_means": {str(k): v for k, v in cell_means.items()},
        },
    }
    with open("data/processed/grid4x4_label_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n저장 완료:")
    print(f"  data/processed/grid4x4_train.csv")
    print(f"  data/processed/grid4x4_val.csv")
    print(f"  data/processed/grid4x4_test.csv")
    print(f"  data/processed/grid4x4_{{train,val,test}}_normalized.csv")
    print(f"  data/processed/grid4x4_label_stats.json")


if __name__ == "__main__":
    main()
