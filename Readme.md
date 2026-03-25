# CLIP Tactile: Regression & Contrastive Retrieval

촉각(tactile) 이미지 → 6축 force/torque 예측

---

## 폴더 구조

```
clip_tactile/
│
├── requirements.txt
├── .gitignore
├── Readme.md
│
├── configs/                        # 실험 설정
│   ├── regression.yaml                 # A-1: encoder freeze
│   ├── regression_partial.yaml         # A-2: 마지막 2 layer unfreeze
│   ├── regression_full.yaml            # A-3: 전체 fine-tuning
│   ├── regression_padded.yaml          # A-1p: freeze + letterbox padding
│   ├── regression_partial_padded.yaml  # A-2p: partial + letterbox padding
│   ├── regression_full_padded.yaml     # A-3p: full + letterbox padding
│   ├── contrastive.yaml                # B-1: contrastive retrieval
│   ├── contrastive_padded.yaml         # B-1p: contrastive + letterbox padding
│   ├── grid4x4_regression.yaml         # C-1: grid 양자화 regression
│   └── grid4x4_contrastive.yaml        # C-2: grid 양자화 contrastive
│
├── model.py                        # CLIPVisionRegressor
├── contrastive_model.py            # CLIPContrastive
├── dataset.py                      # regression dataset (커스텀 정규화 지원)
├── contrastive_dataset.py          # contrastive dataset (6축→문자열, 커스텀 정규화 지원)
├── utils.py                        # seed, metrics, checkpoint
│
├── train.py                        # regression 학습
├── evaluate.py                     # regression 평가
├── contrastive_train.py            # contrastive 학습
├── contrastive_eval.py             # retrieval 평가
├── benchmark.py                    # 모델 벤치마크
├── visualize.py                    # 결과 시각화
│
├── scripts/                        # 유틸리티 / 일회성 스크립트
│   ├── prepare_data.py                 # 데이터 셋업 (심볼릭 링크 or 복사)
│   ├── prepare_padded_images.py        # letterbox padding 전처리 + 통계 계산
│   ├── prepare_grid_data.py            # 4x4 격자 라벨 양자화
│   ├── plot_loss.py                    # 학습 loss 시각화
│   ├── visualize_preprocess.py         # 원본 vs CLIP crop vs padding 비교
│   └── test_scale_comparison.py        # 정규화 스케일 비교 실험
│
├── md_file/                        # 프로젝트 문서 (gitignore 대상)
│   ├── contrastive_explanation.md
│   ├── execution_flow.md
│   ├── finetuning_strategy.md
│   ├── multi_contact_comparison.md
│   ├── multi_contact_project_design.md
│   ├── normalization_and_inference.md
│   └── normalization_similarity.md
│
├── data/                           # 데이터 (gitignore 대상)
│   ├── train/images/                   # 학습 촉각 이미지
│   ├── val/images/                     # 검증 촉각 이미지
│   ├── test/images/                    # 테스트 촉각 이미지
│   ├── {split}/images_padded/          # letterbox padding된 이미지 (prepare_padded_images.py로 생성)
│   └── processed/
│       ├── train_labels_normalized.csv
│       ├── val_labels_normalized.csv
│       ├── test_labels_normalized.csv
│       ├── label_stats.json                # 라벨 mean/std (역변환용)
│       ├── image_stats_padded.json         # padded 이미지 mean/std (prepare_padded_images.py로 생성)
│       ├── grid4x4_{train,val,test}.csv                # grid 양자화 라벨 (prepare_grid_data.py로 생성)
│       ├── grid4x4_{train,val,test}_normalized.csv     # grid 양자화 + z-score 정규화
│       └── grid4x4_label_stats.json                    # grid용 mean/std + 격자 경계
│
└── outputs/                        # 체크포인트 & 로그 (gitignore 대상)
    ├── clip_vision/                    # A-1 결과
    ├── clip_vision_partial/            # A-2 결과
    ├── clip_vision_full/               # A-3 결과
    └── clip_contrastive/               # B-1 결과
```

---

## .gitignore 안내

아래 항목들은 Git에 포함되지 않습니다. 클론 후 반드시 별도로 준비해야 합니다.

| 대상 | 내용 | 준비 방법 |
|------|------|-----------|
| `data/*` | 촉각 이미지 + 라벨 CSV + 통계 JSON | `python scripts/prepare_data.py`로 생성 |
| `outputs/*` | 학습된 모델 체크포인트(`.pt`) + 학습 로그 | `train.py` / `contrastive_train.py`로 학습 |
| `md_file/*` | 프로젝트 내부 설계 문서 | 별도 공유 필요 |
| `__pycache__/*` | Python 캐시 | 자동 생성 |

> **주의:** `data/`, `outputs/`, `md_file/`은 레포지토리에 포함되지 않으므로, 클론만으로는 학습/평가를 바로 실행할 수 없습니다. 데이터 원본을 확보한 뒤 아래 초기 셋업 절차를 따라야 합니다.

---

## 초기 셋업

```bash
cd clip_tactile
pip install -r requirements.txt
```

### 1단계: 데이터 연결

```bash
# 같은 프로젝트 안에 있으면 자동 탐색
python scripts/prepare_data.py

# 다른 곳에서 가져온 경우 source 지정
python scripts/prepare_data.py --source /path/to/data

# 완전히 독립시키려면 실제 복사
python scripts/prepare_data.py --source /path/to/data --copy
```

실행 후 `data/` 디렉토리에 이미지와 CSV가 준비됩니다.

### 2단계 (선택): Letterbox Padding 전처리

CLIP 기본 전처리(resize + center crop)는 640x480 이미지의 상하를 잘라냅니다. 정보 손실을 줄이려면 letterbox padding을 적용할 수 있습니다.

```bash
python scripts/prepare_padded_images.py
```

이 스크립트가 생성하는 것:
- `data/{train,val,test}/images_padded/` — 비율 유지 + 검정 패딩 224x224 이미지
- `data/processed/image_stats_padded.json` — padded 이미지 기준 RGB mean/std

> padded 관련 config (`*_padded.yaml`)를 사용하려면 이 단계가 **필수**입니다.

### 3단계 (선택): Grid 양자화 데이터 생성

x-y 좌표 공간을 4x4 격자(16개 셀)로 나누어 연속 라벨을 이산화합니다.

```bash
python scripts/prepare_grid_data.py
```

이 스크립트가 생성하는 것:
- `data/processed/grid4x4_{train,val,test}.csv` — 원본 스케일 grid 라벨
- `data/processed/grid4x4_{train,val,test}_normalized.csv` — z-score 정규화
- `data/processed/grid4x4_label_stats.json` — mean/std + 격자 경계 정보

> `grid4x4_*.yaml` config를 사용하려면 이 단계가 **필수**입니다.

---

## 사용 데이터

| 파일 | 설명 |
|------|------|
| `data/processed/train_labels_normalized.csv` | 학습 (4,000행) |
| `data/processed/val_labels_normalized.csv` | 검증 (500행) |
| `data/processed/test_labels_normalized.csv` | 테스트 (500행) |
| `data/{train,val,test}/images/` | 원본 촉각 이미지 |
| `data/{train,val,test}/images_padded/` | letterbox padding 이미지 (선택) |
| `data/processed/label_stats.json` | 라벨 mean/std (역변환용) |
| `data/processed/image_stats_padded.json` | padded 이미지 RGB mean/std (선택) |

라벨 컬럼: **dX, dY, dZ, Fx, Fy, Fz** (z-score normalized)

### 이미지 전처리 모드

config에 `image_stats` 필드 유무에 따라 두 가지 전처리 모드가 자동 선택됩니다.

| 모드 | 조건 | 동작 |
|------|------|------|
| **기본 (crop)** | `image_stats` 없음 | CLIP resize + center crop 224x224, 프로젝트 데이터셋 mean/std 적용 |
| **Padding** | `image_stats` 있음 | resize/crop 비활성화, 외부 JSON의 mean/std 적용 (이미 224x224인 padded 이미지 사용) |

기본 모드 프로젝트 통계:

| 항목 | R | G | B |
|------|---|---|---|
| mean | 0.4161 | 0.3432 | 0.3261 |
| std  | 0.3450 | 0.3239 | 0.3235 |

---

## Exp A. Regression (MLP Head)

CLIP Vision Encoder feature → MLP head → 6축 연속값 회귀.

```
tactile image → CLIPVisionModel → pooler_output [768]
              → Linear(768,256) → ReLU → Dropout → Linear(256,6) → 6축 값
```

| 항목 | 값 |
|------|----|
| 입력 | `pixel_values [B, 3, 224, 224]` |
| 출력 | `[B, 6]` (dX, dY, dZ, Fx, Fy, Fz) |
| Loss | SmoothL1Loss |
| 평가 | MAE, RMSE, Euclidean Distance |

### Freeze 전략

| 실험 | config | 이미지 | 학습 범위 | lr | epochs |
|------|--------|--------|----------|-----|--------|
| A-1 | `regression.yaml` | crop | head만 | 3e-4 | 3000 |
| A-2 | `regression_partial.yaml` | crop | head + 마지막 2 layer | 5e-5 | 300 |
| A-3 | `regression_full.yaml` | crop | 전체 | 1e-5 | 100 |
| A-1p | `regression_padded.yaml` | **padded** | head만 | 3e-4 | 3000 |
| A-2p | `regression_partial_padded.yaml` | **padded** | head + 마지막 2 layer | 5e-5 | 3000 |
| A-3p | `regression_full_padded.yaml` | **padded** | 전체 | 1e-5 | 3000 |

### 실행

```bash
# A-1: freeze (기본 crop)
python train.py --config configs/regression.yaml --seed 42
python evaluate.py --config configs/regression.yaml \
  --checkpoint outputs/clip_vision/best.pt \
  --label_stats data/processed/label_stats.json

# A-2: partial
python train.py --config configs/regression_partial.yaml --seed 42
python evaluate.py --config configs/regression_partial.yaml \
  --checkpoint outputs/clip_vision_partial/best.pt \
  --label_stats data/processed/label_stats.json

# A-3: full fine-tuning
python train.py --config configs/regression_full.yaml --seed 42
python evaluate.py --config configs/regression_full.yaml \
  --checkpoint outputs/clip_vision_full/best.pt \
  --label_stats data/processed/label_stats.json

# A-1p: freeze + letterbox padding (padded 데이터 필요)
python train.py --config configs/regression_padded.yaml --seed 42
python evaluate.py --config configs/regression_padded.yaml \
  --checkpoint outputs/clip_vision/best.pt \
  --label_stats data/processed/label_stats.json
```

> padded 실험(A-1p ~ A-3p)은 동일한 명령어 구조이며, config만 `*_padded.yaml`로 교체합니다.

---

## Exp B. Contrastive Retrieval

CLIP image encoder + text encoder. 6축 값을 문자열로 변환하여 유사도 기반 검색.

```
Image: tactile image → CLIPVisionModel → visual_projection [512] → L2 norm
Text:  "dX -0.96, dY -0.45, ..." → CLIPTextModel → text_projection [512] → L2 norm
                              → cosine similarity → contrastive loss
```

| 항목 | 값 |
|------|----|
| Image 입력 | `pixel_values [B, 3, 224, 224]` |
| Text 입력 | `"dX -0.9605, dY -0.4460, dZ -0.0143, Fx 0.2122, Fy -0.2083, Fz 0.2855"` |
| 출력 | `logits [B, B]` similarity matrix |
| Loss | Symmetric Cross-Entropy |
| 평가 | Top-1/5 accuracy, MAE, RMSE |

### 실행

```bash
# B-1: contrastive (기본 crop)
python contrastive_train.py --config configs/contrastive.yaml --seed 42
python contrastive_eval.py --config configs/contrastive.yaml \
  --checkpoint outputs/clip_contrastive/best.pt \
  --label_stats data/processed/label_stats.json

# B-1p: contrastive + letterbox padding (padded 데이터 필요)
python contrastive_train.py --config configs/contrastive_padded.yaml --seed 42
python contrastive_eval.py --config configs/contrastive_padded.yaml \
  --checkpoint outputs/clip_contrastive/best.pt \
  --label_stats data/processed/label_stats.json
```

---

## Exp C. Grid 양자화 실험

x-y 좌표 공간을 4x4 격자(16셀)로 나누어 연속 라벨을 이산화한 뒤 학습. 라벨 다양성을 ~4000 → 16으로 줄여 contrastive 학습의 클러스터 형성을 돕는 실험입니다.

```bash
# C-1: grid regression
python train.py --config configs/grid4x4_regression.yaml --seed 42

# C-2: grid contrastive
python contrastive_train.py --config configs/grid4x4_contrastive.yaml --seed 42
```

> grid 실험은 원본 이미지(crop 모드)를 사용하며, grid 전용 CSV와 label_stats를 참조합니다.

---

## 유틸리티 스크립트 (`scripts/`)

| 스크립트 | 설명 | 사용법 |
|---------|------|--------|
| `prepare_data.py` | 데이터 디렉토리 구성 (symlink/copy) | `python scripts/prepare_data.py [--source PATH] [--copy]` |
| `prepare_padded_images.py` | letterbox padding + 이미지 통계 계산 | `python scripts/prepare_padded_images.py` |
| `prepare_grid_data.py` | 4x4 격자 라벨 양자화 | `python scripts/prepare_grid_data.py` |
| `plot_loss.py` | 학습 로그 CSV → loss/accuracy 그래프 | `python scripts/plot_loss.py --log outputs/.../xxx.csv` |
| `visualize_preprocess.py` | 원본 vs CLIP crop vs padding 비교 시각화 | `python scripts/visualize_preprocess.py --image_dir data/test/images` |
| `test_scale_comparison.py` | 정규화 vs 원본 스케일 성능 비교 실험 | `python scripts/test_scale_comparison.py` |

---

## 의존성

```
torch
torchvision
transformers
pandas
numpy
Pillow
pyyaml
tqdm
```

```bash
pip install -r requirements.txt
```

---

## 핵심 개념: 정규화와 추론

- 모델은 항상 **z-score 정규화된 텍스트/라벨**로 학습됩니다.
- 평가 시 예측값을 원본 스케일로 역변환할 때는 `label_stats.json`의 mean/std를 사용합니다.
- Contrastive 모델의 text encoder는 정규화된 문자열만 입력받습니다. 원본 스케일 변환은 검색된 후보의 수치를 산술적으로 역변환하는 것이지, text encoder를 다시 호출하는 것이 아닙니다.
- 자세한 설명은 `md_file/normalization_and_inference.md`, `md_file/normalization_similarity.md` 참고 (gitignore 대상이므로 별도 확보 필요).
