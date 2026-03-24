# CLIP Regression & Contrastive Retrieval

촉각(tactile) 이미지 → 6축 force/torque 예측

---

## 폴더 구조

```
clip_tactile/
│
├── prepare_data.py          # 데이터 셋업 (심볼릭 링크 or 복사)
├── requirements.txt
├── .gitignore
├── Readme.md
│
├── configs/
│   ├── regression.yaml          # A-1: encoder freeze
│   ├── regression_partial.yaml  # A-2: 마지막 2 layer unfreeze
│   ├── regression_full.yaml     # A-3: 전체 fine-tuning
│   └── contrastive.yaml         # B:   contrastive retrieval
│
├── model.py                 # CLIPVisionRegressor
├── contrastive_model.py     # CLIPContrastive
├── dataset.py               # regression dataset
├── contrastive_dataset.py   # contrastive dataset (6축→문자열)
├── utils.py                 # seed, metrics, checkpoint
│
├── train.py                 # regression 학습
├── evaluate.py              # regression 평가
├── contrastive_train.py     # contrastive 학습
├── contrastive_eval.py      # retrieval 평가
├── benchmark.py             # 모델 벤치마크
├── plot_loss.py             # 학습 loss 시각화
├── visualize.py             # 결과 시각화
│
├── md_file/                 # 프로젝트 문서
│   ├── CLAUDE.md
│   ├── contrastive_explanation.md
│   ├── execution_flow.md
│   ├── finetuning_strategy.md
│   ├── multi_contact_comparison.md
│   └── multi_contact_project_design.md
│
├── data/                    # prepare_data.py로 생성
│   ├── train/images/
│   ├── val/images/
│   ├── test/images/
│   └── processed/
│       ├── train_labels_normalized.csv
│       ├── val_labels_normalized.csv
│       ├── test_labels_normalized.csv
│       └── label_stats.json
│
└── outputs/                 # 체크포인트 & 로그 저장
    ├── clip_vision/
    │   ├── best.pt
    │   └── logs/
    ├── clip_vision_partial/
    │   ├── best.pt
    │   └── logs/
    ├── clip_vision_full/
    │   ├── best.pt
    │   └── logs/
    └── clip_contrastive/
        ├── best.pt
        └── logs/
```

---

## 초기 셋업

```bash
cd clip_tactile
pip install -r requirements.txt

# 데이터 연결 (같은 프로젝트 안에 있으면 자동 탐색)
python prepare_data.py

# 다른 곳에서 가져온 경우 source 지정
python prepare_data.py --source /path/to/data

# 완전히 독립시키려면 실제 복사
python prepare_data.py --source /path/to/data --copy
```

---

## 사용 데이터

| 파일 | 설명 |
|------|------|
| `data/processed/train_labels_normalized.csv` | 학습 (4,000행) |
| `data/processed/val_labels_normalized.csv` | 검증 (500행) |
| `data/processed/test_labels_normalized.csv` | 테스트 (500행) |
| `data/{train,val,test}/images/` | 촉각 이미지 (224x224) |
| `data/processed/label_stats.json` | 라벨 mean/std (역변환용) |

라벨 컬럼: **dX, dY, dZ, Fx, Fy, Fz** (z-score normalized)

### 이미지 전처리

CLIP 기본 normalization 대신 프로젝트 데이터셋 통계 사용:

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

| 실험 | config | 학습 범위 | lr | weight_decay | epochs |
|------|--------|----------|-----|-------------|--------|
| A-1 | `configs/regression.yaml` | head만 (0%) | 3e-4 | 1e-4 | 3000 |
| A-2 | `configs/regression_partial.yaml` | head + 마지막 2 layer (16.2%) | 5e-5 | 1e-3 | 300 |
| A-3 | `configs/regression_full.yaml` | 전체 (100%) | 1e-5 | 1e-2 | 100 |

### 실행

```bash
# A-1: freeze
python train.py --config configs/regression.yaml --seed 42
python evaluate.py --config configs/regression.yaml --checkpoint outputs/clip_vision/best.pt --label_stats data/processed/label_stats.json

# A-2: partial
python train.py --config configs/regression_partial.yaml --seed 42
python evaluate.py --config configs/regression_partial.yaml --checkpoint outputs/clip_vision_partial/best.pt --label_stats data/processed/label_stats.json

# A-3: full
python train.py --config configs/regression_full.yaml --seed 42
python evaluate.py --config configs/regression_full.yaml --checkpoint outputs/clip_vision_full/best.pt --label_stats data/processed/label_stats.json
```

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
# 학습
python contrastive_train.py --config configs/contrastive.yaml --seed 42

# 평가 (후보 ~5000개 대비 검색)
python contrastive_eval.py --config configs/contrastive.yaml --checkpoint outputs/clip_contrastive/best.pt --label_stats data/processed/label_stats.json
```


<!-- 
python visualize.py --mode regression --config configs/regression.yaml --checkpoint outputs/clip_vision/best.pt --num_samples 5 


 python visualize.py --mode contrastive --config configs/contrastive.yaml --checkpoint outputs/clip_contrastive/best.pt --num_samples 5  -->



   # contrastive                                                                
  python plot_loss.py --log outputs/clip_contrastive/logs/clip_contrastive_20260324_102955.csv           
                                                                    
  # regression도 동일하게                                                      
  python plot_loss.py --log outputs/clip_vision/logs/clip_vision_20260324_102946.csv
                         
