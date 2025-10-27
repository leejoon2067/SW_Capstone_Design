# CQ500 Anomaly Detection - AE/AE-U 구현

## 📌 프로젝트 개요

CQ500 CT 데이터셋을 사용한 뇌출혈(ICH) 이상 탐지
- **데이터**: CQ500 DICOM (2D 슬라이스 방식)
- **모델**: Autoencoder (AE), Autoencoder with Uncertainty (AE-U), Variational Autoencoder (VAE)
- **목표**: 정상 뇌 CT로 학습 후 출혈(Herrmohage) 감지

---

## ✅ 구현 완료 현황

### 1. **Data Pipeline** (100%)
```
✓ dataloaders/dataload.py     - CQ500Dataset, DataModule
✓ dataloaders/preprocess.py   - DICOM 로더
✓ dataloaders/data_utils.py   - 유틸리티
✓ 2D Direct Load (422배 빠름!)
✓ Train: 4,555 slices (HEALTHY)
✓ Val/Test: ~50,000 slices
```

### 2. **Model Architecture** (100%)
```
✓ networks/ae.py              - Autoencoder
✓ networks/aeu.py             - AE with Uncertainty
✓ networks/vae.py             - Variational Autoencoder
✓ networks/base_units/        - Building blocks
✓ AE parameters: 4.5M
✓ AE-U parameters: 4.5M
```

### 3. **Training Infrastructure** (100%)
```
✓ utils/losses.py             - AELoss, AEULoss
✓ utils/ae_worker.py          - AE trainer
✓ utils/aeu_worker.py         - AE-U trainer
✓ utils/vae_worker.py         - VAE trainer
✓ utils/base_worker.py        - Base class
✓ utils/util.py               - Utilities
✓ train.py                    - 학습 스크립트
✓ test.py                     - 테스트 스크립트
✓ options.py                  - 설정 관리
```

---

## 🚀 사용 방법

### 학습 실행
```bash
# AE 모델 학습
python train.py --model-name ae --dataset cq500 --input-size 64 --batch-size 64 --train-epochs 250

# AE-U 모델 학습
python train.py --model-name aeu --dataset cq500 --input-size 64 --batch-size 64 --train-epochs 250
```

### 주요 파라미터
```
--input-size 64          : 2D 이미지 크기
--batch-size 64          : 배치 사이즈
--train-epochs 250       : 학습 에폭
--latent-size 16         : Latent 차원
--base-width 16          : 채널 수
--train-lr 0.001         : Learning rate
```

---

## 📊 데이터셋 정보

### CQ500 Dataset
```
전체: 364 patients
  - HEALTHY: 113 patients (31%)
  - ICH: 247 patients (69%)

Train/Val/Test Split (70%/15%/15%):
  - Train: 79 patients → 4,555 slices (100% HEALTHY)
  - Val: 263 patients → 25,273 slices (6% HEALTHY, 94% ICH)
  - Test: 265 patients → ~25,000 slices
```

### 데이터 특성
```
- 2D 슬라이스: 각 DICOM 파일이 1개 슬라이스
- 이미지 크기: 512×512 → 64×64로 리사이즈
- Windowing: Brain window (center=40, width=80)
- 정규화: 0-1 범위
```

---

## 🏗️ 아키텍처 (Basic AE pipeline)

### AE (Autoencoder)
```
Input (B, 1, 64, 64)
    ↓
Encoder (4 blocks, downsample)
    ↓
Bottleneck (Latent: 16)
    ↓
Decoder (4 blocks, upsample)
    ↓
Output (B, 1, 64, 64)

Loss: MSE(input, reconstruction)
```
---

## 📁 AE_baseline 모델 폴더 구조

```
AE_baseline/
├── dataloaders/
│   ├── __init__.py
│   ├── dataload.py          ✓ Dataset/DataModule
│   ├── preprocess.py        ✓ DICOM loader
│   └── data_utils.py        ✓ Utilities
│
├── networks/
│   ├── __init__.py
│   ├── ae.py                ✓ AE model
│   ├── aeu.py               ✓ AE-U model
│   └── base_units/
│       ├── __init__.py
│       ├── blocks.py        ✓ Building blocks
│       └── conv_layers.py   ✓ Conv layers
│
├── utils/
│   ├── __init__.py
│   ├── util.py              ✓ General utilities
│   ├── losses.py            ✓ Loss functions
│   ├── base_worker.py       ✓ Base trainer
│   ├── ae_worker.py         ✓ AE trainer
│   └── aeu_worker.py        ✓ AE-U trainer
│
├── train.py                 ✓ 학습 스크립트
├── test.py                  ✓ 테스트 스크립트
└── options.py               ✓ 설정 관리
```

---

## 🎯 성능 메트릭

### 평가 지표
- **AUROC**: ROC 곡선 아래 면적
- **AUPRC**: Precision-Recall 곡선 아래 면적

### 학습 모니터링
- Training loss (reconstruction error, re + log-loss)
- Validation AUROC/AUPRC

---

**Last Updated**: 2025-10-08
**Version**: 1.0
**Status**: 학습 준비 완료!








