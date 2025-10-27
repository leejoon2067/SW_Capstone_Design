"""
CQ500 데이터 유틸리티 함수
- Windowing, 정규화, Transform 등
"""

import os
import numpy as np
import torch
from typing import Optional, Tuple
from monai.transforms import (
    Compose, RandRotate90d, RandFlipd, 
    RandGaussianNoised, RandAdjustContrastd
)


def apply_ct_windowing(
    volume: np.ndarray, 
    window_center: int = 40, 
    window_width: int = 80
) -> np.ndarray:
    """
    CT 이미지에 windowing 적용
    
    Args:
        volume: CT 볼륨 (HU 값)
        window_center: Window center (default: 40 for brain)
        window_width: Window width (default: 80 for brain)
    
    Returns:
        Windowed volume (0-1 normalized)
    """
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    
    volume = np.clip(volume, lower, upper)
    volume = (volume - lower) / (upper - lower)
    
    return volume.astype(np.float32)


def normalize_volume(
    volume: np.ndarray,
    method: str = "minmax",
    min_val: float = 0.0,
    max_val: float = 1.0
) -> np.ndarray:
    """
    볼륨 정규화
    
    Args:
        volume: 입력 볼륨
        method: 'minmax', 'zscore' 중 선택
        min_val: minmax 정규화 시 최소값
        max_val: minmax 정규화 시 최대값
    
    Returns:
        정규화된 볼륨
    """
    if method == "minmax":
        vol_min = volume.min()
        vol_max = volume.max()
        if vol_max - vol_min > 0:
            volume = (volume - vol_min) / (vol_max - vol_min)
            volume = volume * (max_val - min_val) + min_val
    elif method == "zscore":
        mean = volume.mean()
        std = volume.std()
        if std > 0:
            volume = (volume - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return volume.astype(np.float32)


def get_train_transforms(
    use_augmentation: bool = True,
    rotation_prob: float = 0.5,
    flip_prob: float = 0.5,
    noise_prob: float = 0.3,
    contrast_prob: float = 0.3
) -> Optional[Compose]:
    """
    학습용 Data Augmentation transforms
    
    Args:
        use_augmentation: augmentation 사용 여부
        rotation_prob: 90도 회전 확률
        flip_prob: Flip 확률
        noise_prob: Gaussian noise 추가 확률
        contrast_prob: Contrast 조정 확률
    
    Returns:
        MONAI Compose transform 또는 None
    """
    if not use_augmentation:
        return None
    
    return Compose([
        RandRotate90d(keys=["image"], prob=rotation_prob, spatial_axes=(0, 1)),
        RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=0),
        RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=1),
        RandGaussianNoised(keys=["image"], prob=noise_prob, std=0.05),
        RandAdjustContrastd(keys=["image"], prob=contrast_prob, gamma=(0.8, 1.2)),
    ])


def get_window_presets() -> dict:
    """
    CT windowing 프리셋
    
    Returns:
        window_center, window_width 딕셔너리
    """
    return {
        "brain": (40, 80),
        "subdural": (80, 200),
        "stroke": (40, 40),
        "brain_bone": (40, 380),
        "brain_soft": (40, 40),
        "bone": (600, 2000),
        "soft_tissue": (50, 350),
    }


def clip_and_normalize_hu(
    volume: np.ndarray,
    hu_min: int = -100,
    hu_max: int = 200
) -> np.ndarray:
    """
    HU 값 클리핑 및 정규화
    
    Args:
        volume: CT 볼륨 (HU 값)
        hu_min: 최소 HU 값
        hu_max: 최대 HU 값
    
    Returns:
        0-1 범위로 정규화된 볼륨
    """
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)
    return volume.astype(np.float32)


def get_data_split_indices(
    n_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    데이터 분할 인덱스 생성
    
    Args:
        n_samples: 전체 샘플 수
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        seed: Random seed
    
    Returns:
        train_indices, val_indices, test_indices
    """
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    return train_indices, val_indices, test_indices


def convert_sample_name(sample_name: str) -> str:
    """
    CSV의 sample name을 폴더명으로 변환
    
    Args:
        sample_name: CSV의 name (예: CT_0, CT_10)
    
    Returns:
        실제 폴더명 (예: CQ500-CT-0, CQ500-CT-10)
    """
    if sample_name.startswith("CT_"):
        return sample_name.replace("CT_", "CQ500-CT-")
    return sample_name


def get_data_statistics(volume: np.ndarray) -> dict:
    """
    볼륨 데이터 통계 정보
    
    Args:
        volume: 입력 볼륨
    
    Returns:
        통계 정보 딕셔너리
    """
    return {
        "shape": volume.shape,
        "dtype": volume.dtype,
        "min": float(volume.min()),
        "max": float(volume.max()),
        "mean": float(volume.mean()),
        "std": float(volume.std()),
        "median": float(np.median(volume))
    }
