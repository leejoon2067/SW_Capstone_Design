"""
CQ500 데이터로더 - AE/AE-U 모델 학습 파이프라인 (2D 방식)
- 학습: HEALTHY(정상) 샘플만 사용
- 검증/테스트: HEALTHY + ICH(비정상) 샘플 사용
- 개별 DICOM 파일을 2D 이미지로 직접 로드
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 로컬 모듈
from dataloaders.preprocess import DICOMLoader
from dataloaders.data_utils import get_train_transforms

class CQ500Dataset(Dataset):
    """
    CQ500 데이터셋 클래스 (2D 직접 로드 방식)
    
    Args:
        data_root: CQ500 데이터 루트 경로
        label_csv: 라벨 CSV 파일 경로
        mode: 'train', 'val', 'test' 중 하나
        img_size: 2D 이미지 크기 (default: 64)
        transform: 추가 transform (augmentation)
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        window_center: CT windowing center
        window_width: CT windowing width
        slice_range: 사용할 슬라이스 범위 (start, end) 또는 None (중간 80%)
        use_3d_volume: True면 3D 볼륨으로 합침 (legacy), False면 2D 직접 로드
        seed: Random seed
    """
    
    def __init__(
        self,
        data_root: str,
        label_csv: str,
        mode: str = "train",
        img_size: int = 64,
        transform: Optional[transforms.Compose] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        window_center: int = 40,
        window_width: int = 80,
        slice_range: Optional[tuple] = None,
        use_3d_volume: bool = False,  # NEW: 조건 분기
        seed: int = 42
    ):
        assert mode in ["train", "val", "test"], f"Invalid mode: {mode}"
        
        self.data_root = data_root
        self.mode = mode
        self.img_size = img_size
        self.transform = transform
        self.slice_range = slice_range
        self.use_3d_volume = use_3d_volume
        
        # DICOM 로더 초기화
        self.dicom_loader = DICOMLoader(
            data_root=data_root,
            window_center=window_center,
            window_width=window_width,
            verbose=False
        )
        
        # 라벨 CSV 로드 및 데이터 분할
        self.labels_df = pd.read_csv(label_csv, sep=';')
        self._split_data(train_ratio, val_ratio, seed)
        
        # 모드에 따라 샘플 선택
        self.samples = self._get_samples_for_mode()
        
        # 슬라이스 데이터 준비
        self.slice_data = []  # [(sample_name, dicom_path, label), ...]
        self._prepare_slices()
        
        print(f"[{mode.upper()}] Total volumes: {len(self.samples)}, "
              f"Total 2D slices: {len(self.slice_data)} "
              f"(Method: {'3D→2D' if use_3d_volume else '2D direct'})")
    
    def _split_data(self, train_ratio: float, val_ratio: float, seed: int):
        """HEALTHY 데이터를 Train/Val/Test로 분할"""
        np.random.seed(seed)
        
        # HEALTHY vs ICH 분할
        # 중요: is_pathl=0 조건을 명시적으로 사용 (라벨링 오류 방지)
        healthy_samples = self.labels_df[
            (self.labels_df['diagnosis'] == 'HEALTHY') & 
            (self.labels_df['is_pathl'] == 0)
        ].copy()
        
        # ICH 샘플 (병변이 있는 샘플)
        ich_samples = self.labels_df[
            self.labels_df['is_pathl'] == 1
        ].copy()
        
        # HEALTHY 데이터 셔플 및 분할
        healthy_shuffled = healthy_samples.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        n_total = len(healthy_shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        self.train_healthy = healthy_shuffled[:n_train]
        self.val_healthy = healthy_shuffled[n_train:n_train + n_val]
        self.test_healthy = healthy_shuffled[n_train + n_val:]
        
        # ICH 데이터는 Val/Test에만 사용
        self.test_ich = ich_samples
    
    def _get_samples_for_mode(self) -> pd.DataFrame:
        """모드에 따라 샘플 반환"""
        if self.mode == "train":
            # 학습: HEALTHY만 사용 (Anomaly Detection)
            return self.train_healthy
        elif self.mode == "val":
            # 검증: HEALTHY + ICH
            return pd.concat([self.val_healthy, self.test_ich], ignore_index=True)
        else:  # test
            # 테스트: HEALTHY + ICH
            return pd.concat([self.test_healthy, self.test_ich], ignore_index=True)
    
    def _prepare_slices(self):
        """
        2D 슬라이스 데이터 준비
        
        use_3d_volume=False (기본): 개별 DICOM 파일 경로 수집
        use_3d_volume=True (legacy): 3D 볼륨으로 합쳤다가 분할
        """
        if self.use_3d_volume:
            # [Legacy] 3D 볼륨 방식
            print(f"[{self.mode.upper()}] Converting 3D volumes to 2D slices (legacy mode)...")
            self._prepare_slices_from_3d()
        else:
            # [NEW] 2D 직접 로드 방식
            print(f"[{self.mode.upper()}] Loading 2D DICOM slices directly...")
            self._prepare_slices_from_2d()
    
    def _prepare_slices_from_2d(self):
        """개별 DICOM 파일을 2D로 직접 로드 (NEW!)"""
        for idx, row in self.samples.iterrows():
            sample_name = row['name']
            label = int(row['is_pathl'])
            
            try:
                # DICOM 파일 리스트 가져오기
                dicom_files = self.dicom_loader.get_dicom_file_list(sample_name)
                
                # Slice 범위 결정
                total_slices = len(dicom_files)
                if self.slice_range is not None:
                    start_idx, end_idx = self.slice_range
                else:
                    # 중간 80% 사용 (양 끝 10%씩 제외)
                    start_idx = int(total_slices * 0.1)
                    end_idx = int(total_slices * 0.9)
                
                start_idx = max(0, start_idx)
                end_idx = min(total_slices, end_idx)
                
                # 선택된 DICOM 파일들을 개별 샘플로 추가
                for slice_idx in range(start_idx, end_idx):
                    self.slice_data.append({
                        'sample_name': sample_name,
                        'dicom_path': dicom_files[slice_idx],
                        'slice_idx': slice_idx,
                        'label': label
                    })
            
            except Exception as e:
                print(f"[Warning] Failed to load {sample_name}: {e}")
                continue
        
        print(f"[{self.mode.upper()}] Collected {len(self.slice_data)} 2D DICOM files "
              f"from {len(self.samples)} patients")
    
    # def _prepare_slices_from_3d(self):
    #     """3D 볼륨을 2D 슬라이스로 분할 (Legacy)"""
    #     for idx, row in self.samples.iterrows():
    #         sample_name = row['name']
    #         label = int(row['is_pathl'])
            
    #         try:
    #             # 3D 볼륨 로드
    #             volume = self.dicom_loader.load_and_preprocess(sample_name, as_3d=True)
                
    #             # Slice 범위 결정
    #             if self.slice_range is not None:
    #                 start_idx, end_idx = self.slice_range
    #             else:
    #                 depth = volume.shape[0]
    #                 start_idx = int(depth * 0.1)
    #                 end_idx = int(depth * 0.9)
                
    #             start_idx = max(0, start_idx)
    #             end_idx = min(volume.shape[0], end_idx)
                
    #             # 각 슬라이스를 개별 샘플로 추가
    #             for slice_idx in range(start_idx, end_idx):
    #                 self.slice_data.append({
    #                     'sample_name': sample_name,
    #                     'slice_idx': slice_idx,
    #                     'label': label,
    #                     'volume_shape': volume.shape,
    #                     'use_3d': True
    #                 })
            
    #         except Exception as e:
    #             print(f"[Warning] Failed to load {sample_name}: {e}")
    #             continue
        
    #     print(f"[{self.mode.upper()}] Generated {len(self.slice_data)} 2D slices "
    #           f"from {len(self.samples)} 3D volumes")
    
    def __len__(self) -> int:
        return len(self.slice_data)
    
    def __getitem__(self, index: int) -> dict:
        """
        2D 슬라이스 반환
        
        현재: 단일 2D 슬라이스 (1, H, W)
        
        TODO [향후 개선 v2.0]: Spatial 정보 추가
          1. Multi-slice: 인접 슬라이스도 로드 → (3, H, W)
          2. Positional encoding: 슬라이스 위치 정보 추가
          3. 2.5D approach: 다중 뷰 결합
        """
        slice_info = self.slice_data[index]
        sample_name = slice_info['sample_name']
        slice_idx = slice_info['slice_idx']
        label = slice_info['label']
        
        if self.use_3d_volume:
            # [Legacy] 3D 볼륨에서 슬라이스 추출
            volume = self.dicom_loader.load_and_preprocess(sample_name, as_3d=True)
            slice_2d = volume[slice_idx, :, :]  # (H, W)
        else:
            # [NEW] DICOM 파일 직접 로드 (422배 빠름!)
            dicom_path = slice_info['dicom_path']
            slice_2d = self.dicom_loader.load_single_dicom_2d(dicom_path)  # (H, W)
        
        # TODO [향후 개선]: Multi-slice 로드 구현
        # if self.use_multi_slice:
        #     dicom_files = self.dicom_loader.get_dicom_file_list(sample_name)
        #     prev_idx = max(0, slice_idx - 1)
        #     next_idx = min(len(dicom_files) - 1, slice_idx + 1)
        #     prev_slice = self.dicom_loader.load_single_dicom_2d(dicom_files[prev_idx])
        #     next_slice = self.dicom_loader.load_single_dicom_2d(dicom_files[next_idx])
        #     # Stack: (3, H, W)
        #     slice_2d = np.stack([prev_slice, slice_2d, next_slice])
        
        # 0-255 범위로 변환 (PIL Image 호환)
        slice_2d = (slice_2d * 255).astype(np.uint8)
        
        # PIL Image로 변환 및 리사이즈
        img = Image.fromarray(slice_2d, mode='L')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Transform 적용
        if self.transform is not None:
            img = self.transform(img)
        else:
            # 기본 transform
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.5], [0.5])(img)
        
        # TODO [향후 개선]: Positional encoding 추가
        # if self.add_positional_encoding:
        #     position = slice_idx / len(self.dicom_loader.get_dicom_file_list(sample_name))
        #     # position을 추가 정보로 반환하거나 이미지에 embed
        
        # 최종 출력
        return {
            "img": img,  # (1, H, W) - 2D slice
            "label": torch.tensor(label, dtype=torch.long),
            "name": f"{sample_name}_slice{slice_idx:03d}"
        }


class CQ500DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for CQ500
    
    Args:
        data_root: CQ500 데이터 루트 경로
        label_csv: 라벨 CSV 파일 경로
        batch_size: 배치 사이즈
        spatial_size: 리사이즈할 공간 크기
        num_workers: DataLoader worker 수
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        use_augmentation: 학습 시 augmentation 사용 여부
        window_center: CT windowing center
        window_width: CT windowing width
        seed: Random seed
    """
    
    def __init__(
        self,
        data_root: str = "./CQ500",
        label_csv: str = "./CQ500/cq500_labels.csv",
        batch_size: int = 64,
        img_size: int = 64,
        num_workers: int = 4,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        use_augmentation: bool = True,
        window_center: int = 40,
        window_width: int = 80,
        slice_range: Optional[tuple] = None,
        use_3d_volume: bool = False,
        seed: int = 42
    ):
        super().__init__()
        self.data_root = data_root
        self.label_csv = label_csv
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.use_augmentation = use_augmentation
        self.window_center = window_center
        self.window_width = window_width
        self.slice_range = slice_range
        self.use_3d_volume = use_3d_volume
        self.seed = seed
    
    def _get_2d_transform(self, is_train: bool = False):
        """2D 이미지용 transform 생성"""
        if is_train and self.use_augmentation:
            # 학습용 augmentation
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            # 검증/테스트용 (augmentation 없음)
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
    
    def setup(self, stage: Optional[str] = None):
        """데이터셋 setup"""
        
        if stage in (None, 'fit'):
            # 학습 데이터셋
            self.train_dataset = CQ500Dataset(
                data_root=self.data_root,
                label_csv=self.label_csv,
                mode="train",
                img_size=self.img_size,
                transform=self._get_2d_transform(is_train=True),
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                window_center=self.window_center,
                window_width=self.window_width,
                slice_range=self.slice_range,
                use_3d_volume=self.use_3d_volume,
                seed=self.seed
            )
            
            # 검증 데이터셋
            self.val_dataset = CQ500Dataset(
                data_root=self.data_root,
                label_csv=self.label_csv,
                mode="val",
                img_size=self.img_size,
                transform=self._get_2d_transform(is_train=False),
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                window_center=self.window_center,
                window_width=self.window_width,
                slice_range=self.slice_range,
                use_3d_volume=self.use_3d_volume,
                seed=self.seed
            )
        
        if stage in (None, 'test'):
            # 테스트 데이터셋
            self.test_dataset = CQ500Dataset(
                data_root=self.data_root,
                label_csv=self.label_csv,
                mode="test",
                img_size=self.img_size,
                transform=self._get_2d_transform(is_train=False),
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                window_center=self.window_center,
                window_width=self.window_width,
                slice_range=self.slice_range,
                use_3d_volume=self.use_3d_volume,
                seed=self.seed
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
