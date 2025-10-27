"""
CQ500 DICOM 전처리 모듈
- DICOM 파일 로드 및 3D 볼륨 변환
- 전처리 파이프라인
"""

import os
import glob
import numpy as np
import SimpleITK as sitk
from typing import Optional, Tuple
from pathlib import Path

from .data_utils import apply_ct_windowing, convert_sample_name


class DICOMLoader:
    """
    DICOM 파일 로더 클래스
    """
    
    def __init__(
        self,
        data_root: str,
        window_center: int = 40,
        window_width: int = 80,
        verbose: bool = False
    ):
        """
        Args:
            data_root: CQ500 데이터 루트 경로
            window_center: CT windowing center
            window_width: CT windowing width
            verbose: 상세 로그 출력 여부
        """
        self.data_root = data_root
        self.window_center = window_center
        self.window_width = window_width
        self.verbose = verbose
    
    def get_dicom_series_path(self, sample_name: str) -> str:
        """
        DICOM 시리즈 폴더 경로 가져오기
        
        Args:
            sample_name: CSV의 샘플명 (예: CT_0)
        
        Returns:
            DICOM 시리즈가 있는 폴더 경로
        """
        # CT_xxx -> CQ500-CT-xxx 변환
        folder_name = convert_sample_name(sample_name)
        base_path = os.path.join(self.data_root, "extracted", folder_name)
        
        if not os.path.exists(base_path):
            raise FileNotFoundError(
                f"Patient folder not found: {base_path}\n"
                f"Original sample name: {sample_name}"
            )
        
        # DICOM 파일들이 있는 폴더 찾기 (재귀적으로 검색)
        dicom_files = glob.glob(os.path.join(base_path, "**", "*.dcm"), recursive=True)
        
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in: {base_path}")
        
        # 첫 번째 DICOM 파일의 디렉토리 반환 (같은 시리즈)
        series_path = os.path.dirname(dicom_files[0])
        
        if self.verbose:
            print(f"Found {len(dicom_files)} DICOM files in: {series_path}")
        
        return series_path
    
    def load_dicom_series(self, series_path: str) -> sitk.Image:
        """
        DICOM 시리즈를 SimpleITK 이미지로 로드
        
        Args:
            series_path: DICOM 시리즈 폴더 경로
        
        Returns:
            SimpleITK Image 객체
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(series_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        if self.verbose:
            print(f"Loaded DICOM series: {image.GetSize()}, "
                  f"Spacing: {image.GetSpacing()}, "
                  f"Origin: {image.GetOrigin()}")
        
        return image
    
    def get_dicom_file_list(self, sample_name: str) -> list:
        """
        환자의 모든 DICOM 파일 경로 리스트 반환
        
        Args:
            sample_name: CSV의 샘플명
        
        Returns:
            정렬된 DICOM 파일 경로 리스트
        """
        series_path = self.get_dicom_series_path(sample_name)
        
        # DICOM 파일들 찾기
        dicom_files = glob.glob(os.path.join(series_path, "*.dcm"))
        dicom_files.sort()  # 파일명으로 정렬
        
        if self.verbose:
            print(f"Found {len(dicom_files)} DICOM files in: {series_path}")
        
        return dicom_files
    
    def load_single_dicom_2d(self, dicom_path: str) -> np.ndarray:
        """
        단일 DICOM 파일을 2D 이미지로 로드 (NEW!)
        
        Args:
            dicom_path: DICOM 파일 경로
        
        Returns:
            전처리된 2D 이미지 (H, W), 0-1 normalized
        """
        # 단일 DICOM 파일 읽기
        image = sitk.ReadImage(dicom_path)
        
        # NumPy 배열로 변환
        array = sitk.GetArrayFromImage(image)  # Shape: (1, H, W) or (H, W)
        
        # 2D로 변환 (depth 차원 제거)
        if array.ndim == 3:
            array = array[0, :, :]  # (H, W)
        
        # Windowing 적용 (HU -> 0-1 정규화)
        array = apply_ct_windowing(
            array,
            self.window_center,
            self.window_width
        )
        
        return array  # (H, W)
    
    def load_and_preprocess(self, sample_name: str, as_3d: bool = True) -> np.ndarray:
        """
        DICOM 로드 및 전처리 (windowing 포함)
        
        Args:
            sample_name: CSV의 샘플명
            as_3d: True면 3D 볼륨으로, False면 2D 슬라이스 리스트로
        
        Returns:
            as_3d=True: 전처리된 3D 볼륨 (D, H, W)
            as_3d=False: 2D 슬라이스 경로 리스트
        """
        if not as_3d:
            # 2D 방식: 파일 경로만 반환
            return self.get_dicom_file_list(sample_name)
        
        # 3D 방식: 기존 로직
        series_path = self.get_dicom_series_path(sample_name)
        image = self.load_dicom_series(series_path)
        volume = sitk.GetArrayFromImage(image)  # Shape: (D, H, W)
        
        # Windowing 적용
        volume = apply_ct_windowing(
            volume, 
            self.window_center, 
            self.window_width
        )
        
        if self.verbose:
            print(f"Preprocessed volume shape: {volume.shape}, "
                  f"Range: [{volume.min():.3f}, {volume.max():.3f}]")
        
        return volume


class VolumePreprocessor:
    """
    3D 볼륨 전처리 파이프라인
    """
    
    def __init__(
        self,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        clip_range: Optional[Tuple[float, float]] = None,
        apply_normalization: bool = True
    ):
        """
        Args:
            target_spacing: 타겟 spacing (리샘플링) (z, y, x)
            clip_range: 값 클리핑 범위 (min, max)
            apply_normalization: 정규화 적용 여부
        """
        self.target_spacing = target_spacing
        self.clip_range = clip_range
        self.apply_normalization = apply_normalization
    
    def resample_volume(
        self, 
        image: sitk.Image, 
        target_spacing: Tuple[float, float, float]
    ) -> sitk.Image:
        """
        볼륨 리샘플링 (spacing 변경)
        
        Args:
            image: SimpleITK Image
            target_spacing: 타겟 spacing (z, y, x)
        
        Returns:
            리샘플링된 SimpleITK Image
        """
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        # 새로운 크기 계산
        new_size = [
            int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
            for i in range(3)
        ]
        
        # 리샘플링
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(image.GetPixelIDValue())
        resampler.SetInterpolator(sitk.sitkLinear)
        
        return resampler.Execute(image)
    
    def preprocess(self, volume: np.ndarray) -> np.ndarray:
        """
        볼륨 전처리 적용
        
        Args:
            volume: 입력 볼륨
        
        Returns:
            전처리된 볼륨
        """
        # 클리핑
        if self.clip_range is not None:
            volume = np.clip(volume, self.clip_range[0], self.clip_range[1])
        
        # 정규화
        if self.apply_normalization:
            vol_min = volume.min()
            vol_max = volume.max()
            if vol_max - vol_min > 0:
                volume = (volume - vol_min) / (vol_max - vol_min)
        
        return volume.astype(np.float32)


def load_cq500_volume(
    sample_name: str,
    data_root: str,
    window_center: int = 40,
    window_width: int = 80,
    verbose: bool = False
) -> np.ndarray:
    """
    CQ500 DICOM 볼륨 로드 (간편 함수)
    
    Args:
        sample_name: 샘플명 (예: CT_0)
        data_root: CQ500 데이터 루트 경로
        window_center: CT windowing center
        window_width: CT windowing width
        verbose: 상세 로그 출력
    
    Returns:
        전처리된 3D 볼륨
    """
    loader = DICOMLoader(data_root, window_center, window_width, verbose)
    return loader.load_and_preprocess(sample_name)


# 사용 예제
if __name__ == "__main__":
    """
    사용 예제
    """
    # DICOM Loader 초기화
    loader = DICOMLoader(
        data_root="./CQ500",
        window_center=40,
        window_width=80,
        verbose=True
    )
    
    # 샘플 로드
    try:
        volume = loader.load_and_preprocess("CT_0")
        print(f"\n로드 성공!")
        print(f"Volume shape: {volume.shape}")
        print(f"Volume range: [{volume.min():.3f}, {volume.max():.3f}]")
        print(f"Volume dtype: {volume.dtype}")
    except Exception as e:
        print(f"에러 발생: {e}")
