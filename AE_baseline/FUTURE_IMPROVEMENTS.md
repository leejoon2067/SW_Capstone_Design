# 향후 개선 사항 (Spatial 정보 추가)

## 현재 상태 (v1.0)
- ✅ 2D 직접 로드 방식 구현
- ✅ 422배 빠른 속도
- ✅ 메모리 효율적
- ❌ Spatial context 없음

---

## 🎯 향후 개선 방향

### Phase 1: Multi-Slice Input (2.5D) 
**목표**: 인접 슬라이스 정보 활용

#### 구현 방법:
```python
# 현재 (1-slice)
input: (1, H, W)  # 단일 슬라이스

# 개선 (3-slice)
input: (3, H, W)  # [이전, 현재, 다음] 슬라이스
```

#### 코드 수정 위치:
```
dataload.py:
  - __init__(): use_multi_slice 파라미터 추가
  - __getitem__(): 인접 슬라이스 로드 로직 추가
  - _load_adjacent_slice(): 헬퍼 함수 구현

networks/ae.py:
  - in_planes=1 → in_planes=3 (3채널 입력)
```

#### 예상 효과:
- ✅ 일부 depth 정보 복원
- ✅ 경계 정보 개선
- ⚠️ 메모리 3배 증가
- ⚠️ 약간의 속도 저하

---

### Phase 2: Positional Encoding
**목표**: 슬라이스 위치 정보 제공

#### 구현 방법:
```python
# 슬라이스 위치를 모델에 제공
slice_position = slice_idx / total_depth  # 0~1 정규화

# 옵션 1: 추가 입력 채널
pos_channel = np.ones((H, W)) * slice_position
input = np.stack([slice_2d, pos_channel])  # (2, H, W)

# 옵션 2: Concatenate to latent
latent = encoder(img)
latent_with_pos = torch.cat([latent, pos_embedding], dim=1)
```

#### 코드 수정 위치:
```
dataload.py:
  - __getitem__(): position encoding 추가
  
networks/ae.py:
  - forward(): positional info 처리
```

#### 예상 효과:
- ✅ 슬라이스 위치 정보 활용
- ✅ 상/중/하 부위별 특성 학습 가능
- ✅ 계산량 증가 거의 없음

---

### Phase 3: 3D Patch-based Approach
**목표**: 작은 3D patch로 학습

#### 구현 방법:
```python
# 전체 볼륨 대신 작은 3D patch 사용
patch_size = (16, 64, 64)  # (D, H, W)

# Random patch sampling
start_d = random.randint(0, depth - 16)
patch = volume[start_d:start_d+16, :, :]
```

#### 코드 수정 위치:
```
dataload.py:
  - __getitem__(): 3D patch extraction 추가
  
networks/:
  - Conv2d → Conv3d 변환
  - 3D 아키텍처 구현
```

#### 예상 효과:
- ✅ 완전한 3D context 복원
- ✅ 작은 patch로 메모리 절약
- ❌ 코드 대폭 수정 필요
- ❌ 학습 시간 증가

---

### Phase 4: Attention Mechanism
**목표**: Slice 간 관계 학습

#### 구현 방법:
```python
# Transformer-based attention
# 여러 슬라이스를 sequence로 취급

slice_sequence = [slice_1, slice_2, ..., slice_N]
attended_features = transformer_encoder(slice_sequence)
```

#### 예상 효과:
- ✅ Long-range dependency 학습
- ✅ Slice 간 관계 모델링
- ❌ 모델 복잡도 대폭 증가

---

## 구현 우선순위

### 즉시 가능 (현재 코드 기반):
1. **2D Direct Load** (완료!)
2. AE/AE-U 모델 학습 (reconstruction 코드 활용)
3. 결과 평가 및 baseline 설정

### 장/단기 개선 process (1-2주):
4. **Multi-Slice Input (2.5D)**
   - 코드 수정 최소
   - 성능 향상 기대
   
5. 🎯 **Positional Encoding**
   - 구현 간단
   - 추가 정보 제공
6. 🔮 3D Patch-based
7. 🔮 Attention Mechanism

---

### 향후 파라미터 추가 예정:
```python
CQ500Dataset(
    ...
    use_multi_slice=False,      # TODO: 2.5D 구현 시
    num_adjacent_slices=1,      # TODO: 인접 슬라이스 개수
    add_positional_encoding=False,  # TODO: 위치 정보 추가
    ...
)
```
