# UWB 딥러닝 실내 측위 (UWB Indoor Positioning)

UWB 신호로 실내 위치(x, y)를 추정하는 딥러닝 회귀 프로젝트.
장애물로 인한 **NLOS·multipath** 때문에 부정확해지는 전통 측위를, 신호 파형을 학습하는 딥러닝으로 개선했습니다.

## 문제

- 실내는 GPS 불가 → 앵커(비콘) 신호로 위치를 추정해야 함
- 벽·장애물로 신호가 반사·굴절(NLOS/multipath)되면 도달 시간만으로는 오차가 큼
- **신호 파형(CIR)까지 학습**해 오차, 특히 큰 오차(꼬리)를 줄이는 것이 목표

## 데이터 & 입력 특징

- 입력: **8개 앵커 × [ ToA(1) + CIR(128) + CIR Energy(1) ] = 130** 차원
- **CIR Energy(RSRP)** 는 원본에 없던 파생 특징 — `sum(CIR²)` 로 직접 계산해 추가 (신호 세기 → 앵커 신뢰도 판단에 활용)
- 전처리: `log1p` 로 동적 범위 압축 → `StandardScaler` 정규화
- 출력: (x, y) 좌표 회귀

## 모델 발전 과정

| 모델 | 핵심 | 테스트 RMSE |
| --- | --- | --- |
| DNN | 베이스라인 | ≈ 1.88 m |
| 1D CNN | CIR 시간축 지역 패턴, RSRP 추가 | ≈ 1.49 m |
| **2D CNN + 어텐션** | 앵커×시간 공간관계 + 신뢰도 가중 | **≈ 1.27 m** |

**최종 모델(2D CNN) 구성**
- **멀티스케일 conv**: `(3×5)` ∥ `(5×9)` 커널 병렬 후 concat — sharp한 LOS 피크와 넓게 퍼진 multipath를 동시에 포착
- **Dilated conv**: 파라미터를 늘리지 않고 넓은 수용영역 확보
- **듀얼 어텐션**: Channel(SE 블록) + AP(앵커별) 어텐션으로 NLOS 앵커·잡음 특징 억제
- swish 활성화 + BatchNorm, Dense head에 residual 블록

## 결과

- DNN → 1D CNN → 2D CNN 순으로 RMSE 감소, 최종 **테스트 RMSE ≈ 1.27 m**
- 단순 평균 오차보다 **큰 오차(꼬리) 억제**에 초점 → RMSE·90% error·CDF·공간 오차 히트맵으로 평가
- 어텐션·2D 구조가 NLOS 구간의 큰 오차를 효과적으로 완화



### 모델별 성능 비교

<img width="600" height="350" alt="image" src="https://github.com/user-attachments/assets/dde3bbff-7a42-444f-bf9a-9120b5eee5a4" />

| 모델 | RMSE[m] | Median[m] | 90% error[m] | ≤2.0m |
| --- | --- | --- | --- | --- |
| DNN | 1.889 | 1.057 | 3.157 | 73.4% |
| 1D CNN | 1.496 | 0.837 | 2.553 | 81.1% |
| **2D CNN** | **1.270** | **0.828** | **2.078** | **88.8%** |

### 오차 분포 비교 (CDF)

<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/7c95f219-4edc-4b8a-bdd6-14db0cea51fe" />

2D CNN(초록)이 전 구간에서 가장 낮은 오차 분포를 보이며, 특히 오차가 큰 구간에서 격차가 벌어집니다 — NLOS 환경의 큰 오차를 효과적으로 억제했음을 의미합니다.

### 최종 모델 (2D CNN) 상세

**학습 곡선 및 공간 오차 히트맵**

<img width="45%" height="400" alt="image" src="https://github.com/user-attachments/assets/464cd679-6112-49b6-833d-c72d23f0d164" /> <img width="45%" height="845" alt="image" src="https://github.com/user-attachments/assets/6a20fca7-b51b-41a6-b2d9-2be748ba4ba3" />

- 학습 곡선 : train/val loss가 안정적으로 수렴
- 오차 히트맵 : 측정 경로 전반에서 오차가 대체로 낮게(파란색) 유지되며, 큰 오차 구간이 국소적으로만 나타남

## 기술 스택

`Python` · `TensorFlow/Keras` · `NumPy` · `pandas` · `scikit-learn` · `matplotlib`

## 파일

- `uwb_final.py` — 최종 모델(2D CNN + 멀티스케일 + 듀얼 어텐션) 전체 파이프라인

## 배운 점

- RF 신호(ToA·CIR·energy)에 대한 도메인 이해가 특징 설계에 직결됨
- "평균 오차가 작은 모델"보다 "큰 오차를 얼마나 억제하느냐"가 실내 측위의 핵심
