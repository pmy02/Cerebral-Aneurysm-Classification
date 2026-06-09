<!-- Language switcher -->
[English](README.md) | **한국어**

# 뇌동맥류 분류 (Cerebral Aneurysm Classification)

익명화된 뇌혈관조영술 영상에서 뇌동맥류의 **존재 여부**와 **발생 위치**를 분류하는 딥러닝 파이프라인입니다.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-ee4c2c)
![Task](https://img.shields.io/badge/task-medical%20image%20classification-informational)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

> ⚠️ **연구·교육용 프로젝트 (2023.06–07).** 임상 의사결정 도구가 아니며 진단 용도로 검증되지 않았습니다. 실제 진료에 사용하지 마세요.

---

## 개요

뇌동맥류는 두개내 동맥의 국소적 팽창으로, 파열 시 생명을 위협할 수 있어 혈관조영 영상에서의 조기 발견이 임상적으로 중요합니다. 본 프로젝트는 익명화된 뇌혈관조영술 영상에 대해 검출을 두 개의 연결된 학습 문제로 정의합니다.

1. **동맥류 검출** — 동맥류 존재 여부의 이진 분류
2. **위치 분류** — 어떤 동맥 분절에 동맥류가 있는지에 대한 다중 레이블 분류

환자 한 명당 8장의 조영 영상이 있으며, 한 장의 영상으로 모든 동맥 분절을 볼 수 없기 때문에 학습 전에 영상과 레이블을 뇌혈류의 전방/후방 순환에 따라 나눕니다.

<!-- TODO: 그림을 저장소에 커밋한 assets/ 폴더로 옮기고 상대 경로(예: ![목표](assets/goal.png))로 참조하면 이 저장소 밖에서도 README가 정상 렌더링됩니다 -->
![프로젝트 목표](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/aaab8df7-92f5-4f89-b43a-7a954261ea40)

## 문제 정의

뇌혈류는 전방(내경동맥, ICA)과 후방(추골동맥, VA) 순환으로 나뉘므로, 레이블도 이에 맞춰 묶었습니다.

| 순환 | 주 혈관 | 분절(레이블) |
|------|---------|--------------|
| 전방 | ICA | ICA, AntChor, ACA, ACOM, MCA |
| 후방 | VA  | VA, PICA, SCA, BA, PCA, PCOM |

이로써 **4개의 데이터셋**이 생깁니다: 이진 검출용과 다중 레이블용을 각각 전방·후방 하위 집합으로 나눈 형태입니다.

## 파이프라인

![파이프라인](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/becfd58a-7d74-4cb5-ac28-f3407dc530ad)

전체 흐름: 영상 전처리 → 순환별 분할 → 이진 검출 → 양성 케이스에 대해 다중 레이블 위치 분류 → 환자 단위 예측 통합.

## 데이터와 전처리

![데이터셋 구성](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/e0f9f236-abef-4d41-a3b9-b2c0a522d707)

**레이블 구성 (다중 레이블 / 위치).** I/V 표식을 기준으로 영상 경로를 전방/후방으로 나눕니다. `train.csv`에서 L/R, I/V 기준으로 해당 컬럼을 재배치해 새로운 전방·후방 데이터프레임을 만듭니다. 환자당 8장이 전/후방으로 나뉘므로 이 데이터프레임의 행 수는 원본의 약 4배입니다. 좌/우(L/R)는 레이블 단계에서 하나로 통합하되, 같은 레이블이라도 L/R 영상에 따라 양성·음성이 달라지므로 추론 시 다시 분리합니다.

**레이블 구성 (이진 / 검출).** 각 행에서 위치 레이블이 하나라도 양성이면 이진 레이블을 1, 아니면 0으로 둡니다. 이렇게 다중 레이블용 `anterior`/`posterior`에 대응하는 `binary_anterior`/`binary_posterior`를 만듭니다.

**영상 전처리.** 학습 영상은 세 가지 형태였고 각각 다음과 같이 처리했습니다.
- *여백·글자 없음* — 그대로 사용.
- *글자 + 여백* — 픽셀값으로 여백을 제거하고, 글자가 고정 위치에 있으므로 고정 영역을 지정해 인접 영역에서 샘플링한 평균 배경색으로 덮음.
- *여백만 있음* — 여백 크기가 영상마다 달라, 중심 기준 수평·수직 라인을 스캔해 회색이 시작되는 좌표를 찾아 그 지점에서 잘라냄.

> 기반 데이터셋은 비공개·익명화된 의료 영상으로 여기서 배포하지 않습니다. 입력 형식은 [재현](#재현) 항목을 참고하세요.

## 방법

![모델 구조](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/57600112-1e36-4d0b-9d0c-9761659d3bae)

**이진 검출 (전방·후방 공통).**
- 백본: **MedNet** — 회색조 의료 영상으로 사전학습된 가중치([MedicalNet-Resnet18](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18/blob/main/resnet_18.pth))로 초기화한 ResNet-18.
- 손실: `BCELoss`. 증강: resize, normalize. 추가 처리: `cv2.bitwise_not`, `cv2.medianBlur`.

**다중 레이블 위치 분류 (순환별 별도 모델).**
- 전방: **SwinV2** — `timm`의 사전학습 `swinv2_cr_tiny_ns_224`.
- 후방: **ResNet-18** — `torchvision` 사전학습 모델.
- 손실: **Asymmetric Loss** ([Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL)). 증강: resize, normalize.

**판정 규칙.**
1. 이진 검출을 실행해 임계점을 넘는 행은 동맥류 양성으로 보고, 나머지는 위치 레이블을 모두 0으로 둡니다. 임계점은 `train.csv`에 대한 이진 예측값의 중앙값입니다.
2. 양성 행은 다중 레이블 분류를 실행하고, 각 위치는 해당 위치 예측값의 90백분위수(총 21개)를 임계점으로 둡니다.
3. 예측값은 9016×1 형태(환자당 8장)이므로, 전방·후방 각 4장의 평균을 구하고 그중 큰 값으로 환자 단위 동맥류 점수를 결정합니다.

## 사전학습 모델

`Model/` 디렉터리에 학습된 체크포인트가 있습니다.

| 파일 | 역할 | 백본 |
|------|------|------|
| `MedNet_ant_binary.pt`      | 전방 이진 검출 | ResNet-18 (MedicalNet) |
| `MedNet_pos_binary.pt`      | 후방 이진 검출 | ResNet-18 (MedicalNet) |
| `SwinNet_ant_multilabel.pt` | 전방 위치 분류 | SwinV2 (timm) |
| `ResNet_pos_multilabel.pt`  | 후방 위치 분류 | ResNet-18 (torchvision) |
| `resnet_18.pth`             | MedicalNet 기반 가중치 | ResNet-18 |

> **저장소 위생 참고.** `SwinNet_ant_multilabel.pt`와 `resnet_18.pth`는 Git LFS로 추적되지만, 나머지 세 체크포인트는 `.gitattributes` 없이 원시 바이너리로 커밋돼 있습니다. 히스토리를 가볍게 유지하려면 모든 모델 파일을 LFS로 일관되게 추적하세요(`.gitattributes` 제공).

## 결과

지표는 [`scripts/evaluate.py`](scripts/evaluate.py)로 계산합니다. 체크포인트를 불러와
검증 분할에서 추론하고 아래 표를 출력합니다. 이진 과제는 중앙값 임계점에서 F1/AUROC를,
위치 과제는 클래스별 90백분위수 임계점(본 프로젝트의 판정 규칙)에서 macro-F1을 보고합니다.
실제 값은 데이터에 직접 돌려 채우세요 — 아래 표는 추정하지 않고 일부러 비워 두었습니다.

```bash
python -m scripts.evaluate --config configs/binary_anterior.yaml \
    --checkpoint Model/MedNet_ant_binary.pt
```

<!-- scripts/evaluate.py 출력으로 채우기; 추정 금지 -->
| 과제 | 모델 | 지표 | 값 |
|------|------|------|-----|
| 이진 검출 (전방) | MedNet | F1 / AUROC | _evaluate.py 실행_ |
| 이진 검출 (후방) | MedNet | F1 / AUROC | _evaluate.py 실행_ |
| 위치 분류 (전방) | SwinV2 | macro-F1 | _evaluate.py 실행_ |
| 위치 분류 (후방) | ResNet-18 | macro-F1 | _evaluate.py 실행_ |

## 설명가능성 (Explainability)

글자·여백을 제거한 영상에 대해(증강 없이) Grad-CAM을 적용하여 ResNet-18의 판단 근거 영역을 살펴봤습니다. 히트맵은 해석 가능했지만, 정답 정확도를 기준으로 두지 않고 적용했기에 엄밀한 귀인보다는 정성적 확인 용도로 사용했습니다.

![Grad-CAM 예시](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/df6a2bb5-2efd-476e-8871-a1934169d5e7)

## 탐색적 실험

시도한 접근과 결과를 부정적 결과까지 포함해 솔직하게 정리했습니다.

- **규칙 기반 크롭** — 양성 레이블 위치 기준으로 영상을 잘라 학습. 한 분절에 동맥류가 2개 이상이어도 위치 레이블은 1 하나로만 표기되어 정확한 위치를 복원할 수 없어 중단.
- **위치/각도/방향별 분할 학습** — 하위 집합별로 weighted sampling과 class weight를 적용. 클래스 불균형의 영향이 더 커서 데이터가 적은 클래스 성능이 낮았고, 가중치 기법의 개선폭은 미미.
- **자기지도 사전학습** — 불균형에 강한 특성을 학습하기 위한 오토인코더(200 epoch) 후 지도 이진 분류, 그리고 ResNet-50 특성 기반 대조학습(같은 클래스는 가깝게, 다른 클래스는 멀게). 불균형으로 지도학습 성능은 여전히 제한적.
- **fc→conv 변형 ResNet-18** — 공간 정보 보존 목적. 모든 epoch에서 F1이 53.68%로 일정.
- **DenseNet** — 더 강한 특성 추출을 위해 시도.

## 저장소 구조

```
Cerebral-Aneurysm-Classification/
├── Model/                  # 학습된 체크포인트 (사전학습 모델 참고)
├── src/
│   ├── data.py             # 데이터 구성, 전방/후방 분할, Dataset
│   ├── preprocessing.py    # 여백/글자 제거 및 크롭
│   ├── models.py           # MedNet, SwinV2, ResNet-18 빌더
│   ├── losses.py           # Asymmetric Loss (다중 레이블)
│   └── metrics.py          # F1 / AUROC / macro-F1 / mAP
├── scripts/
│   ├── train.py            # config 기반 학습
│   ├── evaluate.py         # 체크포인트 -> 지표 표
│   ├── infer.py            # 전체 판정 규칙 파이프라인
│   └── gradcam.py          # Grad-CAM 히트맵
├── configs/                # 과제/순환별 YAML
├── requirements.txt
├── .gitattributes          # 모델 파일 Git LFS 추적
├── README.md               # 영어 (기본)
└── README.ko.md            # 한국어 (이 파일)
```

> `src/`·`scripts/` 코드는 문서화된 방법을 동작 가능한 스켈레톤으로 재구성한 것입니다.
> 학습 전에 각 모듈을 원래 구현과 대조해 검증하고, `src/data.py`의 데이터 스키마
> TODO를 채우세요.

## 시작하기

```bash
git clone https://github.com/pmy02/Cerebral-Aneurysm-Classification.git
cd Cerebral-Aneurysm-Classification
git lfs install && git lfs pull          # LFS 추적 체크포인트 받기
pip install -r requirements.txt          # 재현을 위해 버전 고정
```

학습·평가·추론 (모두 config 기반):
```bash
python -m scripts.train    --config configs/binary_anterior.yaml
python -m scripts.evaluate --config configs/binary_anterior.yaml \
    --checkpoint Model/MedNet_ant_binary.pt
python -m scripts.gradcam  --checkpoint Model/MedNet_ant_binary.pt --image path/to/frame.png
```

> 데이터 로딩은 `train.csv` 스키마가 확정되기 전까지 placeholder에 연결돼 있습니다 —
> `src/data.py`의 TODO를 참고하세요.

## 재현

다른 사람이 작업을 재현할 수 있도록 다음을 문서화하세요(채울 항목).
- **환경** — Python·PyTorch 버전, CUDA 버전. <!-- TODO -->
- **의존성** — `timm`, `torchvision`, `opencv-python` 등 `requirements.txt`에 버전 고정. <!-- TODO -->
- **데이터** — `train.csv` 스키마와 이미지 디렉터리 구조, (비공개) 데이터 획득 방법. <!-- TODO -->
- **하드웨어·시간** — GPU 모델과 대략적 학습 시간. <!-- TODO -->

## 인용

```bibtex
@misc{cerebral_aneurysm_classification_2023,
  title  = {Cerebral Aneurysm Classification},
  author = {<!-- TODO: 저자명 -->},
  year   = {2023},
  url    = {https://github.com/pmy02/Cerebral-Aneurysm-Classification}
}
```

## 감사의 글

- [MedicalNet (TencentMedicalNet)](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18) — 회색조 의료 영상 사전학습 가중치.
- [Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL) — Asymmetric Loss.
- [`timm`](https://github.com/huggingface/pytorch-image-models), `torchvision` — 모델 백본.

## 라이선스

<!-- TODO: LICENSE 파일이 없습니다. 공개 포트폴리오 저장소라면 코드·가중치 사용 조건을 명확히 하기 위해 라이선스(예: MIT) 추가를 고려하세요. -->

## 연락처

박민영 (Minyoung Park) — [LinkedIn](https://www.linkedin.com/in/minyoung-park-672754237) · minyo0119@gmail.com
