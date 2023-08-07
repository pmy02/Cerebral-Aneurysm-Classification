# Cerebral Aneurysm Classification
- 프로젝트 소개 : 익명화된 뇌혈관조영술 영상을 기반으로 뇌동맥류 여부, 위치를 진단하는 소프트웨어 개발 
- 프로젝트 기간 : 2023.06 ~ 2023.07
# Dataset
크게 Aneurysm 예측을 위한 Dataset과 각 위치들을 예측하기위한 Dataset으로 나누고, 두 Dataset을
각각 anterior(carotid), posterior(vertebral)로 나눔. 총 4개의 Dataset이 존재<br><br>
<strong>위치 Dataset - multilabel classification</strong>
1) 사진에서 I,V를 기준으로 전/후방 image path들을 나눔
2) train.csv에서 L/R, I/V를 기준으로 각각에 맞는 Column들을 할당한 새 Dataframe 생성

![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/3df2bd34-b342-46e9-b98b-9f7c5ab8cd6c)

3) train.csv 크기(len)의 4배, 칼럼은 동일, 새로운 Dataframe anterior, posterior 생성 <br>
(=> 한 명당 8장의 이미지인 데이터를 전/후방으로 나누었기에 크기가 4배임)

![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/de1babb5-f4f6-44d6-8afe-c0ac48b235f0)

4) anterior/posterior에 각 이미지에서 보이는 레이블에 해당하는 값들을 적절한 간격으로
L/R_atnterior, L/R_posterior의 값으로 할당

![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/37d3b648-d618-4c0b-b814-614c09a93771)
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/85962cb8-5735-4a3c-8b29-8c51ad4121ed)
*anterior과 posterior의 레이블에서 L/R을 하나로 통합했지만, 각각 같은 값을 가지는 것이 아닌 사진의
L/R에 따라서 같은 레이블이라도 1인 경우와 0인 경우가 따로 나타나게 됨 (추론 시 L/R 레이블을 다시 생성)

<strong>Aneurysm dataset - binary classi fication</strong>
1) 만들어진 anterior과 posterior을 기준으로 각 행에 대하여 1이 한 개라도 있으면 ‘sum’이라는
label에 1로 할당, 그 외엔 0

![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/48de51e1-e285-4306-8b4e-d7fd1b775a7a)

학습에 사용되는 데이터는 binary classification을 위한 binary_anterior, binary_posterior과 multilabel classifcation을 위한 anterior, posterior <br>
=> 총 4개

# Class 나눈 기준
한 명당 8장의 사진 중,각각의 사진이 train.csv의 레이블에 해당하는 모든 위치들을 볼 수 없다고 판단하였음. <br>
조사 결과, 뇌 혈액 순환은 전/후방에 따라 ICA/VA로 나누어 진다는 사실을 알게 되었고, 이에 따라 전/후방에 위치한 혈관으로 레이블을 나눔. <br><br>
ICA: ICA, AntChor, ACA, ACOM, MCA <br>
VA : VA, PICA, SCA, BA, PCA, PCOM <br>

# Image preprocessing
train dataset은 아래와 같이 3가지 타입으로 나뉘어져 있었으며, 여백과 글자가 없는 경우를 제외한 나머지는 이미지 전처리를 해줬음.
- 여백과 글자가 없는 경우 <br>
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/9dc2357d-b4ca-4176-8778-76d9307fb453)
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/8cd2dc93-dc0f-49bd-8d29-bd7f4d1538d3)
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/b370932d-080a-4bda-b533-34a77b0ab5ad)

- 글자와 여백이 모두 존재하는 경우 <br>
픽셀값을 이용하여 여백을 모두 지워주고 난 뒤, 글자가 모두 같은 위치에 있었기 때문에 고정좌표로 두 번째 사진과 같이 초록색 영역을 설정함. 이후 사진에 맞는 배경색을 가지고 오기 위해서 파란색 영역의 픽셀값을 평균내어 초록색 영역을 덮음. <br>
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/f65d2744-e427-4c09-bfd7-15f96c1a0e34)


- 여백이 있는 경우 <br>
이미지마다 여백의 크기가 달라, 이미지의 center를 기준으로 horizontal line과 vertical line을 슬라이스 한 뒤, 반복문을 돌려 gray color가 <br>
들어오기 시작하는 좌표를 찾아 이미지를 자름 <br>
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/afbe0efe-5c51-4d68-9e3c-2ef80c8285e6)
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/ffb52686-f7f4-4313-a6b4-50a26ce3769f)
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/12454466-b96f-455b-862a-8d7a4d4c77ed)

# 모델 예측 방식
1) Binary classification으로 Aneurysm을 예측하고, 임계점을 넘어 가면 그 행(row)는 위치 label을 모두 0으로 대체 - 임계점은 train.csv에 대한 Aneurysm의 예측값의 중앙값
2) 임계점을 넘는 행은 Multi label classification으로 각 위치에 대한 값을 예측하고, 임계점을 두어 1 또는 0으로 대체 - 임계점은 train.csv에 대한 각 위치 예측값의 90백분위수들 (21개)

* Aneurysm 예측값의 형태는 9016x1 (한 사람 당 8장 이미지에 대한 모든 Aneurysm 예측) 이므로,
anterior, posterior 각각 4장에 대하여 평균 2개를 구하고, 그 중 가장 큰 값으로 Aneurysm 값 결정

# Model
1. binary_classification - anterior/posterior 같은 모델 사용
* MedNet (resnet18) - 여러 medical, gray-scale로 학습된 가중치를 사용
* https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18/blob/main/resnet_18.pth
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/b5e6eee4-6da7-4a20-9d46-856fd48b4f54)
* Loss : BCELoss
* augmen ta tion : resize, no rmalize
* 이미지에 cv2.bitwise_not, cv2.medianBlur를 적용 <br><br>

2. multi label classification - 전/후방 다른 모델 사용
* SwinNet - timm 라이브러리 pretrained swinv2_cr_tiny_ns_224 사용 (anterior dataset)
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/1e2aa568-e0fc-48e0-b895-44725800a8ee)
* ResNet18 - torchvision pretrained resnet18 사용(posterior dataset)
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/110916d0-9352-4ffa-a47e-4a8dcce26f14)
* Loss : Asymmetric Loss
  - https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
* augmentation : resize, normalize

# etc
