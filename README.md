# Cerebral Aneurysm Classification
- 프로젝트 소개 : 익명화된 뇌혈관조영술 영상을 기반으로 뇌동맥류 여부, 위치를 진단하는 소프트웨어 개발 
- 프로젝트 기간 : 2023.06 ~ 2023.07

![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/ffa40320-946b-4aa1-aa31-2f67a6c3b1dc)


# 프로젝트 목표
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/aaab8df7-92f5-4f89-b43a-7a954261ea40)


# Pipeline
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/becfd58a-7d74-4cb5-ac28-f3407dc530ad)


# Dataset
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/e0f9f236-abef-4d41-a3b9-b2c0a522d707)
크게 Aneurysm 예측을 위한 Dataset과 각 위치들을 예측하기위한 Dataset으로 나누고, 두 Dataset을
각각 anterior(carotid), posterior(vertebral)로 나눔. 총 4개의 Dataset이 존재<br><br>
<strong>위치 Dataset - multilabel classification</strong>
1) 사진에서 I,V를 기준으로 전/후방 image path들을 나눔
2) train.csv에서 L/R, I/V를 기준으로 각각에 맞는 Column들을 할당한 새 Dataframe 생성 <br>
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/3df2bd34-b342-46e9-b98b-9f7c5ab8cd6c)

3) train.csv 크기(len)의 4배, 칼럼은 동일, 새로운 Dataframe anterior, posterior 생성 <br>
(=> 한 명당 8장의 이미지인 데이터를 전/후방으로 나누었기에 크기가 4배임) <br>
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/de1babb5-f4f6-44d6-8afe-c0ac48b235f0)

4) anterior/posterior에 각 이미지에서 보이는 레이블에 해당하는 값들을 적절한 간격으로
L/R_atnterior, L/R_posterior의 값으로 할당 <br>
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/37d3b648-d618-4c0b-b814-614c09a93771)
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/85962cb8-5735-4a3c-8b29-8c51ad4121ed)
*anterior과 posterior의 레이블에서 L/R을 하나로 통합했지만, 각각 같은 값을 가지는 것이 아닌 사진의
L/R에 따라서 같은 레이블이라도 1인 경우와 0인 경우가 따로 나타나게 됨 (추론 시 L/R 레이블을 다시 생성)

<strong>Aneurysm dataset - binary classification</strong>
1) 만들어진 anterior과 posterior을 기준으로 각 행에 대하여 1이 한 개라도 있으면 ‘sum’이라는 label에 1로 할당, 그 외엔 0 <br>
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/48de51e1-e285-4306-8b4e-d7fd1b775a7a)

학습에 사용되는 데이터는 binary classification을 위한 binary_anterior, binary_posterior과 multilabel classifcation을 위한 anterior, posterior <br>
=> 총 4개


# Class 나눈 기준
한 명당 8장의 사진 중,각각의 사진이 train.csv의 레이블에 해당하는 모든 위치들을 볼 수 없다고 판단하였음. <br>
조사 결과, 뇌 혈액 순환은 전/후방에 따라 ICA/VA로 나누어 진다는 사실을 알게 되었고, 이에 따라 전/후방에 위치한 혈관으로 레이블을 나눔. <br><br>
ICA: ICA, AntChor, ACA, ACOM, MCA <br>
VA : VA, PICA, SCA, BA, PCA, PCOM <br>


# Image preprocessing
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/c4bdb6e7-8b5f-4c4b-87a4-f54c71012bbd)
train dataset은 아래와 같이 3가지 타입으로 나뉘어져 있었으며, 여백과 글자가 없는 경우를 제외한 나머지는 이미지 전처리를 해줬음.
- 여백과 글자가 없는 경우 <br>
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/9dc2357d-b4ca-4176-8778-76d9307fb453)
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/8cd2dc93-dc0f-49bd-8d29-bd7f4d1538d3)
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/b370932d-080a-4bda-b533-34a77b0ab5ad)

- 글자와 여백이 모두 존재하는 경우 <br>
픽셀값을 이용하여 여백을 모두 지워주고 난 뒤, 글자가 모두 같은 위치에 있었기 때문에 고정좌표로 두 번째 사진과 같이 초록색 영역을 설정함. 이후 사진에 맞는 배경색을 가지고 오기 위해서 파란색 영역의 픽셀값을 평균내어 초록색 영역을 덮음. <br>
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/f65d2744-e427-4c09-bfd7-15f96c1a0e34)


- 여백이 있는 경우 <br>
이미지마다 여백의 크기가 달라, 이미지의 center를 기준으로 horizontal line과 vertical line을 슬라이스 한 뒤, <br>
반복문을 돌려 gray color가 들어오기 시작하는 좌표를 찾아 이미지를 자름 <br>
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/afbe0efe-5c51-4d68-9e3c-2ef80c8285e6)
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/ffb52686-f7f4-4313-a6b4-50a26ce3769f)
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/12454466-b96f-455b-862a-8d7a4d4c77ed)


# 모델 예측 방식
1) Binary classification으로 Aneurysm을 예측하고, 임계점을 넘어 가면 그 행(row)는 위치 label을 모두 0으로 대체 - 임계점은 train.csv에 대한 Aneurysm의 예측값의 중앙값
2) 임계점을 넘는 행은 Multi label classification으로 각 위치에 대한 값을 예측하고, 임계점을 두어 1 또는 0으로 대체 - 임계점은 train.csv에 대한 각 위치 예측값의 90백분위수들 (21개)

* Aneurysm 예측값의 형태는 9016x1 (한 사람 당 8장 이미지에 대한 모든 Aneurysm 예측) 이므로,
anterior, posterior 각각 4장에 대하여 평균 2개를 구하고, 그 중 가장 큰 값으로 Aneurysm 값 결정


# Model
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/57600112-1e36-4d0b-9d0c-9761659d3bae)
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

![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/92a87a45-0556-48c4-8a6b-628cd4ffd98d)
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/82bc40ae-058c-491d-a4df-7bcd58ecea53)


# etc
<strong>전처리</strong>
1. Rule-base Crop을 이용한 학습 이미지 전처리 <br>
방식 : 동맥류 양성 레이블에 따라 이미지를 해당 위치 기준으로 Rule-base Crop해 학습하고자 함 <br>
이유 : 고화질의 이미지를 Resize 하는 과정에서 이미지의 화질이 저하되고, 이는 Feature loss로 이어질 것이라 판단 <br>
결과/분석 : 뇌동맥류가 특정위치에 2개 이상 있더라도 위치정보에는1로 표기되어 있어 정확한 위치를 특정하지 못해 진행하지 못함 <br>

2. 이미지 데이터를 위치/각도/방향에 따라 분할한 뒤 각각 학습 <br>
방식 : 위치, 각도, 방향에 따라 이미지셋을 분할 한 뒤, 그 이미지에서만 볼 수 있는 레이블만 이용해 따로따로 학습하고자 함 <br>
(Ex : LI-A 이미지 + LI-A에서만 보이는 레이블만 학습한 모델 등) <br>
이유 : 위치,각도,방향이 같은 이미지들은 비슷한 형태를 띄고, 판단해야 할 레이블 수도 적어 학습에 유리할 것으로 판단. <br>
weighted sampler + weightes classes 모두 이용 <br>
결과/분석 : 데이터 분할 시 양/음성 데이터 간 불균형에 영향을 더 많이 받아 데이터가 적은 클래스의 경우 예측을 제대로 하지 못함. <br>
weight 기법들로인한 정확도 증가 미미

<strong>Self-Supervised Learning</strong>
1. Autoencoder - image classification 연결
방식 : 데이터셋 label에 있어 불균형이 있었는데, 불균형에 영향을 적게 받으며 이미지의 특성을 학습하는 autoencoder의 특성을 이용하였다. <br>
원리 : 입력을 저차원의 특성 벡터로 인코딩한다. 그 후 인코더에서 생성된 잠재 표현을 사용하여 원본 데이터를 복구하는데, 이렇게 재구성한 이미지가 원본 이미지와 비슷하도록 loss를 줄여나간다. <br>
그 결과로 재구성된 이미지는 원본이미지와 유사하되, 특성을 학습하게 되는 원리이다. 200번의 epoch를 돌린 후, path에 있던 이미지들을 autoencoder의 pt파일을 통과하도록 하여 학습된 특성을 강조하도록 하였고, <br>
그 후 binary classification으로 label이 존재하는 supervised learning을 진행하였다. <br>
supervised learning : 지도학습을 할 때는 처음에는 전체 데이터에 대해 진행하였는데, loss가 떠도
불균형이 너무 심했기에 의미가 없었고, 두 가지 시도를 하였다.
label의 개수를 어느정도 맞추기 위해 많은 개수를 가지는 label을 다운샘플링(랜덤으로 고르고 나머지는 버림)
으로 했지만 데이터의 개수가 너무 작았음, weighted sampling을 하였을 때도 다운샘플링과 결과는
비슷했다.

2. Contrastive Learning
- 이미지 데이터셋에서 라벨 불균형을 극복하기 위해 Contrastive Learning을 적용하였다. 이미지는 사전
훈련된 ResNet50 모델을 통해 특성을 추출하고, 이를 기반으로 같은 클래스의 이미지는 가까운 임베딩
공간으로, 다른 클래스의 이미지는 멀리 놓음으로써 유사성을 학습한다. 이 과정은 라벨 불균형이 덜 영향을
미치며, 각 이미지의 특성을 더 정확하게 반영할 수 있다.

<strong>ResNet18</strong>(fc-layer → conv layer) <br>
일반적인 image classification을 하는 CNN모델들은 last layer가 fully connected layer(fc layer)로 이루어져 있음. <br>
fc layer는 픽셀의 위치 정보가 사라져 segment에 취약함. fc layer를 convlayer로 바꿔 전처리와 augmentation을 하지 않은 데이터셋을 이용하여, ResNet18 모델을 학습시킴. 결과는 모든 에폭에서 같은 53.68%(f1 score)가 나옴. <br>
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/9f60b60f-4f6f-41ac-b021-f32447a86749)

<strong>DenseNet</strong> <br>
DenseNet은 ResNet의 skip connection을 업그레이드 시킨 버전으로, 주어진 이미지의 feature를 파악하는 것이 가장 중요하다고 생각함. feature extraction 부분에서 성능이 높은 전처리와 augmentation을 하지 않은 데이터셋을 이용하여, DenseNet을 학습시킴.
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/8d6a20c2-8b0f-471d-9487-f2175d75b407)

<strong>Grad-Cam</strong> <br>
여러 ResNet, DenseNet, VGG16 등 여러 CNN 모델을 돌려 보았지만, 이 모델이 이미지의 어떤 부분을
토대로 이러한 판단을 내렸는지 알 수 없었음. 또한, image augmentation을 하기 위해서는 이미지의 어떠한
부분이 중요한 지 알아야 했음. 설명 가능한 인공지능(explainable AI)의 기법 중 하나인 Grad-Cam을
적용하여 이미지 분석을 시도함. 전처리를 통해 글자와 여백을 제거하고 난 이미지를 이용하였고, augmentation은 하지 않았음. <br>
ResNet18모델이 판단의 기준으로 이용한 부분을 히트맵으로 볼 수 있었으며, 모델의 정확도나 정답을 모르는
상태에서 Grad-Cam을 적용하였기 때문에 결과가 나왔지만 정확한 분석을 할 수 없었음. <br>
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/df6a2bb5-2efd-476e-8871-a1934169d5e7)
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/752b5bcb-5a2a-41ef-b2bd-f5249714aa27)
![image](https://github.com/pmy02/Cerebral-Aneurysm-Classification/assets/62882579/192eabd5-d7ea-4467-83f0-03e074233926)


