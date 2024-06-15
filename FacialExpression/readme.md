# **😀😢감정 인식 프로젝트😠😨 (2022.11 ~ 2023.02)**

### **1️⃣ 프로젝트 소개 & 기획 의도**
사람의 감정은 단 하나의 감정으로 분류하기엔 어렵습니다. 예를 들어 ‘100% 행복’한 표정이라는 것은 존재하기 힘듭니다. 아무리 행복한 표정일지라도 사람의 표정에는 동시에 슬픔, 분노, 두려움 등 다양한 감정들이 표현되기 마련이기 때문입니다. 하지만, 여러 경험들을 통해 특정한 얼굴 표현이 어떠한 감정을 나타내는지를 짐작하는 것은 가능할 것입니다. 

연구에 따르면 의사소통에서 말이나 대화 등 언어적 소통이 차지하는 부분은 **7%** 남짓이고, 비언어적 표현이 **90% 이상**을 차지합니다.  
**(메라비언의 법칙)**  
이때, 표정 등의 시각적 표현이 차지하는 부분은 **55%** 로, 감정 표현의 파악은 의사소통에서 상당히 중요한 부분을 차지합니다. 

따라서, 감정 표현이 드러나는 사진을 감정에 따라 7가지의 카테고리로 분류하고, 많은 학습을 통해 해당 표정이 어떠한 감정을 드러내고 있는지를 분류해 낼 수 있는 **분류 모델**을 만드는 것을 목표로 하였습니다.

- - -

### **2️⃣ Data Description**
- **활용 데이터 셋**
  - [Facial Expression Dataset](https://www.kaggle.com/datasets/aadityasinghal/facial-expression-dataset)
- **데이터 살펴보기**
  - train, test dataset으로 이미 분리가 되어있는 상태
    - 각각의 데이터는 다시 7개의 감정(angry, disgust, fear, happy, neutral, sad, surprise)으로 분류되어 있음
    - disgust의 경우 다른 데이터에 비해 개수가 매우 적음 -> **Data Augmentation** 적용, **가중 손실 함수** 활용
  - 각 사진의 크기는 48 * 48 size의 흑백 사진
  - 사진들의 이름이 제각각임 -> 일정한 형식으로 통일하기 위해 전처리 수행
  - 오분류된 사진, 잘못된 사진(ex. 캐릭터 사진 등) 전처리

### **3️⃣ 전처리**
**1) 데이터 재분배**
- train: valid: test = 8:1:1이 되도록 데이터를 재분할 후, 각각의 폴더 경로에 저장

**[데이터 구조 변경 전]** 

<img src = "https://user-images.githubusercontent.com/98953721/209521079-a34e5197-37c4-40ea-a4a6-2b97d34f0d29.JPG" width = 500 height = 150>  


**[데이터 구조 변경 후]** 

<img src = "https://user-images.githubusercontent.com/98953721/209520743-656a5800-8217-4165-91b3-7cc69a969e51.JPG" width = 500 height = 120>  

<img src = "https://user-images.githubusercontent.com/98953721/209521143-87391b68-75f6-4586-bb45-0ef77745bb75.JPG" width = 500 height = 150>  

**2) 파일 이름 재정의**
- **감정_번호** 형식으로 통일
<img src = "https://user-images.githubusercontent.com/98953721/209521622-47e2f7b8-fd6f-4645-b063-0a37d82fada1.png" width = 200 height = 400>

* 이후 데이터를 효율적으로 불러오기 위해 train 데이터의 경우 csv 파일로 저장


### **4️⃣ 모델링**
- 프로젝트 목표: 7개의 카테고리로 분류된 감정들 중 **가장 적합한** 감정을 예측하는 모델 생성  
  -> **분류(Classification)** 문제, **1개**의 정답  
- 활용 모듈: Pytorch(version: 1.12.1 + cu116)
- 전이 학습(transfer learning) 모델 활용
- 활용한 분류 모델  
  [⚙️VGG19](https://github.com/chasubeen/Projects/tree/main/FacialExpression/VGG19)    
  [⚙️ResNet50](https://github.com/chasubeen/Projects/tree/main/FacialExpression/ResNet)    
  - 개별 분류 모델에 각각 데이터 셋을 학습
  - learning rate, batch size,optimizer 등의 **parameter**들을 조정하며 여러 조건들에서 학습 진행
- 이후 **앙상블 기법**을 적용하여 여러 분류기를 합쳐 성능 향상

### **5️⃣ 최종 결과**

### **6️⃣ 의의/ 한계**
