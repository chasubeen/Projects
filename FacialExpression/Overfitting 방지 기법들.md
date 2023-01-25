# **0. Reference**
[기술블로그_과적합 방지를 통한 모델 성능 개선](https://yeong-jin-data-blog.tistory.com/entry/%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98-%EC%8A%A4%ED%84%B0%EB%94%94-%EA%B3%BC%EC%A0%81%ED%95%A9-%EB%B0%A9%EC%A7%80%EB%A5%BC-%ED%86%B5%ED%95%9C-%EB%AA%A8%EB%8D%B8-%EC%84%B1%EB%8A%A5-%EA%B0%9C%EC%84%A0-%EB%B0%A9%EB%B2%95)

# **1. 데이터 증식(Data Augmentation)**
- 학습에 필요한 추가 데이터 수집이 어려운 경우 기존 데이터를 증식하는 방법
- 구체적인 방법으로는 통계적 기법, 단순 변형, 생성 모델 사용 등이 있음

### **1-1.  torchvision.transforms**
- 파이토치에서 공식적으로 제공하는 모듈
- 먼저 **DataLoader**를 정의한 후, DataLoader 클래스의 **__getitem__** 메소드에서 transform 호출
  - DataLoader 클래스는 **__getitem__** 메소드를 통해 이미지를 불러온 후 데이터 augmentation을 진행 
- 이후 transform 파라미터에 저장되어 있는 augmentation 규칙에 따라 augmentation이 진행됨
- 코드
  - [Reference](https://pseudo-lab.github.io/Tutorial-Book/chapters/object-detection/Ch3-preprocessing.html) 

**1) DataLoader 정의**  
```Python
from PIL import Image
import cv2
import numpy as np
import time
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations
import albumentations.pytorch
from matplotlib import pyplot as plt
import os
import random

class TorchvisionMaskDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.imgs = list(sorted(os.listdir(self.path)))
        self.transform = transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        file_image = self.imgs[idx]
        file_label = self.imgs[idx][:-3] + 'xml'
        img_path = os.path.join(self.path, file_image)
        
        if 'test' in self.path:
            label_path = os.path.join("test_annotations/", file_label)
        else:
            label_path = os.path.join("annotations/", file_label)

        img = Image.open(img_path).convert("RGB")
        
        target = generate_target(label_path)
        
        if self.transform:
            img = self.transform(img)

        return img, target
```

**2) transform 정의**  
```Python  
torchvision_transform = transforms.Compose([
    transforms.Resize((300, 300)), 
    transforms.RandomCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomHorizontalFlip(p = 1),
    transforms.ToTensor(),
])
```
- Resize(W,H): 이미지 크기 조정
- RandomCrop(size): 사진을 임의로 size 크기로 자르기
- ColorJitter(brightness, contrast, saturation, hue): 이미지의 밝기(brightness), 대비(contrast), 채도(saturation), 색조(hue) 변형
- RandomHorizontalFlip(p): 정의한 p의 확률로 좌우 반전

**3) 활용**  
- 데이터를 불러올 때 transform 수행  
```Python
torchvision_dataset = TorchvisionMaskDataset(
    path = 'images/',
    transform = torchvision_transform
) 
```

- 객체 탐지용 모델 구축을 위한 이미지 augmentation 기능은 albumentations에서만 제공한다는 단점이 존재  
(사실 우리 프로젝트와는 상관이 없어 그래서..^-^)


### **1-2. albumentations 모듈**
- OpenCV와 같은 오픈 소스 컴퓨터 비젼 라이브러리를 최적화한 라이브러리
- 다른 라이브러리보다 더 빠른 처리 속도 및 기타 기능을 제공


# **2. 조기 종료(Early Stopping)**

# **3. L2 정규화(L2 규제)**
- 규제(Regularization): 학습이 과대적합되는 것을 방지하고자 일종의 penalty를 부여하는 것

### **3-1.L2 정규화(L2 규제)**
- 각 가중치 제곱의 합에 규제 강도를 곱한 값($Error = MSE + α𝑤^2$)
- 원형의 경계를 만들어서 학습 데이터셋의 최적 지점인 w* 에 도달하지 못하게 하고 경계 내부의 v* 까지만 도달할 수 있도록 하는 방식
- Optimizer로 Adam을 사용할 경우 **weight_decay** 파라미터를 추가할 수 있음
  - 값이 클수록 규제 강도가 강한 것을 의미 -> 가중치가 더 많이 감소됨
  - 릿지(Ridge) 모델에 적용된다.
  - 코드
  ```Python
  optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-3)
  ```

### **3-2. L1 정규화(L1 규제)**
- 가중치의 합을 더한 값에 규제 강도를 곱하여 오차에 더한 값($Error = MSE + α|w|$)
- 어떤 가중치는 실제로 0이 된다. => 모델에서 완전히 제외되는 특성이 발생할 수 있음
- 라쏘(Lasso) 모델에 적용된다.


# **4. 드롭아웃(Dropout)**
- 훈련(train)할 때 **일정 비율**의 뉴런만 사용하고, 나머지 뉴런에 해당하는 가중치는 업데이트하지 않는 방법
  - 노드를 임의로 끄면서 학습하는 방법
  - 은닉층(hidden layer)에 배치된 노드 중 일부를 임의로 끄면서 학습
- 학습 데이터에 대한 과적합(overfitting)을 억제할 수 있음
- 훈련 시간이 길어지는 단점이 존재
- 출력층에서는 예측값을 산출해야 하기 때문에 드롭아웃을 사용해서는 안되고, **은닉층**에 대해서만 사용해야 함
- 모델 평가 단계에서는 드롭아웃을 실시하지 않은 모델로 평가해야 함
  - 파이토치에서는 .eval() 를 통해서 원래 전체 모델을 사용할 수 있음

# **5. 배치 정규화(Batch Normalization)**
- 
