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


# **2. 
