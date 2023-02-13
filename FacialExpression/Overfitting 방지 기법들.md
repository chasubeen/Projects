# **0. Reference**
- [기술블로그_과적합 방지를 통한 모델 성능 개선](https://yeong-jin-data-blog.tistory.com/entry/%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98-%EC%8A%A4%ED%84%B0%EB%94%94-%EA%B3%BC%EC%A0%81%ED%95%A9-%EB%B0%A9%EC%A7%80%EB%A5%BC-%ED%86%B5%ED%95%9C-%EB%AA%A8%EB%8D%B8-%EC%84%B1%EB%8A%A5-%EA%B0%9C%EC%84%A0-%EB%B0%A9%EB%B2%95)  
- [교란 라벨링 관련 논문](https://arxiv.org/abs/1605.00055)

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
  - 파이토치에서는 ```.eval()``` 를 통해서 원래 전체 모델을 사용할 수 있음

# **5. 배치 정규화(Batch Normalization)**
- 기울기 소멸(gradient vanishing)이나 기울기 폭발(gradient exploding)과 같은 문제를 해결하기 위한 방법
- 일반적으로 기울기 소멸이나 폭발 문제를 해결하기 위해 손실 함수로 ```ReLu```를 사용하거나 초깃값 튜닝, 학습률(learning rate) 등을 조정
- 분산된 분포를 정규분포로 만들기 위해 표준화와 유사한 방식을 미니 배치에 적용하여 **평균 = 0, 표준편차 = 1**로 유지되도록 함
- 단계>
  1) 미니 배치 평균 구하기
  2) 미니 배치의 분산, 표준편차 구하기
  3) 정규화 수행
  4) 스케일(scale) 조정(데이터 분포 조정)
- 장점: 매 단계마다 활성화 함수를 거치면서 데이터셋 분포를 **일정**하게 유지시킬 수 있음 => 속도 향상
- 단점
  - 배치 크기가 작을 때는 정규화 값이 기존 값과 다른 방향으로 훈련될 수 있음
    - 분산이 0인 경우 정규화 자체가 수행되지 않는 경우가 생길 수 있음
  - RNN의 경우 네트워크 계층별로 미니 정규화를 적용해야 함 -> 모델이 더 복잡해지면서 비효율적일 수 있음
- 코드
```Python
torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
```

# **6. 교란 라벨 (Disturb Label) / 교란 값(Disturb Value)**
- 분류(classification) 문제에서 일정 비율의 라벨을 의도적으로 **잘못된** 라벨로 만들어서 학습을 방해하는 방식
  - 단순한 방식이지만 과적합을 효과적으로 막을 수 있음
- 손실층(loss layer)에 규제를 두는 방식
- 코드  
**1. DisturbLabel 객체 정의**  
```Python
class DisturbLabel(torch.nn.Module):
    def __init__(self, alpha, num_classes): 
    #alpha: 교란 라벨로 처리할 비율, num_classes: 데이터의 클래스 개수
        super(DisturbLabel, self).__init__()
        self.alpha = alpha
        self.C = num_classes
        self.p_c = (1 - ((self.C - 1) / self.C) * (alpha / 100)) # 실제 라벨을 뽑을 확률
        self.p_i = (1-self.p_c)/(self.C-1) # 나머지
 
    def forward(self, y):
        y_tensor = y.type(torch.LongTensor).view(-1, 1)      
        depth = self.C
        y_one_hot = torch.ones(y_tensor.size()[0], depth) * self.p_i
        y_one_hot.scatter_(1, y_tensor, self.p_c)
        y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,))) # create disturbed labels    
        distribution = torch.distributions.OneHotCategorical(y_one_hot) # sample from Multinoulli distribution
        y_disturbed = distribution.sample()
        y_disturbed = y_disturbed.max(dim=1)[1]
        return y_disturbed
```
  
**2. 교란 라벨 추가 & 학습 진행**  
```Python
disturblabels = DisturbLabel(alpha = 30, num_classes = 10) # 교란 라벨 생성
 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=1e-3)
loss_ = [] # 그래프를 그리기 위한 loss 저장용 리스트 
n = len(trainloader) # 배치 개수
 
for epoch in range(50): 
    running_loss = 0.0
    for data in trainloader:
        inputs, labels = data[0].to(device), data[1].to(device) # 배치 데이터 
        optimizer.zero_grad()
        
        outputs = resnet(inputs) # 예측값 산출 
        labels = disturblabels(labels).to(device) # 기존 라벨 -> 교란 라벨
        
        loss = criterion(outputs, labels) # 손실함수 계산
        loss.backward() # 손실함수 기준으로 역전파 선언
        optimizer.step() # 가중치 최적화
        running_loss += loss.item()
 
    loss_.append(running_loss / n)    
    print('[%d] loss: %.3f' %(epoch + 1, running_loss / n))
```

# **7. 라벨 스무딩(Label Smoothing)**
- 일반적인 분류 문제에서는 **softmax**나 **sigmoid** 함수를 통해서 0 또는 1의 값을 예측
- CrossEntropyLoss를 계산할 때, 실제값을 0 or 1이 아닌 0.2 or 0.8로 구성해서 과적합을 방지하는 방식
- 라벨 1을 예측 시 확률 값이 0.7로 나타나면 원래는 1로 나오도록 학습이 진행됨 
  - 이때 기준을 0.8로 낮추면 보다 적게 실제값과 가까워지는 방향으로 학습하고, 이를 통해서 과적합을 완화할 수 있음
- 파이토치에서 제공하는 nn.CrossEntropyLoss() 함수는 실제 라벨의 원-핫 벡터를 입력받을 수 없다.
  - 따라서 라벨 스무딩을 적용하려면 One - hot 벡터를 사용할 수 있도록 별도로 손실 함수를 정의해야 함
- 코드
```Python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing = 0.0, dim = -1):
    # classes: 데이터셋의 클래스 개수, smoothing: 스무딩 비율, dim: 차원
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
 
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim) # CrossEntropy 부분의 log softmax 미리 계산하기
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred) # 예측값과 동일한 크기의 영(zero) 텐서 만들기
            true_dist.fill_(self.smoothing / (self.cls - 1)) # alpha/(K-1)을 만들어 줌(alpha/K로 할 수도 있음)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) # (1-alpha)y + alpha/(K-1)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim)) # CrossEntropyLoss 계산
```

```Python
criterion = LabelSmoothingLoss(classes=10, smoothing=0.2)
optimizer = optim.Adam(resnet.parameters(), lr=1e-3)
```








