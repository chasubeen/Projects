# **VGG19 모델링 결과**

### **1️⃣Baseline**
- **VGG19?**
  - 이미지 분류를 위한 CNN 모델 중 하나
  - 몇 개의 층(layer)로 구성되어 있는지에 따라, 16개 층으로 구성된 경우 VGG16, 19개 층으로 구성된 경우 VGG19로 구분
  - VGGNet 연구의 핵심: **네트워크의 깊이**가 성능에 미치는 영향
    - 합성곱 필터의 커널 사이즈를 최소 사이즈인 **3x3**으로 설정

- **구조**
<img src = "https://user-images.githubusercontent.com/98953721/209606535-39b86461-30c1-4bdc-b52f-75ac868ccb1a.png" width = 500 height = 150>

<img src = "https://user-images.githubusercontent.com/98953721/209606422-8ac1899d-427f-47ec-bf24-740bea388192.png" width = 800 height = 1000>

- 데이터 구조에 맞게 **head 부분** 수정
<img src = "https://user-images.githubusercontent.com/98953721/209606922-f343a3ac-be48-4fb0-aba0-e4500a0b8cf0.png" width = 800 height = 150>

### **2️⃣ 성능 최적화 결과**
✅ **Case 1**
- Optimizer: SGD(Stochastic Gradient Descent/ 확률적 경사 하강법)
```Python
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2, momentum = 0.9)
```
- learning rate
  - 초기: 1e-2
  - lr_scheduler와 early stopping 적용
```Python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience = 5, factor = 0.1, 
                                                       min_lr = 1e-10,verbose = True)  # lr scheduling
early_stopping = EarlyStopping(patience = 10, verbose = False) # 조기 종료(사용자 정의 모듈)
```
- batch size: 64
- Epoch: 100
- **손실함수 & 활성화 함수** 튜닝

|   |sgd_ver1|sgd_ver2|sgd_ver3|
|------|-----|-----|-----|
|손실 함수|CrossEntropyLoss|**가중** CrossEntropyLoss|**가중** CrossEntropyLoss|
|활성화 함수|softmax|softmax|**log**softmax|
|Best Acc|0.6115|0.6133|**0.6654**|



