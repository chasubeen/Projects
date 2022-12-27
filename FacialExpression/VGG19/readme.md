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
  - lr_scheduler, early stopping 적용
```Python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience = 5, factor = 0.1, 
                                                       min_lr = 1e-10,verbose = True)  # lr scheduling
early_stopping = EarlyStopping(patience = 10, verbose = False) # 조기 종료(사용자 정의 모듈)
```
- batch size: 64
- Epoch: 100
- **손실함수 & 활성화 함수** 튜닝

|   |**sgd_ver1**|**sgd_ver2**|**sgd_ver3**|
|------|-------|-------|-------|
|**손실 함수**|CrossEntropyLoss|**가중** CrossEntropyLoss|**가중** CrossEntropyLoss|
|**활성화 함수**|softmax|softmax|**log** softmax|
|**Best Acc**|0.6115|0.6133|**0.6654**|


✅ **Case 2**
- Case 1에서 **batch size**만 변경(64 -> 128)
- batch size가 **클** 때
  - 한 번 학습할 때 많은 데이터로 학습
  - 빠른 학습/수렴 속도 -> local optima에 빠질 확률이 작음
  - 작은 배치에 비해 과적합 위험성 증가(batch가 크면 계산되는 loss값의 편차가 작으므로)
- batch size가 **작을** 때
  - 1 epoch 당 iteration이 크기 때문에 step이 많아짐
  - 작은 데이터로 학습 -> loss의 분산이 커서 정규화 효과가 있음, 조금 더 다양하고 예리하게 학습할 수 있음
  - 긴 학습시간, 많은 step 수로 인해 local minima에 빠질 위험성 증가

|   |**sgd_ver4**|**sgd_ver5**|**sgd_ver6**|
|------|-------|-------|-------|
|**손실 함수**|CrossEntropyLoss|**가중** CrossEntropyLoss|**가중** CrossEntropyLoss|
|**활성화 함수**|softmax|softmax|**log** softmax|
|**Best Acc**|0.6427|**0.6622**|0.6606|

- - -

✅ **Case 3**
- Optimizer 변경(SGD -> **Adam**)
- 실패한 모델들 
  - 손실함수: CrossEntropyLoss
  - 활성화 함수: softmax
  - lr_scheduler, early stopping 적용
 
|   |**adam_ver1**|**adam_ver2**|**adam_ver4**|
|------|-------|-------|-------|
|**batch**|64|128|64|
|**초기 lr**|0.0005|0.001|0.0001|
|**min_lr**|1e-7|1e-8|1e-10|
|**Epoch**|100|200|200|
|**Best Acc**|0.24xx(중단)|(중단)|(중단)|

- 실패 원인 분석: batch size와 learning rate의 관계
<img src = "https://user-images.githubusercontent.com/98953721/209615216-0c5679ab-11db-438d-846d-06bc2f2d6a98.png" width = 300 height = 300>

-> 일반적으로 learning rate와 batch size는 **양의 상관관계**를 보인다.  
-> learning rate를 **줄인** 후 다시 모델링 진행  

✅ **Case 4**
- Case 3에서 **learning rate & batch size** 튜닝 -> 적절한 조합 탐색
- 손실함수: CrossEntropyLoss
- 활성화 함수: softmax
- lr_scheduler, early stopping 적용
- Epoch: 200

|   |**adam_ver3**|**adam_ver5**|**adam_ver6**|
|------|-------|-------|-------|
|**batch**|128|128|64|
|**초기 lr**|1e-4|1e-5|1e-5|
|**min_lr**|1e-10|1e-12|1e-12|
|**Best Acc**|0.6137|**0.6344**|0.6249|

✅ **Case 5**
- Case 4에서 **손실 함수 & 활성화 함수** 튜닝
- batch size: 128
- learning rate
  - 초기: 1e-5
  - lr_scheduler, early stopping 적용(min_lr: 1e-12)
- Epoch: 200 

|   |**adam_ver7**|**adam_ver8**|
|------|-------|-------|
|**손실 함수**|**가중** CrossEntropyLoss|**가중** CrossEntropyLoss|
|**활성화 함수**|softmax|**log** softmax|
|**Best Acc**|0.6238|**0.6444**|


### **#️⃣ Reference**
- [VGG19 관련 논문(Very Deep Convolutional)](https://arxiv.org/abs/1409.1556)  
- [Learning rate & batch size best 조합 찾기(기술 블로그)](https://inhovation97.tistory.com/32)  
- [learning rate& batch size 관련 논문](https://www.sciencedirect.com/science/article/pii/S2405959519303455#fig2)  
